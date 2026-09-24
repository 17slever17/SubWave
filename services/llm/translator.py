import logging
import time

from pathlib import Path

from services.runtime.model_downloads import (
    should_use_translation_mtp,
    translation_mtp_path,
)
from services.runtime.paths import resolve_runtime_path
from services.filters.text import collapse_repeated_text, has_repeated_token_loop, text_preview
from services.llm.llama_server import NativeLlamaServer, get_runtime_backend
from services.llm.prompts import PromptStore


class LLMTranslator:
    _VULKAN_WARMUP_TIMEOUT_S = 180.0
    _VULKAN_WARMUP_MAX_TOKENS = 8

    def __init__(self, config):
        self.root_config = config
        self.config = getattr(config, "translation", config)
        self.logger = logging.getLogger(__name__)

        self.model_path = self._get_val("model_path")
        self.device = str(self._get_val("device", "auto"))
        self.mtp_enabled = bool(self._get_val("mtp_enabled", True))
        self.mtp_model_path = str(self._get_val("mtp_model_path", "auto"))
        self.mtp_n = max(1, int(self._get_val("mtp_n", 1)))
        self.llama_server_path = str(
            self._get_val(
                "llama_server_path",
                "bin/llama.cpp/llama-server.exe",
            )
        )
        resolved_server_path = NativeLlamaServer.resolve_server_path(
            self.llama_server_path
        )
        self.runtime_backend = get_runtime_backend(
            resolved_server_path or self.llama_server_path
        )
        if self.runtime_backend == "cpu" and self.device.strip().lower() in {
            "gpu",
            "cuda",
            "vulkan",
        }:
            self.logger.info(
                "Installed llama.cpp runtime is CPU-only; disabling GPU offload."
            )
            self.device = "cpu"
        self.context_subtitles = min(
            5, max(0, int(self._get_val("context_subtitles", 3)))
        )
        self.n_ctx = 544 + (self.context_subtitles * 160)
        self.n_batch = self._get_val("n_batch", 128)
        self.target_language = self._get_val("target_language", "Russian")
        self.prompt_preset = self._get_val("prompt_preset", "")
        self.source_language = self._get_root_val("language", "en")
        self.max_tokens = int(self._get_val("max_tokens", 128))
        realtime_latency_s = float(
            self._get_val("realtime_max_translation_latency_s", 3.0)
        )
        if self.runtime_backend == "cpu":
            realtime_latency_s = max(30.0, realtime_latency_s)
        self.request_timeout_s = (
            max(30.0, realtime_latency_s + 1.0)
            if self.device.casefold() == "cpu"
            else max(1.0, realtime_latency_s + 1.0)
        )
        self.repeat_penalty = float(self._get_val("repeat_penalty", 1.08))
        self.duplicate_similarity_threshold = float(
            self._get_val("duplicate_similarity_threshold", 0.90)
        )
        self.max_output_length_ratio = float(self._get_val("max_output_length_ratio", 5.0))
        self.max_output_extra_chars = int(self._get_val("max_output_extra_chars", 80))
        self.min_output_length_limit = int(self._get_val("min_output_length_limit", 120))
        self.max_output_chars = int(self._get_val("max_output_chars", 500))
        self.prompt_store = PromptStore(
            Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
        )
        self.llm = None
        self._context_reset_requested = False
        if not self.model_path:
            self.logger.error("Model path not found in config.")
            return
        resolved_model_path = resolve_runtime_path(self.model_path)
        if resolved_model_path and resolved_model_path.exists():
            self.model_path = str(resolved_model_path)

        self.logger.info("Loading translator model from %s...", self.model_path)
        configured_model_path = str(self._get_val("model_path", self.model_path))
        use_mtp = False
        if self.runtime_backend == "vulkan":
            self.logger.info("Automatic MTP is disabled for the Vulkan runtime.")
        else:
            use_mtp = should_use_translation_mtp(
                configured_model_path,
                self.mtp_model_path,
                self.device,
                self.mtp_enabled,
            )
        draft_path = None
        if use_mtp:
            candidate = translation_mtp_path(
                configured_model_path,
                self.mtp_model_path,
            )
            if candidate and candidate.is_file():
                draft_path = candidate
            else:
                self.logger.warning(
                    "MTP model is unavailable for %s; using native standard inference.",
                    configured_model_path,
                )
        elif self.mtp_enabled and self.runtime_backend != "vulkan":
            self.logger.info(
                "MTP is not compatible with model=%s on device=%s; using native standard inference.",
                configured_model_path,
                self.device,
            )
        else:
            self.logger.info("MTP is disabled; using native standard inference.")

        server_kwargs = {
            "server_path": self.llama_server_path,
            "model_path": self.model_path,
            "device": self.device,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "request_timeout_s": self.request_timeout_s,
        }
        if draft_path is not None:
            server_kwargs.update(
                mtp_model_path=str(draft_path),
                mtp_n=self.mtp_n,
            )
        try:
            self.llm = NativeLlamaServer(**server_kwargs)
            if self.runtime_backend == "vulkan":
                try:
                    self._warmup_vulkan_runtime()
                except Exception as exc:
                    self.logger.error(
                        "Vulkan translator warmup failed; disabling the translator "
                        "to avoid a cold first subtitle: %s",
                        exc,
                    )
                    try:
                        self.llm.close()
                    except Exception as close_exc:
                        self.logger.warning(
                            "Could not stop llama-server after Vulkan warmup failure: %s",
                            close_exc,
                        )
                    self.llm = None
                    return
            if draft_path is not None:
                self.logger.info(
                    "Translator model loaded with native MTP. "
                    "device=%s mtp_model=%s mtp_n=%s n_ctx=%s n_batch=%s",
                    self.device,
                    draft_path,
                    self.mtp_n,
                    self.n_ctx,
                    self.n_batch,
                )
            else:
                self.logger.info(
                    "Translator model loaded with native standard inference. "
                    "device=%s n_ctx=%s n_batch=%s",
                    self.device,
                    self.n_ctx,
                    self.n_batch,
                )
        except Exception as exc:
            self.logger.error("Failed to load native llama-server translator model: %s", exc)
            self.llm = None

    def _warmup_vulkan_runtime(self) -> None:
        synthetic_source = (
            "これは翻訳ランタイムの初期化確認です。"
            if self.source_language.strip().lower().startswith("ja")
            else "This is a synthetic translation runtime warmup check."
        )
        started_at = time.monotonic()
        self.logger.info(
            "Starting Vulkan translator warmup (timeout=%.0fs, max_tokens=%s).",
            self._VULKAN_WARMUP_TIMEOUT_S,
            min(self._VULKAN_WARMUP_MAX_TOKENS, max(1, self.max_tokens)),
        )
        response = self.llm.create_chat_completion(
            messages=self._build_messages(synthetic_source),
            max_tokens=min(self._VULKAN_WARMUP_MAX_TOKENS, max(1, self.max_tokens)),
            stop=[
                "[PREVIOUS_SOURCE]",
                "[PREVIOUS_TRANSLATION]",
                "[CURRENT_SOURCE]",
                "<thought>",
                "</think>",
            ],
            temperature=0.0,
            min_p=0.1,
            top_k=3,
            repeat_penalty=self.repeat_penalty,
            timeout=self._VULKAN_WARMUP_TIMEOUT_S,
        )
        if not response.get("choices"):
            raise RuntimeError("llama-server returned no warmup completion")
        self.logger.info(
            "Vulkan translator warmup completed in %.2fs.",
            time.monotonic() - started_at,
        )

    def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()

    def _get_val(self, key, default=None):
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _get_root_val(self, key, default=None):
        if isinstance(self.root_config, dict):
            return self.root_config.get(key, default)
        return getattr(self.root_config, key, default)

    def _build_system_prompt(self) -> str:
        return self.prompt_store.get_prompt(
            preset_id=self.prompt_preset,
            source_language=self.source_language,
            target_language=self.target_language,
        )

    def _build_messages(
        self,
        text: str,
        context_text: str = "",
        context_translation: str = "",
    ) -> list[dict[str, str]]:
        system_content = self._build_system_prompt()
        user_parts: list[str] = []
        if context_text:
            user_parts.append(f"[PREVIOUS_SOURCE]\n{context_text.strip()}")
        if context_translation:
            user_parts.append(f"[PREVIOUS_TRANSLATION]\n{context_translation.strip()}")
        user_parts.append(f"[CURRENT_SOURCE]\n{text.strip()}")
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def translate(
        self,
        text: str,
        context_text: str = "",
        context_translation: str = "",
    ) -> str:
        if not self.llm:
            return ""

        self._context_reset_requested = False
        try:
            generation_options = {
                "temperature": 0.0,
                "min_p": 0.1,
                "top_k": 3,
                "repeat_penalty": self.repeat_penalty,
            }
            inference_started_at = time.monotonic()
            response = self.llm.create_chat_completion(
                messages=self._build_messages(text, context_text, context_translation),
                max_tokens=self.max_tokens,
                stop=[
                    "[PREVIOUS_SOURCE]",
                    "[PREVIOUS_TRANSLATION]",
                    "[CURRENT_SOURCE]",
                    "<thought>",
                    "</think>",
                ],
                **generation_options,
            )
            inference_elapsed_s = time.monotonic() - inference_started_at
            if inference_elapsed_s >= 2.0:
                timings = response.get("timings") or {}
                self.logger.warning(
                    "Slow LLM request: total=%.2fs prompt_ms=%s predicted_ms=%s "
                    "predicted_tokens=%s predicted_per_second=%s",
                    inference_elapsed_s,
                    timings.get("prompt_ms", "n/a"),
                    timings.get("predicted_ms", "n/a"),
                    timings.get("predicted_n", "n/a"),
                    timings.get("predicted_per_second", "n/a"),
                )
            result = response["choices"][0]["message"]["content"].strip()

            prefixes = [
                "Translation:",
                "Russian:",
                "Result:",
                "Output:",
                "Перевод:",
                "Ответ:",
            ]
            for prefix in prefixes:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix) :].lstrip(": ").strip()
                    break
            return self._stabilize_output(
                source_text=text,
                result=result,
                finish_reason=response["choices"][0].get("finish_reason"),
            )
        except TimeoutError as exc:
            self.logger.error("Translation timed out: %s", exc)
            return ""
        except Exception as exc:
            self.logger.error("Translation failed: %s", exc)
            return ""

    def _stabilize_output(self, source_text: str, result: str, finish_reason: str | None = None) -> str:
        had_token_loop = has_repeated_token_loop(result)
        cleaned, collapsed = collapse_repeated_text(
            result,
            similarity_threshold=self.duplicate_similarity_threshold,
        )
        if collapsed or had_token_loop:
            self._context_reset_requested = True
            self.logger.warning(
                "Translation repetition detected; collapsed=%s finish_reason=%s chars=%s->%s",
                collapsed,
                finish_reason,
                len(result),
                len(cleaned),
            )

        if has_repeated_token_loop(cleaned):
            self.logger.error(
                "Skipping translation with unresolved token loop: %r",
                text_preview(cleaned),
            )
            return ""

        if finish_reason == "length" and not collapsed:
            self._context_reset_requested = True
            self.logger.error("Skipping translation truncated at max_tokens: chars=%s", len(cleaned))
            return ""

        source_length = max(1, len(source_text.strip()))
        allowed_length = max(
            self.min_output_length_limit,
            int(source_length * self.max_output_length_ratio) + self.max_output_extra_chars,
        )
        allowed_length = min(self.max_output_chars, allowed_length)
        if len(cleaned) > allowed_length:
            self._context_reset_requested = True
            self.logger.error(
                "Skipping abnormally long translation: source_chars=%s output_chars=%s limit=%s finish_reason=%s",
                source_length,
                len(cleaned),
                allowed_length,
                finish_reason,
            )
            return ""
        return cleaned

    def consume_context_reset_request(self) -> bool:
        requested = self._context_reset_requested
        self._context_reset_requested = False
        return requested
