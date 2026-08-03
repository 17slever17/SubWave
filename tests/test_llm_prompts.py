import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from configs.settings import Config, TranslationConfig
from services.llm.translator import LLMTranslator


class TestLLMPrompts(unittest.TestCase):
    def test_translation_message_construction(self):
        config = Config()
        config.language = "ja"
        config.translation = TranslationConfig(
            model_path="",
            target_language="English",
            prompt_preset="ja_to_en",
        )

        translator = LLMTranslator(config)
        messages = translator._build_messages(
            text="こんにちは、世界。",
            context_text="最初の字幕。\nあの変なケーブルについての前の文。",
            context_translation="First subtitle.\nPrevious sentence about the strange cable.",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Translate Japanese subtitles into English", messages[0]["content"])
        self.assertIn("STYLE: friendly", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("[PREVIOUS_SOURCE]", messages[1]["content"])
        self.assertIn("最初の字幕。\nあの変なケーブル", messages[1]["content"])
        self.assertIn("あの変なケーブルについての前の文。", messages[1]["content"])
        self.assertIn("[PREVIOUS_TRANSLATION]", messages[1]["content"])
        self.assertIn("First subtitle.\nPrevious sentence", messages[1]["content"])
        self.assertIn("Previous sentence about the strange cable.", messages[1]["content"])
        self.assertIn("[CURRENT_SOURCE]", messages[1]["content"])
        self.assertIn("こんにちは、世界。", messages[1]["content"])

    def test_translation_cpu_mode_disables_gpu_layers(self):
        config = Config()
        config.translation = TranslationConfig(
            model_path="",
            device="cpu",
            n_gpu_layers=-1,
        )
        translator = LLMTranslator(config)
        self.assertEqual(translator._resolve_n_gpu_layers(), 0)

    def test_translation_gpu_mode_repairs_stale_zero_layers(self):
        config = Config()
        config.translation = TranslationConfig(
            model_path="",
            device="gpu",
            n_gpu_layers=0,
        )
        translator = LLMTranslator(config)
        self.assertEqual(translator._resolve_n_gpu_layers(), -1)

    @patch("llama_cpp.Llama")
    def test_translate_gemma_can_use_gpu_layers(self, mock_llama):
        config = Config()
        config.translation = TranslationConfig(
            model_path="models/translate_gemma4_sub-E4B-Q4_K_XL.gguf",
            device="gpu",
            n_gpu_layers=-1,
        )

        translator = LLMTranslator(config)

        self.assertEqual(translator._resolve_n_gpu_layers(), -1)

    def test_translation_duplicate_is_collapsed_and_requests_context_reset(self):
        config = Config()
        config.translation = TranslationConfig(model_path="")
        translator = LLMTranslator(config)

        result = translator._stabilize_output(
            source_text="Thank you for coming.",
            result="Спасибо, что пришли. Спасибо, что пришли.",
        )

        self.assertEqual(result, "Спасибо, что пришли")
        self.assertTrue(translator.consume_context_reset_request())
        self.assertFalse(translator.consume_context_reset_request())

    def test_abnormally_long_translation_is_skipped(self):
        config = Config()
        config.translation = TranslationConfig(model_path="")
        translator = LLMTranslator(config)
        long_result = " ".join(f"слово{index}" for index in range(80))

        result = translator._stabilize_output(source_text="Hi", result=long_result)

        self.assertEqual(result, "")
        self.assertTrue(translator.consume_context_reset_request())

    def test_truncated_translation_is_skipped(self):
        config = Config()
        config.translation = TranslationConfig(model_path="")
        translator = LLMTranslator(config)

        result = translator._stabilize_output(
            source_text="A normal source sentence.",
            result="Незавершённый перевод",
            finish_reason="length",
        )

        self.assertEqual(result, "")
        self.assertTrue(translator.consume_context_reset_request())

    def test_native_mtp_is_used_when_matching_draft_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "translate_gemma4_sub-E4B-Q4_K_XL.gguf"
            draft = Path(temp_dir) / "mtp-gemma-4-E4B-it.gguf"
            model.touch()
            draft.touch()
            config = Config()
            config.translation = TranslationConfig(
                model_path=str(model),
                mtp_enabled=True,
                mtp_model_path=str(draft),
                mtp_n=2,
            )

            with (
                patch(
                    "services.llm.translator.NativeLlamaServer"
                ) as native,
                patch("llama_cpp.Llama") as fallback,
            ):
                translator = LLMTranslator(config)

            self.assertIs(translator.llm, native.return_value)
            native.assert_called_once()
            self.assertEqual(native.call_args.kwargs["mtp_model_path"], str(draft))
            self.assertEqual(native.call_args.kwargs["mtp_n"], 2)
            fallback.assert_not_called()

    def test_native_mtp_failure_falls_back_to_llama_cpp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "translate_gemma4_sub-E4B-Q4_K_XL.gguf"
            draft = Path(temp_dir) / "mtp-gemma-4-E4B-it.gguf"
            model.touch()
            draft.touch()
            config = Config()
            config.translation = TranslationConfig(
                model_path=str(model),
                mtp_enabled=True,
                mtp_model_path=str(draft),
            )

            with (
                patch(
                    "services.llm.translator.NativeLlamaServer",
                    side_effect=RuntimeError("unsupported draft"),
                ) as native,
                patch("llama_cpp.Llama") as fallback,
            ):
                translator = LLMTranslator(config)

            native.assert_called_once()
            fallback.assert_called_once()
            self.assertIs(translator.llm, fallback.return_value)

    def test_e2b_gpu_does_not_start_native_mtp_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "translate_gemma4_sub-E2B-Q4_K_XL.gguf"
            draft = Path(temp_dir) / "mtp-gemma-4-E2B-it.gguf"
            model.touch()
            draft.touch()
            config = Config()
            config.translation = TranslationConfig(
                model_path=str(model),
                device="gpu",
                mtp_enabled=True,
                mtp_model_path=str(draft),
            )

            with (
                patch("services.llm.translator.NativeLlamaServer") as native,
                patch("llama_cpp.Llama") as fallback,
            ):
                translator = LLMTranslator(config)

            native.assert_not_called()
            fallback.assert_called_once()
            self.assertIs(translator.llm, fallback.return_value)

    def test_custom_model_can_use_explicit_mtp_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "gemma-4-12b.gguf"
            draft = Path(temp_dir) / "gemma-4-12b-mtp.gguf"
            model.touch()
            draft.touch()
            config = Config()
            config.translation = TranslationConfig(
                model_path=str(model),
                device="gpu",
                mtp_enabled=True,
                mtp_model_path=str(draft),
            )

            with (
                patch("services.llm.translator.NativeLlamaServer") as native,
                patch("llama_cpp.Llama") as fallback,
            ):
                translator = LLMTranslator(config)

            native.assert_called_once()
            fallback.assert_not_called()
            self.assertIs(translator.llm, native.return_value)

    def test_disabling_mtp_never_starts_native_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "translate_gemma4_sub-E4B-Q4_K_XL.gguf"
            model.touch()
            config = Config()
            config.translation = TranslationConfig(
                model_path=str(model),
                mtp_enabled=False,
            )

            with (
                patch(
                    "services.llm.translator.NativeLlamaServer"
                ) as native,
                patch("llama_cpp.Llama") as fallback,
            ):
                translator = LLMTranslator(config)

            native.assert_not_called()
            fallback.assert_called_once()
            self.assertIs(translator.llm, fallback.return_value)

    def test_translate_uses_deterministic_options_and_strips_prefix(self):
        config = Config()
        config.translation = TranslationConfig(model_path="")
        translator = LLMTranslator(config)
        fake_llm = unittest.mock.MagicMock()
        fake_llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {"content": "Translation: Привет!"},
                    "finish_reason": "stop",
                }
            ]
        }
        translator.llm = fake_llm

        result = translator.translate("Hello!")

        self.assertEqual(result, "Привет!")
        kwargs = fake_llm.create_chat_completion.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["top_k"], 3)
        self.assertEqual(kwargs["min_p"], 0.1)

    def test_translate_drops_timed_out_native_request(self):
        config = Config()
        config.translation = TranslationConfig(model_path="")
        translator = LLMTranslator(config)
        translator.llm = unittest.mock.MagicMock()
        translator.llm.create_chat_completion.side_effect = TimeoutError(
            "llama-server request exceeded 4.0s"
        )

        self.assertEqual(translator.translate("Hello!"), "")


if __name__ == "__main__":
    unittest.main()
