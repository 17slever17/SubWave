from __future__ import annotations

import copy
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ru": "Russian",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ko": "Korean",
    "pt": "Portuguese",
    "zh": "Chinese",
}


FACTORY_PROMPTS: dict[str, dict[str, str]] = {
    "ja_to_en": {
        "name": "Japanese to English",
        "source_language": "ja",
        "target_language": "English",
        "template": (
            "TASK: Translate Japanese subtitles into English.\n"
            "STYLE: friendly.\n"
            "Translate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only. "
            "Preserve meaning, tone, slang, profanity, uncertainty, repetitions and incomplete speech. "
            "Return only the final translation without labels or commentary."
        ),
    },
}
DEFAULT_PROMPT_ID = "ja_to_en"


class PromptStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def ensure_exists(self) -> None:
        with self._lock:
            if self.path.exists():
                return
            logger.info("[prompts.ensure] creating default prompt store path=%s", self.path)
            self.save({"presets": copy.deepcopy(FACTORY_PROMPTS), "active": DEFAULT_PROMPT_ID})

    def load(self) -> dict[str, Any]:
        with self._lock:
            self.ensure_exists()
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except Exception:
                logger.exception("[prompts.load] failed path=%s; using factory defaults", self.path)
                data = {}
            presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
            merged = copy.deepcopy(FACTORY_PROMPTS)
            for key, value in presets.items():
                if isinstance(value, dict):
                    merged[str(key)] = {**merged.get(str(key), {}), **value}
            active = str(data.get("active") or DEFAULT_PROMPT_ID)
            if active not in merged:
                active = DEFAULT_PROMPT_ID
            return {"presets": merged, "active": active, "factory_ids": sorted(FACTORY_PROMPTS.keys())}

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            raw_presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
            presets = {str(key): value for key, value in raw_presets.items() if isinstance(value, dict)}
            names: set[str] = set()
            for preset in presets.values():
                if not isinstance(preset, dict):
                    continue
                normalized_name = _prompt_display_name(preset).casefold()
                if normalized_name and normalized_name in names:
                    raise ValueError(f"Prompt preset name already exists: {preset.get('name')}")
                names.add(normalized_name)
            active = str(data.get("active") or DEFAULT_PROMPT_ID)
            payload = {"active": active, "presets": presets}
            self._atomic_write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            )
            logger.info("[prompts.save] saved path=%s active=%s preset_count=%s", self.path, active, len(presets))
            return self.load()

    def delete(self, preset_id: str) -> dict[str, Any]:
        with self._lock:
            if preset_id in FACTORY_PROMPTS:
                raise ValueError("Factory prompt presets cannot be deleted")
            current = self.load()
            if preset_id not in current["presets"]:
                raise KeyError(preset_id)
            current["presets"].pop(preset_id)
            if current["active"] == preset_id:
                current["active"] = DEFAULT_PROMPT_ID
            logger.info("[prompts.delete] removed custom preset=%s", preset_id)
            return self.save(current)

    def reset(self, preset_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            current = self.load()
            if preset_id:
                if preset_id in FACTORY_PROMPTS:
                    current["presets"][preset_id] = copy.deepcopy(FACTORY_PROMPTS[preset_id])
                    logger.info("[prompts.reset] reset preset=%s", preset_id)
                else:
                    current["presets"].pop(preset_id, None)
                    logger.info("[prompts.reset] removed custom preset=%s", preset_id)
                return self.save(current)
            logger.info("[prompts.reset] reset all prompts")
            return self.save({"presets": copy.deepcopy(FACTORY_PROMPTS), "active": DEFAULT_PROMPT_ID})

    def _atomic_write_text(self, text: str) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get_prompt(self, preset_id: str | None, source_language: str, target_language: str) -> str:
        return self._get_prompt_field(
            preset_id,
            source_language,
            target_language,
            "template",
        )

    def _get_prompt_field(
        self,
        preset_id: str | None,
        source_language: str,
        target_language: str,
        field: str,
    ) -> str:
        data = self.load()
        resolved_id = preset_id or data.get("active") or self._infer_preset(source_language, target_language)
        preset = data["presets"].get(str(resolved_id))
        if not preset:
            inferred = self._infer_preset(source_language, target_language)
            logger.warning("[prompts.get] missing preset=%s; fallback=%s", resolved_id, inferred)
            preset = data["presets"].get(inferred, FACTORY_PROMPTS[DEFAULT_PROMPT_ID])
        template = str(preset.get(field, "")).strip()
        return (
            template
            .replace("{source_language}", source_language)
            .replace("{target_language}", target_language)
        )

    @staticmethod
    def _infer_preset(source_language: str, target_language: str) -> str:
        return DEFAULT_PROMPT_ID


def _prompt_display_name(preset: dict[str, Any]) -> str:
    source_code = str(preset.get("source_language", "")).strip().lower()
    source = LANGUAGE_NAMES.get(source_code, source_code.upper() or "Unknown")
    target_raw = str(preset.get("target_language", "")).strip()
    target = LANGUAGE_NAMES.get(target_raw.lower(), target_raw or "Unknown")
    direction = f"{source} to {target}"
    name = str(preset.get("name", "")).strip()
    return name or direction
