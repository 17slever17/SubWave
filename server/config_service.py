from __future__ import annotations

import copy
import hashlib
import logging
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from configs.settings import Config, context_window_for_subtitles

logger = logging.getLogger(__name__)


def default_config_dict() -> dict[str, Any]:
    return asdict(Config())


class ConfigService:
    def __init__(self, path: Path, reset_path: Path | None = None):
        self.path = Path(path)
        self.reset_path = Path(reset_path) if reset_path else self.path.with_name("config.reset.yaml")
        self._lock = threading.RLock()

    def read(self) -> dict[str, Any]:
        with self._lock:
            defaults = default_config_dict()
            if not self.path.exists():
                logger.warning("[config.read] missing path=%s; using defaults", self.path)
                return self._with_meta(defaults)
            try:
                loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except Exception:
                logger.exception("[config.read] failed path=%s; using defaults", self.path)
                loaded = {}
            if not isinstance(loaded, dict):
                logger.warning(
                    "[config.read] expected mapping path=%s; using defaults",
                    self.path,
                )
                loaded = {}
            merged = self._deep_merge(defaults, loaded)
            self._normalize_translation_context(merged)
            return self._with_meta(merged)

    def write(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self._without_meta(data)
            self._normalize_translation_context(payload)
            self._atomic_write_text(
                self.path,
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            )
            logger.info("[config.write] path=%s hash=%s", self.path, self.hash(payload))
            return self.read()

    def patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self._without_meta(self.read())
            changed_keys = sorted(patch.keys())
            next_config = self._deep_merge(current, patch)
            self.write(next_config)
            logger.info("[config.patch] changed_keys=%s hash=%s", changed_keys, self.hash(next_config))
            return self.read()

    def preview_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self._without_meta(self.read())
            result = self._deep_merge(current, patch)
            self._normalize_translation_context(result)
            return result

    def reset_section(self, section: str) -> dict[str, Any]:
        with self._lock:
            defaults = self.reset_defaults()
            current = self._without_meta(self.read())
            if section not in defaults:
                raise KeyError(section)
            current[section] = copy.deepcopy(defaults[section])
            self.write(current)
            logger.info("[config.reset] section=%s", section)
            return self.read()

    def ensure_reset_baseline(self) -> None:
        with self._lock:
            if self.reset_path.exists():
                return
            if self.path.exists():
                baseline = self._without_meta(self.read())
            else:
                baseline = default_config_dict()
            self._atomic_write_text(
                self.reset_path,
                yaml.safe_dump(baseline, allow_unicode=True, sort_keys=False),
            )
            logger.info("[config.reset_baseline] created path=%s", self.reset_path)

    def ensure_current_config(self) -> None:
        with self._lock:
            if self.path.exists():
                return
            self.write(self.reset_defaults())
            logger.info("[config.ensure] created path=%s from reset baseline", self.path)

    def reset_defaults(self) -> dict[str, Any]:
        with self._lock:
            self.ensure_reset_baseline()
            defaults = default_config_dict()
            try:
                loaded = yaml.safe_load(self.reset_path.read_text(encoding="utf-8")) or {}
            except Exception:
                logger.exception("[config.reset_defaults] failed path=%s; using dataclass defaults", self.reset_path)
                loaded = {}
            if not isinstance(loaded, dict):
                logger.warning(
                    "[config.reset_defaults] expected mapping path=%s; using dataclass defaults",
                    self.reset_path,
                )
                loaded = {}
            result = self._deep_merge(defaults, loaded)
            self._normalize_translation_context(result)
            return result

    @staticmethod
    def _normalize_translation_context(data: dict[str, Any]) -> None:
        translation = data.get("translation")
        if not isinstance(translation, dict):
            return
        try:
            count = int(translation.get("context_subtitles", 3))
        except (TypeError, ValueError):
            logger.warning(
                "[config.normalize] invalid context_subtitles=%r; using 3",
                translation.get("context_subtitles"),
            )
            count = 3
        count = min(5, max(0, count))
        translation["context_subtitles"] = count
        translation["n_ctx"] = context_window_for_subtitles(count)

    def _with_meta(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = self._without_meta(data)
        result = copy.deepcopy(payload)
        result["_hash"] = self.hash(payload)
        return result

    @staticmethod
    def _without_meta(data: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in copy.deepcopy(data).items() if not str(k).startswith("_")}

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in (patch or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def hash(data: dict[str, Any]) -> str:
        raw = yaml.safe_dump(data, allow_unicode=True, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
