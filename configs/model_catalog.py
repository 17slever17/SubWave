from __future__ import annotations

from pathlib import Path
from typing import Any

from services.stt.models import DEFAULT_MODEL_BY_LANGUAGE, model_options


LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "kk": "Kazakh", "km": "Khmer",
    "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Myanmar",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Filipino",
    "tr": "Turkish", "ug": "Uyghur", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
    "zh-hant": "Traditional Chinese",
}

# Keep this UI list conservative: languages covered by the available STT models
# and the translation directions supported by the bundled Gemma models.
GEMMA_LANGUAGES = frozenset(
    {
        "ar", "bg", "bn", "bo", "cs", "da", "de", "el", "en", "es", "et",
        "fa", "fi", "fr", "gu", "he", "hi", "hr", "hu", "id", "it", "ja",
        "kk", "km", "ko", "lt", "lv", "mn", "mr", "ms", "mt", "my", "nl",
        "pl", "pt", "ro", "ru", "sk", "sl", "sv", "ta", "te", "th", "tl",
        "tr", "ug", "uk", "ur", "vi", "yue", "zh", "zh-hant",
    }
)

TRANSLATION_MODELS: tuple[dict[str, Any], ...] = (
    {
        "value": "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf",
        "label": "Translate Gemma Sub E4B Q4 (better)",
        "family": "gemma",
        "languages": GEMMA_LANGUAGES,
    },
    {
        "value": "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
        "label": "Translate Gemma Sub E2B Q4 (faster)",
        "family": "gemma",
        "languages": GEMMA_LANGUAGES,
    },
)


def normalize_language(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    for code, name in LANGUAGE_NAMES.items():
        if normalized in {code, name.lower()}:
            return code
    return normalized


def translation_model_for_path(path: str) -> dict[str, Any] | None:
    name = Path(path).name.casefold()
    for model in TRANSLATION_MODELS:
        if Path(str(model["value"])).name.casefold() == name:
            return {**model, "value": path}
    return None


def compatible_translation_models(source: str, target: str) -> list[dict[str, Any]]:
    source_code = normalize_language(source)
    target_code = normalize_language(target)
    return [
        model for model in TRANSLATION_MODELS
        if source_code in model["languages"] and target_code in model["languages"]
    ]


def capabilities(models_root: Path) -> dict[str, Any]:
    stt_models = model_options(models_root)
    available_source_languages = sorted(
        {
            language
            for model in stt_models
            for language in model["languages"]
            if language in GEMMA_LANGUAGES
        }
    )
    target_languages = sorted(GEMMA_LANGUAGES)
    return {
        "languages": [
            {"value": code, "label": f"{LANGUAGE_NAMES.get(code, code)} ({code})"}
            for code in sorted(GEMMA_LANGUAGES)
        ],
        "source_languages": available_source_languages,
        "target_languages": target_languages,
        "stt_models": stt_models,
        "stt_defaults": DEFAULT_MODEL_BY_LANGUAGE,
        "translation_models": [
            {**model, "languages": sorted(model["languages"])} for model in TRANSLATION_MODELS
        ],
    }


def validate_runtime_config(config: dict[str, Any]) -> None:
    def section(name: str) -> dict[str, Any]:
        value = config.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        return value

    def number(
        values: dict[str, Any],
        key: str,
        *,
        minimum: float,
        maximum: float,
        default: float,
    ) -> float:
        value = values.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} must be between {minimum} and {maximum}"
            ) from exc
        if not minimum <= numeric <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return numeric

    source = normalize_language(str(config.get("language", "")))
    translation = section("translation")
    target = normalize_language(str(translation.get("target_language", "")))
    stt = section("stt")
    audio = section("audio")
    overlay = section("overlay")
    selected_stt = str(stt.get("sherpa_model_id", "auto"))
    provider = str(stt.get("sherpa_onnx_provider", "cpu")).lower()

    number(translation, "n_batch", minimum=1, maximum=4096, default=256)
    number(translation, "n_ctx", minimum=128, maximum=131072, default=1024)
    number(translation, "max_tokens", minimum=1, maximum=4096, default=128)
    number(translation, "mtp_n", minimum=1, maximum=64, default=1)
    number(translation, "context_subtitles", minimum=0, maximum=5, default=3)
    translation_device = str(translation.get("device", "gpu")).lower()
    if translation_device not in {"cpu", "gpu", "cuda"}:
        raise ValueError(f"Unsupported translation device: {translation_device!r}")

    capture_mode = str(audio.get("capture_mode", "browser_tab")).lower()
    if capture_mode not in {"browser_tab", "input_device"}:
        raise ValueError(f"Unsupported audio capture mode: {capture_mode!r}")
    number(
        audio,
        "browser_audio_port",
        minimum=1,
        maximum=65535,
        default=8766,
    )
    number(
        audio,
        "sampling_rate",
        minimum=8000,
        maximum=192000,
        default=16000,
    )
    number(
        audio,
        "buffer_max_chunks",
        minimum=1,
        maximum=65536,
        default=1024,
    )
    number(audio, "input_gain", minimum=0, maximum=10, default=1)
    overlay_mode = str(overlay.get("mode", "browser")).lower()
    if overlay_mode not in {"browser", "desktop"}:
        raise ValueError(f"Unsupported subtitle overlay mode: {overlay_mode!r}")
    number(
        overlay,
        "browser_bridge_port",
        minimum=1,
        maximum=65535,
        default=8765,
    )
    number(overlay, "opacity", minimum=0, maximum=1, default=0.82)
    number(overlay, "font_size", minimum=8, maximum=96, default=23)
    number(overlay, "source_font_size", minimum=8, maximum=72, default=15)

    process_interval = number(
        stt,
        "process_interval_s",
        minimum=0.1,
        maximum=60,
        default=2,
    )
    number(
        stt,
        "chunk_size_ms",
        minimum=10,
        maximum=1000,
        default=60,
    )
    max_phrase = number(
        stt,
        "max_phrase_s",
        minimum=0.1,
        maximum=300,
        default=10,
    )
    if max_phrase < process_interval:
        raise ValueError(
            "max_phrase_s must be greater than or equal to process_interval_s"
        )

    available_stt = [
        model for model in model_options()
        if source in model["languages"]
    ]
    if not available_stt:
        raise ValueError(f"No STT model supports source language {source!r}")
    if selected_stt != "auto" and not any(model["value"] == selected_stt for model in available_stt):
        raise ValueError(f"STT model {selected_stt!r} does not support {source!r}")
    if provider not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported STT device: {provider!r}")
    if provider == "cuda":
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ValueError("STT GPU mode requires sherpa-onnx") from exc
        if "+cuda" not in str(getattr(sherpa_onnx, "__version__", "")).lower():
            raise ValueError("STT GPU mode requires a CUDA-enabled sherpa-onnx wheel")

    model = translation_model_for_path(str(translation.get("model_path", "")))
    if model is None:
        custom_path = str(translation.get("model_path", "")).strip()
        if not custom_path or Path(custom_path).suffix.casefold() != ".gguf":
            raise ValueError("Select a built-in model or an existing custom .gguf model")
        return
    if source not in model["languages"] or target not in model["languages"]:
        raise ValueError(
            f"{model['label']} does not support {source!r} to {target!r} translation"
        )
