from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SherpaModelSpec:
    model_id: str
    label: str
    languages: tuple[str, ...]
    architecture: str
    model_type: str = ""
    blank_penalty: float = 0.0

    @property
    def directory(self) -> Path:
        return Path("models") / self.model_id


SHERPA_MODELS: tuple[SherpaModelSpec, ...] = (
    SherpaModelSpec(
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        "European languages - Parakeet v3 (25 languages)",
        (
            "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
            "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
            "sl", "es", "sv", "ru", "uk",
        ),
        "transducer",
        "nemo_transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-nemo-fast-conformer-ctc-es-1424-int8",
        "Spanish - FastConformer CTC (better for loud speech)",
        ("es",),
        "nemo_ctc",
    ),
    SherpaModelSpec(
        "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
        "Chinese - Zipformer CTC",
        ("zh", "cmn"),
        "zipformer_ctc",
    ),
    SherpaModelSpec(
        "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16",
        "Russian - GigaAM v3 (better)",
        ("ru",),
        "transducer",
        "nemo_transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
        "Japanese - Parakeet 0.6B (better)",
        ("ja",),
        "nemo_ctc",
    ),
    SherpaModelSpec(
        "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
        "Japanese - ReazonSpeech (faster)",
        ("ja",),
        "transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
        "English - Parakeet 0.6B (better)",
        ("en",),
        "transducer",
        "nemo_transducer",
        blank_penalty=0.3,
    ),
    SherpaModelSpec(
        "sherpa-onnx-zipformer-thai-2024-06-20",
        "Thai - Zipformer",
        ("th",),
        "transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-zipformer-vi-int8-2025-04-20",
        "Vietnamese - Zipformer (better)",
        ("vi",),
        "transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        "Vietnamese - Zipformer 30M (faster)",
        ("vi",),
        "transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09",
        "Bengali - Streaming Zipformer",
        ("bn",),
        "online_transducer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03",
        "Wu Chinese - WeNetSpeech CTC",
        ("wuu", "wu"),
        "wenet_ctc",
    ),
    SherpaModelSpec(
        "sherpa-onnx-paraformer-zh-int8-2025-10-07",
        "Chinese (Sichuanese) - Paraformer",
        ("zh", "cmn"),
        "paraformer",
    ),
    SherpaModelSpec(
        "sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10",
        "Cantonese - WeNetSpeech CTC",
        ("yue",),
        "wenet_ctc",
    ),
)

SHERPA_MODELS_BY_ID = {spec.model_id: spec for spec in SHERPA_MODELS}

DEFAULT_MODEL_BY_LANGUAGE = {
    "bn": "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09",
    "cmn": "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
    "en": "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
    "ja": "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
    "ru": "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16",
    "th": "sherpa-onnx-zipformer-thai-2024-06-20",
    "vi": "sherpa-onnx-zipformer-vi-int8-2025-04-20",
    "wu": "sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03",
    "wuu": "sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03",
    "yue": "sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10",
    "zh": "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
}

_EUROPEAN_PARAKEET_ID = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
for _language in SHERPA_MODELS_BY_ID[_EUROPEAN_PARAKEET_ID].languages:
    DEFAULT_MODEL_BY_LANGUAGE.setdefault(_language, _EUROPEAN_PARAKEET_ID)


def get_sherpa_model(model_id: str, language: str) -> SherpaModelSpec:
    selected = model_id.strip()
    if not selected or selected == "auto":
        language_key = language.strip().lower().split("-", 1)[0]
        selected = DEFAULT_MODEL_BY_LANGUAGE.get(language_key, "")
        if not selected:
            supported = ", ".join(sorted(DEFAULT_MODEL_BY_LANGUAGE))
            raise ValueError(
                f"No automatic sherpa-onnx model for language {language!r}. "
                f"Select a model explicitly; supported auto languages: {supported}"
            )
    try:
        return SHERPA_MODELS_BY_ID[selected]
    except KeyError as exc:
        raise ValueError(f"Unknown sherpa-onnx model: {selected!r}") from exc


def model_options(root: Path | None = None) -> list[dict[str, object]]:
    return [
        {
            "value": spec.model_id,
            "label": spec.label,
            "languages": list(spec.languages),
            "architecture": spec.architecture,
            "installed": root is None or (root / spec.model_id).is_dir(),
        }
        for spec in SHERPA_MODELS
    ]


def models_for_language(language: str, root: Path | None = None) -> list[SherpaModelSpec]:
    language_key = language.strip().lower().split("-", 1)[0]
    return [
        spec
        for spec in SHERPA_MODELS
        if language_key in spec.languages and (root is None or (root / spec.model_id).is_dir())
    ]


def resolve_model_files(spec: SherpaModelSpec, root: Path) -> dict[str, str]:
    directory = root / spec.model_id
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Sherpa model is not installed: {directory}. "
            "Run realtime_translator/scripts/download_sherpa_models.py"
        )

    def first(label: str, patterns: tuple[str, ...]) -> str:
        for pattern in patterns:
            matches = sorted(directory.rglob(pattern))
            if matches:
                return str(matches[0])
        raise FileNotFoundError(f"No {label} file found in {directory}")

    files = {"tokens": first("tokens", ("tokens.txt",))}
    if spec.architecture in {"transducer", "online_transducer"}:
        files.update(
            encoder=first("encoder", ("*encoder*.int8.onnx", "*encoder*.onnx")),
            decoder=first("decoder", ("*decoder*.int8.onnx", "*decoder*.onnx")),
            joiner=first("joiner", ("*joiner*.int8.onnx", "*joiner*.onnx")),
        )
    else:
        files["model"] = first("model", ("model.int8.onnx", "*model*.int8.onnx", "model.onnx", "*.onnx"))
    return files


def create_recognizer(
    spec: SherpaModelSpec,
    files: dict[str, str],
    provider: str = "cpu",
    num_threads: int = 4,
    sample_rate: int = 16000,
    blank_penalty: float | None = None,
    decoding_method: str = "greedy_search",
    max_active_paths: int = 4,
):
    import sherpa_onnx

    provider = provider.strip().lower()
    if provider == "cuda":
        if "+cuda" not in str(getattr(sherpa_onnx, "__version__", "")).lower():
            raise RuntimeError(
                "STT GPU mode requires a CUDA-enabled sherpa-onnx wheel"
            )
    elif provider != "cpu":
        raise ValueError(f"Unsupported sherpa-onnx provider: {provider!r}")

    common = {
        "tokens": files["tokens"],
        "num_threads": num_threads,
        "provider": provider,
        "sample_rate": sample_rate,
        "decoding_method": decoding_method,
    }
    transducer_options = {
        **common,
        "blank_penalty": (
            spec.blank_penalty if blank_penalty is None else float(blank_penalty)
        ),
        "max_active_paths": max(1, int(max_active_paths)),
    }
    if spec.architecture == "online_transducer":
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            **transducer_options,
        )
    if spec.architecture == "transducer":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            model_type=spec.model_type or "transducer",
            **transducer_options,
        )
    if spec.architecture == "paraformer":
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=files["model"], **common
        )
    factory = getattr(sherpa_onnx.OfflineRecognizer, f"from_{spec.architecture}")
    return factory(model=files["model"], **common)
