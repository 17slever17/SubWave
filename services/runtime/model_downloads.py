from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from services.runtime.paths import APP_DIR, resolve_runtime_path
from services.stt.models import get_sherpa_model
from scripts.download_parallel import download as parallel_download


ProgressCallback = Callable[[str], None]
MODELS_DIR = APP_DIR / "models"

TRANSLATION_DOWNLOADS: dict[str, dict[str, Any]] = {
    "translate_gemma4_sub-e2b-q4_k_xl.gguf": {
        "repo_id": "17slever17/translate-gemma-4-sub-E2B-GGUF",
        "preferred_filename": "translate_gemma4_sub-E2B-Q4_K_XL.gguf",
        "mtp_filename": "mtp-gemma-4-E2B-it.gguf",
        "mtp_devices": frozenset({"cpu"}),
    },
    "translate_gemma4_sub-e4b-q4_k_xl.gguf": {
        "repo_id": "17slever17/translate-gemma-4-sub-E4B-GGUF",
        "preferred_filename": "translate_gemma4_sub-E4B-Q4_K_XL.gguf",
        "mtp_filename": "mtp-gemma-4-E4B-it.gguf",
        "mtp_devices": frozenset({"cpu", "gpu", "cuda"}),
    },
}
FASTENHANCER_LANGUAGES = frozenset({"en", "ja", "ko", "es"})
FASTENHANCER_FILENAME = "fastenhancer_b_48khz.onnx"
FASTENHANCER_URL = (
    "https://github.com/aask1357/fastenhancer/releases/download/"
    "onnx-48khz-convstft-v0.1.0/b.onnx"
)


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _select_gguf(repo_files: list[str], preferred: str) -> str:
    by_name = {Path(item).name.casefold(): item for item in repo_files}
    exact = by_name.get(preferred.casefold())
    if exact:
        return exact

    candidates = [item for item in repo_files if item.casefold().endswith(".gguf")]
    for marker in ("q4_k_xl", "q4_k_m", "q4"):
        match = next((item for item in candidates if marker in Path(item).name.casefold()), None)
        if match:
            return match
    if candidates:
        return candidates[0]
    raise FileNotFoundError("The Hugging Face repository does not contain a GGUF file")


def _download_hf_file(
    repo_id: str,
    filename: str,
    destination: Path,
    callback: ProgressCallback | None,
    label: str,
) -> Path:
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename))
        if metadata.size is None:
            raise OSError("Hugging Face did not report the model size")
        parallel_download(
            metadata.location,
            destination,
            int(metadata.size),
            connections=8,
            label=label,
            callback=callback,
        )
        return destination
    except Exception as exc:
        _emit(callback, f"[download:{label}] Parallel download unavailable; using fallback: {exc}")
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=destination.parent,
            )
        )


def ensure_translation_model(
    configured_path: str,
    callback: ProgressCallback | None = None,
) -> Path:
    resolved = resolve_runtime_path(configured_path)
    if resolved and resolved.is_file():
        return resolved

    requested = Path(configured_path)
    spec = TRANSLATION_DOWNLOADS.get(requested.name.casefold())
    if spec is None:
        raise FileNotFoundError(
            f"Custom translation model was not found: {resolved or requested}. "
            "Select an existing .gguf file in Advanced settings."
        )

    from huggingface_hub import HfApi

    repo_id = spec["repo_id"]
    preferred = spec["preferred_filename"]
    _emit(callback, f"Downloading translation model from {repo_id}...")
    repo_files = HfApi().list_repo_files(repo_id=repo_id)
    filename = _select_gguf(repo_files, preferred)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    desired = MODELS_DIR / requested.name
    downloaded = _download_hf_file(
        repo_id,
        filename,
        desired,
        callback,
        "Translation model",
    )
    if downloaded.resolve() != desired.resolve() and not desired.exists():
        try:
            os.link(downloaded, desired)
        except OSError:
            downloaded.replace(desired)
    _emit(callback, f"Translation model ready: {desired}")
    return desired


def translation_mtp_path(
    configured_model_path: str,
    configured_mtp_path: str = "auto",
) -> Path | None:
    if configured_mtp_path.strip().lower() != "auto":
        return resolve_runtime_path(configured_mtp_path)
    spec = TRANSLATION_DOWNLOADS.get(Path(configured_model_path).name.casefold())
    if spec is None:
        return None
    return MODELS_DIR / spec["mtp_filename"]


def should_use_translation_mtp(
    configured_model_path: str,
    configured_mtp_path: str,
    device: str,
    enabled: bool = True,
) -> bool:
    """Apply built-in compatibility rules without restricting custom model pairs."""
    if not enabled:
        return False

    spec = TRANSLATION_DOWNLOADS.get(Path(configured_model_path).name.casefold())
    if spec is not None:
        return device.strip().lower() in spec["mtp_devices"]

    # Custom targets can use MTP when the user explicitly supplies a matching draft.
    return configured_mtp_path.strip().lower() != "auto"


def ensure_translation_mtp_model(
    configured_model_path: str,
    configured_mtp_path: str = "auto",
    callback: ProgressCallback | None = None,
) -> Path | None:
    desired = translation_mtp_path(configured_model_path, configured_mtp_path)
    if desired is None:
        _emit(callback, "MTP disabled for custom translation model.")
        return None
    if desired.is_file():
        return desired

    spec = TRANSLATION_DOWNLOADS.get(Path(configured_model_path).name.casefold())
    if spec is None:
        return None

    from huggingface_hub import HfApi

    repo_id = spec["repo_id"]
    filename = spec["mtp_filename"]
    repo_files = HfApi().list_repo_files(repo_id=repo_id)
    match = next(
        (item for item in repo_files if Path(item).name.casefold() == filename.casefold()),
        None,
    )
    if match is None:
        _emit(
            callback,
            f"MTP model is not published in {repo_id} yet; continuing without MTP.",
        )
        return None

    _emit(callback, f"Downloading MTP model from {repo_id}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = _download_hf_file(
        repo_id,
        match,
        desired,
        callback,
        "MTP model",
    )
    if downloaded.resolve() != desired.resolve() and not desired.exists():
        try:
            os.link(downloaded, desired)
        except OSError:
            downloaded.replace(desired)
    _emit(callback, f"MTP model ready: {desired}")
    return desired


def ensure_stt_model(
    stt_config: dict[str, Any],
    language: str,
    callback: ProgressCallback | None = None,
) -> Path:
    spec = get_sherpa_model(str(stt_config.get("sherpa_model_id", "auto")), language)
    destination = MODELS_DIR / spec.model_id
    if destination.is_dir() and any(destination.rglob("*.onnx")):
        return destination

    _emit(callback, f"Downloading STT model: {spec.label}...")
    from scripts.download_sherpa_models import install

    result = install(spec.model_id, callback=callback)
    _emit(callback, f"STT model ready: {result}")
    return result


def ensure_endpoint_model(
    stt_config: dict[str, Any],
    callback: ProgressCallback | None = None,
) -> Path | None:
    if not stt_config.get("end_vad_enabled", True):
        return None
    backend = str(stt_config.get("end_vad_backend", "energy")).strip().lower()
    if backend not in {"ten", "firered"}:
        return None

    if backend == "firered":
        configured_path = str(
            stt_config.get(
                "firered_vad_model_path",
                "models/FireRedVAD/Stream-VAD",
            )
        )
        resolved = resolve_runtime_path(configured_path)
        if (
            resolved
            and resolved.is_dir()
            and (resolved / "fireredvad_stream_vad_with_cache.onnx").is_file()
            and (resolved / "cmvn.ark").is_file()
        ):
            return resolved
        if Path(configured_path).as_posix().casefold() != "models/fireredvad/stream-vad":
            raise FileNotFoundError(f"Custom FireRedVAD model was not found: {resolved}")

        _emit(callback, "Downloading FireRedVAD endpoint model...")
        from scripts.download_sherpa_models import install_firered_vad

        result = install_firered_vad(callback=callback)
        _emit(callback, f"FireRedVAD model ready: {result}")
        return result

    configured_path = str(
        stt_config.get("ten_vad_model_path", "models/ten-vad.int8.onnx")
    )
    resolved = resolve_runtime_path(configured_path)
    if resolved and resolved.is_file():
        return resolved
    if Path(configured_path).name.casefold() != "ten-vad.int8.onnx":
        raise FileNotFoundError(f"Custom TEN VAD model was not found: {resolved}")

    _emit(callback, "Downloading TEN VAD endpoint model...")
    from scripts.download_sherpa_models import install_ten_vad

    result = install_ten_vad(callback=callback)
    _emit(callback, f"TEN VAD model ready: {result}")
    return result


def ensure_fastenhancer_model(
    audio_config: dict[str, Any],
    language: str,
    callback: ProgressCallback | None = None,
) -> Path | None:
    fast_config = audio_config.get("fastenhancer", {})
    language_key = str(language).strip().lower().split("-", 1)[0]
    supported = {
        str(item).strip().lower()
        for item in fast_config.get(
            "languages",
            FASTENHANCER_LANGUAGES,
        )
    }
    if not fast_config.get("enabled", True) or language_key not in supported:
        return None

    configured_path = str(
        fast_config.get(
            "model_path",
            f"models/fastenhancer/{FASTENHANCER_FILENAME}",
        )
    )
    resolved = resolve_runtime_path(configured_path)
    if resolved and resolved.is_file():
        return resolved
    if Path(configured_path).name.casefold() != FASTENHANCER_FILENAME:
        raise FileNotFoundError(
            f"Custom FastEnhancer model was not found: {resolved}"
        )

    destination = MODELS_DIR / "fastenhancer" / FASTENHANCER_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    _emit(callback, "Downloading FastEnhancer-B 48 kHz...")
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(FASTENHANCER_URL, temporary)
            temporary.replace(destination)
            break
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            delay_s = 2 ** (attempt - 1)
            _emit(
                callback,
                f"FastEnhancer download interrupted; retrying in {delay_s}s "
                f"({attempt + 1}/{attempts})...",
            )
            time.sleep(delay_s)
    temporary.unlink(missing_ok=True)
    _emit(callback, f"FastEnhancer model ready: {destination}")
    return destination


def ensure_runtime_models(
    config: dict[str, Any],
    callback: ProgressCallback | None = None,
) -> None:
    stt_config = config.get("stt", {})
    language = str(config.get("language", "en"))
    ensure_endpoint_model(stt_config, callback)
    ensure_stt_model(stt_config, language, callback)
    ensure_fastenhancer_model(config.get("audio", {}), language, callback)
    translation = config.get("translation", {})
    model_path = str(translation.get("model_path", ""))
    ensure_translation_model(model_path, callback)
    mtp_path = str(translation.get("mtp_model_path", "auto"))
    translation_device = str(translation.get("device", "gpu"))
    if should_use_translation_mtp(
        model_path,
        mtp_path,
        translation_device,
        bool(translation.get("mtp_enabled", True)),
    ):
        ensure_translation_mtp_model(
            model_path,
            mtp_path,
            callback,
        )
    elif translation.get("mtp_enabled", True):
        _emit(
            callback,
            f"MTP is not used for {Path(model_path).name} on {translation_device}.",
        )
