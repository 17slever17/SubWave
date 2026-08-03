from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import urllib.error
from typing import Callable

from scripts.download_parallel import download as parallel_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
TEN_VAD_FILENAME = "ten-vad.int8.onnx"
FIRERED_VAD_REVISION = "c30ec49e8cc69642b0ee65362eba11b9d11c6e54"
FIRERED_VAD_BASE = (
    "https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/"
    f"{FIRERED_VAD_REVISION}/pretrained_models/onnx_models"
)


MODEL_IDS = (
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "sherpa-onnx-nemo-fast-conformer-ctc-es-1424-int8",
    "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
    "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16",
    "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
    "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
    "sherpa-onnx-zipformer-thai-2024-06-20",
    "sherpa-onnx-zipformer-vi-int8-2025-04-20",
    "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
    "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09",
    "sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03",
    "sherpa-onnx-paraformer-zh-int8-2025-10-07",
    "sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10",
)


ProgressCallback = Callable[[str], None]


def _remote_size(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "realtime-translator-installer"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def _download(
    url: str,
    destination: Path,
    callback: ProgressCallback | None = None,
    label: str = "STT model",
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            expected_bytes = _remote_size(url)
            parallel_download(
                url,
                destination,
                expected_bytes,
                connections=8,
                label=label,
                callback=callback,
            )
            return
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            message = f"[download:{label}] network retry {attempt}/5: {exc}"
            callback(message) if callback else print(message, flush=True)
            time.sleep(min(10, attempt * 2))
    try:
        _download_once(url, destination, callback=callback, label=label)
        return
    except Exception as exc:
        raise RuntimeError(f"Download failed after 5 attempts: {url}") from (last_error or exc)


def _download_once(
    url: str,
    destination: Path,
    callback: ProgressCallback | None = None,
    label: str = "model",
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if downloaded:
        request.add_header("Range", f"bytes={downloaded}-")

    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response:
        if downloaded and response.status != 206:
            downloaded = 0
            partial.unlink(missing_ok=True)
        total = response.headers.get("Content-Length")
        total_bytes = downloaded + int(total) if total else 0
        mode = "ab" if downloaded else "wb"
        with partial.open(mode) as output:
            copied = downloaded
            next_report = time.monotonic()
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                copied += len(chunk)
                now = time.monotonic()
                if now >= next_report:
                    percent = copied * 100 / total_bytes if total_bytes else 0
                    speed = copied / max(1.0, now - started) / 1024 / 1024
                    message = (
                        f"[download:{label}] {percent:5.1f}% | "
                        f"{copied / 1024**2:,.0f}/{total_bytes / 1024**2:,.0f} MiB | "
                        f"{speed:.1f} MiB/s"
                    )
                    callback(message) if callback else print(message, flush=True)
                    next_report = now + 5
    partial.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r:bz2") as bundle:
        bundle.extractall(staging, filter="data")

    children = list(staging.iterdir())
    source = children[0] if len(children) == 1 and children[0].is_dir() else staging
    if destination.exists():
        shutil.rmtree(destination)
    if source == staging:
        staging.replace(destination)
    else:
        source.replace(destination)
        staging.rmdir()

    if not any(destination.rglob("*.onnx")):
        raise RuntimeError(f"No ONNX files found after extracting {archive.name}")


def install(
    model_id: str,
    keep_archives: bool = False,
    callback: ProgressCallback | None = None,
) -> Path:
    destination = MODELS_DIR / model_id
    if destination.is_dir() and any(destination.rglob("*.onnx")):
        print(f"[skip] {model_id}")
        return destination

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODELS_DIR / f"{model_id}.tar.bz2"
    if not archive.exists():
        print(f"[download] {model_id}", flush=True)
        _download(
            f"{RELEASE_BASE}/{archive.name}",
            archive,
            callback=callback,
            label=f"STT {model_id}",
        )
    print(f"[extract] {archive.name}", flush=True)
    _safe_extract(archive, destination)
    if not keep_archives:
        archive.unlink(missing_ok=True)
    print(f"[ready] {destination}", flush=True)
    return destination


def install_ten_vad(callback: ProgressCallback | None = None) -> Path:
    destination = MODELS_DIR / TEN_VAD_FILENAME
    if destination.is_file():
        print(f"[skip] {TEN_VAD_FILENAME}")
        return destination

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] {TEN_VAD_FILENAME}", flush=True)
    _download(
        f"{RELEASE_BASE}/{TEN_VAD_FILENAME}",
        destination,
        callback=callback,
        label="TEN VAD",
    )
    print(f"[ready] {destination}", flush=True)
    return destination


def install_firered_vad(callback: ProgressCallback | None = None) -> Path:
    destination = MODELS_DIR / "FireRedVAD" / "Stream-VAD"
    model_file = destination / "fireredvad_stream_vad_with_cache.onnx"
    cmvn_file = destination / "cmvn.ark"
    if model_file.is_file() and cmvn_file.is_file():
        print("[skip] FireRedVAD Stream-VAD")
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    for filename in (model_file.name, cmvn_file.name):
        target = destination / filename
        if not target.is_file():
            print(f"[download] FireRedVAD/{filename}", flush=True)
            _download(
                f"{FIRERED_VAD_BASE}/{filename}",
                target,
                callback=callback,
                label=f"FireRedVAD {filename}",
            )
    print(f"[ready] {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Install CPU sherpa-onnx STT models")
    parser.add_argument("models", nargs="*", choices=MODEL_IDS)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    for model_id in args.models or MODEL_IDS:
        install(model_id, args.keep_archives)


if __name__ == "__main__":
    main()
