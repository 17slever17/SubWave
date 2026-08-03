from __future__ import annotations

from typing import Any

PRESETS: list[dict[str, Any]] = [
    {
        "id": "max",
        "name": "Max",
        "vram": "8GB",
        "quality": "Best",
        "speed": "Good",
        "description": "Best translation quality for 8GB GPUs.",
        "patch": {
            "stt": {
                "sherpa_model_id": "auto",
                "sherpa_onnx_provider": "cpu",
            },
            "translation": {
                "model_path": "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf",
                "device": "gpu",
                "n_gpu_layers": -1,
                "mtp_enabled": True,
                "mtp_model_path": "auto",
                "mtp_n": 1,
                "n_ctx": 1024,
                "n_batch": 256,
                "context_subtitles": 3,
            },
        },
    },
    {
        "id": "medium",
        "name": "Medium",
        "vram": "6GB",
        "quality": "Balanced",
        "speed": "Good",
        "description": "Smaller translation model for 6GB GPUs.",
        "patch": {
            "stt": {
                "sherpa_model_id": "auto",
                "sherpa_onnx_provider": "cpu",
            },
            "translation": {
                "model_path": "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
                "device": "gpu",
                "n_gpu_layers": -1,
                "mtp_enabled": True,
                "mtp_model_path": "auto",
                "mtp_n": 1,
                "n_ctx": 1024,
                "n_batch": 256,
                "context_subtitles": 3,
            },
        },
    },
    {
        "id": "potato",
        "name": "Potato",
        "vram": "CPU",
        "quality": "Basic",
        "speed": "No GPU required",
        "description": "Runs ASR and translation on CPU; no graphics card is required.",
        "patch": {
            "stt": {
                "sherpa_model_id": "auto",
                "sherpa_onnx_provider": "cpu",
            },
            "translation": {
                "model_path": "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
                "device": "cpu",
                "n_gpu_layers": 0,
                "mtp_enabled": True,
                "mtp_model_path": "auto",
                "mtp_n": 1,
                "n_ctx": 1024,
                "n_batch": 256,
                "context_subtitles": 3,
            },
        },
    },
]


def list_presets() -> list[dict[str, Any]]:
    return [*PRESETS]


def get_preset(preset_id: str) -> dict[str, Any] | None:
    for preset in list_presets():
        if preset["id"] == preset_id:
            return preset
    return None
