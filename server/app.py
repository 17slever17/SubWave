from __future__ import annotations

import asyncio
import importlib
import json
import logging
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.stt.audio import _hostapi_name, describe_audio_devices, query_audio_devices  # noqa: E402
from .config_service import ConfigService  # noqa: E402
from .paths import CONFIG_PATH, PROMPTS_PATH, RESET_CONFIG_PATH, STATIC_DIR, resolve_project_path  # noqa: E402
from . import presets as preset_module  # noqa: E402
from .runtime import RuntimeController  # noqa: E402
from services.llm.prompts import PromptStore  # noqa: E402
from services.llm.llama_server import NativeLlamaServer  # noqa: E402
from services.stt.models import get_sherpa_model, model_options  # noqa: E402
from configs.model_catalog import (  # noqa: E402
    capabilities,
    compatible_translation_models,
    normalize_language,
    validate_runtime_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("realtime_webui")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config_service.ensure_reset_baseline()
    config_service.ensure_current_config()
    prompt_store.ensure_exists()
    logger.info(
        "[startup] config=%s reset=%s prompts=%s static=%s",
        CONFIG_PATH,
        RESET_CONFIG_PATH,
        PROMPTS_PATH,
        STATIC_DIR,
    )
    try:
        yield
    finally:
        logger.info("[shutdown] stopping realtime translator runtime")
        await runtime.stop()


app = FastAPI(
    title="Realtime Subtitle Translator WebUI",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config_service = ConfigService(CONFIG_PATH, RESET_CONFIG_PATH)
prompt_store = PromptStore(PROMPTS_PATH)
runtime = RuntimeController()
_extension_last_seen = 0.0
_EXTENSION_HEARTBEAT_TTL_S = 5.0


def presets() -> Any:
    """Reload preset definitions in dev so edits to presets.py are visible immediately."""
    return importlib.reload(preset_module)


def _push_overlay_preview(config: dict[str, Any]) -> None:
    overlay = config.get("overlay", {})
    host = str(overlay.get("browser_bridge_host", "127.0.0.1"))
    port = int(overlay.get("browser_bridge_port", 8765))
    payload = json.dumps(
        {
            "mode": overlay.get("mode", "browser"),
            "show_source": overlay.get("show_source", True),
            "opacity": overlay.get("opacity", 0.82),
            "font_size": overlay.get("font_size", 23),
            "source_font_size": overlay.get("source_font_size", 15),
        }
    ).encode("utf-8")
    bridge_request = urllib_request.Request(
        f"http://{host}:{port}/overlay_config",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(bridge_request, timeout=0.2):
            pass
    except OSError:
        # The translator may be stopped; persisted settings apply on its next start.
        return


def _browser_bridge_health(host: str, port: int) -> tuple[bool, str]:
    address = f"{host}:{port}"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) != 0:
            return True, f"{address} free"
    except OSError as exc:
        return False, f"{address} unavailable: {exc}"
    finally:
        sock.close()

    try:
        bridge_request = urllib_request.Request(
            f"http://{address}/overlay_subtitles",
            method="GET",
        )
        with urllib_request.urlopen(bridge_request, timeout=0.3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if (
            response.status == 200
            and isinstance(payload, dict)
            and payload.get("ok") is True
            and "sequence" in payload
        ):
            return True, f"{address} translator bridge active"
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return False, f"{address} occupied by another process"


@app.get("/api/status")
def get_status(request: Request) -> dict[str, Any]:
    global _extension_last_seen
    if request.query_params.get("client") == "extension":
        _extension_last_seen = time.monotonic()
    return {
        "service": "realtime-translator",
        "runtime": runtime.status(),
        "config_hash": config_service.read().get("_hash"),
    }


def _extension_health(host: str, port: int) -> tuple[bool, str]:
    age = time.monotonic() - _extension_last_seen
    if _extension_last_seen > 0 and age <= _EXTENSION_HEARTBEAT_TTL_S:
        return True, "Browser extension connected"
    try:
        status_request = urllib_request.Request(
            f"http://{host}:{port}/overlay_status",
            method="GET",
        )
        with urllib_request.urlopen(status_request, timeout=0.3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if response.status == 200 and payload.get("connected") is True:
            return True, "Browser extension connected through active tab"
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return False, "Not detected; enable capture once from the browser toolbar"


@app.post("/api/control/start")
async def start_runtime() -> dict[str, Any]:
    try:
        validate_runtime_config(config_service.read())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"runtime": await runtime.start()}


@app.post("/api/control/pause")
async def pause_runtime() -> dict[str, Any]:
    return {"runtime": await runtime.pause()}


@app.post("/api/control/resume")
async def resume_runtime() -> dict[str, Any]:
    return {"runtime": await runtime.resume()}


@app.post("/api/control/stop")
async def stop_runtime() -> dict[str, Any]:
    return {"runtime": await runtime.stop()}


@app.post("/api/control/restart")
async def restart_runtime() -> dict[str, Any]:
    return {"runtime": await runtime.restart()}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return config_service.read()


@app.post("/api/config")
async def save_config(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    try:
        validate_runtime_config(body)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    saved = config_service.write(body)
    await asyncio.to_thread(_push_overlay_preview, saved)
    return {"config": saved}


@app.post("/api/config/patch")
async def patch_config(request: Request) -> dict[str, Any]:
    body = await request.json()
    patch = body.get("patch", body)
    if not isinstance(patch, dict):
        raise HTTPException(status_code=400, detail="patch must be an object")
    candidate = config_service.preview_patch(patch)
    try:
        validate_runtime_config(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    saved = config_service.patch(patch)
    await asyncio.to_thread(_push_overlay_preview, saved)
    return {"config": saved}


@app.post("/api/config/overlay-preview")
async def preview_overlay_config(request: Request) -> dict[str, bool]:
    body = await request.json()
    current = config_service.read()
    overlay = current.get("overlay", {})
    preview = {
        **current,
        "overlay": {
            **overlay,
            **{
                key: body[key]
                for key in ("opacity", "font_size", "source_font_size", "show_source")
                if key in body
            },
        },
    }
    await asyncio.to_thread(_push_overlay_preview, preview)
    return {"ok": True}


@app.post("/api/config/reset-section")
async def reset_config_section(request: Request) -> dict[str, Any]:
    body = await request.json()
    section = str(body.get("section", "")).strip()
    if not section:
        raise HTTPException(status_code=400, detail="section is required")
    try:
        saved = config_service.reset_section(section)
        await asyncio.to_thread(_push_overlay_preview, saved)
        return {"config": saved}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown section: {section}") from None


@app.get("/api/presets")
def get_presets() -> dict[str, Any]:
    return {"presets": presets().list_presets()}


@app.post("/api/presets/apply")
async def apply_preset(request: Request) -> dict[str, Any]:
    body = await request.json()
    preset_id = str(body.get("id", "")).strip()
    preset = presets().get_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")
    candidate = config_service.preview_patch(preset["patch"])
    try:
        validate_runtime_config(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info("[presets.apply] id=%s", preset_id)
    saved = config_service.patch(preset["patch"])
    await asyncio.to_thread(_push_overlay_preview, saved)
    return {"preset": preset, "config": saved}


@app.get("/api/prompts")
def get_prompts() -> dict[str, Any]:
    return prompt_store.load()


@app.post("/api/prompts")
async def save_prompts(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="prompt state must be an object")
    try:
        for preset in body.get("presets", {}).values():
            if not isinstance(preset, dict):
                continue
            source = str(preset.get("source_language", ""))
            target = str(preset.get("target_language", ""))
            source_code = normalize_language(source)
            available_stt = [
                model for model in model_options()
                if source_code in model["languages"]
            ]
            if not available_stt:
                raise ValueError(f"No STT model supports {source!r}")
            if not compatible_translation_models(source, target):
                raise ValueError(f"No translation model supports {source!r} to {target!r}")
        return prompt_store.save(body)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/prompts/reset")
async def reset_prompts(request: Request) -> dict[str, Any]:
    body = await request.json()
    preset_id = body.get("id")
    return prompt_store.reset(str(preset_id) if preset_id else None)


@app.post("/api/prompts/delete")
async def delete_prompt(request: Request) -> dict[str, Any]:
    preset_id = str((await request.json()).get("id", "")).strip()
    if not preset_id:
        raise HTTPException(status_code=400, detail="id is required")
    try:
        return prompt_store.delete(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt preset: {preset_id}") from None


@app.get("/api/audio/devices")
def get_audio_devices() -> dict[str, Any]:
    try:
        devices_by_name: dict[str, dict[str, Any]] = {}
        available = query_audio_devices()
        has_full_cable = any(
            str(item.get("name", "")) == "CABLE Output (VB-Audio Virtual Cable)"
            and "WASAPI" in _hostapi_name(item)
            for item in available
        )
        for device in available:
            name = str(device.get("name", "")).strip()
            if not name or device.get("max_input_channels", 0) <= 0:
                continue
            hostapi = _hostapi_name(device) or "Unknown"
            if has_full_cable and name.startswith("CABLE Output (VB-Audio Virtual") and not name.endswith(")"):
                continue
            recommended = name == "CABLE Output (VB-Audio Virtual Cable)" and "WASAPI" in hostapi
            candidate = {"name": name, "hostapi": hostapi, "recommended": recommended}
            key = name.casefold()
            existing = devices_by_name.get(key)
            if existing is None or recommended or ("WASAPI" in hostapi and "WASAPI" not in existing["hostapi"]):
                devices_by_name[key] = candidate
        devices = list(devices_by_name.values())
        devices.sort(key=lambda item: (not item["recommended"], item["name"].casefold()))
        return {"devices": devices}
    except Exception as exc:
        logger.exception("[audio.devices] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models/status")
def get_models_status() -> dict[str, Any]:
    config = config_service.read()
    stt_config = config.get("stt", {})
    try:
        spec = get_sherpa_model(str(stt_config.get("sherpa_model_id", "auto")), str(config.get("language", "en")))
        stt_path = resolve_project_path(f"models/{spec.model_id}")
    except (ValueError, TypeError):
        stt_path = None
    llm_path = resolve_project_path(config.get("translation", {}).get("model_path"))
    return {
        "stt": {"path": str(stt_path) if stt_path else "", "exists": bool(stt_path and stt_path.exists())},
        "translation": {"path": str(llm_path) if llm_path else "", "exists": bool(llm_path and llm_path.exists())},
    }


@app.get("/api/stt/models")
def get_stt_models() -> dict[str, Any]:
    return {"models": model_options(resolve_project_path("models"))}


@app.get("/api/capabilities")
def get_capabilities() -> dict[str, Any]:
    payload = capabilities(resolve_project_path("models"))
    try:
        import sherpa_onnx
    except Exception:
        sherpa_onnx = None
    payload["stt_providers"] = [
        {"value": "cpu", "label": "CPU (Recommended)"},
        *(
            [{"value": "cuda", "label": "GPU (CUDA)"}]
            if "+cuda" in str(getattr(sherpa_onnx, "__version__", "")).lower()
            else []
        ),
    ]
    return payload


@app.post("/api/models/browse")
async def browse_model_path(request: Request) -> dict[str, Any]:
    body = await request.json()
    path = resolve_project_path(str(body.get("path", "")))
    return {"path": str(path) if path else "", "exists": bool(path and path.exists())}


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    config = config_service.read()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("python", True, sys.executable)
    try:
        import torch

        add("torch", True, f"{torch.__version__}, cuda={torch.cuda.is_available()}")
    except Exception as exc:
        add("torch", False, str(exc))
    for module_name in ["torchaudio", "sherpa_onnx"]:
        try:
            __import__(module_name)
            add(module_name, True)
        except Exception as exc:
            add(module_name, False, str(exc))
    translation_config = config.get("translation") or {}
    configured_server_path = str(
        translation_config.get(
            "llama_server_path",
            "bin/llama.cpp/llama-server.exe",
        )
    )
    try:
        server_path = NativeLlamaServer.resolve_server_path(configured_server_path)
        if server_path is None:
            add(
                "llama_server",
                False,
                "Native llama-server executable was not found; run start.bat to install it.",
            )
        else:
            add("llama_server", True, f"Native llama-server installed: {server_path}")
    except Exception as exc:
        add("llama_server", False, str(exc))
    try:
        from services.filters.denoise import ensure_torchaudio_compat

        ensure_torchaudio_compat()
        from df.enhance import df_features  # noqa: F401
        from df.model import ModelParams  # noqa: F401
        from df.utils import get_device  # noqa: F401

        add("deepfilternet", True)
    except Exception as exc:
        add("deepfilternet", False, str(exc))
    model_status = get_models_status()
    add("stt_model", bool(model_status["stt"]["exists"]), model_status["stt"]["path"])
    add("translation_model", bool(model_status["translation"]["exists"]), model_status["translation"]["path"])
    host = str(config.get("overlay", {}).get("browser_bridge_host", "127.0.0.1"))
    port = int(config.get("overlay", {}).get("browser_bridge_port", 8765))
    extension_ok, extension_detail = _extension_health(host, port)
    add("browser_extension", extension_ok, extension_detail)
    bridge_ok, bridge_detail = _browser_bridge_health(host, port)
    add("browser_bridge_port", bridge_ok, bridge_detail)
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = runtime.subscribe_logs()
    try:
        while True:
            line = await queue.get()
            await websocket.send_json({"type": "log", "text": line})
    except WebSocketDisconnect:
        pass
    finally:
        runtime.unsubscribe_logs(queue)


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = runtime.subscribe_status()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json({"type": "status", "runtime": payload})
    except WebSocketDisconnect:
        pass
    finally:
        runtime.unsubscribe_status(queue)


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    if not full_path:
        return RedirectResponse(url="/control", status_code=307)
    index = STATIC_DIR / "index.html"
    candidate = STATIC_DIR / full_path
    if candidate.exists() and candidate.is_file():
        return FileResponse(str(candidate))
    if index.exists():
        return FileResponse(str(index), headers={"Cache-Control": "no-store"})
    return JSONResponse({"detail": "Frontend not built. Run start-dev.bat for development."}, status_code=503)


if __name__ == "__main__":
    import threading
    import time
    import webbrowser

    import uvicorn

    def open_browser() -> None:
        time.sleep(1.0)
        webbrowser.open("http://127.0.0.1:7860")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("server.app:app", host="127.0.0.1", port=7860, reload=False, app_dir=str(ROOT))
