from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from services.runtime.paths import APP_DIR, resolve_runtime_path
from services.runtime.windows_job import assign_kill_on_close_job, close_windows_handle


def _assign_kill_on_close_job(process: subprocess.Popen) -> int | None:
    return assign_kill_on_close_job(pid=process.pid, process_handle=process._handle)


def _close_windows_handle(handle: int | None) -> None:
    close_windows_handle(handle)


def get_runtime_backend(server_path: str | Path | None) -> str | None:
    if not server_path:
        return None
    resolved = resolve_runtime_path(server_path)
    executable = resolved or Path(server_path)
    bin_dir = executable.parent
    marker_path = bin_dir / "runtime-backend.txt"
    marker = None
    try:
        value = marker_path.read_text(encoding="ascii").strip().lower()
        if value in {"cuda", "vulkan", "cpu"}:
            marker = value
    except OSError:
        pass

    try:
        dll_names = {path.name.lower() for path in bin_dir.iterdir() if path.is_file()}
    except OSError:
        dll_names = set()
    has_cuda = any(
        name.startswith(("ggml-cuda", "cublas", "cudart")) and name.endswith(".dll")
        for name in dll_names
    )
    has_vulkan = any(
        name.startswith("ggml-vulkan") and name.endswith(".dll")
        for name in dll_names
    )

    if has_cuda and has_vulkan:
        return "mixed"
    detected = "cuda" if has_cuda else "vulkan" if has_vulkan else None
    if marker:
        if detected and detected != marker:
            return "mixed"
        if marker == "cpu" and detected:
            return "mixed"
        return marker
    if detected:
        return detected
    if executable.is_file():
        return "cpu"
    return None


class NativeLlamaServer:
    def __init__(
        self,
        *,
        server_path: str,
        model_path: str,
        device: str,
        n_ctx: int,
        n_batch: int,
        mtp_model_path: str | None = None,
        mtp_n: int = 1,
        request_timeout_s: float = 4.0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.server_path = self.resolve_server_path(server_path)
        if self.server_path is None:
            raise FileNotFoundError(
                "llama-server was not found. Run start.bat to install the native runtime."
            )
        self.runtime_backend = get_runtime_backend(self.server_path)
        if self.runtime_backend == "mixed":
            raise RuntimeError(
                "llama.cpp runtime contains conflicting backend markers/DLLs; "
                "run start.bat to reinstall a clean runtime."
            )
        self.logger.info(
            "Using native llama.cpp runtime backend=%s",
            self.runtime_backend or "unknown",
        )
        self.port = self._reserve_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.log_path = APP_DIR / "logs" / "llama-server.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        self._job_handle = None
        self.process = subprocess.Popen(
            self.build_command(
                server_path=self.server_path,
                model_path=Path(model_path),
                mtp_model_path=Path(mtp_model_path) if mtp_model_path else None,
                mtp_n=mtp_n,
                device=device,
                runtime_backend=self.runtime_backend,
                n_ctx=n_ctx,
                n_batch=n_batch,
                port=self.port,
            ),
            cwd=APP_DIR,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
        )
        try:
            self._job_handle = _assign_kill_on_close_job(self.process)
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    @staticmethod
    def resolve_server_path(configured_path: str) -> Path | None:
        candidates: list[Path] = []
        resolved = resolve_runtime_path(configured_path)
        if resolved:
            candidates.append(resolved)
        candidates.extend(
            [
                APP_DIR / "bin" / "llama.cpp" / "llama-server.exe",
                APP_DIR / "bin" / "llama.cpp" / "llama-server",
            ]
        )
        executable = shutil.which("llama-server")
        if executable:
            candidates.append(Path(executable))
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def build_command(
        *,
        server_path: Path,
        model_path: Path,
        device: str,
        n_ctx: int,
        n_batch: int,
        port: int,
        mtp_model_path: Path | None = None,
        mtp_n: int = 1,
        runtime_backend: str | None = None,
    ) -> list[str]:
        if runtime_backend is None:
            runtime_backend = get_runtime_backend(server_path)
        if runtime_backend == "mixed":
            raise ValueError("llama.cpp runtime contains conflicting backend DLLs")

        command = [
            str(server_path),
            "--model",
            str(model_path),
            "--ctx-size",
            str(n_ctx),
            "--batch-size",
            str(n_batch),
            "--ubatch-size",
            str(min(n_batch, 256)),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            "1",
            "--no-webui",
            "--jinja",
            "--reasoning",
            "off",
        ]
        use_gpu = (
            device.strip().lower() in {"gpu", "cuda", "vulkan"}
            and runtime_backend != "cpu"
        )
        if use_gpu:
            command.extend(
                [
                    "--gpu-layers",
                    "all",
                    "--flash-attn",
                    "auto" if runtime_backend == "vulkan" else "on",
                ]
            )
        else:
            command.extend(
                [
                    "--device",
                    "none",
                    "--gpu-layers",
                    "0",
                ]
            )
        if mtp_model_path:
            command.extend(
                [
                    "--spec-type",
                    "draft-mtp",
                    "--spec-draft-model",
                    str(mtp_model_path),
                    "--spec-draft-n-max",
                    str(max(1, mtp_n)),
                ]
            )
            if use_gpu:
                command.extend(["--spec-draft-ngl", "all"])
            else:
                command.extend(
                    [
                        "--spec-draft-device",
                        "none",
                        "--spec-draft-ngl",
                        "0",
                    ]
                )
        return command

    @staticmethod
    def _reserve_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(
                f"llama-server request exceeded {timeout:.1f}s"
            ) from exc

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                self._request("GET", "/health", timeout=2.0)
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.2)
        raise TimeoutError(f"llama-server did not become ready; see {self.log_path}")

    def create_chat_completion(
        self,
        *,
        timeout: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        payload.setdefault("model", "translation")
        return self._request(
            "POST",
            "/v1/chat/completions",
            payload,
            timeout=(
                self.request_timeout_s
                if timeout is None
                else max(1.0, float(timeout))
            ),
        )

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_file = getattr(self, "_log_file", None)
        if log_file is not None and not log_file.closed:
            log_file.close()
        job_handle = getattr(self, "_job_handle", None)
        if job_handle:
            _close_windows_handle(job_handle)
            self._job_handle = None
