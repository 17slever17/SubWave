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
    ) -> list[str]:
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
        if device.strip().lower() in {"gpu", "cuda"}:
            command.extend(
                [
                    "--gpu-layers",
                    "all",
                    "--flash-attn",
                    "on",
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
            if device.strip().lower() in {"gpu", "cuda"}:
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

    def create_chat_completion(self, **payload: Any) -> dict[str, Any]:
        payload.setdefault("model", "translation")
        return self._request(
            "POST",
            "/v1/chat/completions",
            payload,
            timeout=self.request_timeout_s,
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
