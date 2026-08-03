from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_DIR / "frontend"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.runtime.windows_job import assign_kill_on_close_job, close_windows_handle


def _stream_output(process: subprocess.Popen[str], prefix: str) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()
        process.wait(timeout=3)


def main() -> int:
    backend_port = int(os.environ.get("REALTIME_WEBUI_BACKEND_PORT", "7860"))
    frontend_port = int(os.environ.get("REALTIME_WEBUI_DEV_PORT", "5174"))
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    stop_requested = threading.Event()
    processes: list[subprocess.Popen[str]] = []
    job_handles: list[int | None] = []

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server.app:app",
                "--app-dir",
                str(APP_DIR),
                "--host",
                "127.0.0.1",
                "--port",
                str(backend_port),
            ],
            cwd=APP_DIR,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        processes.append(backend)
        job_handles.append(
            assign_kill_on_close_job(
                pid=backend.pid,
                process_handle=getattr(backend, "_handle", None),
            )
        )

        frontend_command = [
            "cmd.exe",
            "/d",
            "/s",
            "/c",
            f"npm run dev -- --host 127.0.0.1 --port {frontend_port} --strictPort",
        ] if os.name == "nt" else [
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(frontend_port),
            "--strictPort",
        ]
        frontend = subprocess.Popen(
            frontend_command,
            cwd=FRONTEND_DIR,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        processes.append(frontend)
        job_handles.append(
            assign_kill_on_close_job(
                pid=frontend.pid,
                process_handle=getattr(frontend, "_handle", None),
            )
        )

        for process, prefix in ((backend, "backend"), (frontend, "frontend")):
            threading.Thread(
                target=_stream_output,
                args=(process, prefix),
                daemon=True,
            ).start()

        if os.environ.get("REALTIME_WEBUI_OPEN_BROWSER", "1") != "0":
            threading.Timer(
                1.5,
                lambda: webbrowser.open(f"http://127.0.0.1:{frontend_port}/control"),
            ).start()

        while not stop_requested.wait(0.2):
            if backend.poll() is not None:
                print(f"[start-dev] Backend exited with code {backend.returncode}.", flush=True)
                return backend.returncode or 1
            if frontend.poll() is not None:
                print(f"[start-dev] Frontend exited with code {frontend.returncode}.", flush=True)
                return frontend.returncode or 1
        return 0
    finally:
        for process in reversed(processes):
            _stop_process(process)
        for handle in job_handles:
            close_windows_handle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
