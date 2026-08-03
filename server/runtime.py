from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import sys
import yaml
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .paths import APP_DIR, CONFIG_PATH, MAIN_SCRIPT, VENV_PYTHON
from services.runtime.model_downloads import ensure_runtime_models
from services.runtime.events import RUNTIME_EVENT_PREFIX
from services.runtime.windows_job import assign_kill_on_close_job, close_windows_handle

logger = logging.getLogger(__name__)

MAX_LOG_HISTORY = 1500
MAX_LOG_REPLAY = 200
MAX_LOG_QUEUE = 600
MAX_STATUS_QUEUE = 20


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


@dataclass
class RuntimeStatus:
    state: str = "idle"
    phase: str = "idle"
    pid: int | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    exit_code: int | None = None
    last_error: str = ""
    paused: bool = False


class RuntimeController:
    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._job_handle: int | None = None
        self._status = RuntimeStatus()
        self._lock = asyncio.Lock()
        self._start_generation = 0
        self._start_in_progress = False
        self._logs: deque[str] = deque(maxlen=MAX_LOG_HISTORY)
        self._log_subscribers: set[asyncio.Queue[str]] = set()
        self._status_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._stt_active = False
        self._llm_active = False

    def status(self) -> dict[str, Any]:
        return asdict(self._status)

    async def start(self) -> dict[str, Any]:
        async with self._lock:
            if self._process and self._process.returncode is None:
                logger.info("[runtime.start] already running pid=%s", self._process.pid)
                return self.status()
            if self._start_in_progress:
                logger.info("[runtime.start] model preparation is already in progress")
                return self.status()
            self._start_generation += 1
            start_generation = self._start_generation
            self._start_in_progress = True
            self._status = RuntimeStatus(state="starting", phase="starting")
            await self._publish_status()
            try:
                config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                self._start_in_progress = False
                self._status.state = "error"
                self._status.last_error = str(exc)
                await self._append_log(f"Could not read runtime config: {exc}")
                await self._publish_status()
                return self.status()
            self._status.state = "downloading"
            self._status.phase = "checking_models"
            await self._append_log("Checking required STT and translation models...")
            await self._publish_status()

        try:
            loop = asyncio.get_running_loop()

            def report_model_progress(message: str) -> None:
                asyncio.run_coroutine_threadsafe(self._append_log(message), loop)

            await asyncio.to_thread(
                ensure_runtime_models,
                config,
                report_model_progress,
            )
        except asyncio.CancelledError:
            async with self._lock:
                self._start_in_progress = False
                if start_generation == self._start_generation:
                    self._start_generation += 1
                    self._status.state = "idle"
                    self._status.phase = "idle"
                    await self._publish_status()
            raise
        except Exception as exc:
            async with self._lock:
                self._start_in_progress = False
                if start_generation != self._start_generation:
                    logger.info("[runtime.start] ignored model preparation failure after cancellation")
                    return self.status()
                logger.exception("[runtime.start] model preparation failed")
                self._status.state = "error"
                self._status.phase = "error"
                self._status.last_error = str(exc)
                await self._append_log(f"Model preparation failed: {exc}")
                await self._publish_status()
                return self.status()

        async with self._lock:
            self._start_in_progress = False
            if start_generation != self._start_generation:
                await self._append_log("Runtime start cancelled before process launch.")
                return self.status()
            await self._append_log("Required models are ready.")
            self._status.state = "starting"
            self._status.phase = "starting"
            await self._publish_status()
            python_exe = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
            command = [python_exe, str(MAIN_SCRIPT)]
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env["REALTIME_TRANSLATOR_CONFIG"] = str(CONFIG_PATH)
            env["REALTIME_TRANSLATOR_PHASE_EVENTS"] = "1"
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            logger.info(
                "[runtime.start] cwd=%s cmd=%s config=%s hidden=%s",
                APP_DIR,
                command_text(command),
                CONFIG_PATH,
                bool(creationflags),
            )
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(APP_DIR),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.PIPE,
                    env=env,
                    creationflags=creationflags,
                )
                self._job_handle = assign_kill_on_close_job(pid=self._process.pid)
            except Exception as exc:
                logger.exception("[runtime.start] process launch failed")
                if self._process and self._process.returncode is None:
                    await self._kill_process_tree(self._process)
                    await self._process.wait()
                self._process = None
                self._release_job_handle()
                self._status.state = "error"
                self._status.phase = "error"
                self._status.last_error = str(exc)
                await self._append_log(f"Runtime process launch failed: {exc}")
                await self._publish_status()
                return self.status()
            loop = asyncio.get_running_loop()
            self._status = RuntimeStatus(
                state="running",
                phase="starting",
                pid=self._process.pid,
                started_at=loop.time(),
            )
            await self._append_log(f"Started realtime translator pid={self._process.pid}")
            await self._publish_status()
            asyncio.create_task(self._stream_process(self._process))
            return self.status()

    async def pause(self) -> dict[str, Any]:
        logger.info("[runtime.pause] requested")
        return await self._set_paused(True)

    async def resume(self) -> dict[str, Any]:
        logger.info("[runtime.resume] requested")
        return await self._set_paused(False)

    async def stop(self) -> dict[str, Any]:
        logger.info("[runtime.stop] requested")
        await self._stop_process(next_state="idle", paused=False)
        return self.status()

    async def restart(self) -> dict[str, Any]:
        logger.info("[runtime.restart] requested")
        await self._stop_process(next_state="idle", paused=False)
        return await self.start()

    async def _set_paused(self, paused: bool) -> dict[str, Any]:
        async with self._lock:
            process = self._process
            if not process or process.returncode is not None or process.stdin is None:
                self._status.last_error = "Realtime translator is not running"
                await self._publish_status()
                return self.status()
            command = "PAUSE" if paused else "RESUME"
            process.stdin.write(f"{command}\n".encode("ascii"))
            await process.stdin.drain()
            self._status.state = "paused" if paused else "running"
            self._status.phase = "paused" if paused else "ready"
            self._status.paused = paused
            self._stt_active = False
            self._llm_active = False
            self._status.last_error = ""
            await self._append_log(f"[FIX:runtime-pause] {command.lower()} requested; process {process.pid} remains loaded")
            await self._publish_status()
            return self.status()

    def subscribe_logs(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_LOG_QUEUE)
        for line in list(self._logs)[-MAX_LOG_REPLAY:]:
            put_latest(queue, line)
        self._log_subscribers.add(queue)
        return queue

    def unsubscribe_logs(self, queue: asyncio.Queue[str]) -> None:
        self._log_subscribers.discard(queue)

    def subscribe_status(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=MAX_STATUS_QUEUE)
        put_latest(queue, self.status())
        self._status_subscribers.add(queue)
        return queue

    def unsubscribe_status(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._status_subscribers.discard(queue)

    async def _stop_process(self, next_state: str, paused: bool) -> None:
        async with self._lock:
            self._start_generation += 1
            process = self._process
            if not process or process.returncode is not None:
                self._release_job_handle()
                self._process = None
                self._status.state = next_state
                self._status.phase = next_state
                self._status.paused = paused
                await self._publish_status()
                return
            self._status.state = "stopping"
            self._status.phase = "stopping"
            await self._publish_status()
            await self._append_log("Stopping realtime translator...")
            graceful_shutdown_requested = False
            if process.stdin is not None:
                try:
                    process.stdin.write(b"SHUTDOWN\n")
                    await process.stdin.drain()
                    graceful_shutdown_requested = True
                except (BrokenPipeError, ConnectionResetError):
                    logger.info("[runtime.stop] control pipe already closed")
            if not graceful_shutdown_requested:
                await self._kill_process_tree(process)
                await process.wait()
            else:
                try:
                    await asyncio.wait_for(process.wait(), timeout=8.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[runtime.stop] graceful shutdown timeout; killing process tree pid=%s",
                        process.pid,
                    )
                    await self._kill_process_tree(process)
                    await process.wait()
            self._status.state = next_state
            self._status.phase = next_state
            self._status.paused = paused
            self._status.exit_code = process.returncode
            self._status.stopped_at = asyncio.get_running_loop().time()
            self._status.pid = None
            self._release_job_handle()
            self._process = None
            await self._append_log(f"Realtime translator stopped exit_code={process.returncode}")
            await self._publish_status()

    async def _kill_process_tree(self, process: asyncio.subprocess.Process) -> None:
        if sys.platform != "win32":
            process.kill()
            return
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        await killer.wait()

    async def _stream_process(self, process: asyncio.subprocess.Process) -> None:
        async def read_stream(stream, stream_name: str) -> None:
            if stream is None:
                return
            buffer = b""
            while True:
                chunk = await stream.read(512)
                if not chunk:
                    break
                buffer += chunk
                parts = buffer.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
                buffer = parts[-1]
                for part in parts[:-1]:
                    text = part.decode("utf-8", errors="replace").rstrip()
                    if text:
                        await self._append_log(text)
            if buffer:
                text = buffer.decode("utf-8", errors="replace").rstrip()
                if text:
                    await self._append_log(text)
            logger.debug("[runtime.stream] stream=%s eof", stream_name)

        try:
            await asyncio.gather(read_stream(process.stdout, "stdout"), read_stream(process.stderr, "stderr"))
            code = await process.wait()
            if self._process is process:
                self._release_job_handle()
                self._status.state = "idle" if code == 0 else "error"
                self._status.phase = "idle" if code == 0 else "error"
                self._status.paused = False
                self._status.exit_code = code
                self._status.pid = None
                self._status.stopped_at = asyncio.get_running_loop().time()
                self._status.last_error = "" if code == 0 else f"Process exited with code {code}"
                self._process = None
                await self._append_log(f"Realtime translator exited code={code}")
                await self._publish_status()
        except Exception as exc:
            logger.exception("[runtime.stream] failed")
            self._status.state = "error"
            self._status.phase = "error"
            self._status.last_error = str(exc)
            await self._publish_status()

    def _release_job_handle(self) -> None:
        handle = self._job_handle
        self._job_handle = None
        close_windows_handle(handle)

    async def _append_log(self, line: str) -> None:
        if line.startswith(RUNTIME_EVENT_PREFIX):
            await self._handle_runtime_event(line[len(RUNTIME_EVENT_PREFIX):])
            return
        progress_key = line.split("]", 1)[0] if line.startswith("[download:") else ""
        if progress_key and self._logs and self._logs[-1].startswith(progress_key + "]"):
            self._logs[-1] = line
        else:
            self._logs.append(line)
        dead: list[asyncio.Queue[str]] = []
        for queue in self._log_subscribers:
            try:
                put_latest(queue, line)
            except Exception:
                dead.append(queue)
        for queue in dead:
            self._log_subscribers.discard(queue)

    async def _handle_runtime_event(self, event: str) -> None:
        if event == "initializing_stt":
            self._status.phase = "initializing_stt"
        elif event == "initializing_llm":
            self._status.phase = "initializing_llm"
        elif event == "initialization_ready":
            self._stt_active = False
            self._llm_active = False
            self._status.phase = "ready"
        elif event == "stt_start":
            self._stt_active = True
        elif event == "stt_end":
            self._stt_active = False
        elif event == "llm_start":
            self._llm_active = True
        elif event == "llm_end":
            self._llm_active = False
        else:
            return

        if event in {"stt_start", "stt_end", "llm_start", "llm_end"}:
            self._status.phase = (
                "translating"
                if self._llm_active
                else "transcribing"
                if self._stt_active
                else "ready"
            )
        await self._publish_status()

    async def _publish_status(self) -> None:
        payload = self.status()
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self._status_subscribers:
            try:
                put_latest(queue, payload)
            except Exception:
                dead.append(queue)
        for queue in dead:
            self._status_subscribers.discard(queue)


def put_latest(queue: asyncio.Queue[Any], item: Any) -> None:
    while queue.full():
        queue.get_nowait()
    queue.put_nowait(item)
