import asyncio
import contextlib
import logging
import sys
import os
import time
import threading
from collections import deque
from collections.abc import Awaitable, Callable

# Ensure the root of the project is in path or relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from datetime import datetime
from services.stt.audio import describe_audio_devices

from services.stt.service import STTService
from services.llm.translator import LLMTranslator
from services.browser.overlay import SubtitleOverlay
from services.browser.status import BrowserStatusServer, STORE
from services.runtime.events import emit_runtime_event
from services.filters.text import is_context_leak, strip_repeated_subtitle_prefix

logger = logging.getLogger(__name__)


class RealtimeUtteranceState:
    def __init__(self) -> None:
        self.latest_sequence = 0


async def stream_realtime_utterances(
    stt_service: STTService,
    max_pending: int = 2,
    max_age_s: float = 3.0,
    state: RealtimeUtteranceState | None = None,
):
    """Keep STT consuming live PCM while downstream translation is busy."""
    queue: asyncio.Queue[tuple[str, int, float, str | None]] = asyncio.Queue(
        maxsize=max(1, int(max_pending))
    )
    state = state or RealtimeUtteranceState()
    producer_error: BaseException | None = None
    dropped = 0

    async def produce() -> None:
        nonlocal producer_error, dropped
        try:
            async for utterance in stt_service.stream_utterances():
                state.latest_sequence += 1
                if queue.full():
                    queue.get_nowait()
                    dropped += 1
                    if dropped == 1 or dropped % 10 == 0:
                        logger.warning(
                            "Dropped %s queued STT utterance(s); translation "
                            "is behind realtime audio",
                            dropped,
                        )
                queue.put_nowait(
                    (
                        "utterance",
                        state.latest_sequence,
                        time.monotonic(),
                        utterance,
                    )
                )
        except BaseException as exc:
            producer_error = exc
        finally:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(("done", state.latest_sequence, time.monotonic(), None))

    producer_task = asyncio.create_task(
        produce(),
        name="realtime-stt-producer",
    )
    try:
        while True:
            kind, sequence, queued_at, utterance = await queue.get()
            if kind == "done":
                if producer_error is not None:
                    raise producer_error
                return

            stale_skipped = 0
            done_after_latest = False
            if max_age_s > 0 and time.monotonic() - queued_at > max_age_s:
                newest_found = False
                while not queue.empty():
                    candidate = queue.get_nowait()
                    if candidate[0] == "done":
                        done_after_latest = True
                        continue
                    kind, sequence, queued_at, utterance = candidate
                    newest_found = True
                    stale_skipped += 1
                logger.warning(
                    "Skipped stale translation backlog%s "
                    "(discarded=%s newest_age=%.2fs)",
                    "; using newest utterance" if newest_found else "",
                    stale_skipped if newest_found else 1,
                    time.monotonic() - queued_at,
                )
                if done_after_latest:
                    queue.put_nowait(
                        ("done", state.latest_sequence, time.monotonic(), None)
                    )
                if (
                    not newest_found
                    or time.monotonic() - queued_at > max_age_s
                ):
                    continue

            if utterance is not None:
                yield sequence, utterance
    finally:
        producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task


class SubtitleDisplayHistory:
    """Track only subtitles that were actually published to the viewer."""

    def __init__(self) -> None:
        self._last_translation = ""

    def advance(self, translation: str) -> tuple[str, str]:
        current = translation.strip()
        previous = self._last_translation
        self._last_translation = current
        return previous, current


def subtitle_display_duration(
    translation: str,
    pending_subtitles: int = 0,
    words_per_second: float = 4.0,
    minimum_seconds: float = 1,
    maximum_seconds: float = 4.0,
    catch_up_factor: float = 0.75,
) -> float:
    word_count = len(translation.split())
    reading_seconds = word_count / words_per_second if words_per_second > 0 else maximum_seconds
    display_seconds = max(minimum_seconds, min(maximum_seconds, reading_seconds))
    if pending_subtitles >= 2:
        display_seconds = max(minimum_seconds, display_seconds * catch_up_factor)
    return display_seconds


class SubtitleDisplayManager:
    """Show translated subtitles one at a time for a readable duration."""

    def __init__(
        self,
        publish: Callable[[str, str], None],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_pending: int = 2,
        max_age_s: float = 4.0,
    ) -> None:
        self._publish = publish
        self._sleep = sleep
        self._history = SubtitleDisplayHistory()
        self._queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue(
            maxsize=max(1, int(max_pending))
        )
        self._max_age_s = max(0.0, float(max_age_s))
        self._pending_changed = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._display_subtitles(),
                name="subtitle-display",
            )

    def add(self, translation: str) -> None:
        current = translation.strip()
        if current:
            if self._queue.full():
                dropped, _ = self._queue.get_nowait()
                logger.warning(
                    "Dropped stale pending subtitle to stay realtime: %r",
                    dropped,
                )
            self._queue.put_nowait((current, time.monotonic()))
            if self._queue.qsize() >= 2:
                self._pending_changed.set()

    async def close(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None

    async def _display_subtitles(self) -> None:
        while True:
            translation, queued_at = await self._queue.get()
            queued_for_s = time.monotonic() - queued_at
            if self._max_age_s > 0 and queued_for_s > self._max_age_s:
                logger.warning(
                    "Skipped stale subtitle display backlog (age=%.2fs): %r",
                    queued_for_s,
                    translation,
                )
                continue
            previous, current = self._history.advance(translation)
            self._publish(previous, current)
            rendered_at = time.monotonic()
            pending_subtitles = self._queue.qsize()
            display_seconds = subtitle_display_duration(
                current,
                pending_subtitles=pending_subtitles,
            )
            logger.debug(
                "Rendered subtitle for %.2fs (pending=%d): %r",
                display_seconds,
                pending_subtitles,
                current,
            )
            await self._wait_for_display_time(current, rendered_at)

    async def _wait_for_display_time(
        self,
        translation: str,
        rendered_at: float,
    ) -> None:
        while True:
            display_seconds = subtitle_display_duration(
                translation,
                pending_subtitles=self._queue.qsize(),
            )
            remaining_seconds = display_seconds - (time.monotonic() - rendered_at)
            if remaining_seconds <= 0:
                return

            self._pending_changed.clear()
            sleep_task = asyncio.create_task(self._sleep(remaining_seconds))
            backlog_task = asyncio.create_task(self._pending_changed.wait())
            try:
                done, pending = await asyncio.wait(
                    (sleep_task, backlog_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in (sleep_task, backlog_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    sleep_task,
                    backlog_task,
                    return_exceptions=True,
                )

            if sleep_task in done:
                return


class RuntimeCommandListener:
    def __init__(
        self,
        stt_service: STTService,
        request_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self.stt_service = stt_service
        self.request_shutdown = request_shutdown
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stdin or not sys.stdin.readable():
            return
        self._thread = threading.Thread(target=self._listen, name="runtime-control", daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        for raw_line in sys.stdin:
            command = raw_line.strip().upper()
            if command == "PAUSE":
                self.stt_service.set_paused(True)
            elif command == "RESUME":
                self.stt_service.set_paused(False)
            elif command == "SHUTDOWN" and self.request_shutdown is not None:
                self.request_shutdown()
            elif command:
                logger.warning("Unknown runtime control command: %s", command)


def append_translation_log(log_path: str, payload: str, max_bytes: int) -> None:
    encoded_size = len(payload.encode("utf-8"))
    if max_bytes > 0 and os.path.exists(log_path):
        try:
            if os.path.getsize(log_path) + encoded_size > max_bytes:
                stem, extension = os.path.splitext(log_path)
                backup_path = f"{stem}.1{extension}"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.replace(log_path, backup_path)
                logger.info("Rotated translation log to: %s", backup_path)
        except OSError as exc:
            logger.warning("Failed to rotate translation log: %s", exc)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(payload)


def is_translatable_text(text: str) -> bool:
    if not text:
        return False
    alnum_count = sum(ch.isalnum() for ch in text)
    return alnum_count >= 3


def is_identity_translation(source_text: str, translated_text: str) -> bool:
    source_norm = " ".join(source_text.strip().split()).casefold()
    translated_norm = " ".join(translated_text.strip().split()).casefold()
    return bool(source_norm) and source_norm == translated_norm

async def main():
    exit_code = 0
    config = load_config()
    
    # Configure logging
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("sherpa_onnx").setLevel(logging.INFO)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    
    logger.info("Starting Realtime Screen Translator")
    logger.info("Config path env: %s", os.environ.get("REALTIME_TRANSLATOR_CONFIG", ""))
    logger.info(
        "Loaded settings: stt_model=%s stt_provider=%s translation_model=%s",
        getattr(config.stt, "sherpa_model_id", "auto"),
        getattr(config.stt, "sherpa_onnx_provider", "cpu"),
        getattr(config.translation, "model_path", ""),
    )
    logger.info("Audio capture mode: %s", config.audio.capture_mode)
    if config.audio.capture_mode == "input_device" and config.audio.device_name:
        logger.info("Preferred audio device match: %s", config.audio.device_name)
    if config.audio.capture_mode == "input_device":
        device_lines = describe_audio_devices()
        if device_lines:
            logger.info("Available audio devices:")
            for line in device_lines:
                logger.info("  %s", line)
        else:
            logger.warning("No audio devices were enumerated by sounddevice.")
    
    # Initialize services
    emit_runtime_event("initializing_stt")
    logger.info("Initializing STT Service...")
    stt_service = STTService(config)
    runtime_loop = asyncio.get_running_loop()
    runtime_task = asyncio.current_task()

    def request_runtime_shutdown() -> None:
        if runtime_task is not None:
            runtime_loop.call_soon_threadsafe(runtime_task.cancel)

    runtime_commands = RuntimeCommandListener(stt_service, request_runtime_shutdown)
    runtime_commands.start()
    emit_runtime_event("initializing_llm")
    logger.info("Initializing LLM Translator Service...")
    translator_service = LLMTranslator(config)
    overlay = None
    browser_bridge = None
    display_manager = None
    if getattr(config, "overlay", None) and getattr(config.overlay, "enabled", False):
        STORE.set_overlay_config(
            mode=str(getattr(config.overlay, "mode", "desktop")),
            show_source=bool(getattr(config.overlay, "show_source", True)),
            opacity=float(getattr(config.overlay, "opacity", 0.82)),
            font_size=int(getattr(config.overlay, "font_size", 23)),
            source_font_size=int(getattr(config.overlay, "source_font_size", 15)),
        )
        if getattr(config.overlay, "browser_bridge_enabled", False):
            browser_bridge = BrowserStatusServer(
                host=str(getattr(config.overlay, "browser_bridge_host", "127.0.0.1")),
                port=int(getattr(config.overlay, "browser_bridge_port", 8765)),
            )
            browser_bridge.start()
        if str(getattr(config.overlay, "mode", "desktop")).strip().lower() == "desktop":
            logger.info("Initializing desktop subtitle overlay...")
            overlay = SubtitleOverlay(config.overlay)
            overlay.start()
        else:
            logger.info("Browser subtitle overlay mode enabled; desktop overlay is disabled.")

    def publish_subtitle(previous: str, current: str) -> None:
        if browser_bridge:
            STORE.set_subtitles(previous, current)
        if overlay:
            overlay.publish(previous, current)

    display_manager = SubtitleDisplayManager(publish_subtitle)
    display_manager.start()
    
    # Setup translation log file
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translations_log.txt")
    logger.info(f"Translations will be saved to: {log_file_path}")
    
    logger.info("Initialization complete. Entering main loop...")
    emit_runtime_event("initialization_ready")
    
    try:
        context_subtitles = max(
            0,
            min(5, int(getattr(config.translation, "context_subtitles", 3))),
        )
        context_history: deque[tuple[str, str]] = deque(
            maxlen=max(1, context_subtitles)
        )
        previous_context_at = 0.0
        realtime_utterance_state = RealtimeUtteranceState()
        
        async for utterance_sequence, text_chunk in stream_realtime_utterances(
            stt_service,
            max_pending=int(
                getattr(config.translation, "realtime_queue_max_utterances", 2)
            ),
            max_age_s=float(
                getattr(config.translation, "realtime_max_utterance_age_s", 3.0)
            ),
            state=realtime_utterance_state,
        ):
            if text_chunk and is_translatable_text(text_chunk):
                if stt_service.is_paused:
                    continue
                context_max_age_s = float(getattr(config.translation, "context_max_age_s", 15.0))
                if (
                    context_history
                    and (
                        context_max_age_s <= 0
                        or (time.monotonic() - previous_context_at) > context_max_age_s
                    )
                ):
                    context_history.clear()
                    previous_context_at = 0.0
                context_is_fresh = (
                    context_subtitles > 0
                    and bool(context_history)
                )
                context_for_translation = (
                    "\n".join(source for source, _ in context_history)
                    if context_is_fresh
                    else ""
                )
                previous_translation_for_context = (
                    "\n".join(translation for _, translation in context_history)
                    if context_is_fresh
                    else ""
                )
                previous_source_text, previous_translation = (
                    context_history[-1] if context_is_fresh else ("", "")
                )
                
                # Perform translation
                # Run translation in thread to avoid blocking asyncio event loop
                translation_started_at = time.monotonic()
                emit_runtime_event("llm_start")
                try:
                    translated_text = await asyncio.to_thread(
                        translator_service.translate,
                        text_chunk,
                        context_for_translation,
                        previous_translation_for_context,
                    )
                finally:
                    emit_runtime_event("llm_end")
                translation_elapsed_s = time.monotonic() - translation_started_at
                if translation_elapsed_s >= 2.0:
                    logger.warning(
                        "Slow translation completed: elapsed=%.2fs source=%r",
                        translation_elapsed_s,
                        text_chunk,
                    )
                max_translation_latency_s = float(
                    getattr(
                        config.translation,
                        "realtime_max_translation_latency_s",
                        3.0,
                    )
                )
                if (
                    max_translation_latency_s > 0
                    and translation_elapsed_s >= max_translation_latency_s
                    and utterance_sequence
                    < realtime_utterance_state.latest_sequence
                ):
                    logger.warning(
                        "Discarding stale translation result: sequence=%s "
                        "latest=%s elapsed=%.2fs",
                        utterance_sequence,
                        realtime_utterance_state.latest_sequence,
                        translation_elapsed_s,
                    )
                    continue
                if stt_service.is_paused:
                    logger.info("[FIX:runtime-pause] Discarded translation completed after pause")
                    continue
                should_reset_context = translator_service.consume_context_reset_request()
                leak_settings = config.translation
                if is_context_leak(
                    current_source=text_chunk,
                    previous_source=context_for_translation,
                    translation=translated_text,
                    previous_translation=previous_translation_for_context,
                    translation_similarity_threshold=float(
                        getattr(leak_settings, "context_leak_similarity_threshold", 0.68)
                    ),
                    source_similarity_limit=float(
                        getattr(leak_settings, "context_leak_source_similarity_limit", 0.55)
                    ),
                ):
                    logger.warning(
                        "Possible translation context leak detected; retrying without previous context"
                    )
                    emit_runtime_event("llm_start")
                    try:
                        translated_text = await asyncio.to_thread(
                            translator_service.translate,
                            text_chunk,
                            "",
                        )
                    finally:
                        emit_runtime_event("llm_end")
                    should_reset_context = True
                    translator_service.consume_context_reset_request()
                    if is_context_leak(
                        current_source=text_chunk,
                        previous_source=context_for_translation,
                        translation=translated_text,
                        previous_translation=previous_translation_for_context,
                        translation_similarity_threshold=float(
                            getattr(leak_settings, "context_leak_similarity_threshold", 0.68)
                        ),
                        source_similarity_limit=float(
                            getattr(leak_settings, "context_leak_source_similarity_limit", 0.55)
                        ),
                    ):
                        logger.warning("Skipping translation after repeated context leak: %r", text_chunk)
                        context_history.clear()
                        previous_context_at = 0.0
                        continue
                translated_text, removed_duplicate_prefix, duplicate_similarity = (
                    strip_repeated_subtitle_prefix(
                        previous_translation=previous_translation,
                        current_translation=translated_text,
                        previous_source=previous_source_text,
                        current_source=text_chunk,
                        source_language=str(getattr(config, "language", "")),
                        similarity_threshold=float(
                            getattr(
                                leak_settings,
                                "subtitle_duplicate_similarity_threshold",
                                0.82,
                            )
                        ),
                        min_duplicate_chars=int(
                            getattr(leak_settings, "subtitle_duplicate_min_chars", 12)
                        ),
                        source_repeat_similarity_threshold=float(
                            getattr(
                                leak_settings,
                                "subtitle_source_repeat_similarity_threshold",
                                0.78,
                            )
                        ),
                        cjk_source_repeat_similarity_threshold=float(
                            getattr(
                                leak_settings,
                                "subtitle_cjk_source_repeat_similarity_threshold",
                                0.90,
                            )
                        ),
                    )
                )
                if removed_duplicate_prefix:
                    should_reset_context = True
                    logger.warning(
                        "Removed repeated previous-subtitle prefix: similarity=%.3f remainder=%r",
                        duplicate_similarity,
                        translated_text,
                    )
                if not translated_text or not is_translatable_text(translated_text):
                    logger.warning("Skipping unstable or empty translation for: %r", text_chunk)
                    if should_reset_context:
                        context_history.clear()
                        previous_context_at = 0.0
                    continue
                if is_identity_translation(text_chunk, translated_text):
                    logger.info("Skipping identity translation: %r", text_chunk)
                    if should_reset_context:
                        context_history.clear()
                        previous_context_at = 0.0
                    continue
                
                logger.debug(
                    "Queued subtitle for display: current=%r context_items=%d",
                    translated_text,
                    len(context_history),
                )
                display_manager.add(translated_text)
                
                # Print to console for the user to see easily
                print(f"\n[Source] {text_chunk}")
                print(f"[Transl] {translated_text}\n")
                
                # Write to log file
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_translation_log(
                    log_path=log_file_path,
                    payload=f"{current_time}\n{text_chunk}\n{translated_text}\n\n",
                    max_bytes=int(getattr(config, "translation_log_max_bytes", 10_000_000)),
                )
                
                if should_reset_context:
                    context_history.clear()
                if context_subtitles > 0:
                    context_history.append((text_chunk, translated_text))
                    previous_context_at = time.monotonic()
            elif text_chunk:
                logger.info("Skipping low-information STT chunk: %r", text_chunk)
                
    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
        exit_code = 130
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        exit_code = 1
    finally:
        translator_service.close()
        if display_manager:
            await display_manager.close()
        if overlay:
            overlay.stop()
        if browser_bridge:
            browser_bridge.stop()
        # Cleanup
        logger.info("Graceful shutdown complete.")
    return exit_code

if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        if os.name == "nt":
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(exit_code)
        raise SystemExit(exit_code)
