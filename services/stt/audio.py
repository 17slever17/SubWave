from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Optional
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np
import sounddevice as sd
import soxr

logger = logging.getLogger(__name__)


@dataclass
class AudioSourceSpec:
    capture_mode: str
    device_id: int | None = None
    device_name: str = ""
    native_samplerate: int = 16000
    channels: int = 1


@dataclass
class PCMChunk:
    samples: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float
    source_name: str


class PCMChunkBuffer:
    def __init__(self, max_chunks: int = 256, drop_oldest_on_overflow: bool = True):
        self._queue: Queue[PCMChunk] = Queue(maxsize=max_chunks)
        self.max_chunks = max_chunks
        self.drop_oldest_on_overflow = drop_oldest_on_overflow
        self.dropped_chunks = 0
        self.total_pushed = 0

    def push(self, chunk: PCMChunk) -> None:
        try:
            self._queue.put_nowait(chunk)
            self.total_pushed += 1
            return
        except Full:
            if not self.drop_oldest_on_overflow:
                self.dropped_chunks += 1
                return

        try:
            self._queue.get_nowait()
        except Empty:
            pass

        try:
            self._queue.put_nowait(chunk)
            self.total_pushed += 1
            self.dropped_chunks += 1
        except Full:
            self.dropped_chunks += 1

    def get_nowait(self) -> PCMChunk:
        return self._queue.get_nowait()

    def qsize(self) -> int:
        return self._queue.qsize()


def query_audio_devices() -> list[dict]:
    try:
        return list(sd.query_devices())
    except Exception as exc:
        logger.error("Failed to query audio devices: %s", exc)
        return []


def describe_audio_devices() -> list[str]:
    lines: list[str] = []
    for idx, device in enumerate(query_audio_devices()):
        hostapi = _hostapi_name(device) or "Unknown"
        lines.append(
            f"[{idx}] {device.get('name', '<unknown>')} | hostapi={hostapi} | "
            f"in={device.get('max_input_channels', 0)} | out={device.get('max_output_channels', 0)} | "
            f"default_sr={int(device.get('default_samplerate', 0) or 0)}"
        )
    return lines


def _hostapi_name(device: dict) -> str:
    try:
        return str(sd.query_hostapis(device["hostapi"])["name"])
    except Exception:
        return ""


def _contains_name(device: dict, target_name: str) -> bool:
    return target_name.lower() in str(device.get("name", "")).lower()


def find_audio_device(target_name: str = "CABLE Output", fallback_id: Optional[int] = None) -> Optional[int]:
    devices = query_audio_devices()
    best_match_id = None
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0 and _contains_name(device, target_name):
            hostapi = _hostapi_name(device)
            if "WASAPI" in hostapi:
                logger.info("Found target input device (WASAPI): %s (ID: %s)", device["name"], i)
                return i
            if best_match_id is None:
                best_match_id = i

    if best_match_id is not None:
        logger.info("Found target input device: %s (ID: %s)", devices[best_match_id]["name"], best_match_id)
        return best_match_id

    logger.warning("Target audio input containing '%s' not found.", target_name)

    if fallback_id is not None and fallback_id < len(devices):
        device = devices[fallback_id]
        if device.get("max_input_channels", 0) > 0:
            logger.info("Using fallback input device: %s (ID: %s)", device["name"], fallback_id)
            return fallback_id

    try:
        default_input = sd.default.device[0]
        if default_input is not None:
            device = devices[default_input]
            if device.get("max_input_channels", 0) > 0:
                logger.info("Using default input device: %s (ID: %s)", device["name"], default_input)
                return default_input
    except Exception as exc:
        logger.error("Failed to get default input audio device: %s", exc)

    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            logger.info("Using first available input device: %s (ID: %s)", device["name"], i)
            return i

    logger.error("No input audio devices found.")
    return None


def build_audio_source(config) -> AudioSourceSpec:
    audio_conf = getattr(config, "audio", config)
    capture_mode = str(
        getattr(audio_conf, "capture_mode", "input_device")
    ).strip().lower()
    if capture_mode == "browser_tab":
        return AudioSourceSpec(
            capture_mode="browser_tab",
            device_name="browser-tab",
            native_samplerate=48000,
            channels=1,
        )
    target_name = getattr(audio_conf, "device_name", "")
    fallback_id = getattr(audio_conf, "fallback_mic_id", None)
    device_id = find_audio_device(target_name=target_name or "CABLE Output", fallback_id=fallback_id)
    if device_id is None:
        return AudioSourceSpec(capture_mode=capture_mode)
    device = query_audio_devices()[device_id]
    return AudioSourceSpec(
        capture_mode="input_device",
        device_id=device_id,
        device_name=str(device["name"]),
        native_samplerate=int(device["default_samplerate"]),
        channels=max(1, int(device["max_input_channels"])),
    )


class BrowserPCMWebSocketCapture:
    """Receives mono float32 PCM from the browser extension."""

    def __init__(
        self,
        host: str,
        port: int,
        target_samplerate: int,
        gain: float,
        buffer: PCMChunkBuffer,
    ):
        self.host = str(host)
        self.port = int(port)
        self.target_samplerate = int(target_samplerate)
        self.gain = float(gain)
        self.buffer = buffer
        self._active_connection = None
        self._active_session_started_at = 0
        self._connection_lock = asyncio.Lock()

    async def _handle_connection(self, websocket) -> None:
        request = getattr(websocket, "request", None)
        request_path = getattr(request, "path", "")
        request_headers = getattr(request, "headers", {})
        origin = request_headers.get("Origin", "") if request_headers else ""
        parsed_request = urlsplit(request_path)
        if parsed_request.path != "/audio":
            await websocket.close(code=1008, reason="Invalid audio endpoint")
            return
        if origin:
            parsed_origin = urlsplit(origin)
            if (
                parsed_origin.scheme != "chrome-extension"
                or not parsed_origin.hostname
            ):
                await websocket.close(code=1008, reason="Invalid origin")
                return
        query = parse_qs(parsed_request.query)
        try:
            session_started_at = int(query.get("session_started_at", [""])[0])
        except ValueError:
            session_started_at = 0
        if session_started_at <= 0:
            session_started_at = int(time.time() * 1000)
        async with self._connection_lock:
            previous_connection = self._active_connection
            if (
                previous_connection is not None
                and session_started_at < self._active_session_started_at
            ):
                await websocket.close(
                    code=4001,
                    reason="A newer browser capture is active",
                )
                return
            self._active_connection = websocket
            self._active_session_started_at = session_started_at
            if previous_connection is not None:
                await previous_connection.close(
                    code=4001,
                    reason="Capture moved to another browser",
                )

        resampler = StreamingAudioResampler(48000, self.target_samplerate)
        expected_sequence: int | None = None
        logger.info("Browser audio extension connected.")
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "ready",
                        "format": "f32le",
                        "target_sample_rate": self.target_samplerate,
                    }
                )
            )
            async for message in websocket:
                if isinstance(message, str):
                    payload = json.loads(message)
                    if payload.get("type") != "stream_start":
                        continue
                    input_rate = int(payload.get("sample_rate", 48000))
                    channels = int(payload.get("channels", 1))
                    sample_format = str(payload.get("format", "f32le"))
                    if input_rate < 8000 or input_rate > 192000:
                        raise ValueError(
                            f"Unsupported browser sample rate: {input_rate}"
                        )
                    if channels != 1 or sample_format != "f32le":
                        raise ValueError(
                            "Unsupported browser PCM format: "
                            f"{sample_format}/{channels}ch"
                        )
                    resampler = StreamingAudioResampler(
                        input_rate,
                        self.target_samplerate,
                    )
                    expected_sequence = None
                    logger.info(
                        "Browser PCM stream started: input_sr=%s target_sr=%s",
                        input_rate,
                        self.target_samplerate,
                    )
                    continue

                if len(message) < 4 or len(message) > 1_048_580:
                    raise ValueError(
                        f"Invalid browser PCM frame size: {len(message)}"
                    )
                sequence = int.from_bytes(
                    message[:4],
                    "little",
                    signed=False,
                )
                if expected_sequence is not None and sequence != expected_sequence:
                    logger.warning(
                        "Browser PCM sequence gap: expected=%s received=%s",
                        expected_sequence,
                        sequence,
                    )
                    resampler.reset()
                expected_sequence = (sequence + 1) & 0xFFFFFFFF
                samples = np.frombuffer(message, dtype="<f4", offset=4)
                if not samples.size or not np.all(np.isfinite(samples)):
                    continue
                pcm = resampler.process(samples)
                if pcm.size:
                    self.buffer.push(
                        PCMChunk(
                            samples=np.clip(
                                pcm * self.gain,
                                -1.0,
                                1.0,
                            ),
                            sample_rate=self.target_samplerate,
                            channels=1,
                            timestamp=time.time(),
                            source_name="browser-tab",
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Browser audio connection ended with error: %s",
                exc,
            )
        finally:
            async with self._connection_lock:
                if self._active_connection is websocket:
                    self._active_connection = None
                    self._active_session_started_at = 0
            logger.info("Browser audio extension disconnected.")

    async def serve(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError(
                "Browser tab capture requires the 'websockets' package."
            ) from exc

        async with serve(
            self._handle_connection,
            self.host,
            self.port,
            compression=None,
            max_size=1_048_580,
            ping_interval=20,
            ping_timeout=20,
        ):
            logger.info(
                "Browser PCM bridge listening on ws://%s:%s/audio",
                self.host,
                self.port,
            )
            await asyncio.Future()





def normalize_mono_audio_chunk(
    audio_data: np.ndarray,
    native_samplerate: int,
    target_samplerate: int,
    gain: float = 1.0,
) -> np.ndarray:
    audio_data = np.asarray(audio_data, dtype=np.float32)
    audio_data = audio_data.astype(np.float32)

    if native_samplerate != target_samplerate and audio_data.size:
        audio_data = soxr.resample(
            audio_data,
            native_samplerate,
            target_samplerate,
            quality="HQ",
        ).astype(np.float32, copy=False)

    return np.clip(audio_data * gain, -1.0, 1.0)


class StreamingAudioResampler:
    """Anti-aliased mono resampler that preserves filter state across callbacks."""

    def __init__(self, input_rate: int, output_rate: int, quality: str = "HQ"):
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        self.quality = quality
        self._stream = None
        if self.input_rate != self.output_rate:
            self._stream = soxr.ResampleStream(
                self.input_rate,
                self.output_rate,
                num_channels=1,
                dtype="float32",
                quality=quality,
            )

    def process(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._stream is None:
            return audio
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)
        return self._stream.resample_chunk(audio, last=False).astype(np.float32, copy=False)

    def flush(self) -> np.ndarray:
        if self._stream is None:
            return np.empty(0, dtype=np.float32)
        return self._stream.resample_chunk(
            np.empty(0, dtype=np.float32),
            last=True,
        ).astype(np.float32, copy=False)

    def reset(self) -> None:
        if self._stream is not None:
            self._stream.clear()


class DevicePCMStreamCapture:
    def __init__(
        self,
        spec: AudioSourceSpec,
        target_samplerate: int,
        chunk_size_ms: int,
        gain: float,
        buffer: PCMChunkBuffer,
        processor=None,
    ):
        self.spec = spec
        self.target_samplerate = int(target_samplerate)
        self.chunk_size_ms = int(chunk_size_ms)
        self.gain = float(gain)
        self.buffer = buffer
        self.processor = processor
        self._resampler = StreamingAudioResampler(
            input_rate=int(self.spec.native_samplerate),
            output_rate=self.target_samplerate,
        )
        self._stream = None

    @property
    def is_active(self) -> bool:
        return bool(self._stream is not None and self._stream.active)

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio stream status: %s", status)
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.copy()
        mono = np.asarray(mono, dtype=np.float32)
        if self.processor is not None:
            try:
                mono = self.processor.process(mono)
            except Exception as exc:
                logger.exception("Audio processor failed, using raw PCM for this chunk: %s", exc)
        pcm = self._resampler.process(mono)
        pcm = np.clip(pcm * self.gain, -1.0, 1.0)
        self._push_pcm(pcm)

    def _push_pcm(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        self.buffer.push(
            PCMChunk(
                samples=pcm,
                sample_rate=self.target_samplerate,
                channels=1,
                timestamp=time.time(),
                source_name=self.spec.device_name or f"device:{self.spec.device_id}",
            )
        )

    def start(self):
        self._resampler.reset()
        blocksize = int(self.chunk_size_ms * int(self.spec.native_samplerate) / 1000)
        self._stream = make_sounddevice_stream(self.spec, self._callback, blocksize)
        self._stream.start()
        logger.info(
            "PCM capture started: mode=%s device=%s (%s) native_sr=%s target_sr=%s buffer=%s",
            self.spec.capture_mode,
            self.spec.device_id,
            self.spec.device_name,
            self.spec.native_samplerate,
            self.target_samplerate,
            self.buffer.max_chunks,
        )
        return self

    def stop(self):
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        tail = self._resampler.flush()
        self._push_pcm(np.clip(tail * self.gain, -1.0, 1.0))
        logger.info(
            "PCM capture stopped. total_pushed=%s dropped=%s remaining=%s",
            self.buffer.total_pushed,
            self.buffer.dropped_chunks,
            self.buffer.qsize(),
        )
def make_sounddevice_stream(spec: AudioSourceSpec, callback, blocksize: int):
    if spec.device_id is None:
        raise RuntimeError("Audio source device is not configured.")
    stream_kwargs = {
        "samplerate": spec.native_samplerate,
        "channels": spec.channels,
        "callback": callback,
        "blocksize": blocksize,
        "dtype": "float32",
        "device": spec.device_id,
    }
    return sd.InputStream(**stream_kwargs)
