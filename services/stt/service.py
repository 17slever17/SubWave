from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import hashlib
import logging
import os
from queue import Empty
import re
import sys
import time
from pathlib import Path

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.append(root_dir)

from services.stt.audio import (  # noqa: E402
    BrowserPCMWebSocketCapture,
    PCMChunkBuffer,
    DevicePCMStreamCapture,
    build_audio_source,
)
from services.filters.denoise import build_denoiser, build_stt_denoiser
from services.runtime.paths import resolve_runtime_path
from services.runtime.events import emit_runtime_event
from services.stt.models import create_recognizer, get_sherpa_model, resolve_model_files
from services.vad.ten import TenVadEndpointDetector
from services.vad.firered import FireRedVadEndpointDetector
from services.filters.text import (
    cap_repeated_token_sequences,
    normalize_for_comparison,
    text_preview,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+([,.!?:;])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not re.search(r"[^\W_]", text):
        return ""

    garbage_pattern = r"^[.\s,。、!?！？\-\(\)\[\]]*$"
    if re.match(garbage_pattern, text):
        return ""

    return text


def is_meaningful_text(
    text: str,
    min_words: int = 2,
    min_alnum_chars: int = 6,
    min_single_word_chars: int = 3,
) -> bool:
    if not text:
        return False

    # 1. Считаем буквенно-цифровые символы (Unicode-safe)
    alnum_count = sum(ch.isalnum() for ch in text)

    # 2. Проверка на CJK (Японский: Hiragana, Katakana, Kanji; Китайский; Корейский)
    # Диапазоны: \u3040-\u30ff (кана), \u4e00-\u9faf (иероглифы), \uac00-\ud7af (хангыль)
    is_cjk = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9faf\uac00-\ud7af]", text))

    if is_cjk:
        # В CJK один иероглиф — это часто целое слово. 
        # Порог в 6 символов избыточен, достаточно 2 символов для смысла.
        # Логика min_words игнорируется, так как разделителей-пробелов нет.
        return alnum_count >= 2

    # 3. Стандартная логика для языков с пробелами (RU, EN и т.д.)
    words = [w for w in re.split(r"\s+", text) if any(ch.isalnum() for ch in w)]
    if len(words) == 1:
        return alnum_count >= min_single_word_chars

    return alnum_count >= min_alnum_chars and len(words) >= min_words


class STTService:
    def __init__(self, config):
        self.config = config

        audio_conf = getattr(self.config, "audio", None) or self.config
        stt_conf = getattr(self.config, "stt", None) or self.config

        self.sherpa_model_id = str(getattr(stt_conf, "sherpa_model_id", "auto"))
        self.lang = getattr(self.config, "language", "en")
        self.sampling_rate = int(getattr(audio_conf, "sampling_rate", 16000))
        self.chunk_size_ms = int(getattr(stt_conf, "chunk_size_ms", 60))
        self.process_interval_s = float(getattr(stt_conf, "process_interval_s", 2.0))
        self.window_slack_s = float(getattr(stt_conf, "window_slack_s", 1.0))
        self.process_batch_ms = int(getattr(stt_conf, "process_batch_ms", 720))
        self.window_overlap_s = float(getattr(stt_conf, "window_overlap_s", 0.0))
        self.end_vad_enabled = bool(getattr(stt_conf, "end_vad_enabled", True))
        self.end_vad_backend = str(getattr(stt_conf, "end_vad_backend", "energy")).strip().lower()
        self.firered_vad_model_path = str(
            getattr(
                stt_conf,
                "firered_vad_model_path",
                "models/FireRedVAD/Stream-VAD",
            )
        )
        self.firered_vad_threshold = float(
            getattr(stt_conf, "firered_vad_threshold", 0.4)
        )
        self.firered_vad_min_speech_ms = max(
            0, int(getattr(stt_conf, "firered_vad_min_speech_ms", 80))
        )
        self.firered_vad_rms_confirmation = bool(
            getattr(stt_conf, "firered_vad_rms_confirmation", True)
        )
        self.ten_vad_model_path = str(
            getattr(stt_conf, "ten_vad_model_path", "models/ten-vad.int8.onnx")
        )
        self.ten_vad_threshold = float(getattr(stt_conf, "ten_vad_threshold", 0.20))
        self.ten_vad_min_speech_ms = max(
            0, int(getattr(stt_conf, "ten_vad_min_speech_ms", 100))
        )
        self.end_vad_silence_ms = int(getattr(stt_conf, "end_vad_silence_ms", 450))
        self.long_phrase_after_s = float(getattr(stt_conf, "long_phrase_after_s", 8.0))
        self.long_phrase_end_vad_silence_ms = int(getattr(stt_conf, "long_phrase_end_vad_silence_ms", 200))
        self.max_phrase_s = float(getattr(stt_conf, "max_phrase_s", 10.0))
        self.forced_split_search_s = max(
            0.5,
            float(getattr(stt_conf, "forced_split_search_s", 3.0)),
        )
        self.end_vad_rms_threshold = float(getattr(stt_conf, "end_vad_rms_threshold", 0.010))
        self.end_vad_frame_ms = max(10, int(getattr(stt_conf, "end_vad_frame_ms", 25)))
        self.end_vad_max_active_ratio = float(
            np.clip(getattr(stt_conf, "end_vad_max_active_ratio", 0.15), 0.0, 1.0)
        )
        self.end_vad_recent_ms = max(25, int(getattr(stt_conf, "end_vad_recent_ms", 75)))
        self.speech_activity_rms_threshold = float(getattr(stt_conf, "speech_activity_rms_threshold", 0.018))
        self.speech_activity_min_ratio = float(getattr(stt_conf, "speech_activity_min_ratio", 0.12))
        self.sherpa_onnx_provider = str(getattr(stt_conf, "sherpa_onnx_provider", "cpu"))
        self.sherpa_onnx_num_threads = max(1, int(getattr(stt_conf, "sherpa_onnx_num_threads", 4)))
        self.sherpa_blank_penalty = getattr(stt_conf, "sherpa_blank_penalty", None)
        self.blocked_phrases = list(getattr(stt_conf, "blocked_phrases", []))
        self.duplicate_similarity_threshold = float(
            getattr(stt_conf, "duplicate_similarity_threshold", 0.90)
        )
        self.max_consecutive_duplicate_utterances = max(
            1, int(getattr(stt_conf, "max_consecutive_duplicate_utterances", 1))
        )
        self.max_consecutive_decode_failures = max(
            1, int(getattr(stt_conf, "max_consecutive_decode_failures", 3))
        )
        self.debug_audio_enabled = bool(getattr(stt_conf, "debug_audio_enabled", False))
        self.debug_audio_max_files = max(
            1, int(getattr(stt_conf, "debug_audio_max_files", 100))
        )
        self._debug_audio_index = 0
        self._debug_audio_limit_logged = False
        self._debug_audio_dir = (
            Path(__file__).resolve().parents[2]
            / "debug_audio"
            / f"stt_windows_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        if self.debug_audio_enabled:
            self._debug_audio_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "STT input recording enabled: directory=%s max_files=%s",
                self._debug_audio_dir,
                self.debug_audio_max_files,
            )

        self.input_gain = float(getattr(audio_conf, "input_gain", 1.0))
        self.browser_audio_host = str(
            getattr(audio_conf, "browser_audio_host", "127.0.0.1")
        )
        self.browser_audio_port = int(
            getattr(audio_conf, "browser_audio_port", 8766)
        )
        self.buffer_max_chunks = int(getattr(audio_conf, "buffer_max_chunks", 1024))
        self.drop_oldest_on_overflow = bool(getattr(audio_conf, "drop_oldest_on_overflow", True))
        self.max_pcm_backlog_s = max(
            0.0,
            float(getattr(audio_conf, "max_pcm_backlog_s", 1.5)),
        )
        self._pcm_discontinuity_detected = False

        self.audio_source = build_audio_source(self.config)
        self.pcm_buffer = PCMChunkBuffer(
            max_chunks=self.buffer_max_chunks,
            drop_oldest_on_overflow=self.drop_oldest_on_overflow,
        )
        self._stop_audio = False
        self._paused = False
        self._last_utterance_normalized = ""
        self._consecutive_utterance_count = 0
        denoiser_sample_rate = int(
            self.audio_source.native_samplerate or self.sampling_rate
        )
        self._denoiser = build_denoiser(
            self.config,
            denoiser_sample_rate,
        )
        self._stt_denoiser = build_stt_denoiser(
            self.config,
            denoiser_sample_rate,
            self.lang,
            dfn_fallback=self._denoiser,
        )
        self._denoiser_enabled = bool(getattr(self._denoiser, "enabled", False))
        self._stt_denoiser_enabled = bool(
            getattr(self._stt_denoiser, "enabled", False)
        )
        dfn2_conf = getattr(audio_conf, "dfn2", None)
        self._dfn_vad_atten_lim_db = getattr(
            dfn2_conf,
            "vad_atten_lim_db",
            getattr(dfn2_conf, "atten_lim_db", None),
        )
        self._dfn_stt_atten_lim_db = getattr(dfn2_conf, "stt_atten_lim_db", 24.0)
        self._dfn_vad_context_s = max(1.2, float(getattr(dfn2_conf, "vad_context_s", 3.0)))
        self._vad_check_samples = max(1, int(self.sampling_rate * 0.20))
        logger.info(
            "Audio denoisers: vad=%s stt=%s language=%s",
            type(self._denoiser).__name__ if getattr(self._denoiser, "enabled", False) else "disabled",
            type(self._stt_denoiser).__name__
            if self._stt_denoiser_enabled
            else "disabled",
            self.lang,
        )
        self._endpoint_detectors: dict[int, object] = {}
        self._firered_vad_detector: FireRedVadEndpointDetector | None = None
        self._firered_rms_recheck_generation: int | None = None
        self._init_endpoint_detectors()

        self._sherpa_recognizer = None
        self._sherpa_online = False
        self._init_sherpa_onnx_model()

    def _init_endpoint_detectors(self) -> None:
        if not self.end_vad_enabled or self.end_vad_backend not in {"ten", "firered"}:
            return
        if self.sampling_rate != 16000:
            logger.warning(
                "%s VAD requires 16 kHz audio; falling back to energy endpointing (sample_rate=%s)",
                self.end_vad_backend.upper(),
                self.sampling_rate,
            )
            return

        use_firered = self.end_vad_backend == "firered"
        configured_path = (
            self.firered_vad_model_path if use_firered else self.ten_vad_model_path
        )
        model_path = resolve_runtime_path(configured_path)
        if model_path is None or not model_path.exists():
            logger.warning(
                "%s VAD model is missing at %s; falling back to energy endpointing",
                self.end_vad_backend.upper(),
                model_path,
            )
            return

        try:
            threshold = (
                self.firered_vad_threshold if use_firered else self.ten_vad_threshold
            )
            min_speech_ms = (
                self.firered_vad_min_speech_ms
                if use_firered
                else self.ten_vad_min_speech_ms
            )
            if use_firered:
                self._firered_vad_detector = FireRedVadEndpointDetector(
                    model_path=model_path,
                    sample_rate=self.sampling_rate,
                    threshold=threshold,
                    min_silence_ms=max(0, self.end_vad_silence_ms),
                    min_speech_ms=min_speech_ms,
                    max_speech_s=max(30.0, self.max_phrase_s * 2),
                )
            else:
                silence_values = {
                    max(0, self.end_vad_silence_ms),
                    max(0, self.long_phrase_end_vad_silence_ms),
                }
                for silence_ms in silence_values:
                    self._endpoint_detectors[silence_ms] = TenVadEndpointDetector(
                        model_path=model_path,
                        sample_rate=self.sampling_rate,
                        threshold=threshold,
                        min_silence_ms=silence_ms,
                        min_speech_ms=min_speech_ms,
                        max_speech_s=max(30.0, self.max_phrase_s * 2),
                    )
        except Exception:
            self._endpoint_detectors.clear()
            self._firered_vad_detector = None
            logger.exception(
                "%s VAD initialization failed; falling back to energy endpointing",
                self.end_vad_backend.upper(),
            )
            return

        logger.info(
            "Endpoint detector: %s VAD threshold=%.2f silence=%sms long_silence=%sms",
            "FireRed" if use_firered else "TEN",
            threshold,
            self.end_vad_silence_ms,
            self.long_phrase_end_vad_silence_ms,
        )

    def _accept_endpoint_audio(self, samples: np.ndarray) -> None:
        if self._firered_vad_detector is not None:
            self._firered_vad_detector.accept(samples)
        for detector in self._endpoint_detectors.values():
            detector.accept(samples)

    def _reset_endpoint_detectors(self, retained_audio: np.ndarray | None = None) -> None:
        self._firered_rms_recheck_generation = None
        if self._firered_vad_detector is not None:
            self._firered_vad_detector.reset()
        for detector in self._endpoint_detectors.values():
            detector.reset()
        if retained_audio is not None and retained_audio.size:
            self._accept_endpoint_audio(retained_audio)

    def _active_endpoint_detector(
        self, phrase_duration_s: float
    ) -> object | None:
        if self._firered_vad_detector is not None:
            silence_ms = (
                self.long_phrase_end_vad_silence_ms
                if phrase_duration_s >= self.long_phrase_after_s
                else self.end_vad_silence_ms
            )
            self._firered_vad_detector.set_min_silence_ms(silence_ms)
            return self._firered_vad_detector
        if not self._endpoint_detectors:
            return None
        silence_ms = (
            self.long_phrase_end_vad_silence_ms
            if phrase_duration_s >= self.long_phrase_after_s
            else self.end_vad_silence_ms
        )
        return self._endpoint_detectors.get(max(0, silence_ms))

    def _init_sherpa_onnx_model(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx backend requires the 'sherpa-onnx' package"
            ) from exc

        spec = get_sherpa_model(self.sherpa_model_id, self.lang)
        self._resolved_sherpa_model_id = spec.model_id
        models_root = resolve_runtime_path("models")
        if models_root is None:
            raise FileNotFoundError("Models directory could not be resolved")
        files = resolve_model_files(spec, models_root)
        logger.info(
            "Loading sherpa-onnx model=%s architecture=%s provider=%s threads=%s language=%s blank_penalty=%s",
            spec.model_id,
            spec.architecture,
            self.sherpa_onnx_provider,
            self.sherpa_onnx_num_threads,
            self.lang,
            (
                spec.blank_penalty
                if self.sherpa_blank_penalty is None
                else self.sherpa_blank_penalty
            ),
        )
        self._sherpa_online = spec.architecture == "online_transducer"
        self._sherpa_recognizer = create_recognizer(
            spec,
            files,
            provider=self.sherpa_onnx_provider,
            num_threads=self.sherpa_onnx_num_threads,
            sample_rate=self.sampling_rate,
            blank_penalty=self.sherpa_blank_penalty,
        )
        logger.info("STT initialization complete. backend=sherpa_onnx")

    def _drain_pcm_batch(self) -> np.ndarray:
        max_chunks = max(1, self.process_batch_ms // self.chunk_size_ms)
        chunks: list[np.ndarray] = []
        stale_chunks = 0
        now = time.time()
        while len(chunks) < max_chunks:
            try:
                pcm_chunk = self.pcm_buffer.get_nowait()
            except Empty:
                break
            if (
                self.max_pcm_backlog_s > 0
                and now - pcm_chunk.timestamp > self.max_pcm_backlog_s
            ):
                stale_chunks += 1
                continue
            chunks.append(pcm_chunk.samples)
        if stale_chunks:
            self._pcm_discontinuity_detected = True
            logger.warning(
                "Dropped %s stale PCM chunks to restore realtime audio "
                "(max_backlog=%.2fs)",
                stale_chunks,
                self.max_pcm_backlog_s,
            )
        if not chunks:
            raise Empty()
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def _max_audio_window_samples(self) -> int:
        window_seconds = self.max_phrase_s + self.window_overlap_s + self.window_slack_s
        return max(1, int(window_seconds * self.sampling_rate))

    def _audio_duration_s(self, audio: np.ndarray) -> float:
        return float(audio.size) / self.sampling_rate

    @staticmethod
    def _raise_if_audio_source_stopped(producer_task: asyncio.Task) -> None:
        if not producer_task.done():
            return
        if producer_task.cancelled():
            raise RuntimeError("Audio source task was cancelled unexpectedly")
        error = producer_task.exception()
        if error is not None:
            raise RuntimeError("Audio source task failed") from error
        raise RuntimeError("Audio source task stopped unexpectedly")

    def _transcribe_window_text(
        self,
        audio_window: np.ndarray,
        raw_audio_window: np.ndarray | None = None,
        vad_audio_window: np.ndarray | None = None,
    ) -> TranscriptionResult:
        if audio_window.size == 0:
            return TranscriptionResult("")

        # Keep one owned float32 buffer for both the debug WAV and Sherpa. This
        # makes the recording an exact replay of the samples accepted by STT.
        samples = np.array(audio_window, dtype=np.float32, order="C", copy=True)
        input_hash = self._audio_sha256(samples)
        debug_path = self._save_debug_stt_window(samples)
        raw_path = None
        raw_hash = None
        vad_path = None
        vad_hash = None
        if debug_path is not None and raw_audio_window is not None:
            raw_samples = np.array(raw_audio_window, dtype=np.float32, order="C", copy=True)
            raw_hash = self._audio_sha256(raw_samples)
            raw_path = debug_path.with_name(f"{debug_path.stem}_raw.wav")
            self._write_verified_float_wav(raw_path, raw_samples)
        if debug_path is not None and vad_audio_window is not None:
            vad_samples = np.array(vad_audio_window, dtype=np.float32, order="C", copy=True)
            vad_hash = self._audio_sha256(vad_samples)
            vad_path = debug_path.with_name(f"{debug_path.stem}_vad.wav")
            self._write_verified_float_wav(vad_path, vad_samples)
        result = self._transcribe_sherpa_onnx(samples)
        output_hash = self._audio_sha256(samples)
        if output_hash != input_hash:
            logger.error(
                "STT input buffer changed during decode: before=%s after=%s",
                input_hash,
                output_hash,
            )
        if debug_path is not None:
            debug_path.with_suffix(".txt").write_text(
                (
                    f"sha256: {input_hash}\n"
                    f"samples: {samples.size}\n"
                    f"sample_rate: {self.sampling_rate}\n"
                    f"raw_file: {raw_path.name if raw_path is not None else 'none'}\n"
                    f"raw_sha256: {raw_hash or 'none'}\n"
                    f"vad_file: {vad_path.name if vad_path is not None else 'none'}\n"
                    f"vad_sha256: {vad_hash or 'none'}\n"
                    f"text: {result.text}\n"
                    f"buffer_unchanged_after_decode: {output_hash == input_hash}\n"
                ),
                encoding="utf-8",
            )
            logger.info(
                "STT debug mapping: wav=%s raw=%s sha256=%s text=%r",
                debug_path.name,
                raw_path.name if raw_path is not None else "none",
                input_hash[:12],
                result.text,
            )
        return result

    def _enhance_audio_window(
        self,
        audio_window: np.ndarray,
        *,
        for_stt: bool = False,
    ) -> np.ndarray:
        samples = np.asarray(audio_window, dtype=np.float32)
        denoiser = (
            getattr(self, "_stt_denoiser", self._denoiser)
            if for_stt
            else self._denoiser
        )
        denoiser_enabled = bool(getattr(denoiser, "enabled", False))
        if not denoiser_enabled:
            return samples
        started = time.perf_counter()
        try:
            atten_lim_db = (
                self._dfn_stt_atten_lim_db
                if for_stt
                else self._dfn_vad_atten_lim_db
            )
            enhanced = denoiser.process_window(
                samples,
                self.sampling_rate,
                atten_lim_db=atten_lim_db,
            )
        except Exception:
            logger.exception(
                "Whole-window %s processing failed; using raw audio",
                type(denoiser).__name__,
            )
            enhanced = samples
        enhanced = np.clip(enhanced * self.input_gain, -1.0, 1.0).astype(np.float32, copy=False)
        logger.debug(
            "%s window processed: audio=%.2fs wall=%.3fs",
            type(denoiser).__name__,
            self._audio_duration_s(samples),
            time.perf_counter() - started,
        )
        return enhanced

    @staticmethod
    def _audio_sha256(samples: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(samples, dtype=np.float32)
        return hashlib.sha256(contiguous.tobytes()).hexdigest()

    def _save_debug_stt_window(self, audio_window: np.ndarray) -> Path | None:
        if not self.debug_audio_enabled:
            return None
        if self._debug_audio_index >= self.debug_audio_max_files:
            if not self._debug_audio_limit_logged:
                logger.warning(
                    "STT input recording limit reached: max_files=%s directory=%s",
                    self.debug_audio_max_files,
                    self._debug_audio_dir,
                )
                self._debug_audio_limit_logged = True
            return None

        samples = np.ascontiguousarray(audio_window, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        peak = float(np.max(np.abs(samples), initial=0.0))
        duration = self._audio_duration_s(samples)
        self._debug_audio_index += 1
        path = self._debug_audio_dir / (
            f"{self._debug_audio_index:04d}_{duration:05.2f}s_"
            f"rms{rms:.4f}_peak{peak:.4f}.wav"
        )
        self._write_verified_float_wav(path, samples)
        logger.info(
            "Saved exact float32 STT input window: %s sha256=%s",
            path,
            self._audio_sha256(samples)[:12],
        )
        return path

    def _save_debug_rejected_window(
        self,
        raw_audio: np.ndarray,
        vad_audio: np.ndarray,
        stt_audio: np.ndarray,
        endpoint_detector: object | None,
    ) -> None:
        if not self.debug_audio_enabled:
            return
        if self._debug_audio_index >= self.debug_audio_max_files:
            return

        samples = np.ascontiguousarray(stt_audio, dtype=np.float32)
        raw_samples = np.ascontiguousarray(raw_audio, dtype=np.float32)
        vad_samples = np.ascontiguousarray(vad_audio, dtype=np.float32)
        duration = self._audio_duration_s(samples)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        peak = float(np.max(np.abs(samples), initial=0.0))
        self._debug_audio_index += 1
        stem = (
            f"{self._debug_audio_index:04d}_{duration:05.2f}s_"
            f"rms{rms:.4f}_peak{peak:.4f}_rejected_no_speech"
        )
        stt_path = self._debug_audio_dir / f"{stem}.wav"
        raw_path = self._debug_audio_dir / f"{stem}_raw.wav"
        vad_path = self._debug_audio_dir / f"{stem}_vad.wav"
        self._write_verified_float_wav(stt_path, samples)
        self._write_verified_float_wav(raw_path, raw_samples)
        self._write_verified_float_wav(vad_path, vad_samples)
        stt_path.with_suffix(".txt").write_text(
            (
                "status: rejected_before_asr\n"
                "reason: no_speech_activity\n"
                f"detector: {type(endpoint_detector).__name__ if endpoint_detector else 'rms'}\n"
                f"speech_seen: {getattr(endpoint_detector, 'speech_seen', 'n/a')}\n"
                f"raw_probability: {getattr(endpoint_detector, 'last_raw_probability', 'n/a')}\n"
                f"smoothed_probability: {getattr(endpoint_detector, 'last_smoothed_probability', 'n/a')}\n"
                f"raw_peak: {float(np.max(np.abs(raw_samples), initial=0.0)):.6f}\n"
                f"vad_peak: {float(np.max(np.abs(vad_samples), initial=0.0)):.6f}\n"
                f"stt_peak: {peak:.6f}\n"
            ),
            encoding="utf-8",
        )
        logger.warning(
            "Saved ASR-rejected debug window: stt=%s raw=%s vad=%s",
            stt_path.name,
            raw_path.name,
            vad_path.name,
        )

    def _write_verified_float_wav(self, path: Path, samples: np.ndarray) -> None:
        import soundfile as sf

        sf.write(path, samples, self.sampling_rate, format="WAV", subtype="FLOAT")
        saved, saved_rate = sf.read(path, dtype="float32", always_2d=False)
        if saved_rate != self.sampling_rate or not np.array_equal(saved, samples):
            raise RuntimeError(f"Debug STT WAV verification failed: {path}")

    def _transcribe_sherpa_onnx(self, audio_window: np.ndarray) -> TranscriptionResult:
        if self._sherpa_recognizer is None:
            return TranscriptionResult("")

        stream = self._sherpa_recognizer.create_stream()
        stream.accept_waveform(
            self.sampling_rate,
            audio_window,
        )
        if self._sherpa_online:
            stream.input_finished()
            while self._sherpa_recognizer.is_ready(stream):
                self._sherpa_recognizer.decode_stream(stream)
        else:
            self._sherpa_recognizer.decode_stream(stream)
        result = (
            self._sherpa_recognizer.get_result(stream)
            if self._sherpa_online
            else stream.result
        )
        text = clean_text(result if isinstance(result, str) else getattr(result, "text", ""))
        return TranscriptionResult(text=text)

    def _transcription_rejection_reason(self, result: TranscriptionResult) -> str | None:
        normalized = normalize_for_comparison(result.text)
        if not normalized:
            return "empty text"

        for phrase in self.blocked_phrases:
            blocked = normalize_for_comparison(phrase)
            if blocked and blocked in normalized:
                return f"blocked phrase: {phrase!r}"

        return None

    def _prepare_transcription(self, result: TranscriptionResult) -> str:
        reason = self._transcription_rejection_reason(result)
        if reason:
            logger.warning("Skipping STT result: %s; text=%r", reason, text_preview(result.text))
            return ""

        text, repetition_capped = cap_repeated_token_sequences(
            result.text,
            max_repetitions=3,
        )
        if repetition_capped:
            logger.warning(
                "STT repetition capped at 3: original=%r capped=%r",
                text_preview(result.text),
                text_preview(text),
            )

        normalized = normalize_for_comparison(text)
        if normalized == self._last_utterance_normalized:
            self._consecutive_utterance_count += 1
        else:
            self._last_utterance_normalized = normalized
            self._consecutive_utterance_count = 1
        if self._consecutive_utterance_count > self.max_consecutive_duplicate_utterances:
            logger.warning("Skipping repeated STT utterance: %r", text_preview(text))
            return ""
        return text

    def _has_trailing_silence(self, audio_window: np.ndarray, phrase_duration_s: float) -> bool:
        if not self.end_vad_enabled:
            return True
        silence_ms = (
            self.long_phrase_end_vad_silence_ms
            if phrase_duration_s >= self.long_phrase_after_s
            else self.end_vad_silence_ms
        )
        tail_samples = int(silence_ms * self.sampling_rate / 1000)
        if tail_samples <= 0 or audio_window.size < tail_samples:
            return False
        tail = audio_window[-tail_samples:]
        overall_rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float32))))
        if overall_rms > self.end_vad_rms_threshold:
            return False

        frame_samples = max(1, int(self.end_vad_frame_ms * self.sampling_rate / 1000))
        frame_count = tail.size // frame_samples
        if frame_count <= 0:
            return True
        frames = tail[-frame_count * frame_samples :].reshape(frame_count, frame_samples)
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
        active_ratio = float(np.mean(frame_rms > self.end_vad_rms_threshold))

        recent_samples = min(
            tail.size,
            max(1, int(self.end_vad_recent_ms * self.sampling_rate / 1000)),
        )
        recent = tail[-recent_samples:]
        recent_rms = float(np.sqrt(np.mean(np.square(recent, dtype=np.float32))))
        is_silence = (
            active_ratio <= self.end_vad_max_active_ratio
            and recent_rms <= self.end_vad_rms_threshold
        )
        if not is_silence:
            logger.debug(
                "VAD kept phrase open: overall=%.5f recent=%.5f active_ratio=%.2f",
                overall_rms,
                recent_rms,
                active_ratio,
            )
        return is_silence

    def _vad_context_sample_count(self, silence_ms: int) -> int:
        context_s = max(self._dfn_vad_context_s, (silence_ms / 1000.0) + 0.8)
        return int(self.sampling_rate * context_s)

    def _window_has_speech(self, audio_window: np.ndarray) -> bool:
        if audio_window.size == 0:
            return False

        frame_samples = max(1, int(self.sampling_rate * 0.03))
        if audio_window.size < frame_samples:
            rms = float(np.sqrt(np.mean(np.square(audio_window, dtype=np.float32))))
            return rms >= self.speech_activity_rms_threshold

        usable = (audio_window.size // frame_samples) * frame_samples
        if usable <= 0:
            return False

        frames = audio_window[:usable].reshape(-1, frame_samples)
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
        active_ratio = float(np.mean(rms >= self.speech_activity_rms_threshold))
        peak_rms = float(np.max(rms, initial=0.0))
        return active_ratio >= self.speech_activity_min_ratio or peak_rms >= (self.speech_activity_rms_threshold * 2.0)

    async def _confirm_firered_endpoint(
        self,
        window_audio: np.ndarray,
        phrase_duration_s: float,
    ) -> bool:
        if not self.firered_vad_rms_confirmation or not self._denoiser_enabled:
            return True

        silence_ms = (
            self.long_phrase_end_vad_silence_ms
            if phrase_duration_s >= self.long_phrase_after_s
            else self.end_vad_silence_ms
        )
        context_samples = self._vad_context_sample_count(silence_ms)
        raw_tail = window_audio[-context_samples:].copy()
        enhanced_tail = await asyncio.to_thread(
            self._enhance_audio_window,
            raw_tail,
        )
        confirmed = self._has_trailing_silence(
            enhanced_tail,
            phrase_duration_s,
        )
        if not confirmed:
            logger.debug(
                "Rejected FireRedVAD endpoint: DFN tail still contains speech "
                "(audio=%.2fs raw_prob=%.3f smooth_prob=%.3f)",
                phrase_duration_s,
                self._firered_vad_detector.last_raw_probability
                if self._firered_vad_detector is not None
                else 0.0,
                self._firered_vad_detector.last_smoothed_probability
                if self._firered_vad_detector is not None
                else 0.0,
            )
        return confirmed

    @staticmethod
    def _split_at_firered_endpoint(
        window_audio: np.ndarray,
        endpoint_detector: object | None,
        use_exact_endpoint: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not use_exact_endpoint or endpoint_detector is None:
            return window_audio, np.empty(0, dtype=np.float32)

        endpoint_sample_index = getattr(
            endpoint_detector,
            "endpoint_sample_index",
            None,
        )
        if endpoint_sample_index is None:
            return window_audio, np.empty(0, dtype=np.float32)

        split_at = min(
            window_audio.size,
            max(1, int(endpoint_sample_index)),
        )
        return window_audio[:split_at], window_audio[split_at:]

    def _minimum_endpoint_interval_reached(
        self,
        audio_duration_s: float,
        endpoint_detector: object | None,
    ) -> bool:
        if (
            self.end_vad_backend == "firered"
            and endpoint_detector is not None
        ):
            speech_elapsed_samples = int(
                getattr(endpoint_detector, "speech_elapsed_samples", 0)
            )
            return speech_elapsed_samples >= int(
                self.process_interval_s * self.sampling_rate
            )
        return audio_duration_s >= self.process_interval_s

    def _quiet_forced_split_sample(self, enhanced_audio: np.ndarray) -> int:
        samples = np.asarray(enhanced_audio, dtype=np.float32)
        frame_samples = max(
            1,
            int(self.end_vad_frame_ms * self.sampling_rate / 1000),
        )
        frame_count = samples.size // frame_samples
        if frame_count <= 0:
            return max(1, samples.size // 2)

        framed = samples[: frame_count * frame_samples].reshape(
            frame_count,
            frame_samples,
        )
        rms = np.sqrt(
            np.mean(np.square(framed, dtype=np.float32), axis=1)
        )
        quiet = rms < self.end_vad_rms_threshold
        quiet_runs: list[tuple[int, float, int, int]] = []
        run_start: int | None = None
        for index, is_quiet in enumerate(quiet):
            if is_quiet and run_start is None:
                run_start = index
            if run_start is not None and (
                not is_quiet or index == quiet.size - 1
            ):
                run_end = index + 1 if is_quiet else index
                run_rms = float(np.mean(rms[run_start:run_end]))
                quiet_runs.append(
                    (run_end - run_start, -run_rms, run_start, run_end)
                )
                run_start = None

        if quiet_runs:
            _, _, run_start, run_end = max(quiet_runs)
            split_frame = (run_start + run_end) // 2
        else:
            split_frame = int(np.argmin(rms))

        return min(
            samples.size - 1,
            max(1, split_frame * frame_samples + frame_samples // 2),
        )

    async def _find_forced_split_sample(
        self,
        window_audio: np.ndarray,
    ) -> int:
        search_samples = min(
            window_audio.size,
            max(1, int(self.forced_split_search_s * self.sampling_rate)),
        )
        search_start = window_audio.size - search_samples
        raw_tail = window_audio[search_start:].copy()
        enhanced_tail = await asyncio.to_thread(
            self._enhance_audio_window,
            raw_tail,
        )
        relative_split = self._quiet_forced_split_sample(enhanced_tail)
        split_at = min(
            window_audio.size - 1,
            max(1, search_start + relative_split),
        )
        logger.debug(
            "Selected quiet forced split: audio=%.2fs split=%.2fs "
            "tail=%.2fs",
            self._audio_duration_s(window_audio),
            split_at / self.sampling_rate,
            search_samples / self.sampling_rate,
        )
        return split_at

    async def _produce_from_device(self):
        spec = self.audio_source
        if spec.device_id is None:
            logger.error("Audio capture device is not configured for mode=%s.", spec.capture_mode)
            return

        capture = None
        try:
            while not self._stop_audio:
                if self._paused:
                    if capture is not None:
                        capture.stop()
                        capture = None
                    await asyncio.sleep(0.05)
                    continue
                if capture is None:
                    capture = DevicePCMStreamCapture(
                        spec=spec,
                        target_samplerate=self.sampling_rate,
                        chunk_size_ms=self.chunk_size_ms,
                        gain=1.0 if self._denoiser_enabled else self.input_gain,
                        buffer=self.pcm_buffer,
                        processor=None,
                    ).start()
                await asyncio.sleep(0.1)
                if not capture.is_active:
                    raise RuntimeError("Audio capture stream became inactive")
        finally:
            if capture is not None:
                capture.stop()

    async def _produce_from_browser(self) -> None:
        capture = BrowserPCMWebSocketCapture(
            host=self.browser_audio_host,
            port=self.browser_audio_port,
            target_samplerate=self.sampling_rate,
            gain=1.0 if self._denoiser_enabled else self.input_gain,
            buffer=self.pcm_buffer,
        )
        await capture.serve()

    def set_paused(self, paused: bool) -> None:
        paused = bool(paused)
        if self._paused == paused:
            return
        self._paused = paused
        while True:
            try:
                self.pcm_buffer.get_nowait()
            except Empty:
                break
        logger.info("[FIX:runtime-pause] STT %s; models remain loaded", "paused" if paused else "resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused

    async def _run_audio_source(self):
        if self.audio_source.capture_mode == "browser_tab":
            await self._produce_from_browser()
            return
        await self._produce_from_device()

    async def stream_utterances(self):
        if (
            self.audio_source.capture_mode == "input_device"
            and self.audio_source.device_id is None
        ):
            logger.error("No valid audio source configured. Cannot start STT stream.")
            return

        self._stop_audio = False
        producer_task = asyncio.create_task(self._run_audio_source())
        last_reported_drops = 0
        window_audio = np.array([], dtype=np.float32)
        last_emit_time = time.time()
        consecutive_decode_failures = 0
        samples_at_last_vad_check = 0
        overlap_samples = int(self.window_overlap_s * self.sampling_rate)
        max_window_samples = self._max_audio_window_samples()

        try:
            while True:
                try:
                    self._raise_if_audio_source_stopped(producer_task)
                    if self._paused:
                        window_audio = np.array([], dtype=np.float32)
                        self._reset_endpoint_detectors()
                        last_emit_time = time.time()
                        while True:
                            try:
                                self.pcm_buffer.get_nowait()
                            except Empty:
                                break
                        await asyncio.sleep(0.05)
                        continue
                    if self.pcm_buffer.dropped_chunks > last_reported_drops:
                        logger.warning(
                            "PCM buffer overflow detected: dropped=%s queue=%s/%s",
                            self.pcm_buffer.dropped_chunks,
                            self.pcm_buffer.qsize(),
                            self.pcm_buffer.max_chunks,
                        )
                        last_reported_drops = self.pcm_buffer.dropped_chunks
                        window_audio = np.array([], dtype=np.float32)
                        samples_at_last_vad_check = 0
                        self._reset_endpoint_detectors()
                        last_emit_time = time.time()

                    pcm_batch = self._drain_pcm_batch()
                    if self._pcm_discontinuity_detected:
                        self._pcm_discontinuity_detected = False
                        window_audio = np.array([], dtype=np.float32)
                        samples_at_last_vad_check = 0
                        self._reset_endpoint_detectors()
                        last_emit_time = time.time()
                    self._accept_endpoint_audio(pcm_batch)
                    if window_audio.size == 0:
                        window_audio = pcm_batch
                    else:
                        window_audio = np.concatenate([window_audio, pcm_batch], axis=0)

                    if window_audio.size > max_window_samples:
                        window_audio = window_audio[-max_window_samples:]
                        samples_at_last_vad_check = min(samples_at_last_vad_check, window_audio.size)

                    wall_duration_s = time.time() - last_emit_time
                    audio_duration_s = self._audio_duration_s(window_audio)
                    reached_time_limit = audio_duration_s >= self.max_phrase_s
                    has_phrase_end = False
                    use_exact_firered_endpoint = False
                    forced_split_sample: int | None = None
                    endpoint_detector = self._active_endpoint_detector(audio_duration_s)
                    if endpoint_detector is not None:
                        if (
                            self.end_vad_backend == "firered"
                            and self._firered_rms_recheck_generation is not None
                            and endpoint_detector.speech_generation
                            != self._firered_rms_recheck_generation
                        ):
                            self._firered_rms_recheck_generation = None
                        endpoint_candidate = endpoint_detector.endpoint_pending
                        if self.end_vad_backend == "firered":
                            endpoint_candidate = (
                                endpoint_candidate
                                or self._firered_rms_recheck_generation is not None
                            )
                        has_phrase_end = (
                            self._minimum_endpoint_interval_reached(
                                audio_duration_s,
                                endpoint_detector,
                            )
                            and endpoint_candidate
                        )
                        if (
                            has_phrase_end
                            and not reached_time_limit
                            and self.end_vad_backend == "firered"
                        ):
                            use_exact_firered_endpoint = (
                                endpoint_detector.endpoint_pending
                                and self._firered_rms_recheck_generation is None
                            )
                            has_phrase_end = await self._confirm_firered_endpoint(
                                window_audio,
                                audio_duration_s,
                            )
                            if not has_phrase_end:
                                endpoint_detector.reject_endpoint()
                                self._firered_rms_recheck_generation = (
                                    endpoint_detector.speech_generation
                                )
                            else:
                                self._firered_rms_recheck_generation = None
                    else:
                        should_check_vad = (
                            audio_duration_s >= self.process_interval_s
                            and window_audio.size - samples_at_last_vad_check >= self._vad_check_samples
                        )
                        if should_check_vad and not reached_time_limit:
                            silence_ms = (
                                self.long_phrase_end_vad_silence_ms
                                if audio_duration_s >= self.long_phrase_after_s
                                else self.end_vad_silence_ms
                            )
                            vad_context_samples = self._vad_context_sample_count(silence_ms)
                            vad_raw = window_audio[-vad_context_samples:].copy()
                            vad_audio = await asyncio.to_thread(self._enhance_audio_window, vad_raw)
                            has_phrase_end = self._has_trailing_silence(vad_audio, audio_duration_s)
                            samples_at_last_vad_check = window_audio.size
                    if reached_time_limit or has_phrase_end:
                        if reached_time_limit and not has_phrase_end:
                            forced_split_sample = (
                                await self._find_forced_split_sample(
                                    window_audio,
                                )
                            )
                            decode_window = window_audio[:forced_split_sample]
                            endpoint_remainder = window_audio[
                                forced_split_sample:
                            ]
                        else:
                            decode_window, endpoint_remainder = (
                                self._split_at_firered_endpoint(
                                window_audio,
                                endpoint_detector,
                                use_exact_firered_endpoint and has_phrase_end,
                            )
                            )
                        raw_snapshot = decode_window.copy()
                        snapshot = await asyncio.to_thread(
                            self._enhance_audio_window,
                            raw_snapshot,
                            for_stt=True,
                        )
                        vad_debug_snapshot = (
                            await asyncio.to_thread(
                                self._enhance_audio_window,
                                raw_snapshot,
                            )
                            if self.debug_audio_enabled
                            else None
                        )
                        has_speech = (
                            endpoint_detector.speech_seen
                            if endpoint_detector is not None
                            else self._window_has_speech(snapshot)
                        )
                        if not has_speech:
                            self._save_debug_rejected_window(
                                raw_snapshot,
                                vad_debug_snapshot
                                if vad_debug_snapshot is not None
                                else raw_snapshot,
                                snapshot,
                                endpoint_detector,
                            )
                            logger.debug(
                                "Skipping decode: no speech activity detected in %.2fs audio window",
                                audio_duration_s,
                            )
                            window_audio = np.array([], dtype=np.float32)
                            samples_at_last_vad_check = 0
                            self._reset_endpoint_detectors()
                            last_emit_time = time.time()
                            continue
                        if reached_time_limit and not has_phrase_end:
                            logger.debug(
                                "Forcing STT decode at max phrase length: audio=%.2fs wall=%.2fs",
                                audio_duration_s,
                                wall_duration_s,
                            )
                        try:
                            emit_runtime_event("stt_start")
                            try:
                                result = await asyncio.to_thread(
                                    self._transcribe_window_text,
                                    snapshot,
                                    raw_snapshot,
                                    vad_debug_snapshot,
                                )
                            finally:
                                emit_runtime_event("stt_end")
                            consecutive_decode_failures = 0
                        except Exception:
                            consecutive_decode_failures += 1
                            window_audio = np.array([], dtype=np.float32)
                            samples_at_last_vad_check = 0
                            self._reset_endpoint_detectors()
                            last_emit_time = time.time()
                            logger.exception(
                                "STT decode failed (%s/%s); dropping the failed window and continuing",
                                consecutive_decode_failures,
                                self.max_consecutive_decode_failures,
                            )
                            if consecutive_decode_failures >= self.max_consecutive_decode_failures:
                                raise RuntimeError("STT decode failed repeatedly")
                            continue
                        if self._paused:
                            window_audio = np.array([], dtype=np.float32)
                            samples_at_last_vad_check = 0
                            self._reset_endpoint_detectors()
                            last_emit_time = time.time()
                            continue
                        text = self._prepare_transcription(result)
                        if is_meaningful_text(text):
                            logger.info(
                                "STT utterance ready: wall=%.2fs audio=%.2fs",
                                wall_duration_s,
                                self._audio_duration_s(snapshot),
                            )
                            yield text

                        retained_parts: list[np.ndarray] = []
                        if overlap_samples > 0 and raw_snapshot.size > overlap_samples:
                            retained_parts.append(raw_snapshot[-overlap_samples:])
                        if endpoint_remainder.size:
                            retained_parts.append(endpoint_remainder)
                        if retained_parts:
                            window_audio = np.concatenate(retained_parts)
                            samples_at_last_vad_check = window_audio.size
                        else:
                            window_audio = np.array([], dtype=np.float32)
                            samples_at_last_vad_check = 0
                        self._reset_endpoint_detectors(window_audio)
                        last_emit_time = time.time()

                except Empty:
                    self._raise_if_audio_source_stopped(producer_task)
                    await asyncio.sleep(0.01)
                    continue
        finally:
            self._stop_audio = True
            self._reset_endpoint_detectors()
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
