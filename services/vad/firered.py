from __future__ import annotations

from pathlib import Path

import numpy as np


class FireRedVadEndpointDetector:
    """Adapts FireRedVAD's overlapping 25 ms frames to arbitrary PCM chunks."""

    _FRAME_SAMPLES = 400
    _FRAME_SHIFT_SAMPLES = 160

    def __init__(
        self,
        model_path: Path,
        sample_rate: int,
        threshold: float,
        min_silence_ms: int,
        min_speech_ms: int,
        max_speech_s: float,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("FireRedVAD requires 16 kHz audio")

        import onnxruntime as ort
        from fireredvad.core.audio_feat import AudioFeat
        from fireredvad.core.stream_vad_postprocessor import StreamVadPostprocessor

        model_path = Path(model_path)
        if model_path.is_dir():
            model_file = model_path / "fireredvad_stream_vad_with_cache.onnx"
            cmvn_file = model_path / "cmvn.ark"
        else:
            model_file = model_path
            cmvn_file = model_path.with_name("cmvn.ark")
        if not model_file.is_file():
            raise FileNotFoundError(f"FireRedVAD ONNX model not found: {model_file}")
        if not cmvn_file.is_file():
            raise FileNotFoundError(f"FireRedVAD CMVN file not found: {cmvn_file}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._audio_feat = AudioFeat(str(cmvn_file))
        self._postprocessor = StreamVadPostprocessor(
            smooth_window_size=5,
            speech_threshold=float(threshold),
            pad_start_frame=5,
            min_speech_frame=max(1, round(min_speech_ms / 10)),
            max_speech_frame=max(1, round(max_speech_s * 100)),
            min_silence_frame=max(1, round(min_silence_ms / 10)),
        )
        self._caches = np.zeros((8, 1, 128, 19), dtype=np.float32)
        self._buffer = np.empty(0, dtype=np.float32)
        self.endpoint_pending = False
        self.endpoint_sample_index: int | None = None
        self.first_speech_start_sample_index: int | None = None
        self.latest_processed_sample_index = 0
        self.speech_seen = False
        self.speech_generation = 0
        self.last_raw_probability = 0.0
        self.last_smoothed_probability = 0.0

    def accept(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return

        self._buffer = np.concatenate((self._buffer, samples))
        if self._buffer.size < self._FRAME_SAMPLES:
            return

        frame_count = (
            1
            + (self._buffer.size - self._FRAME_SAMPLES)
            // self._FRAME_SHIFT_SAMPLES
        )
        consumed_samples = frame_count * self._FRAME_SHIFT_SAMPLES
        chunk_samples = (
            self._FRAME_SAMPLES
            + (frame_count - 1) * self._FRAME_SHIFT_SAMPLES
        )
        chunk = self._buffer[:chunk_samples]
        self._buffer = self._buffer[consumed_samples:]

        pcm16 = np.clip(chunk, -1.0, 1.0)
        pcm16 = np.rint(pcm16 * 32767.0).astype(np.int16)
        features, _ = self._audio_feat.extract(pcm16)
        if features.shape[0] == 0:
            return
        probabilities, self._caches = self._session.run(
            ["probs", "caches_out"],
            {
                "feat": features.numpy()[None, ...],
                "caches_in": self._caches,
            },
        )
        for probability in probabilities.reshape(-1):
            result = self._postprocessor.process_one_frame(float(probability))
            self._apply_frame_result(result)

    def _apply_frame_result(self, result) -> None:
        self.latest_processed_sample_index = (
            (int(result.frame_idx) - 1) * self._FRAME_SHIFT_SAMPLES
            + self._FRAME_SAMPLES
        )
        self.last_raw_probability = float(result.raw_prob)
        self.last_smoothed_probability = float(result.smoothed_prob)
        if result.is_speech or result.is_speech_start:
            self.speech_seen = True
        # An endpoint observed before the minimum subtitle duration is stale
        # once a new speech segment starts.
        if result.is_speech_start:
            if self.first_speech_start_sample_index is None:
                self.first_speech_start_sample_index = max(
                    0,
                    (int(result.speech_start_frame) - 1)
                    * self._FRAME_SHIFT_SAMPLES,
                )
            self.speech_generation += 1
            self.endpoint_pending = False
            self.endpoint_sample_index = None
        if result.is_speech_end and result.speech_start_frame >= 0:
            self.speech_seen = True
            self.endpoint_pending = True
            self.endpoint_sample_index = max(
                0,
                (int(result.speech_end_frame) - 1)
                * self._FRAME_SHIFT_SAMPLES,
            )

    def reset(self) -> None:
        self._audio_feat.reset()
        self._postprocessor.reset()
        self._caches.fill(0.0)
        self._buffer = np.empty(0, dtype=np.float32)
        self.endpoint_pending = False
        self.endpoint_sample_index = None
        self.first_speech_start_sample_index = None
        self.latest_processed_sample_index = 0
        self.speech_seen = False
        self.speech_generation = 0
        self.last_raw_probability = 0.0
        self.last_smoothed_probability = 0.0

    def set_min_silence_ms(self, min_silence_ms: int) -> None:
        self._postprocessor.min_silence_frame = max(
            1,
            round(min_silence_ms / 10),
        )

    def reject_endpoint(self) -> None:
        self.endpoint_pending = False
        self.endpoint_sample_index = None

    @property
    def speech_elapsed_samples(self) -> int:
        if self.first_speech_start_sample_index is None:
            return 0
        return max(
            0,
            self.latest_processed_sample_index
            - self.first_speech_start_sample_index,
        )
