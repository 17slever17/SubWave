from __future__ import annotations

from pathlib import Path

import numpy as np


class TenVadEndpointDetector:
    """Incremental TEN VAD endpoint detector for 16 kHz mono audio."""

    def __init__(
        self,
        *,
        model_path: Path,
        sample_rate: int,
        threshold: float,
        min_silence_ms: int,
        min_speech_ms: int,
        max_speech_s: float,
    ) -> None:
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.ten_vad.model = str(model_path)
        config.ten_vad.threshold = float(threshold)
        config.ten_vad.min_silence_duration = max(0.0, min_silence_ms / 1000.0)
        config.ten_vad.min_speech_duration = max(0.0, min_speech_ms / 1000.0)
        config.ten_vad.max_speech_duration = max(0.1, float(max_speech_s))
        config.sample_rate = int(sample_rate)
        config.validate()

        self._vad = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=max(30.0, max_speech_s * 2),
        )
        self._window_size = int(config.ten_vad.window_size)
        self._pending_samples = np.empty(0, dtype=np.float32)
        self.endpoint_pending = False
        self.speech_seen = False

    def accept(self, samples: np.ndarray) -> None:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size == 0:
            return
        self._pending_samples = np.concatenate((self._pending_samples, incoming))

        while self._pending_samples.size >= self._window_size:
            frame = self._pending_samples[: self._window_size]
            self._pending_samples = self._pending_samples[self._window_size :]
            self._vad.accept_waveform(frame)

            completed = False
            while not self._vad.empty():
                completed = True
                self.speech_seen = True
                self._vad.pop()

            has_current_speech = len(self._vad.current_segment.samples) > 0
            if completed:
                self.endpoint_pending = True
            elif has_current_speech:
                self.speech_seen = True
                self.endpoint_pending = False

    def reset(self) -> None:
        self._vad.reset()
        self._pending_samples = np.empty(0, dtype=np.float32)
        self.endpoint_pending = False
        self.speech_seen = False
