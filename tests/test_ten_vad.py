import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.vad.ten import TenVadEndpointDetector


class _FakeTenConfig:
    def __init__(self):
        self.model = ""
        self.threshold = 0.0
        self.min_silence_duration = 0.0
        self.min_speech_duration = 0.0
        self.max_speech_duration = 0.0
        self.window_size = 256


class _FakeVadConfig:
    def __init__(self):
        self.ten_vad = _FakeTenConfig()
        self.sample_rate = 0

    def validate(self):
        return None


class _FakeVad:
    def __init__(self, config, buffer_size_in_seconds):
        self.config = config
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.is_speech_detected = False
        self.current_segment = types.SimpleNamespace(
            samples=np.empty(0, dtype=np.float32)
        )
        self._completed = 0

    def accept_waveform(self, frame):
        if np.max(np.abs(frame)) > 0.1:
            self.is_speech_detected = True
            self.current_segment.samples = frame
        elif self.is_speech_detected:
            self.is_speech_detected = False
            self.current_segment.samples = np.empty(0, dtype=np.float32)
            self._completed += 1

    def empty(self):
        return self._completed == 0

    def pop(self):
        self._completed -= 1

    def reset(self):
        self.is_speech_detected = False
        self.current_segment.samples = np.empty(0, dtype=np.float32)
        self._completed = 0


class TenVadEndpointDetectorTests(unittest.TestCase):
    def test_buffers_frames_and_marks_completed_speech(self):
        fake_module = types.SimpleNamespace(
            VadModelConfig=_FakeVadConfig,
            VoiceActivityDetector=_FakeVad,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": fake_module}):
            detector = TenVadEndpointDetector(
                model_path=Path("ten-vad.int8.onnx"),
                sample_rate=16000,
                threshold=0.2,
                min_silence_ms=230,
                min_speech_ms=100,
                max_speech_s=8.0,
            )

            detector.accept(np.ones(128, dtype=np.float32))
            self.assertFalse(detector.speech_seen)
            detector.accept(np.ones(128, dtype=np.float32))
            self.assertTrue(detector.speech_seen)
            self.assertFalse(detector.endpoint_pending)

            detector.accept(np.zeros(256, dtype=np.float32))
            self.assertTrue(detector.endpoint_pending)

            detector.reset()
            self.assertFalse(detector.speech_seen)
            self.assertFalse(detector.endpoint_pending)


if __name__ == "__main__":
    unittest.main()
