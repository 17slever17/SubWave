import tempfile
import unittest
from pathlib import Path

import numpy as np

from configs.settings import load_config
from services.filters.denoise import (
    DeepFilterNet2Denoiser,
    DeepFilterNet2Settings,
    resample_mono_audio,
)


class _FakeDenoiser(DeepFilterNet2Denoiser):
    def __init__(self):
        self.input_sr = 16000
        self.settings = DeepFilterNet2Settings(block_ms=40)
        self._input_block_samples = 640
        self._input_carry = np.zeros(0, dtype=np.float32)

    def _process_block(self, audio: np.ndarray) -> np.ndarray:
        return audio * 0.5


class TestDenoiseConfig(unittest.TestCase):
    def test_dfn_resampler_suppresses_out_of_band_aliases(self):
        source_rate = 48000
        target_rate = 16000
        t = np.arange(source_rate, dtype=np.float32) / source_rate
        in_band = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        out_of_band = np.sin(2 * np.pi * 12000 * t).astype(np.float32)

        low = resample_mono_audio(in_band, source_rate, target_rate)
        aliased = resample_mono_audio(out_of_band, source_rate, target_rate)

        low_rms = float(np.sqrt(np.mean(np.square(low[256:-256]))))
        alias_rms = float(np.sqrt(np.mean(np.square(aliased[256:-256]))))
        self.assertGreater(low_rms, 0.6)
        self.assertLess(alias_rms, low_rms * 0.02)

    def test_load_nested_dfn2_config(self):
        yaml_text = """
audio:
  dfn2:
    enabled: true
    input_sample_rate: 48000
    block_ms: 40
    vad_atten_lim_db: 18
    vad_context_s: 3
    post_filter: false
"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(yaml_text, encoding="utf-8")
            config = load_config(str(cfg_path))

        self.assertTrue(config.audio.dfn2.enabled)
        self.assertEqual(config.audio.dfn2.input_sample_rate, 48000)
        self.assertEqual(config.audio.dfn2.block_ms, 40)
        self.assertEqual(config.audio.dfn2.vad_atten_lim_db, 18)
        self.assertEqual(config.audio.dfn2.vad_context_s, 3)
        self.assertFalse(config.audio.dfn2.post_filter)

    def test_block_processing_preserves_cumulative_stream(self):
        denoiser = _FakeDenoiser()
        first = np.ones(960, dtype=np.float32)
        second = np.ones(960, dtype=np.float32) * 2

        out1 = denoiser.process(first)
        out2 = denoiser.process(second)
        combined = np.concatenate([out1, out2])

        self.assertEqual(len(out1), 640)
        self.assertEqual(len(out2), 1280)
        self.assertEqual(len(combined), len(first) + len(second))
        self.assertTrue(np.allclose(combined[:960], 0.5, atol=1e-6))
        self.assertTrue(np.allclose(combined[960:], 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
