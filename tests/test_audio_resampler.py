import unittest

import numpy as np

from services.stt.audio import StreamingAudioResampler


def _run_stream(samples: np.ndarray, chunk_sizes: list[int]) -> np.ndarray:
    resampler = StreamingAudioResampler(48000, 16000)
    chunks: list[np.ndarray] = []
    offset = 0
    chunk_index = 0
    while offset < samples.size:
        size = chunk_sizes[chunk_index % len(chunk_sizes)]
        chunks.append(resampler.process(samples[offset : offset + size]))
        offset += size
        chunk_index += 1
    chunks.append(resampler.flush())
    return np.concatenate(chunks)


class StreamingAudioResamplerTests(unittest.TestCase):
    def test_chunk_boundaries_do_not_change_output(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.2, 48000).astype(np.float32)

        whole = _run_stream(samples, [samples.size])
        chunked = _run_stream(samples, [1920, 3840, 960, 2880])

        self.assertEqual(whole.size, chunked.size)
        np.testing.assert_allclose(whole, chunked, atol=2e-6, rtol=2e-5)

    def test_preserves_speech_band_tone(self):
        time = np.arange(48000, dtype=np.float32) / 48000.0
        samples = np.sin(2.0 * np.pi * 1000.0 * time).astype(np.float32)

        output = _run_stream(samples, [1920])
        input_rms = float(np.sqrt(np.mean(np.square(samples))))
        output_rms = float(np.sqrt(np.mean(np.square(output))))

        self.assertAlmostEqual(output.size / 16000.0, 1.0, places=3)
        self.assertGreater(output_rms / input_rms, 0.97)
        self.assertLess(output_rms / input_rms, 1.03)

    def test_suppresses_frequencies_above_output_nyquist(self):
        time = np.arange(48000, dtype=np.float32) / 48000.0
        samples = np.sin(2.0 * np.pi * 12000.0 * time).astype(np.float32)

        output = _run_stream(samples, [1920])
        naive = samples[::3]
        output_rms = float(np.sqrt(np.mean(np.square(output))))
        naive_rms = float(np.sqrt(np.mean(np.square(naive))))

        self.assertLess(output_rms, naive_rms * 0.02)

    def test_same_rate_is_lossless(self):
        samples = np.linspace(-0.8, 0.8, 257, dtype=np.float32)
        resampler = StreamingAudioResampler(16000, 16000)

        output = resampler.process(samples)

        np.testing.assert_array_equal(output, samples)
        self.assertEqual(resampler.flush().size, 0)

    def test_emits_audio_with_bounded_startup_latency(self):
        resampler = StreamingAudioResampler(48000, 16000)
        block = np.zeros(1920, dtype=np.float32)

        first_output_block = next(
            index
            for index in range(1, 6)
            if resampler.process(block).size > 0
        )

        self.assertLessEqual(first_output_block * 40, 200)

    def test_can_be_reset_after_flush(self):
        samples = np.ones(4800, dtype=np.float32)
        resampler = StreamingAudioResampler(48000, 16000)
        first = np.concatenate([resampler.process(samples), resampler.flush()])

        resampler.reset()
        second = np.concatenate([resampler.process(samples), resampler.flush()])

        np.testing.assert_allclose(first, second, atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
