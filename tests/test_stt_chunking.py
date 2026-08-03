import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from configs.settings import Config, STTConfig
from services.stt.service import (
    STTService,
    TranscriptionResult,
    clean_text,
    is_meaningful_text,
)
from services.stt.audio import (
    AudioSourceSpec,
    PCMChunk,
    PCMChunkBuffer,
)


class TestSTTChunking(unittest.TestCase):
    def test_stale_pcm_is_dropped_before_it_reaches_vad(self):
        service = STTService.__new__(STTService)
        service.process_batch_ms = 200
        service.chunk_size_ms = 40
        service.max_pcm_backlog_s = 1.5
        service._pcm_discontinuity_detected = False
        service.pcm_buffer = PCMChunkBuffer(max_chunks=8)
        old = PCMChunk(
            samples=np.full(640, 0.1, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time() - 5.0,
            source_name="test",
        )
        current = PCMChunk(
            samples=np.full(640, 0.2, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp=time.time(),
            source_name="test",
        )
        service.pcm_buffer.push(old)
        service.pcm_buffer.push(current)

        batch = service._drain_pcm_batch()

        np.testing.assert_allclose(batch, current.samples)
        self.assertTrue(service._pcm_discontinuity_detected)

    def test_vad_uses_enough_dfn_context_without_changing_silence_duration(self):
        service = STTService.__new__(STTService)
        service.sampling_rate = 16000
        service._dfn_vad_context_s = 3.0

        self.assertEqual(service._vad_context_sample_count(300), 48000)
        self.assertEqual(service._vad_context_sample_count(170), 48000)

    def test_vad_does_not_cut_when_sparse_speech_continues_at_tail(self):
        service = STTService.__new__(STTService)
        service.end_vad_enabled = True
        service.end_vad_silence_ms = 300
        service.long_phrase_end_vad_silence_ms = 170
        service.long_phrase_after_s = 6.0
        service.sampling_rate = 16000
        service.end_vad_rms_threshold = 0.01
        service.end_vad_frame_ms = 25
        service.end_vad_max_active_ratio = 0.15
        service.end_vad_recent_ms = 75

        mostly_quiet = np.ones(4800, dtype=np.float32) * 0.001
        mostly_quiet[-800:] = 0.02
        self.assertFalse(service._has_trailing_silence(mostly_quiet, 4.0))

        actual_silence = np.ones(4800, dtype=np.float32) * 0.001
        self.assertTrue(service._has_trailing_silence(actual_silence, 4.0))

    def test_stt_window_uses_speech_preservation_limit_but_vad_does_not(self):
        service = STTService.__new__(STTService)
        service._denoiser_enabled = True
        service._dfn_vad_atten_lim_db = None
        service._dfn_stt_atten_lim_db = 24.0
        service._denoiser = Mock()
        service._denoiser.process_window.side_effect = lambda samples, *_args, **_kwargs: samples
        service.sampling_rate = 16000
        service.input_gain = 1.0
        samples = np.ones(1600, dtype=np.float32) * 0.01

        service._enhance_audio_window(samples, for_stt=False)
        self.assertIsNone(service._denoiser.process_window.call_args.kwargs["atten_lim_db"])

        service._enhance_audio_window(samples, for_stt=True)
        self.assertEqual(service._denoiser.process_window.call_args.kwargs["atten_lim_db"], 24.0)

    def test_debug_recording_writes_exact_stt_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = STTService.__new__(STTService)
            service.debug_audio_enabled = True
            service.debug_audio_max_files = 2
            service._debug_audio_index = 0
            service._debug_audio_limit_logged = False
            service._debug_audio_dir = Path(temp_dir)
            service.sampling_rate = 16000
            samples = np.concatenate(
                [
                    np.linspace(-0.25, 0.25, 800, dtype=np.float32),
                    np.full(800, 1e-6, dtype=np.float32),
                ]
            )

            expected = TranscriptionResult("texto correcto")
            service._transcribe_sherpa_onnx = lambda _audio: expected
            raw_samples = samples * 0.5
            vad_samples = samples * 0.75
            result = service._transcribe_window_text(samples, raw_samples, vad_samples)

            files = [
                path
                for path in Path(temp_dir).glob("*.wav")
                if not path.stem.endswith(("_raw", "_vad"))
            ]
            self.assertEqual(len(files), 1)
            self.assertEqual(result, expected)
            self.assertIn("texto correcto", files[0].with_suffix(".txt").read_text(encoding="utf-8"))
            saved, sample_rate = sf.read(files[0], dtype="float32", always_2d=False)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(saved.ndim, 1)
            self.assertEqual(saved.size, samples.size)
            np.testing.assert_array_equal(saved, samples)
            self.assertTrue(np.all(saved[-800:] != 0.0))
            metadata = files[0].with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn(f"sha256: {service._audio_sha256(samples)}", metadata)
            self.assertIn("samples: 1600", metadata)
            self.assertIn("sample_rate: 16000", metadata)
            self.assertIn("buffer_unchanged_after_decode: True", metadata)
            raw_file = files[0].with_name(f"{files[0].stem}_raw.wav")
            saved_raw, raw_rate = sf.read(raw_file, dtype="float32", always_2d=False)
            self.assertEqual(raw_rate, 16000)
            np.testing.assert_array_equal(saved_raw, raw_samples)
            self.assertIn(f"raw_file: {raw_file.name}", metadata)
            vad_file = files[0].with_name(f"{files[0].stem}_vad.wav")
            saved_vad, vad_rate = sf.read(vad_file, dtype="float32", always_2d=False)
            self.assertEqual(vad_rate, 16000)
            np.testing.assert_array_equal(saved_vad, vad_samples)
            self.assertIn(f"vad_file: {vad_file.name}", metadata)

    def test_debug_recording_keeps_windows_rejected_before_asr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = STTService.__new__(STTService)
            service.debug_audio_enabled = True
            service.debug_audio_max_files = 2
            service._debug_audio_index = 0
            service._debug_audio_dir = Path(temp_dir)
            service.sampling_rate = 16000
            raw = np.full(1600, 0.4, dtype=np.float32)
            vad = np.full(1600, 0.2, dtype=np.float32)
            stt = np.full(1600, 0.1, dtype=np.float32)
            detector = SimpleNamespace(
                speech_seen=False,
                last_raw_probability=0.1,
                last_smoothed_probability=0.05,
            )

            service._save_debug_rejected_window(raw, vad, stt, detector)

            metadata_file = next(Path(temp_dir).glob("*_rejected_no_speech.txt"))
            metadata = metadata_file.read_text(encoding="utf-8")
            self.assertIn("status: rejected_before_asr", metadata)
            self.assertIn("speech_seen: False", metadata)
            self.assertEqual(len(list(Path(temp_dir).glob("*.wav"))), 3)

    @patch("services.stt.service.STTService._init_sherpa_onnx_model")
    @patch("services.stt.service.build_audio_source")
    def test_trailing_silence_detection(self, mock_build_source, mock_init_model):
        mock_build_source.return_value = AudioSourceSpec(
            capture_mode="input_device",
            device_id=1,
            device_name="Mock Input",
            native_samplerate=16000,
            channels=1,
        )
        mock_init_model.return_value = None

        config = Config()
        config.stt = STTConfig(
            process_interval_s=4,
            end_vad_enabled=True,
            end_vad_silence_ms=450,
            end_vad_rms_threshold=0.01,
        )

        service = STTService(config)
        voiced = np.ones(16000, dtype=np.float32) * 0.1
        voiced_with_silence = np.concatenate([voiced, np.zeros(8000, dtype=np.float32)])

        self.assertFalse(service._has_trailing_silence(voiced, 4.0))
        self.assertTrue(service._has_trailing_silence(voiced_with_silence, 4.0))
        self.assertFalse(service._window_has_speech(np.zeros(16000, dtype=np.float32)))
        speechy = np.concatenate(
            [
                np.zeros(4000, dtype=np.float32),
                np.ones(8000, dtype=np.float32) * 0.08,
                np.zeros(4000, dtype=np.float32),
            ]
        )
        self.assertTrue(service._window_has_speech(speechy))

    def test_text_filters(self):
        self.assertEqual(clean_text(" hello   ,  world ! "), "hello, world!")
        self.assertFalse(is_meaningful_text("."))
        self.assertFalse(is_meaningful_text("me"))
        self.assertTrue(is_meaningful_text("yes"))
        self.assertTrue(is_meaningful_text("Haleo"))
        self.assertTrue(is_meaningful_text("hello world"))

    @patch("services.stt.service.STTService._init_sherpa_onnx_model")
    @patch("services.stt.service.build_audio_source")
    def test_audio_window_keeps_full_max_phrase(self, mock_build_source, mock_init_model):
        mock_build_source.return_value = AudioSourceSpec(
            capture_mode="input_device",
            device_id=1,
            device_name="Mock Input",
            native_samplerate=16000,
            channels=1,
        )
        mock_init_model.return_value = None
        config = Config()
        config.audio.dfn2.enabled = False
        config.stt = STTConfig(
            process_interval_s=3.5,
            window_slack_s=1,
            window_overlap_s=0,
            max_phrase_s=8,
        )

        service = STTService(config)

        self.assertEqual(service._max_audio_window_samples(), 9 * 16000)
        self.assertGreater(service._max_audio_window_samples(), 4 * 16000)
        self.assertEqual(service.process_interval_s, 3.5)
        self.assertEqual(service._audio_duration_s(np.zeros(8 * 16000, dtype=np.float32)), 8.0)

    @patch("services.stt.service.STTService._init_sherpa_onnx_model")
    @patch("services.stt.service.build_audio_source")
    def test_sherpa_model_is_initialized(self, mock_build_source, mock_init_model):
        mock_build_source.return_value = AudioSourceSpec(
            capture_mode="input_device",
            device_id=1,
            device_name="Mock Input",
            native_samplerate=16000,
            channels=1,
        )
        config = Config()
        config.audio.dfn2.enabled = False
        STTService(config)

        mock_init_model.assert_called_once_with()


class TestAudioSourceSupervision(unittest.IsolatedAsyncioTestCase):
    async def test_completed_audio_source_failure_is_propagated(self):
        async def fail_source():
            raise OSError("device disconnected")

        producer_task = asyncio.create_task(fail_source())
        await asyncio.sleep(0)

        with self.assertRaisesRegex(RuntimeError, "Audio source task failed") as raised:
            STTService._raise_if_audio_source_stopped(producer_task)

        self.assertIsInstance(raised.exception.__cause__, OSError)


class TestFireRedEndpointConfirmation(unittest.IsolatedAsyncioTestCase):
    def test_minimum_interval_is_measured_from_first_speech_start(self):
        service = STTService.__new__(STTService)
        service.end_vad_backend = "firered"
        service.process_interval_s = 2.0
        service.sampling_rate = 16000
        detector = Mock(speech_elapsed_samples=16000)

        self.assertFalse(
            service._minimum_endpoint_interval_reached(8.0, detector)
        )

        detector.speech_elapsed_samples = 32000
        self.assertTrue(
            service._minimum_endpoint_interval_reached(8.0, detector)
        )

    def test_forced_split_prefers_center_of_longest_quiet_run(self):
        service = STTService.__new__(STTService)
        service.sampling_rate = 1000
        service.end_vad_frame_ms = 25
        service.end_vad_rms_threshold = 0.01
        audio = np.concatenate(
            (
                np.ones(250, dtype=np.float32) * 0.1,
                np.zeros(200, dtype=np.float32),
                np.ones(150, dtype=np.float32) * 0.1,
                np.zeros(100, dtype=np.float32),
                np.ones(300, dtype=np.float32) * 0.1,
            )
        )

        split_at = service._quiet_forced_split_sample(audio)

        self.assertEqual(split_at, 362)

    def test_forced_split_falls_back_to_quietest_frame(self):
        service = STTService.__new__(STTService)
        service.sampling_rate = 1000
        service.end_vad_frame_ms = 25
        service.end_vad_rms_threshold = 0.01
        audio = np.ones(250, dtype=np.float32) * 0.1
        audio[100:125] = 0.02

        split_at = service._quiet_forced_split_sample(audio)

        self.assertEqual(split_at, 112)

    def test_exact_endpoint_split_preserves_right_hand_audio(self):
        window = np.arange(1200, dtype=np.float32)
        detector = Mock(endpoint_sample_index=880)

        decode, remainder = STTService._split_at_firered_endpoint(
            window,
            detector,
            use_exact_endpoint=True,
        )

        np.testing.assert_array_equal(decode, window[:880])
        np.testing.assert_array_equal(remainder, window[880:])
        np.testing.assert_array_equal(
            np.concatenate((decode, remainder)),
            window,
        )

    def test_endpoint_split_is_disabled_for_rms_rechecks(self):
        window = np.arange(1200, dtype=np.float32)
        detector = Mock(endpoint_sample_index=880)

        decode, remainder = STTService._split_at_firered_endpoint(
            window,
            detector,
            use_exact_endpoint=False,
        )

        self.assertIs(decode, window)
        self.assertEqual(remainder.size, 0)

    async def test_dfn_rms_rejects_endpoint_while_voice_is_still_present(self):
        service = STTService.__new__(STTService)
        service.firered_vad_rms_confirmation = True
        service._denoiser_enabled = True
        service.sampling_rate = 16000
        service.end_vad_enabled = True
        service.end_vad_silence_ms = 230
        service.long_phrase_end_vad_silence_ms = 150
        service.long_phrase_after_s = 6.0
        service.end_vad_rms_threshold = 0.01
        service.end_vad_frame_ms = 25
        service.end_vad_max_active_ratio = 0.15
        service.end_vad_recent_ms = 75
        service._dfn_vad_context_s = 3.0
        service._firered_vad_detector = Mock(
            last_raw_probability=0.1,
            last_smoothed_probability=0.2,
        )
        service._enhance_audio_window = lambda samples: samples

        voiced = np.ones(3 * 16000, dtype=np.float32) * 0.03
        quiet = np.ones(3 * 16000, dtype=np.float32) * 0.001

        self.assertFalse(await service._confirm_firered_endpoint(voiced, 4.0))
        self.assertTrue(await service._confirm_firered_endpoint(quiet, 4.0))


if __name__ == "__main__":
    unittest.main()
