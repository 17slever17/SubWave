import types
import unittest

from services.vad.firered import FireRedVadEndpointDetector


class FireRedVadEndpointDetectorTests(unittest.TestCase):
    def test_new_speech_start_clears_stale_endpoint(self):
        detector = FireRedVadEndpointDetector.__new__(
            FireRedVadEndpointDetector
        )
        detector.endpoint_pending = False
        detector.endpoint_sample_index = None
        detector.first_speech_start_sample_index = None
        detector.latest_processed_sample_index = 0
        detector.speech_seen = False
        detector.speech_generation = 0
        detector.last_raw_probability = 0.0
        detector.last_smoothed_probability = 0.0

        detector._apply_frame_result(
            types.SimpleNamespace(
                raw_prob=0.01,
                smoothed_prob=0.02,
                frame_idx=42,
                is_speech=False,
                is_speech_start=False,
                is_speech_end=True,
                speech_start_frame=10,
                speech_end_frame=42,
            )
        )
        self.assertTrue(detector.endpoint_pending)
        self.assertEqual(detector.endpoint_sample_index, 41 * 160)

        detector._apply_frame_result(
            types.SimpleNamespace(
                raw_prob=0.95,
                smoothed_prob=0.91,
                frame_idx=34,
                is_speech=True,
                is_speech_start=True,
                is_speech_end=False,
                speech_start_frame=30,
                speech_end_frame=-1,
            )
        )
        self.assertFalse(detector.endpoint_pending)
        self.assertIsNone(detector.endpoint_sample_index)
        self.assertTrue(detector.speech_seen)
        self.assertEqual(detector.speech_generation, 1)
        self.assertEqual(detector.first_speech_start_sample_index, 29 * 160)

    def test_endpoint_can_be_rejected_by_secondary_confirmation(self):
        detector = FireRedVadEndpointDetector.__new__(
            FireRedVadEndpointDetector
        )
        detector.endpoint_pending = True
        detector.endpoint_sample_index = 3200

        detector.reject_endpoint()

        self.assertFalse(detector.endpoint_pending)
        self.assertIsNone(detector.endpoint_sample_index)


if __name__ == "__main__":
    unittest.main()
