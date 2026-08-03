from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from configs.settings import Config
from services.filters.denoise import (
    FastEnhancerDenoiser,
    FastEnhancerSettings,
    PassthroughDenoiser,
    build_stt_denoiser,
)


ROOT = Path(__file__).resolve().parents[1]
FASTENHANCER_MODEL = (
    ROOT / "models" / "fastenhancer" / "fastenhancer_b_48khz.onnx"
)


class DenoiseRoutingTests(unittest.TestCase):
    def test_routes_only_supported_languages_to_fastenhancer(self) -> None:
        config = Config()
        fallback = PassthroughDenoiser()
        sentinel = object()

        with patch(
            "services.filters.denoise.FastEnhancerDenoiser",
            return_value=sentinel,
        ) as constructor:
            self.assertIs(
                build_stt_denoiser(config, 16000, "en", fallback),
                sentinel,
            )
            settings = constructor.call_args.args[0]
            self.assertEqual(settings.wet_mix, 0.5)

            self.assertIs(
                build_stt_denoiser(config, 16000, "ja", fallback),
                sentinel,
            )
            settings = constructor.call_args.args[0]
            self.assertEqual(settings.wet_mix, 0.45)

            self.assertIs(
                build_stt_denoiser(config, 16000, "fr", fallback),
                fallback,
            )

    @unittest.skipUnless(
        FASTENHANCER_MODEL.is_file(),
        "FastEnhancer model is not installed",
    )
    def test_fastenhancer_preserves_window_shape(self) -> None:
        denoiser = FastEnhancerDenoiser(
            FastEnhancerSettings(model_path=str(FASTENHANCER_MODEL))
        )
        source = np.zeros(3200, dtype=np.float32)
        enhanced = denoiser.process_window(source, 16000)

        self.assertEqual(enhanced.shape, source.shape)
        self.assertTrue(np.all(np.isfinite(enhanced)))


if __name__ == "__main__":
    unittest.main()
