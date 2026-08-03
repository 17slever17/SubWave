import unittest
import tempfile
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch

from services.runtime.model_downloads import (
    _select_gguf,
    ensure_endpoint_model,
    ensure_fastenhancer_model,
    ensure_runtime_models,
    should_use_translation_mtp,
    translation_mtp_path,
)


class TranslationDownloadSelectionTests(unittest.TestCase):
    def test_prefers_exact_filename(self):
        selected = _select_gguf(
            ["README.md", "nested/model-Q4_K_M.gguf", "exact-Q4_K_XL.gguf"],
            "exact-Q4_K_XL.gguf",
        )
        self.assertEqual(selected, "exact-Q4_K_XL.gguf")

    def test_falls_back_to_q4_k_xl(self):
        selected = _select_gguf(
            ["model-Q8_0.gguf", "model-Q4_K_XL.gguf"],
            "missing.gguf",
        )
        self.assertEqual(selected, "model-Q4_K_XL.gguf")

    def test_rejects_repo_without_gguf(self):
        with self.assertRaises(FileNotFoundError):
            _select_gguf(["README.md"], "model.gguf")

    def test_selects_matching_e2b_mtp_model(self):
        path = translation_mtp_path(
            "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf"
        )
        self.assertEqual(path.name, "mtp-gemma-4-E2B-it.gguf")

    def test_selects_matching_e4b_mtp_model(self):
        path = translation_mtp_path(
            "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf"
        )
        self.assertEqual(path.name, "mtp-gemma-4-E4B-it.gguf")

    def test_does_not_guess_mtp_for_custom_model(self):
        self.assertIsNone(translation_mtp_path("models/custom-model.gguf"))

    def test_explicit_custom_mtp_path_is_allowed_for_custom_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            draft = Path(temp_dir) / "private-draft.gguf"
            self.assertEqual(
                translation_mtp_path(
                    "models/custom-model.gguf",
                    str(draft),
                ),
                draft,
            )

    def test_e2b_mtp_is_cpu_only_even_with_explicit_path(self):
        model = "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf"
        self.assertTrue(should_use_translation_mtp(model, "auto", "cpu"))
        self.assertFalse(should_use_translation_mtp(model, "auto", "gpu"))
        self.assertFalse(
            should_use_translation_mtp(model, "models/e2b-draft.gguf", "cuda")
        )

    def test_e4b_mtp_supports_cpu_and_gpu(self):
        model = "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf"
        self.assertTrue(should_use_translation_mtp(model, "auto", "cpu"))
        self.assertTrue(should_use_translation_mtp(model, "auto", "gpu"))

    def test_custom_model_requires_explicit_mtp_path(self):
        model = "models/gemma-4-12b.gguf"
        self.assertFalse(should_use_translation_mtp(model, "auto", "gpu"))
        self.assertTrue(
            should_use_translation_mtp(model, "models/gemma-4-12b-mtp.gguf", "gpu")
        )
        self.assertFalse(
            should_use_translation_mtp(
                model,
                "models/gemma-4-12b-mtp.gguf",
                "gpu",
                enabled=False,
            )
        )

    def test_runtime_does_not_download_e2b_mtp_for_gpu(self):
        config = {
            "language": "en",
            "audio": {},
            "stt": {},
            "translation": {
                "model_path": "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
                "device": "gpu",
                "mtp_enabled": True,
                "mtp_model_path": "auto",
            },
        }
        with (
            patch("services.runtime.model_downloads.ensure_endpoint_model"),
            patch("services.runtime.model_downloads.ensure_stt_model"),
            patch("services.runtime.model_downloads.ensure_fastenhancer_model"),
            patch("services.runtime.model_downloads.ensure_translation_model"),
            patch(
                "services.runtime.model_downloads.ensure_translation_mtp_model"
            ) as ensure_mtp,
        ):
            ensure_runtime_models(config)

        ensure_mtp.assert_not_called()

    def test_runtime_downloads_e2b_mtp_for_cpu(self):
        config = {
            "language": "en",
            "audio": {},
            "stt": {},
            "translation": {
                "model_path": "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
                "device": "cpu",
                "mtp_enabled": True,
                "mtp_model_path": "auto",
            },
        }
        with (
            patch("services.runtime.model_downloads.ensure_endpoint_model"),
            patch("services.runtime.model_downloads.ensure_stt_model"),
            patch("services.runtime.model_downloads.ensure_fastenhancer_model"),
            patch("services.runtime.model_downloads.ensure_translation_model"),
            patch(
                "services.runtime.model_downloads.ensure_translation_mtp_model"
            ) as ensure_mtp,
        ):
            ensure_runtime_models(config)

        ensure_mtp.assert_called_once()

    def test_fastenhancer_retries_a_transient_disconnect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"

            attempt = 0

            def download_after_disconnect(_url, destination):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise RemoteDisconnected("transient disconnect")
                Path(destination).write_bytes(b"model")

            with (
                patch("services.runtime.model_downloads.MODELS_DIR", models_dir),
                patch("services.runtime.model_downloads.resolve_runtime_path", return_value=None),
                patch(
                    "services.runtime.model_downloads.urllib.request.urlretrieve",
                    side_effect=download_after_disconnect,
                ) as retrieve,
                patch("services.runtime.model_downloads.time.sleep"),
            ):
                result = ensure_fastenhancer_model(
                    {"fastenhancer": {"enabled": True, "languages": ["en"]}},
                    "en",
                )

            self.assertEqual(retrieve.call_count, 2)
            self.assertEqual(result.read_bytes(), b"model")


class EndpointModelDownloadTests(unittest.TestCase):
    def test_skips_endpoint_model_when_ten_is_disabled(self):
        self.assertIsNone(
            ensure_endpoint_model(
                {"end_vad_enabled": False, "end_vad_backend": "ten"}
            )
        )

    @patch("scripts.download_sherpa_models.install_ten_vad")
    def test_installs_default_ten_model_when_missing(self, install_ten_vad):
        install_ten_vad.return_value = "models/ten-vad.int8.onnx"
        with patch("services.runtime.model_downloads.resolve_runtime_path") as resolve:
            resolve.return_value.is_file.return_value = False
            result = ensure_endpoint_model(
                {
                    "end_vad_enabled": True,
                    "end_vad_backend": "ten",
                    "ten_vad_model_path": "models/ten-vad.int8.onnx",
                }
            )
        self.assertEqual(result, "models/ten-vad.int8.onnx")
        install_ten_vad.assert_called_once_with(callback=None)

    @patch("scripts.download_sherpa_models.install_firered_vad")
    def test_installs_default_firered_model_when_missing(self, install_firered_vad):
        install_firered_vad.return_value = "models/FireRedVAD/Stream-VAD"
        with patch("services.runtime.model_downloads.resolve_runtime_path") as resolve:
            resolve.return_value.is_dir.return_value = False
            result = ensure_endpoint_model(
                {
                    "end_vad_enabled": True,
                    "end_vad_backend": "firered",
                    "firered_vad_model_path": "models/FireRedVAD/Stream-VAD",
                }
            )
        self.assertEqual(result, "models/FireRedVAD/Stream-VAD")
        install_firered_vad.assert_called_once_with(callback=None)


if __name__ == "__main__":
    unittest.main()
