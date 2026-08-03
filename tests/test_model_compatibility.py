import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from configs.model_catalog import (
    TRANSLATION_MODELS,
    capabilities,
    compatible_translation_models,
    validate_runtime_config,
)
from services.stt.models import create_recognizer, get_sherpa_model, models_for_language


class TestModelCompatibility(unittest.TestCase):
    def test_fresh_install_exposes_downloadable_stt_languages_and_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = capabilities(Path(temp_dir))

        self.assertTrue({"en", "es", "ja"}.issubset(result["source_languages"]))
        self.assertTrue(result["stt_models"])
        self.assertTrue(all(not model["installed"] for model in result["stt_models"]))

    def test_english_has_specialized_and_multilingual_stt(self):
        model_ids = {model.model_id for model in models_for_language("en")}
        self.assertIn(
            "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
            model_ids,
        )
        self.assertIn("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", model_ids)

    def test_spanish_has_fast_conformer_option(self):
        model_ids = {model.model_id for model in models_for_language("es")}
        self.assertIn(
            "sherpa-onnx-nemo-fast-conformer-ctc-es-1424-int8",
            model_ids,
        )

    def test_english_parakeet_uses_tuned_blank_penalty(self):
        model = get_sherpa_model("auto", "en")
        self.assertEqual(model.blank_penalty, 0.3)

    def test_english_parakeet_passes_tuned_blank_penalty_to_sherpa(self):
        offline_recognizer = types.SimpleNamespace(
            from_transducer=MagicMock(return_value=object())
        )
        fake_sherpa = types.SimpleNamespace(
            __version__="1.0",
            OfflineRecognizer=offline_recognizer,
        )
        model = get_sherpa_model("auto", "en")
        files = {
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        }

        with patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa}):
            create_recognizer(model, files)

        self.assertEqual(
            offline_recognizer.from_transducer.call_args.kwargs["blank_penalty"],
            0.3,
        )

    def test_bulgarian_is_supported_by_both_gemma_sizes(self):
        models = compatible_translation_models("bg", "ru")
        self.assertEqual(len(models), 2)
        self.assertEqual({model["family"] for model in models}, {"gemma"})

    def test_german_auto_selects_multilingual_parakeet(self):
        model = get_sherpa_model("auto", "de")
        self.assertEqual(model.model_id, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")

    def test_custom_gguf_is_allowed_for_developers(self):
        config = {
            "language": "en",
            "stt": {"sherpa_model_id": "auto"},
            "translation": {
                "model_path": "models/custom-translator-Q4_K_M.gguf",
                "target_language": "ru",
                "device": "gpu",
            },
        }
        validate_runtime_config(config)

    def test_non_gguf_custom_model_is_rejected(self):
        config = {
            "language": "en",
            "stt": {"sherpa_model_id": "auto"},
            "translation": {
                "model_path": "models/custom-model.bin",
                "target_language": "ru",
                "device": "gpu",
            },
        }
        with self.assertRaisesRegex(ValueError, "custom .gguf"):
            validate_runtime_config(config)

    def test_translate_gemma_japanese_to_russian_is_valid(self):
        config = {
            "language": "ja",
            "stt": {"sherpa_model_id": "auto"},
            "translation": {
                "model_path": "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf",
                "target_language": "Russian",
                "device": "gpu",
            },
        }
        validate_runtime_config(config)

    def test_absolute_translate_gemma_path_is_valid(self):
        config = {
            "language": "en",
            "stt": {"sherpa_model_id": "auto"},
            "translation": {
                "model_path": (
                    r"C:\models\translate_gemma4_sub-E2B-Q4_K_XL.gguf"
                ),
                "target_language": "ru",
                "device": "gpu",
            },
        }
        validate_runtime_config(config)

    def test_only_translate_gemma_sub_models_are_exposed(self):
        self.assertEqual(
            {model["value"] for model in TRANSLATION_MODELS},
            {
                "models/translate_gemma4_sub-E4B-Q4_K_XL.gguf",
                "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
            },
        )

    def test_rejects_unsafe_runtime_numeric_values(self):
        base = {
            "language": "en",
            "audio": {
                "browser_audio_port": 8766,
                "sampling_rate": 16000,
                "buffer_max_chunks": 1024,
            },
            "overlay": {"browser_bridge_port": 8765},
            "stt": {
                "sherpa_model_id": "auto",
                "sherpa_onnx_provider": "cpu",
                "process_interval_s": 2,
                "chunk_size_ms": 60,
                "max_phrase_s": 10,
            },
            "translation": {
                "model_path": "models/custom-translator.gguf",
                "target_language": "ru",
                "device": "gpu",
                "n_batch": 256,
                "max_tokens": 128,
                "mtp_n": 1,
            },
        }
        invalid_values = (
            ("translation", "n_batch", 0),
            ("translation", "max_tokens", -1),
            ("translation", "mtp_n", 0),
            ("translation", "n_ctx", 0),
            ("translation", "context_subtitles", 6),
            ("audio", "browser_audio_port", 70000),
            ("audio", "buffer_max_chunks", 0),
            ("audio", "input_gain", -1),
            ("overlay", "opacity", 2),
            ("stt", "process_interval_s", 0),
            ("stt", "max_phrase_s", 1),
        )
        for section, key, value in invalid_values:
            with self.subTest(section=section, key=key, value=value):
                config = {
                    **base,
                    section: {**base[section], key: value},
                }
                with self.assertRaisesRegex(ValueError, key):
                    validate_runtime_config(config)

    def test_runtime_modes_are_validated(self):
        base = {
            "language": "en",
            "overlay": {
                "mode": "browser",
                "browser_bridge_port": 8765,
            },
            "audio": {
                "capture_mode": "browser_tab",
                "browser_audio_port": 8766,
                "sampling_rate": 16000,
                "buffer_max_chunks": 1024,
            },
            "stt": {
                "sherpa_model_id": "auto",
                "sherpa_onnx_provider": "cpu",
                "process_interval_s": 2,
                "chunk_size_ms": 60,
                "max_phrase_s": 10,
            },
            "translation": {
                "model_path": "models/custom-translator.gguf",
                "target_language": "ru",
                "device": "gpu",
            },
        }
        for section, key, value in (
            ("audio", "capture_mode", "system_magic"),
            ("overlay", "mode", "floating_magic"),
        ):
            with self.subTest(section=section, key=key):
                config = {
                    **base,
                    section: {**base[section], key: value},
                }
                with self.assertRaises(ValueError):
                    validate_runtime_config(config)


if __name__ == "__main__":
    unittest.main()
