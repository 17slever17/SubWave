from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import yaml

from configs.settings import Config, load_config
from server.config_service import ConfigService


ROOT = Path(__file__).resolve().parents[1]


def nested_keys(value):
    if not isinstance(value, dict):
        return None
    return {key: nested_keys(item) for key, item in value.items()}


class ConfigSchemaContractTests(unittest.TestCase):
    def test_current_and_reset_yaml_match_runtime_schema(self):
        expected = nested_keys(asdict(Config()))
        for filename in ("config.yaml", "config.reset.yaml"):
            loaded = yaml.safe_load(
                (ROOT / "configs" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(
                nested_keys(loaded),
                expected,
                f"{filename} drifted from configs/settings.py",
            )

    def test_all_mtp_fields_survive_service_and_runtime_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            service = ConfigService(path)
            service.write(
                {
                    **asdict(Config()),
                    "translation": {
                        **asdict(Config().translation),
                        "mtp_enabled": False,
                        "mtp_model_path": r"D:\drafts\custom-mtp.gguf",
                        "mtp_n": 4,
                    },
                }
            )

            service_value = service.read()["translation"]
            runtime_value = load_config(str(path)).translation

            self.assertFalse(service_value["mtp_enabled"])
            self.assertEqual(
                service_value["mtp_model_path"],
                r"D:\drafts\custom-mtp.gguf",
            )
            self.assertEqual(service_value["mtp_n"], 4)
            self.assertFalse(runtime_value.mtp_enabled)
            self.assertEqual(runtime_value.mtp_model_path, service_value["mtp_model_path"])
            self.assertEqual(runtime_value.mtp_n, 4)

    def test_context_is_clamped_and_invalid_text_recovers_to_recommended(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigService(Path(temp_dir) / "config.yaml")

            below = service.patch(
                {"translation": {"context_subtitles": -50, "n_ctx": 1}}
            )
            above = service.patch(
                {"translation": {"context_subtitles": 50, "n_ctx": 99999}}
            )
            invalid = service.patch(
                {"translation": {"context_subtitles": "broken"}}
            )

            self.assertEqual(
                (below["translation"]["context_subtitles"], below["translation"]["n_ctx"]),
                (0, 544),
            )
            self.assertEqual(
                (above["translation"]["context_subtitles"], above["translation"]["n_ctx"]),
                (5, 1344),
            )
            self.assertEqual(
                (invalid["translation"]["context_subtitles"], invalid["translation"]["n_ctx"]),
                (3, 1024),
            )

    def test_corrupt_or_non_mapping_yaml_falls_back_to_complete_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            service = ConfigService(path)
            for payload in ("translation: [", "- not\n- a\n- mapping\n"):
                path.write_text(payload, encoding="utf-8")
                loaded = service.read()
                self.assertEqual(loaded["language"], "ja")
                self.assertIn("mtp_enabled", loaded["translation"])
                self.assertIn("_hash", loaded)

    def test_metadata_is_never_persisted_at_any_top_level_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            service = ConfigService(path)
            config = service.read()
            config["_hash"] = "client-stale"
            service.write(config)

            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn("_hash", raw)

    def test_failed_atomic_replace_keeps_previous_config_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            service = ConfigService(path)
            service.write(asdict(Config()))
            before = path.read_text(encoding="utf-8")

            with patch(
                "server.config_service.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    service.patch({"language": "ja"})

            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
