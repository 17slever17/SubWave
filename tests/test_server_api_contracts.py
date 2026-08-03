from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import app as app_module
from server.config_service import ConfigService
from services.llm.prompts import PromptStore


class ServerApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.service = ConfigService(
            root / "config.yaml",
            root / "config.reset.yaml",
        )
        self.service.ensure_current_config()
        self.service.ensure_reset_baseline()
        self.service.patch(
            {
                "language": "en",
                "translation": {
                    "model_path": "models/custom.gguf",
                    "target_language": "Russian",
                },
            }
        )
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def test_config_endpoints_reject_non_objects_instead_of_crashing(self):
        with patch.object(app_module, "config_service", self.service):
            response = self.client.post("/api/config", json=["not", "an", "object"])
            patch_response = self.client.post(
                "/api/config/patch",
                json={"patch": ["not", "an", "object"]},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(patch_response.status_code, 400)

    def test_patch_persists_mtp_and_normalizes_context(self):
        with (
            patch.object(app_module, "config_service", self.service),
            patch.object(app_module, "validate_runtime_config"),
            patch.object(app_module, "_push_overlay_preview"),
        ):
            response = self.client.post(
                "/api/config/patch",
                json={
                    "patch": {
                        "translation": {
                            "mtp_enabled": False,
                            "mtp_model_path": r"D:\draft.gguf",
                            "mtp_n": 3,
                            "context_subtitles": 99,
                        }
                    }
                },
            )

        self.assertEqual(response.status_code, 200)
        translation = response.json()["config"]["translation"]
        self.assertFalse(translation["mtp_enabled"])
        self.assertEqual(translation["mtp_model_path"], r"D:\draft.gguf")
        self.assertEqual(translation["mtp_n"], 3)
        self.assertEqual(translation["context_subtitles"], 5)
        self.assertEqual(translation["n_ctx"], 1344)

    def test_validation_error_is_exposed_as_conflict_without_writing(self):
        before = self.service.read()["_hash"]
        with (
            patch.object(app_module, "config_service", self.service),
            patch.object(
                app_module,
                "validate_runtime_config",
                side_effect=ValueError("unsupported direction"),
            ),
        ):
            response = self.client.post(
                "/api/config/patch",
                json={"patch": {"language": "unsupported"}},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "unsupported direction")
        self.assertEqual(self.service.read()["_hash"], before)

    def test_overlay_preview_accepts_only_live_overlay_fields(self):
        captured = []
        with (
            patch.object(app_module, "config_service", self.service),
            patch.object(
                app_module,
                "_push_overlay_preview",
                side_effect=captured.append,
            ),
        ):
            response = self.client.post(
                "/api/config/overlay-preview",
                json={
                    "opacity": 0.5,
                    "font_size": 30,
                    "unknown": "must-not-pass",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured[0]["overlay"]["opacity"], 0.5)
        self.assertEqual(captured[0]["overlay"]["font_size"], 30)
        self.assertNotIn("unknown", captured[0]["overlay"])

    def test_unknown_preset_and_reset_section_have_stable_status_codes(self):
        with patch.object(app_module, "config_service", self.service):
            preset = self.client.post(
                "/api/presets/apply",
                json={"id": "does-not-exist"},
            )
            reset = self.client.post(
                "/api/config/reset-section",
                json={"section": "does-not-exist"},
            )

        self.assertEqual(preset.status_code, 404)
        self.assertEqual(reset.status_code, 404)

    def test_prompt_can_be_saved_before_its_stt_model_is_downloaded(self):
        prompt_store = PromptStore(Path(self.temp_dir.name) / "prompts.yaml")
        prompt_store.ensure_exists()
        payload = prompt_store.load()
        payload["presets"]["custom_spanish"] = {
            "name": "Shiro1213312",
            "source_language": "es",
            "target_language": "Russian",
            "template": "Translate Spanish subtitles into Russian.",
        }
        payload["active"] = "custom_spanish"

        with patch.object(app_module, "prompt_store", prompt_store):
            response = self.client.post("/api/prompts", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active"], "custom_spanish")

    def test_frontend_root_redirects_to_control(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/control")

    def test_status_identifies_backend_for_extension_port_discovery(self):
        with patch.object(app_module, "config_service", self.service):
            response = self.client.get("/api/status?client=extension")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "realtime-translator")
        self.assertGreater(app_module._extension_last_seen, 0)

    def test_status_allows_chromium_extension_origin(self):
        origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        with patch.object(app_module, "config_service", self.service):
            response = self.client.get(
                "/api/status?client=extension",
                headers={"Origin": origin},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_extension_health_falls_back_to_active_browser_bridge(self):
        bridge_response = mock.MagicMock()
        bridge_response.status = 200
        bridge_response.read.return_value = b'{"ok":true,"connected":true}'
        bridge_context = mock.MagicMock()
        bridge_context.__enter__.return_value = bridge_response
        previous_last_seen = app_module._extension_last_seen
        app_module._extension_last_seen = 0.0
        try:
            with patch.object(
                app_module.urllib_request,
                "urlopen",
                return_value=bridge_context,
            ):
                ok, detail = app_module._extension_health("127.0.0.1", 8765)
        finally:
            app_module._extension_last_seen = previous_last_seen

        self.assertTrue(ok)
        self.assertIn("active tab", detail)

    def test_health_accepts_an_active_translator_bridge(self):
        with (
            patch.object(app_module, "config_service", self.service),
            patch.object(app_module, "_browser_bridge_health") as bridge_health,
            patch.object(app_module, "_extension_health") as extension_health,
            patch.object(app_module, "get_models_status") as model_status,
        ):
            bridge_health.return_value = (True, "127.0.0.1:8765 translator bridge active")
            extension_health.return_value = (True, "Browser extension connected")
            model_status.return_value = {
                "stt": {"exists": True, "path": "stt"},
                "translation": {"exists": True, "path": "llm"},
            }
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        bridge_check = next(
            item for item in response.json()["checks"]
            if item["name"] == "browser_bridge_port"
        )
        self.assertTrue(bridge_check["ok"])
        self.assertIn("translator bridge active", bridge_check["detail"])
        extension_check = next(
            item for item in response.json()["checks"]
            if item["name"] == "browser_extension"
        )
        self.assertTrue(extension_check["ok"])


class ServerLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_stops_runtime_process(self):
        runtime = mock.Mock()
        runtime.stop = mock.AsyncMock(return_value={})
        with (
            patch.object(app_module, "runtime", runtime),
            patch.object(app_module.config_service, "ensure_reset_baseline"),
            patch.object(app_module.config_service, "ensure_current_config"),
            patch.object(app_module.prompt_store, "ensure_exists"),
        ):
            async with app_module.lifespan(app_module.app):
                pass

        runtime.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
