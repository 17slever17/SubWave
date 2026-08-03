from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "browser-extension"


class BrowserExtensionContractTests(unittest.TestCase):
    def test_manifest_resources_and_scripts_exist(self):
        manifest = json.loads(
            (EXTENSION / "manifest.json").read_text(encoding="utf-8")
        )
        referenced = {
            manifest["background"]["service_worker"],
            manifest["action"]["default_popup"],
            *manifest["content_scripts"][0]["js"],
        }
        for sizes in manifest["action"]["default_icon"].values():
            referenced.add(sizes)
        for relative in referenced:
            self.assertTrue(
                (EXTENSION / relative).is_file(),
                f"Missing extension resource: {relative}",
            )

    def test_content_script_covers_youtube_and_twitch(self):
        manifest = json.loads(
            (EXTENSION / "manifest.json").read_text(encoding="utf-8")
        )
        matches = set(manifest["content_scripts"][0]["matches"])
        self.assertIn("https://www.youtube.com/*", matches)
        self.assertIn("https://www.twitch.tv/*", matches)

    def test_audio_worklet_and_local_bridges_are_declared(self):
        manifest = json.loads(
            (EXTENSION / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("tabCapture", manifest["permissions"])
        self.assertIn("offscreen", manifest["permissions"])
        hosts = set(manifest["host_permissions"])
        self.assertIn("http://127.0.0.1/*", hosts)
        self.assertTrue((EXTENSION / "audio-worklet.js").is_file())

    def test_webui_discovery_supports_launcher_fallback_ports(self):
        offscreen = (EXTENSION / "offscreen.js").read_text(encoding="utf-8")
        manifest = json.loads(
            (EXTENSION / "manifest.json").read_text(encoding="utf-8")
        )
        csp = manifest["content_security_policy"]["extension_pages"]

        self.assertIn("WEBUI_PORT_ATTEMPTS", offscreen)
        self.assertIn("realtime-translator", offscreen)
        self.assertIn("http://127.0.0.1:*", csp)

    def test_active_tab_permission_error_has_user_guidance(self):
        popup = (EXTENSION / "capture-popup.js").read_text(encoding="utf-8")
        worker = (EXTENSION / "service-worker.js").read_text(encoding="utf-8")

        self.assertIn("browser toolbar", popup)
        self.assertIn("only need to do this once per tab", popup)
        self.assertIn("activeTabPermission", popup)
        self.assertIn("browser toolbar", worker)
        self.assertIn("only need to do this once per tab", worker)

    def test_page_button_requests_capture_without_programmatic_popup(self):
        worker = (EXTENSION / "service-worker.js").read_text(encoding="utf-8")

        self.assertIn("toggleCaptureFromPage(sender.tab)", worker)
        self.assertIn("chrome.tabCapture.getMediaStreamId", worker)
        self.assertNotIn("chrome.action.openPopup", worker)

    def test_previous_subtitle_state_is_restored_after_overlay_remount(self):
        content_script = (EXTENSION / "content-script.js").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "nextPrev !== lastRenderedPrevious || overlayPrev.textContent !== nextPrev",
            content_script,
        )
        self.assertIn(
            "overlayPrev.classList.toggle('hidden', !showPrev);",
            content_script,
        )
        self.assertIn("lastRenderedPrevious = '';", content_script)
        self.assertIn("lastRenderedShowPrev = false;", content_script)


if __name__ == "__main__":
    unittest.main()
