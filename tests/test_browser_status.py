import json
import socket
import unittest
from urllib import request as urllib_request

from services.browser.status import (
    BrowserStatusServer,
    BrowserStatusStore,
    is_allowed_browser_origin,
)


def test_browser_extension_origin_is_allowed():
    assert is_allowed_browser_origin(
        "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    )


class BrowserStatusStoreTests(unittest.TestCase):
    def test_overlay_preview_is_clamped_and_exposed(self):
        store = BrowserStatusStore()

        store.set_overlay_config(
            mode="browser",
            show_source=False,
            opacity=2.0,
            font_size=500,
            source_font_size=1,
        )

        self.assertEqual(
            store.overlay_config_snapshot(),
            {
                "mode": "browser",
                "show_source": False,
                "opacity": 1.0,
                "font_size": 96,
                "source_font_size": 8,
            },
        )

    def test_bridge_reports_recent_extension_status(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = BrowserStatusServer(port=port)
        server.start()
        try:
            payload = json.dumps({
                "tab_id": "health-test-tab",
                "host": "www.youtube.com",
                "redirected": True,
                "visible": True,
            }).encode("utf-8")
            post = urllib_request.Request(
                f"http://127.0.0.1:{port}/overlay_status",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(post, timeout=1):
                pass
            with urllib_request.urlopen(
                f"http://127.0.0.1:{port}/overlay_status",
                timeout=1,
            ) as response:
                status = json.loads(response.read().decode("utf-8"))

            self.assertTrue(status["connected"])
            self.assertEqual(status["host"], "www.youtube.com")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
