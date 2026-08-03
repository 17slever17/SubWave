from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

MAX_BRIDGE_REQUEST_BYTES = 65_536
ALLOWED_BROWSER_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "twitch.tv",
    "www.twitch.tv",
}


def is_allowed_browser_origin(origin: str) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme == "chrome-extension":
        return bool(parsed.hostname)
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in ALLOWED_BROWSER_HOSTS


@dataclass
class BrowserOverlayState:
    tab_id: str = ""
    host: str = ""
    url: str = ""
    redirected: bool = False
    visible: bool = False
    focused: bool = False
    timestamp: float = 0.0
    last_active_timestamp: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class SubtitleOverlayState:
    previous_translation: str = ""
    translation: str = ""
    timestamp: float = 0.0
    sequence: int = 0


class BrowserStatusStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, BrowserOverlayState] = {}
        self._subtitle_state = SubtitleOverlayState()
        self._overlay_mode = "desktop"
        self._overlay_show_source = True
        self._overlay_opacity = 0.82
        self._overlay_font_size = 23
        self._overlay_source_font_size = 15

    def update(self, payload: dict) -> None:
        now = time.time()
        tab_id = str(payload.get("tab_id", "")).strip() or "default"
        host = str(payload.get("host", "")).strip().lower()
        redirected = bool(payload.get("redirected", False))
        visible = bool(payload.get("visible", False))
        focused = bool(payload.get("focused", False))
        with self._lock:
            previous = self._states.get(tab_id, BrowserOverlayState(tab_id=tab_id))
            last_active_timestamp = previous.last_active_timestamp
            if redirected and visible:
                last_active_timestamp = now
            self._states[tab_id] = BrowserOverlayState(
                tab_id=tab_id,
                host=host,
                url=str(payload.get("url", "")).strip(),
                redirected=redirected,
                visible=visible,
                focused=focused,
                timestamp=now,
                last_active_timestamp=last_active_timestamp,
                raw=dict(payload),
            )
            cutoff = now - 30.0
            self._states = {
                key: state
                for key, state in self._states.items()
                if state.timestamp >= cutoff
            }
    def snapshot(self) -> BrowserOverlayState:
        with self._lock:
            if not self._states:
                return BrowserOverlayState()
            chosen = max(
                self._states.values(),
                key=lambda state: (
                    state.visible,
                    state.redirected,
                    state.last_active_timestamp,
                    state.timestamp,
                ),
            )
            return BrowserOverlayState(**chosen.__dict__)

    def set_subtitles(self, previous_translation: str, translation: str) -> None:
        with self._lock:
            self._subtitle_state = SubtitleOverlayState(
                previous_translation=str(previous_translation or "").strip(),
                translation=str(translation or "").strip(),
                timestamp=time.time(),
                sequence=self._subtitle_state.sequence + 1,
            )

    def subtitle_snapshot(self) -> SubtitleOverlayState:
        with self._lock:
            return SubtitleOverlayState(**self._subtitle_state.__dict__)

    def set_overlay_config(
        self,
        mode: str,
        show_source: bool,
        opacity: float,
        font_size: int = 23,
        source_font_size: int = 15,
    ) -> None:
        with self._lock:
            self._overlay_mode = str(mode or "desktop").strip().lower() or "desktop"
            self._overlay_show_source = bool(show_source)
            self._overlay_opacity = min(1.0, max(0.0, float(opacity)))
            self._overlay_font_size = min(96, max(10, int(font_size)))
            self._overlay_source_font_size = min(72, max(8, int(source_font_size)))

    def overlay_config_snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self._overlay_mode,
                "show_source": self._overlay_show_source,
                "opacity": self._overlay_opacity,
                "font_size": self._overlay_font_size,
                "source_font_size": self._overlay_source_font_size,
            }


STORE = BrowserStatusStore()


class _BrowserStatusHandler(BaseHTTPRequestHandler):
    server_version = "RealtimeTranslatorBridge/1.0"

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin and is_allowed_browser_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _reject_disallowed_origin(self) -> bool:
        origin = self.headers.get("Origin", "")
        if is_allowed_browser_origin(origin):
            return False
        self.send_response(403)
        self.end_headers()
        return True

    def do_OPTIONS(self):
        if self._reject_disallowed_origin():
            return
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self._reject_disallowed_origin():
            return
        request_path = urlsplit(self.path).path
        if request_path not in {"/overlay_status", "/overlay_config"}:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > MAX_BRIDGE_REQUEST_BYTES:
                self.send_response(413)
                self._send_cors_headers()
                self.end_headers()
                return
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Payload must be a JSON object")
            if request_path == "/overlay_config":
                current = STORE.overlay_config_snapshot()
                STORE.set_overlay_config(
                    mode=str(payload.get("mode", current["mode"])),
                    show_source=bool(payload.get("show_source", current["show_source"])),
                    opacity=float(payload.get("opacity", current["opacity"])),
                    font_size=int(payload.get("font_size", current["font_size"])),
                    source_font_size=int(
                        payload.get("source_font_size", current["source_font_size"])
                    ),
                )
            else:
                STORE.update(payload)
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as exc:
            logger.debug("Browser bridge rejected payload: %s", exc)
            self.send_response(400)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":false}')

    def do_GET(self):
        if self._reject_disallowed_origin():
            return
        request_path = urlsplit(self.path).path
        if request_path not in {"/overlay_status", "/overlay_subtitles"}:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            return
        if request_path == "/overlay_status":
            snapshot = STORE.snapshot()
            age = max(0.0, time.time() - snapshot.timestamp) if snapshot.timestamp else None
            payload = {
                "ok": True,
                "connected": bool(age is not None and age <= 5.0),
                "age": age,
                "host": snapshot.host,
                "redirected": snapshot.redirected,
            }
            self._send_json(payload)
            return
        snapshot = STORE.subtitle_snapshot()
        overlay_cfg = STORE.overlay_config_snapshot()
        payload = {
            "ok": True,
            "previous_translation": snapshot.previous_translation,
            "translation": snapshot.translation,
            "timestamp": snapshot.timestamp,
            "sequence": snapshot.sequence,
            "mode": overlay_cfg["mode"],
            "show_source": overlay_cfg["show_source"],
            "opacity": overlay_cfg["opacity"],
            "font_size": overlay_cfg["font_size"],
            "source_font_size": overlay_cfg["source_font_size"],
        }
        self._send_json(payload)

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class BrowserStatusServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._server = ThreadingHTTPServer((self.host, self.port), _BrowserStatusHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="browser-status-bridge",
            daemon=True,
        )
        self._thread.start()
        logger.info("Browser bridge listening on http://%s:%s/overlay_status", self.host, self.port)

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("Browser bridge stopped.")
