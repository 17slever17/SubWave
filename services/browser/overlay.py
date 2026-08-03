from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
import time

from services.browser.status import STORE

logger = logging.getLogger(__name__)

_TRANSPARENT_COLOR = "#ff00ff"
_KNOWN_BROWSERS = {
    "all": {"chrome.exe", "msedge.exe", "opera.exe", "opera_gx.exe", "browser.exe", "yandex.exe", "yandexbrowser.exe"},
    "chrome": {"chrome.exe"},
    "edge": {"msedge.exe"},
    "opera": {"opera.exe", "opera_gx.exe", "browser.exe"},
    "yandex": {"yandex.exe", "yandexbrowser.exe"},
}


@dataclass
class OverlayMessage:
    previous_translation: str
    translation: str


class SubtitleOverlay:
    def __init__(self, config):
        self.config = config
        self._queue: queue.Queue[OverlayMessage] = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stop_requested = threading.Event()
        self._root: tk.Tk | None = None
        self._last_visibility_debug: tuple | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="subtitle-overlay", daemon=True)
        self._thread.start()
        self._started.wait(timeout=3.0)

    def publish(self, previous_translation: str, translation: str) -> None:
        if not self._thread or not self._thread.is_alive():
            return
        self._put_latest(
            OverlayMessage(
                previous_translation=previous_translation.strip(),
                translation=translation.strip(),
            )
        )

    def _put_latest(self, message: OverlayMessage) -> None:
        while self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(message)

    def stop(self) -> None:
        self._stop_requested.set()
        if self._thread and self._thread.is_alive():
            self._put_latest(OverlayMessage(previous_translation="", translation=""))
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            self._run_inner()
        except Exception:
            logger.exception("Subtitle overlay crashed during startup or runtime.")
            self._started.set()

    def _run_inner(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("Realtime Translator Overlay")
        root.configure(bg=_TRANSPARENT_COLOR)
        root.overrideredirect(True)
        root.attributes("-topmost", bool(self.config.topmost))
        root.attributes("-alpha", float(self.config.opacity))
        try:
            root.wm_attributes("-transparentcolor", _TRANSPARENT_COLOR)
        except tk.TclError:
            logger.warning("Transparent rounded corners are not supported in this Tk build.")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        state = {
            "width": int(self.config.width),
            "height": int(self.config.height),
            "x": max(0, (screen_width - int(self.config.width)) // 2),
            "y": max(0, screen_height - int(self.config.height) - int(self.config.bottom_offset)),
            "prev": "",
            "curr": "Waiting for speech...",
            "visible": True,
            "unknown_process_checks": 0,
            "font_scale": 1.0,
        }
        own_process_name = Path(os.path.normpath(os.sys.executable)).name.lower()

        canvas = tk.Canvas(root, bg=_TRANSPARENT_COLOR, highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)

        prev_font = tkfont.Font(family="Segoe UI", size=int(self.config.source_font_size))
        curr_font = tkfont.Font(family="Segoe UI Semibold", size=int(self.config.font_size))
        resize_font = tkfont.Font(family="Segoe UI Symbol", size=12)

        prev_text_id = canvas.create_text(
            0,
            0,
            text="",
            fill=self.config.source_foreground,
            font=prev_font,
            justify="center",
            anchor="n",
            width=state["width"] - 70,
        )
        curr_text_id = canvas.create_text(
            0,
            0,
            text=state["curr"],
            fill=self.config.foreground,
            font=curr_font,
            justify="center",
            anchor="n",
            width=state["width"] - 70,
        )
        grip_id = canvas.create_text(
            0,
            0,
            text="◢",
            fill="#242424",
            font=resize_font,
            anchor="se",
        )

        drag_state = {"mode": None, "x": 0, "y": 0, "font_scale": 1.0}
        horizontal_padding = 24
        vertical_padding = 6
        line_gap = 0

        def rounded_rect(x1, y1, x2, y2, r, **kwargs):
            points = [
                x1 + r, y1,
                x2 - r, y1,
                x2, y1,
                x2, y1 + r,
                x2, y2 - r,
                x2, y2,
                x2 - r, y2,
                x1 + r, y2,
                x1, y2,
                x1, y2 - r,
                x1, y1 + r,
                x1, y1,
            ]
            return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

        def fit_width() -> int:
            longest = max(
                prev_font.measure(state["prev"]) if state["prev"] else 0,
                curr_font.measure(state["curr"]) if state["curr"] else 0,
                260,
            )
            target = longest + (horizontal_padding * 2) + 8
            return max(int(self.config.min_width), min(int(self.config.max_width), target))

        def update_fonts():
            scale = max(0.65, min(1.75, state["font_scale"]))
            prev_font.configure(size=max(10, int(self.config.source_font_size * scale)))
            curr_font.configure(size=max(14, int(self.config.font_size * scale)))

        def redraw():
            update_fonts()
            radius = min(int(self.config.corner_radius), state["height"] // 2 - 4, state["width"] // 6)
            canvas.config(width=state["width"], height=state["height"])
            canvas.delete("bubble")
            rounded_rect(
                4,
                4,
                state["width"] - 4,
                state["height"] - 4,
                max(8, radius),
                fill=self.config.background,
                outline="",
                tags="bubble",
            )
            text_width = max(120, state["width"] - (horizontal_padding * 2))
            canvas.itemconfigure(prev_text_id, width=text_width, text=state["prev"])
            canvas.itemconfigure(curr_text_id, width=text_width, text=state["curr"] or "...")
            root.update_idletasks()
            prev_bbox = canvas.bbox(prev_text_id) if state["prev"] else None
            curr_bbox = canvas.bbox(curr_text_id)
            prev_block_height = (prev_bbox[3] - prev_bbox[1]) if prev_bbox else 0
            curr_block_height = (curr_bbox[3] - curr_bbox[1]) if curr_bbox else curr_font.metrics("linespace")
            total_block_height = prev_block_height + curr_block_height + (line_gap if prev_block_height else 0)
            top_y = max(vertical_padding, int((state["height"] - total_block_height) / 2))
            canvas.coords(prev_text_id, state["width"] / 2, top_y)
            canvas.coords(curr_text_id, state["width"] / 2, top_y + prev_block_height + (line_gap if prev_block_height else 0))
            canvas.coords(grip_id, state["width"] - 12, state["height"] - 10)
            canvas.tag_raise(prev_text_id)
            canvas.tag_raise(curr_text_id)
            canvas.tag_raise(grip_id)
            root.geometry(f"{state['width']}x{state['height']}+{state['x']}+{state['y']}")

        def set_message(prev_text: str, curr_text: str, smooth: bool = False, anchor_mode: str = "center"):
            state["prev"] = prev_text
            state["curr"] = curr_text or "..."
            update_fonts()
            if curr_text:
                center_x = state["x"] + (state["width"] / 2)
                right_x = state["x"] + state["width"]
                bottom_y = state["y"] + state["height"]
                new_width = fit_width()
                target_x = max(0, int(center_x - (new_width / 2)))
                if anchor_mode == "bottom_right":
                    state["x"] = max(0, int(right_x - new_width))
                    if smooth:
                        state["width"] = int((state["width"] * 0.55) + (new_width * 0.45))
                        state["x"] = max(0, int(right_x - state["width"]))
                    else:
                        state["width"] = new_width
                elif smooth:
                    state["x"] = int((state["x"] * 0.65) + (target_x * 0.35))
                    state["width"] = int((state["width"] * 0.65) + (new_width * 0.35))
                else:
                    state["x"] = target_x
                    state["width"] = new_width
            text_width = max(120, state["width"] - (horizontal_padding * 2))
            canvas.itemconfigure(prev_text_id, width=text_width, text=state["prev"])
            canvas.itemconfigure(curr_text_id, width=text_width, text=state["curr"] or "...")
            root.update_idletasks()
            prev_bbox = canvas.bbox(prev_text_id) if state["prev"] else None
            curr_bbox = canvas.bbox(curr_text_id)
            prev_block_height = (prev_bbox[3] - prev_bbox[1]) if prev_bbox else 0
            curr_block_height = (curr_bbox[3] - curr_bbox[1]) if curr_bbox else curr_font.metrics("linespace")
            total_block_height = prev_block_height + curr_block_height + (line_gap if prev_block_height else 0)
            new_height = max(
                int(self.config.min_height),
                min(int(self.config.max_height), int(total_block_height + (vertical_padding * 2))),
            )
            if anchor_mode == "bottom_right":
                if smooth:
                    state["height"] = int((state["height"] * 0.55) + (new_height * 0.45))
                    state["y"] = int(bottom_y - state["height"])
                else:
                    state["height"] = new_height
                    state["y"] = int(bottom_y - state["height"])
            elif smooth:
                state["height"] = int((state["height"] * 0.65) + (new_height * 0.35))
            else:
                state["height"] = new_height
            redraw()

        def on_press(event):
            drag_state["x"] = event.x_root
            drag_state["y"] = event.y_root
            drag_state["font_scale"] = state["font_scale"]
            drag_state["mode"] = "scale" if event.x >= state["width"] - 28 and event.y >= state["height"] - 28 else "move"

        def on_drag(event):
            if drag_state["mode"] == "scale":
                delta = (event.x_root - drag_state["x"]) + (event.y_root - drag_state["y"])
                state["font_scale"] = max(0.65, min(1.75, drag_state["font_scale"] + (delta / 400.0)))
                set_message(state["prev"], state["curr"], smooth=True, anchor_mode="bottom_right")
                return
            state["x"] += event.x_root - drag_state["x"]
            state["y"] += event.y_root - drag_state["y"]
            drag_state["x"] = event.x_root
            drag_state["y"] = event.y_root
            root.geometry(f"+{state['x']}+{state['y']}")

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)

        def browser_targets() -> tuple[set[str], bool]:
            names: set[str] = set()
            all_mode = False
            for raw_name in getattr(self.config, "browser_targets", ["all"]):
                key = str(raw_name).strip().lower()
                if not key:
                    continue
                if key in _KNOWN_BROWSERS:
                    if key == "all":
                        all_mode = True
                    names.update(_KNOWN_BROWSERS[key])
                else:
                    names.add(f"{key}.exe" if not key.endswith(".exe") else key)
            return (names or _KNOWN_BROWSERS["all"], all_mode)

        target_processes, all_mode = browser_targets()
        keywords = [
            str(keyword).strip().lower()
            for keyword in getattr(self.config, "title_keywords", [])
            if str(keyword).strip()
        ]

        def foreground_process_name() -> str:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ""
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = kernel32.OpenProcess(0x1000, False, pid.value)
            if not process:
                return ""
            try:
                buffer = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(len(buffer))
                if ctypes.windll.kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
                    return Path(buffer.value).name.lower()
            finally:
                kernel32.CloseHandle(process)
            return ""

        def foreground_window_title() -> str:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ""
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value.strip().lower()

        def update_visibility():
            if bool(getattr(self.config, "browser_bridge_enabled", False)):
                bridge_state = STORE.snapshot()
                timeout_s = float(getattr(self.config, "browser_bridge_timeout_s", 3.0))
                is_fresh = (time.time() - bridge_state.timestamp) <= timeout_s if bridge_state.timestamp else False
                supported_host = bridge_state.host in {
                    "youtube.com",
                    "www.youtube.com",
                    "m.youtube.com",
                    "music.youtube.com",
                    "youtu.be",
                    "twitch.tv",
                    "www.twitch.tv",
                }
                should_show = supported_host and bridge_state.visible and is_fresh
                debug_tuple = (
                    bridge_state.host,
                    bridge_state.redirected,
                    bridge_state.visible,
                    bridge_state.focused,
                    is_fresh,
                    supported_host,
                    should_show,
                    state["visible"],
                )
                if debug_tuple != self._last_visibility_debug:
                    logger.info(
                        "Overlay bridge decision: host=%s redirected=%s visible=%s focused=%s "
                        "fresh=%s supported=%s should_show=%s current_visible=%s",
                        bridge_state.host or "<empty>",
                        bridge_state.redirected,
                        bridge_state.visible,
                        bridge_state.focused,
                        is_fresh,
                        supported_host,
                        should_show,
                        state["visible"],
                    )
                    self._last_visibility_debug = debug_tuple
                    if should_show != state["visible"]:
                        root.attributes("-alpha", float(self.config.opacity) if should_show else 0.0)
                        state["visible"] = should_show
                root.after(250, update_visibility)
                return

            if all_mode:
                if not state["visible"]:
                    root.attributes("-alpha", float(self.config.opacity))
                    state["visible"] = True
                root.after(250, update_visibility)
                return

            process_name = foreground_process_name()
            title_text = foreground_window_title()
            if process_name == own_process_name:
                root.after(250, update_visibility)
                return
            title_match = any(keyword in title_text for keyword in keywords) if keywords else True
            if process_name:
                state["unknown_process_checks"] = 0
                process_match = process_name in target_processes
                if all_mode:
                    should_show = title_match
                else:
                    should_show = process_match and title_match
            else:
                state["unknown_process_checks"] += 1
                should_show = state["unknown_process_checks"] >= 8 or state["visible"]

            if should_show != state["visible"]:
                root.attributes("-alpha", float(self.config.opacity) if should_show else 0.0)
                state["visible"] = should_show
            root.after(250, update_visibility)

        def poll_queue():
            try:
                while True:
                    message = self._queue.get_nowait()
                    if self._stop_requested.is_set():
                        root.quit()
                        return
                    set_message(message.previous_translation, message.translation)
            except queue.Empty:
                pass
            if self._stop_requested.is_set():
                root.quit()
                return
            root.after(50, poll_queue)

        redraw()
        self._started.set()
        root.after(50, poll_queue)
        root.after(250, update_visibility)
        logger.info("Subtitle overlay started.")
        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass
            self._root = None
        logger.info("Subtitle overlay stopped.")
