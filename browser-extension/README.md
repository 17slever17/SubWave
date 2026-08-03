# Realtime Translator browser extension

The extension captures audio from one YouTube or Twitch tab and sends mono
48 kHz float32 PCM to the local translator at `ws://127.0.0.1:8766/audio`.
It also replaces the Tampermonkey subtitle overlay.

## Install for development

1. Open `chrome://extensions` in Chrome or `edge://extensions` in Edge.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select this `browser-extension` directory.
5. Pin the extension to the browser toolbar.

Start the translator with `audio.capture_mode: browser_tab`, then click either
the extension toolbar button or the matching power button in the YouTube/Twitch
header. Both buttons toggle the same capture and always show the same state.

Button colors:

- gray: this tab is not selected for capture;
- tangerine: the tab is selected, but the WebUI is unavailable;
- blue: the WebUI is available, but the translator runtime is not running;
- green: tab capture and realtime translation are active.

Only one tab is captured at a time. Clicking the toolbar button in another
supported tab moves capture to that tab. The captured tab remains audible
because the extension routes the captured stream back to the default output.

Tab capture is independent from the translator process. Stopping the runtime
does not stop capture: the button becomes blue, disconnected audio frames are
dropped, and the extension reconnects automatically after the runtime starts
again. Closing the WebUI makes the button tangerine. Capture ends only when the
user toggles it off, captures another tab, closes the captured tab, or closes
the browser. The extension never changes the YouTube or Twitch tab favicon.
