from __future__ import annotations

import socket
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn


HOST = "127.0.0.1"
DEFAULT_PORT = 7860
APP_DIR = Path(__file__).resolve().parents[1]


def find_free_port(start: int = DEFAULT_PORT, attempts: int = 40) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free local port found in {start}-{start + attempts - 1}")


def main() -> None:
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    port = find_free_port()
    url = f"http://{HOST}:{port}/control"
    print(f"[webui] Opening {url}", flush=True)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        "server.app:app",
        host=HOST,
        port=port,
        log_level="info",
        app_dir=str(APP_DIR),
    )


if __name__ == "__main__":
    main()
