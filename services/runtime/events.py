from __future__ import annotations

import os


RUNTIME_EVENT_PREFIX = "@@runtime-phase:"


def emit_runtime_event(event: str) -> None:
    if os.environ.get("REALTIME_TRANSLATOR_PHASE_EVENTS") != "1":
        return
    print(f"{RUNTIME_EVENT_PREFIX}{event}", flush=True)
