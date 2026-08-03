from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[2]


def resolve_runtime_path(value: str | Path | None) -> Path | None:
    if not value:
        return None

    raw = Path(value)
    if raw.is_absolute():
        return raw

    app_candidate = APP_DIR / raw
    if app_candidate.exists():
        return app_candidate

    return app_candidate
