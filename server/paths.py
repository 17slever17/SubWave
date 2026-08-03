from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR
CONFIG_DIR = APP_DIR / "configs"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
RESET_CONFIG_PATH = CONFIG_DIR / "config.reset.yaml"
PROMPTS_PATH = CONFIG_DIR / "prompts.yaml"
STATIC_DIR = APP_DIR / "static"
FRONTEND_DIR = APP_DIR / "frontend"
MAIN_SCRIPT = APP_DIR / "main.py"
VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    return APP_DIR / raw
