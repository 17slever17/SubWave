from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_DIR / "frontend"


def run(label: str, command: list[str], cwd: Path) -> bool:
    print(f"\n[test] {label}", flush=True)
    print(f"[test] cwd={cwd}", flush=True)
    print(f"[test] command={' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode:
        print(f"[test] FAILED: {label} (exit={result.returncode})", flush=True)
        return False
    print(f"[test] PASSED: {label}", flush=True)
    return True


def main() -> int:
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    checks = [
        (
            "Python unit and API tests",
            [sys.executable, "-m", "pytest", "-q"],
            APP_DIR,
        ),
    ]
    if npm and (FRONTEND_DIR / "node_modules").is_dir():
        checks.extend(
            [
                ("React component tests", [npm, "test"], FRONTEND_DIR),
                ("Production frontend build", [npm, "run", "build"], FRONTEND_DIR),
            ]
        )
    else:
        print(
            "[test] Frontend dependencies are missing; run npm install in frontend "
            "to enable React tests and the production build check."
        )

    passed = all(run(label, command, cwd) for label, command, cwd in checks)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
