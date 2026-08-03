from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class Part:
    index: int
    start: int
    end: int
    path: Path

    @property
    def size(self) -> int:
        return self.end - self.start + 1


class Progress:
    def __init__(
        self,
        total: int,
        completed: int,
        label: str = "model",
        callback: ProgressCallback | None = None,
    ) -> None:
        self.total = total
        self.completed = completed
        self.transferred = 0
        self.label = label
        self.callback = callback
        self.started_at = time.monotonic()
        self.last_printed_at = 0.0
        self.lock = threading.Lock()

    def add(self, byte_count: int) -> None:
        with self.lock:
            self.completed += byte_count
            self.transferred += byte_count
            now = time.monotonic()
            if now - self.last_printed_at < 0.5 and self.completed < self.total:
                return
            elapsed = max(0.001, now - self.started_at)
            speed_mib = (self.transferred / elapsed) / (1024 * 1024)
            percent = (self.completed / self.total) * 100
            remaining = max(0, self.total - self.completed)
            eta_seconds = remaining / max(1.0, self.transferred / elapsed)
            message = (
                f"[download:{self.label}] {percent:5.1f}% | "
                f"{self.completed / 1024**2:,.0f}/{self.total / 1024**2:,.0f} MiB | "
                f"{speed_mib:.1f} MiB/s | ETA {format_eta(eta_seconds)}"
            )
            if self.callback:
                self.callback(message)
            else:
                print(f"\r{message}", end="", flush=True)
            self.last_printed_at = now

    def finish(self) -> None:
        if not self.callback:
            print(flush=True)


def format_eta(seconds: float) -> str:
    rounded = max(0, round(seconds))
    minutes, remaining = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def build_parts(destination: Path, expected_bytes: int, connections: int) -> list[Part]:
    part_size = (expected_bytes + connections - 1) // connections
    parts = []
    for index in range(connections):
        start = index * part_size
        if start >= expected_bytes:
            break
        end = min(expected_bytes - 1, start + part_size - 1)
        parts.append(
            Part(
                index=index,
                start=start,
                end=end,
                path=destination.with_name(f"{destination.name}.part{index:02d}"),
            )
        )
    return parts


def seed_parts_from_partial(destination: Path, parts: list[Part]) -> None:
    if not destination.exists():
        return
    existing_bytes = destination.stat().st_size
    if existing_bytes <= 0:
        destination.unlink(missing_ok=True)
        return

    with destination.open("rb") as source:
        for part in parts:
            available = min(existing_bytes, part.end + 1) - part.start
            if available <= 0:
                break
            if part.path.exists() and part.path.stat().st_size >= available:
                continue
            source.seek(part.start)
            with part.path.open("wb") as target:
                remaining = available
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise OSError("Unexpected end of partial download")
                    target.write(chunk)
                    remaining -= len(chunk)
    destination.unlink()


def download_part(
    url: str,
    part: Part,
    progress: Progress,
    retries: int = 5,
) -> None:
    for attempt in range(1, retries + 1):
        downloaded = part.path.stat().st_size if part.path.exists() else 0
        if downloaded == part.size:
            return
        if downloaded > part.size:
            part.path.unlink()
            downloaded = 0

        range_start = part.start + downloaded
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "realtime-translator-installer",
                "Range": f"bytes={range_start}-{part.end}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise OSError(
                        f"Server did not honor Range request (HTTP {response.status})"
                    )
                with part.path.open("ab") as target:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        progress.add(len(chunk))
            if part.path.stat().st_size == part.size:
                return
            raise OSError(
                f"Incomplete segment {part.index}: "
                f"{part.path.stat().st_size}/{part.size} bytes"
            )
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(8, 2 ** (attempt - 1)))


def download(
    url: str,
    destination: Path,
    expected_bytes: int,
    connections: int,
    *,
    label: str = "model",
    callback: ProgressCallback | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_bytes:
        print(f"[setup] Using completed download: {destination.name}")
        return
    if destination.exists() and destination.stat().st_size > expected_bytes:
        destination.unlink()

    parts = build_parts(destination, expected_bytes, connections)
    seed_parts_from_partial(destination, parts)
    completed = sum(
        min(part.size, part.path.stat().st_size) if part.path.exists() else 0
        for part in parts
    )
    progress = Progress(expected_bytes, completed, label=label, callback=callback)
    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [
            executor.submit(download_part, url, part, progress)
            for part in parts
        ]
        for future in futures:
            future.result()
    progress.finish()

    assembling = destination.with_name(f"{destination.name}.assembling")
    with assembling.open("wb") as output:
        for part in parts:
            if part.path.stat().st_size != part.size:
                raise OSError(f"Segment size mismatch: {part.path}")
            with part.path.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if assembling.stat().st_size != expected_bytes:
        raise OSError("Assembled download has an unexpected size")
    os.replace(assembling, destination)
    for part in parts:
        part.path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()

    connections = min(16, max(2, args.connections))
    download(args.url, args.output, args.size, connections)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
