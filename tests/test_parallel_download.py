import tempfile
import unittest
from pathlib import Path

from scripts.download_parallel import (
    Progress,
    build_parts,
    seed_parts_from_partial,
)


class TestParallelDownload(unittest.TestCase):
    def test_progress_callback_reports_percent_speed_and_eta(self):
        messages: list[str] = []
        progress = Progress(1024 * 1024, 0, label="STT test", callback=messages.append)

        progress.add(512 * 1024)

        self.assertEqual(len(messages), 1)
        self.assertIn("[download:STT test]", messages[0])
        self.assertIn("50.0%", messages[0])
        self.assertIn("MiB/s", messages[0])
        self.assertIn("ETA", messages[0])

    def test_builds_non_overlapping_parts_covering_the_file(self):
        parts = build_parts(Path("model.zip"), expected_bytes=101, connections=8)

        self.assertEqual(parts[0].start, 0)
        self.assertEqual(parts[-1].end, 100)
        self.assertEqual(sum(part.size for part in parts), 101)
        self.assertEqual(
            [left.end + 1 for left in parts[:-1]],
            [right.start for right in parts[1:]],
        )

    def test_reuses_existing_sequential_partial_download(self):
        payload = bytes(range(100))
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "model.zip"
            destination.write_bytes(payload[:61])
            parts = build_parts(destination, expected_bytes=100, connections=4)

            seed_parts_from_partial(destination, parts)

            self.assertFalse(destination.exists())
            self.assertEqual(parts[0].path.read_bytes(), payload[:25])
            self.assertEqual(parts[1].path.read_bytes(), payload[25:50])
            self.assertEqual(parts[2].path.read_bytes(), payload[50:61])
            self.assertFalse(parts[3].path.exists())
