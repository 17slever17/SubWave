from __future__ import annotations

import unittest

from server.runtime import (
    MAX_LOG_HISTORY,
    MAX_LOG_QUEUE,
    MAX_LOG_REPLAY,
    RuntimeController,
)


class RuntimeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_history_and_new_subscriber_replay_are_bounded(self):
        controller = RuntimeController()
        for index in range(MAX_LOG_HISTORY + 50):
            await controller._append_log(f"line-{index}")

        self.assertEqual(len(controller._logs), MAX_LOG_HISTORY)
        queue = controller.subscribe_logs()
        self.assertLessEqual(queue.qsize(), MAX_LOG_REPLAY)
        replay = [queue.get_nowait() for _ in range(queue.qsize())]
        self.assertEqual(replay[-1], f"line-{MAX_LOG_HISTORY + 49}")

    async def test_slow_log_subscriber_drops_oldest_without_blocking_runtime(self):
        controller = RuntimeController()
        queue = controller.subscribe_logs()
        for index in range(MAX_LOG_QUEUE + 25):
            await controller._append_log(f"new-{index}")

        self.assertEqual(queue.qsize(), MAX_LOG_QUEUE)
        self.assertEqual(queue.get_nowait(), "new-25")

    async def test_pause_without_process_reports_error_without_fake_state(self):
        controller = RuntimeController()
        result = await controller.pause()

        self.assertEqual(result["state"], "idle")
        self.assertFalse(result["paused"])
        self.assertIn("not running", result["last_error"])


if __name__ == "__main__":
    unittest.main()
