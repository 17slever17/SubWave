import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from configs.settings import context_window_for_subtitles
from server.config_service import ConfigService
from server.presets import PRESETS
from server.runtime import RuntimeController
from services.runtime.events import RUNTIME_EVENT_PREFIX
from services.llm.prompts import FACTORY_PROMPTS, PromptStore


class _FakeStdin:
    def __init__(self) -> None:
        self.commands: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.commands.append(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None
        self.stdin = _FakeStdin()


class _GracefulFakeProcess(_FakeProcess):
    async def wait(self) -> int:
        self.returncode = 0
        await asyncio.sleep(0)
        return self.returncode


class RuntimePauseTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_and_resume_keep_the_same_process(self):
        controller = RuntimeController()
        process = _FakeProcess()
        controller._process = process
        controller._status.pid = process.pid
        controller._status.state = "running"

        paused = await controller.pause()
        resumed = await controller.resume()

        self.assertIs(controller._process, process)
        self.assertEqual(process.stdin.commands, [b"PAUSE\n", b"RESUME\n"])
        self.assertEqual(paused["pid"], process.pid)
        self.assertTrue(paused["paused"])
        self.assertEqual(resumed["state"], "running")
        self.assertFalse(resumed["paused"])

    async def test_stop_requests_graceful_child_shutdown_before_killing(self):
        controller = RuntimeController()
        process = _GracefulFakeProcess()
        controller._process = process
        controller._job_handle = 123
        controller._status.pid = process.pid
        controller._status.state = "running"

        with mock.patch(
            "server.runtime.close_windows_handle"
        ) as close_handle:
            stopped = await controller.stop()

        self.assertEqual(process.stdin.commands, [b"SHUTDOWN\n"])
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(stopped["exit_code"], 0)
        close_handle.assert_called_once_with(123)
        self.assertIsNone(controller._job_handle)

    async def test_stop_cancels_pending_start_without_waiting_for_download(self):
        controller = RuntimeController()
        download_started = threading.Event()
        release_download = threading.Event()

        def blocked_model_download(*_args, **_kwargs):
            download_started.set()
            release_download.wait(timeout=5)

        with mock.patch(
            "server.runtime.ensure_runtime_models",
            side_effect=blocked_model_download,
        ), mock.patch(
            "server.runtime.asyncio.create_subprocess_exec",
        ) as create_process:
            start_task = asyncio.create_task(controller.start())
            await asyncio.wait_for(asyncio.to_thread(download_started.wait, 2), timeout=3)

            stopped = await asyncio.wait_for(controller.stop(), timeout=0.5)
            self.assertEqual(stopped["state"], "idle")

            release_download.set()
            started = await asyncio.wait_for(start_task, timeout=3)

        self.assertEqual(started["state"], "idle")
        create_process.assert_not_called()


class RuntimePhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_events_are_hidden_and_track_overlapping_work(self):
        controller = RuntimeController()
        controller._status.state = "running"

        await controller._append_log(f"{RUNTIME_EVENT_PREFIX}stt_start")
        self.assertEqual(controller.status()["phase"], "transcribing")

        await controller._append_log(f"{RUNTIME_EVENT_PREFIX}llm_start")
        await controller._append_log(f"{RUNTIME_EVENT_PREFIX}stt_end")
        self.assertEqual(controller.status()["phase"], "translating")

        await controller._append_log(f"{RUNTIME_EVENT_PREFIX}llm_end")
        self.assertEqual(controller.status()["phase"], "ready")
        self.assertEqual(list(controller._logs), [])

    async def test_download_progress_replaces_previous_history_line(self):
        controller = RuntimeController()

        await controller._append_log("[download:STT model] 10.0%")
        await controller._append_log("[download:STT model] 20.0%")

        self.assertEqual(list(controller._logs), ["[download:STT model] 20.0%"])


class ConfigPersistenceTests(unittest.TestCase):
    def test_missing_current_config_is_created_from_reset_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            reset_path = root / "config.reset.yaml"
            reset_path.write_text(
                "translation:\n  model_path: models/factory.gguf\n",
                encoding="utf-8",
            )
            service = ConfigService(config_path, reset_path)

            service.ensure_current_config()

            self.assertTrue(config_path.is_file())
            self.assertEqual(
                service.read()["translation"]["model_path"],
                "models/factory.gguf",
            )

    def test_translation_model_survives_write_and_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            service = ConfigService(config_path)
            config = service.read()
            config["translation"]["model_path"] = (
                "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf"
            )
            config["translation"]["context_subtitles"] = 5
            config["translation"]["n_ctx"] = 9999

            service.write(config)
            reloaded = service.read()

            self.assertEqual(
                reloaded["translation"]["model_path"],
                "models/translate_gemma4_sub-E2B-Q4_K_XL.gguf",
            )
            self.assertEqual(reloaded["translation"]["context_subtitles"], 5)
            self.assertEqual(reloaded["translation"]["n_ctx"], 1344)

    def test_context_subtitles_control_context_window(self):
        self.assertEqual(context_window_for_subtitles(0), 544)
        self.assertEqual(context_window_for_subtitles(3), 1024)
        self.assertEqual(context_window_for_subtitles(5), 1344)


class PerformancePresetTests(unittest.TestCase):
    def test_presets_cover_8gb_6gb_and_cpu(self):
        self.assertEqual([preset["id"] for preset in PRESETS], ["max", "medium", "potato"])
        self.assertEqual([preset["vram"] for preset in PRESETS], ["8GB", "6GB", "CPU"])
        self.assertTrue(
            all(preset["patch"]["translation"]["n_batch"] == 256 for preset in PRESETS)
        )
        potato = PRESETS[-1]["patch"]["translation"]
        self.assertEqual(potato["device"], "cpu")
        self.assertEqual(potato["n_gpu_layers"], 0)


class PromptStoreSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PromptStore(Path(self.temp_dir.name) / "prompts.yaml")
        self.store.ensure_exists()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_factory_prompt_cannot_be_deleted(self):
        with self.assertRaises(ValueError):
            self.store.delete("ja_to_en")

    def test_duplicate_prompt_names_are_rejected(self):
        presets = dict(FACTORY_PROMPTS)
        presets["custom_duplicate"] = {
            **FACTORY_PROMPTS["ja_to_en"],
            "name": "japanese TO english",
        }
        with self.assertRaises(ValueError):
            self.store.save({"active": "ja_to_en", "presets": presets})

    def test_visible_direction_name_is_used_when_factory_name_is_empty(self):
        presets = dict(FACTORY_PROMPTS)
        presets["ja_to_en"] = {**FACTORY_PROMPTS["ja_to_en"], "name": ""}
        presets["custom_duplicate"] = {
            **FACTORY_PROMPTS["ja_to_en"],
            "name": "Japanese to English",
        }

        with self.assertRaises(ValueError):
            self.store.save({"active": "ja_to_en", "presets": presets})

    def test_custom_prompt_can_be_deleted(self):
        current = self.store.load()
        current["presets"]["custom_test"] = {
            "name": "Custom test",
            "source_language": "en",
            "target_language": "Russian",
            "template": "Translate.",
        }
        current["active"] = "custom_test"
        self.store.save(current)

        result = self.store.delete("custom_test")

        self.assertNotIn("custom_test", result["presets"])
        self.assertEqual(result["active"], "ja_to_en")

    def test_failed_atomic_replace_keeps_previous_prompts_readable(self):
        current = self.store.load()
        current["presets"]["custom_test"] = {
            "name": "Custom test",
            "source_language": "en",
            "target_language": "Russian",
            "template": "Original prompt",
        }
        self.store.save(current)

        changed = self.store.load()
        changed["presets"]["custom_test"]["template"] = "Changed prompt"
        with mock.patch("services.llm.prompts.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                self.store.save(changed)

        loaded = self.store.load()
        self.assertEqual(loaded["presets"]["custom_test"]["template"], "Original prompt")


if __name__ == "__main__":
    unittest.main()
