import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from configs.settings import TranslationConfig
from server.presets import PRESETS
from services.llm.llama_server import (
    NativeLlamaServer,
    _assign_kill_on_close_job,
    _close_windows_handle,
)


class NativeLlamaServerCommandTests(unittest.TestCase):
    def test_chat_completion_uses_realtime_request_timeout(self):
        server = NativeLlamaServer.__new__(NativeLlamaServer)
        server.request_timeout_s = 4.0
        server._request = mock.Mock(return_value={"choices": []})

        server.create_chat_completion(messages=[])

        server._request.assert_called_once_with(
            "POST",
            "/v1/chat/completions",
            {"messages": [], "model": "translation"},
            timeout=4.0,
        )

    @unittest.skipUnless(os.name == "nt", "Windows Job Objects are Windows-only")
    def test_kill_on_close_job_terminates_child_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        job_handle = None
        try:
            job_handle = _assign_kill_on_close_job(process)
            _close_windows_handle(job_handle)
            job_handle = None
            process.wait(timeout=5)
            self.assertIsNotNone(process.returncode)
        finally:
            if job_handle:
                _close_windows_handle(job_handle)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_translation_defaults_use_gpu_e4b_with_stable_mtp_depth(self):
        config = TranslationConfig()
        self.assertIn("E4B", config.model_path)
        self.assertEqual(config.device, "gpu")
        self.assertTrue(config.mtp_enabled)
        self.assertEqual(config.mtp_model_path, "auto")
        self.assertEqual(config.mtp_n, 1)

    def test_cpu_preset_uses_e2b_with_stable_mtp_depth(self):
        potato = next(preset for preset in PRESETS if preset["id"] == "potato")
        translation = potato["patch"]["translation"]
        self.assertIn("E2B", translation["model_path"])
        self.assertEqual(translation["device"], "cpu")
        self.assertTrue(translation["mtp_enabled"])
        self.assertEqual(translation["mtp_n"], 1)

    def test_gpu_uses_full_target_and_draft_offload_with_stable_mtp_depth(self):
        command = NativeLlamaServer.build_command(
            server_path=Path("llama-server.exe"),
            model_path=Path("target.gguf"),
            mtp_model_path=Path("mtp.gguf"),
            mtp_n=1,
            device="gpu",
            n_ctx=1024,
            n_batch=256,
            port=18080,
        )
        self.assertIn("--spec-type", command)
        self.assertIn("draft-mtp", command)
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "1")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "all")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "all")
        self.assertNotIn("--no-warmup", command)

    def test_gpu_standard_command_omits_all_draft_flags(self):
        command = NativeLlamaServer.build_command(
            server_path=Path("llama-server.exe"),
            model_path=Path("target.gguf"),
            device="gpu",
            n_ctx=1024,
            n_batch=256,
            port=18080,
        )
        self.assertNotIn("--spec-type", command)
        self.assertNotIn("--spec-draft-model", command)
        self.assertNotIn("--spec-draft-n-max", command)
        self.assertNotIn("--spec-draft-ngl", command)
        self.assertNotIn("--spec-draft-device", command)
        self.assertEqual(command[command.index("--gpu-layers") + 1], "all")

    def test_cpu_disables_target_and_draft_gpu_offload(self):
        command = NativeLlamaServer.build_command(
            server_path=Path("llama-server.exe"),
            model_path=Path("target.gguf"),
            mtp_model_path=Path("mtp.gguf"),
            mtp_n=1,
            device="cpu",
            n_ctx=1024,
            n_batch=256,
            port=18080,
        )
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
        self.assertEqual(command[command.index("--spec-draft-device") + 1], "none")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "0")

    def test_cpu_standard_command_omits_all_draft_flags(self):
        command = NativeLlamaServer.build_command(
            server_path=Path("llama-server.exe"),
            model_path=Path("target.gguf"),
            device="cpu",
            n_ctx=1024,
            n_batch=256,
            port=18080,
        )
        self.assertNotIn("--spec-type", command)
        self.assertNotIn("--spec-draft-model", command)
        self.assertNotIn("--spec-draft-n-max", command)
        self.assertNotIn("--spec-draft-ngl", command)
        self.assertNotIn("--spec-draft-device", command)
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "0")


if __name__ == "__main__":
    unittest.main()
