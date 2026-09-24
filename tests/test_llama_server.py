import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from configs.settings import TranslationConfig
from server.presets import PRESETS
from services.llm.llama_server import (
    NativeLlamaServer,
    _assign_kill_on_close_job,
    _close_windows_handle,
    get_runtime_backend,
)
from services.llm.translator import LLMTranslator


class NativeLlamaServerCommandTests(unittest.TestCase):
    def test_runtime_backend_marker_and_dlls_are_discovered(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bin_dir = Path(temporary_dir)
            server_path = bin_dir / "llama-server.exe"
            server_path.touch()
            marker_path = bin_dir / "runtime-backend.txt"

            marker_path.write_text("vulkan", encoding="ascii")
            self.assertEqual(get_runtime_backend(server_path), "vulkan")

            (bin_dir / "ggml-cuda.dll").touch()
            self.assertEqual(get_runtime_backend(server_path), "mixed")

            marker_path.write_text("cuda", encoding="ascii")
            (bin_dir / "ggml-vulkan.dll").touch()
            self.assertEqual(get_runtime_backend(server_path), "mixed")

    def test_runtime_backend_is_inferred_for_legacy_install(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bin_dir = Path(temporary_dir)
            server_path = bin_dir / "llama-server.exe"
            server_path.touch()
            self.assertEqual(get_runtime_backend(server_path), "cpu")

            (bin_dir / "ggml-vulkan.dll").touch()
            self.assertEqual(get_runtime_backend(server_path), "vulkan")

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

    def test_vulkan_offloads_layers_and_uses_flash_attention_auto(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bin_dir = Path(temporary_dir)
            server_path = bin_dir / "llama-server.exe"
            server_path.touch()
            (bin_dir / "runtime-backend.txt").write_text("vulkan", encoding="ascii")
            command = NativeLlamaServer.build_command(
                server_path=server_path,
                model_path=Path("target.gguf"),
                device="gpu",
                n_ctx=1024,
                n_batch=256,
                port=18080,
            )
        self.assertEqual(command[command.index("--gpu-layers") + 1], "all")
        self.assertEqual(command[command.index("--flash-attn") + 1], "auto")

    def test_cpu_runtime_marker_disables_gpu_offload_even_for_gpu_setting(self):
        command = NativeLlamaServer.build_command(
            server_path=Path("llama-server.exe"),
            model_path=Path("target.gguf"),
            mtp_model_path=Path("mtp.gguf"),
            device="gpu",
            runtime_backend="cpu",
            n_ctx=1024,
            n_batch=256,
            port=18080,
        )
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
        self.assertEqual(command[command.index("--spec-draft-device") + 1], "none")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "0")

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


class TranslatorBackendTests(unittest.TestCase):
    def _create_config(
        self,
        server_path: Path,
        model_path: Path,
        configured_server_path: Path | None = None,
    ) -> dict:
        return {
            "model_path": str(model_path),
            "device": "gpu",
            "mtp_enabled": True,
            "mtp_model_path": "auto",
            "llama_server_path": str(configured_server_path or server_path),
        }

    def test_vulkan_runtime_disables_automatic_mtp(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bin_dir = Path(temporary_dir)
            server_path = bin_dir / "llama-server.exe"
            server_path.touch()
            configured_server_path = bin_dir / "missing-custom-server.exe"
            (bin_dir / "runtime-backend.txt").write_text("vulkan", encoding="ascii")
            model_path = bin_dir / "translate_gemma4_sub-E4B-Q4_K_XL.gguf"
            model_path.touch()
            draft_path = bin_dir / "draft.gguf"
            draft_path.touch()

            with (
                mock.patch(
                    "services.llm.translator.NativeLlamaServer"
                ) as server_class,
                mock.patch(
                    "services.llm.translator.should_use_translation_mtp",
                    return_value=True,
                ) as should_use_mtp,
                mock.patch(
                    "services.llm.translator.translation_mtp_path",
                    return_value=draft_path,
                ),
            ):
                server_class.resolve_server_path.return_value = server_path
                translator = LLMTranslator(
                    self._create_config(
                        server_path,
                        model_path,
                        configured_server_path=configured_server_path,
                    )
                )

            server_class.resolve_server_path.assert_called_once_with(
                str(configured_server_path)
            )
            should_use_mtp.assert_not_called()
            kwargs = server_class.call_args.kwargs
            self.assertNotIn("mtp_model_path", kwargs)
            self.assertEqual(kwargs["device"], "gpu")
            self.assertEqual(translator.runtime_backend, "vulkan")

    def test_cpu_runtime_fallback_uses_cpu_device_and_timeout(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bin_dir = Path(temporary_dir)
            server_path = bin_dir / "llama-server.exe"
            server_path.touch()
            (bin_dir / "runtime-backend.txt").write_text("cpu", encoding="ascii")
            model_path = bin_dir / "model.gguf"
            model_path.touch()

            with (
                mock.patch(
                    "services.llm.translator.NativeLlamaServer"
                ) as server_class,
                mock.patch(
                    "services.llm.translator.should_use_translation_mtp",
                    return_value=False,
                ) as should_use_mtp,
            ):
                server_class.resolve_server_path.return_value = server_path
                LLMTranslator(self._create_config(server_path, model_path))

            should_use_mtp.assert_called_once()
            self.assertEqual(should_use_mtp.call_args.args[2], "cpu")
            kwargs = server_class.call_args.kwargs
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["request_timeout_s"], 31.0)


if __name__ == "__main__":
    unittest.main()
