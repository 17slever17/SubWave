import asyncio
import contextlib
import json
import socket
from types import SimpleNamespace
import unittest

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from services.stt.audio import (
    BrowserPCMWebSocketCapture,
    PCMChunkBuffer,
    build_audio_source,
)


class BrowserAudioSourceTests(unittest.TestCase):
    def test_browser_source_does_not_require_an_audio_device(self):
        config = SimpleNamespace(
            audio=SimpleNamespace(capture_mode="browser_tab"),
        )

        source = build_audio_source(config)

        self.assertEqual(source.capture_mode, "browser_tab")
        self.assertIsNone(source.device_id)
        self.assertEqual(source.native_samplerate, 48000)


class BrowserPCMBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_browser_connection_takes_over_active_capture(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        capture = BrowserPCMWebSocketCapture(
            host="127.0.0.1",
            port=port,
            target_samplerate=16000,
            gain=1.0,
            buffer=PCMChunkBuffer(max_chunks=8),
        )
        server_task = asyncio.create_task(capture.serve())
        first = None
        second = None
        try:
            for _ in range(50):
                try:
                    first = await connect(
                        f"ws://127.0.0.1:{port}/audio?session_started_at=100"
                    )
                    break
                except OSError:
                    await asyncio.sleep(0.01)
            self.assertIsNotNone(first)
            await first.recv()

            second = await connect(
                f"ws://127.0.0.1:{port}/audio?session_started_at=200"
            )
            ready = json.loads(await asyncio.wait_for(second.recv(), timeout=1))
            self.assertEqual(ready["type"], "ready")
            with self.assertRaises(ConnectionClosed) as closed:
                await asyncio.wait_for(first.recv(), timeout=1)
            self.assertEqual(closed.exception.rcvd.code, 4001)

            stale = await connect(
                f"ws://127.0.0.1:{port}/audio?session_started_at=100"
            )
            with self.assertRaises(ConnectionClosed) as stale_closed:
                await asyncio.wait_for(stale.recv(), timeout=1)
            self.assertEqual(stale_closed.exception.rcvd.code, 4001)
        finally:
            if first is not None:
                await first.close()
            if second is not None:
                await second.close()
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task

    async def test_binary_pcm_reaches_resampled_buffer(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        buffer = PCMChunkBuffer(max_chunks=8)
        capture = BrowserPCMWebSocketCapture(
            host="127.0.0.1",
            port=port,
            target_samplerate=16000,
            gain=1.0,
            buffer=buffer,
        )
        server_task = asyncio.create_task(capture.serve())
        try:
            websocket = None
            for _ in range(50):
                try:
                    websocket = await connect(
                        f"ws://127.0.0.1:{port}/audio"
                    )
                    break
                except OSError:
                    await asyncio.sleep(0.01)
            self.assertIsNotNone(websocket)

            async with websocket:
                ready = json.loads(await websocket.recv())
                self.assertEqual(ready["type"], "ready")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "stream_start",
                            "format": "f32le",
                            "sample_rate": 48000,
                            "channels": 1,
                        }
                    )
                )
                samples = np.full(1920, 0.125, dtype="<f4")
                frame = (0).to_bytes(4, "little") + samples.tobytes()
                await websocket.send(frame)

                for _ in range(50):
                    if buffer.qsize():
                        break
                    await asyncio.sleep(0.01)

            chunk = buffer.get_nowait()
            self.assertEqual(chunk.sample_rate, 16000)
            self.assertEqual(chunk.source_name, "browser-tab")
            # The streaming resampler retains a short filter tail for the
            # following chunk instead of padding the first output block.
            self.assertGreater(chunk.samples.size, 400)
            self.assertTrue(np.all(np.isfinite(chunk.samples)))
        finally:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task


if __name__ == "__main__":
    unittest.main()
