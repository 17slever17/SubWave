import asyncio
import unittest

from main import stream_realtime_utterances


class _FakeSTT:
    def __init__(self) -> None:
        self.consumed = 0
        self.keep_running = asyncio.Event()

    async def stream_utterances(self):
        for value in ("one", "two", "three", "four", "five", "six"):
            self.consumed += 1
            yield value
            await asyncio.sleep(0.01)
        await self.keep_running.wait()


class RealtimeUtteranceQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_stt_keeps_consuming_and_drops_oldest_while_consumer_is_busy(self):
        stt = _FakeSTT()
        stream = stream_realtime_utterances(
            stt,
            max_pending=2,
            max_age_s=1.0,
        )
        self.assertEqual((await anext(stream))[1], "one")

        async with asyncio.timeout(0.5):
            while stt.consumed < 6:
                await asyncio.sleep(0.01)

        self.assertEqual(stt.consumed, 6)
        self.assertEqual((await anext(stream))[1], "five")
        self.assertEqual((await anext(stream))[1], "six")
        await stream.aclose()
