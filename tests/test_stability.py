import asyncio
from io import StringIO
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from main import (
    RuntimeCommandListener,
    SubtitleDisplayHistory,
    SubtitleDisplayManager,
    append_translation_log,
    subtitle_display_duration,
)
from services.browser.status import is_allowed_browser_origin
from services.browser.overlay import OverlayMessage, SubtitleOverlay
from services.stt.service import STTService, TranscriptionResult
from services.filters.text import (
    cap_repeated_token_sequences,
    collapse_repeated_text,
    has_repeated_character_loop,
    has_repeated_token_loop,
    is_context_leak,
    source_repetition_similarity,
    strip_repeated_subtitle_prefix,
    text_preview,
)


def build_filter_service() -> STTService:
    service = STTService.__new__(STTService)
    service.sherpa_model_id = "legacy-test-model"
    service.blocked_phrases = ["Дима Торжок"]
    service.duplicate_similarity_threshold = 0.90
    service.max_consecutive_duplicate_utterances = 1
    service._last_utterance_normalized = ""
    service._consecutive_utterance_count = 0
    return service


class TestRuntimeCommandListener(unittest.TestCase):
    def test_shutdown_command_requests_async_runtime_exit(self):
        shutdown_requests = []
        stt_service = SimpleNamespace(set_paused=lambda _paused: None)
        listener = RuntimeCommandListener(
            stt_service,
            request_shutdown=lambda: shutdown_requests.append(True),
        )

        with mock.patch("main.sys.stdin", StringIO("SHUTDOWN\n")):
            listener._listen()

        self.assertEqual(shutdown_requests, [True])


def transcription(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text)


class TestTextStability(unittest.TestCase):
    def test_caps_single_word_loop_at_three_repetitions(self):
        cleaned, changed = cap_repeated_token_sequences("quiero " * 20)
        self.assertTrue(changed)
        self.assertEqual(cleaned, "quiero quiero quiero")

    def test_caps_repeated_phrase_at_three_repetitions(self):
        cleaned, changed = cap_repeated_token_sequences("no puedo " * 5)
        self.assertTrue(changed)
        self.assertEqual(cleaned, "no puedo no puedo no puedo")

    def test_preserves_natural_spanish_repetition(self):
        text = "No quiero no quiero no quiero, quiero quiero"
        cleaned, changed = cap_repeated_token_sequences(text)
        self.assertFalse(changed)
        self.assertEqual(cleaned, text)

    def test_caps_unspaced_japanese_loop_at_three_repetitions(self):
        cleaned, changed = cap_repeated_token_sequences("押さえたら" * 100)
        self.assertTrue(changed)
        self.assertEqual(cleaned, "押さえたら" * 3)

    def test_collapses_duplicated_translation(self):
        text = "Спасибо, что пришли. Спасибо, что пришли."
        cleaned, changed = collapse_repeated_text(text)
        self.assertTrue(changed)
        self.assertEqual(cleaned, "Спасибо, что пришли")

    def test_preserves_short_human_emphasis(self):
        cleaned, changed = collapse_repeated_text("Очень, очень хорошо!")
        self.assertFalse(changed)
        self.assertEqual(cleaned, "Очень, очень хорошо!")

    def test_detects_repeated_token_loop(self):
        self.assertTrue(has_repeated_token_loop("это было плохо это было плохо это было плохо"))

    def test_detects_single_word_loop(self):
        self.assertTrue(has_repeated_token_loop("да да да да да да да"))

    def test_preserves_local_human_repetition_inside_a_real_utterance(self):
        text = (
            "Uh, wait, wait, wait, I haven't looked at my book in so long. "
            "Give me a second, give me a second"
        )
        self.assertFalse(has_repeated_token_loop(text))

    def test_detects_unspaced_japanese_loop(self):
        self.assertTrue(has_repeated_character_loop("押さえたら" * 100))
        self.assertTrue(has_repeated_token_loop("たら" * 100))

    def test_does_not_treat_short_emphasis_as_loop(self):
        self.assertFalse(has_repeated_token_loop("нет нет нет"))
        self.assertFalse(has_repeated_character_loop("はははは"))

    def test_long_log_preview_is_truncated(self):
        preview = text_preview("たら" * 100, max_chars=20)
        self.assertLess(len(preview), 60)
        self.assertIn("200 chars", preview)

    def test_detects_translation_of_previous_context(self):
        self.assertTrue(
            is_context_leak(
                current_source="whatever. Whatever. Vibe coding.",
                previous_source="I don't mean coding, like, I don't mean how well I do, but I mean like",
                translation="Не то чтобы кодинг, я не имею в виду, насколько хорошо я справляюсь, но я имею в виду типа",
                previous_translation="Я не о кодировании, я не о том, как хорошо я справляюсь, а я имею в виду, типа",
            )
        )

    def test_repeated_source_is_not_a_context_leak(self):
        self.assertFalse(
            is_context_leak(
                current_source="No, no, that's not what I meant.",
                previous_source="No, that's not what I meant.",
                translation="Нет, нет, я не это имел в виду.",
                previous_translation="Нет, я не это имел в виду.",
            )
        )

    def test_strips_previous_subtitle_from_continuation(self):
        cleaned, removed, similarity = strip_repeated_subtitle_prefix(
            previous_translation="Это я. Это буду я, когда приду к тебе со своим...",
            current_translation=(
                "Это я. Это буду я, когда приду к тебе со своими... "
                "Планами по программированию. У меня будет целый план-концепт."
            ),
            previous_source="明日は一人で行くつもり。",
            current_source="プログラミングの計画も全部見せるよ。",
            source_language="ja",
        )
        self.assertTrue(removed)
        self.assertGreaterEqual(similarity, 0.82)
        self.assertEqual(
            cleaned,
            "Планами по программированию. У меня будет целый план-концепт.",
        )

    def test_strips_reordered_near_duplicate_prefix(self):
        cleaned, removed, _ = strip_repeated_subtitle_prefix(
            previous_translation="Я обязательно покажу вам это завтра.",
            current_translation="Завтра я вам это обязательно покажу... А потом всё объясню.",
        )
        self.assertTrue(removed)
        self.assertEqual(cleaned, "А потом всё объясню.")

    def test_preserves_unrelated_and_short_prefixes(self):
        unrelated = strip_repeated_subtitle_prefix(
            previous_translation="Мы закончили первую часть.",
            current_translation="Теперь начинается самое сложное.",
        )
        short = strip_repeated_subtitle_prefix(
            previous_translation="Да.",
            current_translation="Да. Но сначала нужно проверить звук.",
        )
        self.assertEqual(unrelated[:2], ("Теперь начинается самое сложное.", False))
        self.assertEqual(short[:2], ("Да. Но сначала нужно проверить звук.", False))

    def test_preserves_real_repeated_spoken_phrase(self):
        current_translation = "Я покажу вам это завтра. Я обещаю."
        cleaned, removed, _ = strip_repeated_subtitle_prefix(
            previous_translation="Я покажу вам это завтра.",
            current_translation=current_translation,
            previous_source="I will show it to you tomorrow.",
            current_source="I will show it to you tomorrow. I promise.",
            source_language="en",
        )
        self.assertFalse(removed)
        self.assertEqual(cleaned, current_translation)

    def test_preserves_real_cjk_repetition(self):
        self.assertEqual(source_repetition_similarity("明日見せるよ。", "明日見せるよ。本当に。", "ja"), 1.0)
        current_translation = "Я покажу завтра. Правда."
        cleaned, removed, _ = strip_repeated_subtitle_prefix(
            previous_translation="Я покажу завтра.",
            current_translation=current_translation,
            previous_source="明日見せるよ。",
            current_source="明日見せるよ。本当に。",
            source_language="ja",
        )
        self.assertFalse(removed)
        self.assertEqual(cleaned, current_translation)

    def test_strips_false_japanese_to_english_prefix(self):
        cleaned, removed, similarity = strip_repeated_subtitle_prefix(
            previous_translation="That's me. That's what I'll be when I come to you with my...",
            current_translation=(
                "That's me. That's what I'll be when I come to you with mine... "
                "Programming plans. I'll have a whole concept ready."
            ),
            previous_source="昨日は一人で行くつもりだった。",
            current_source="プログラミングの計画を全部見せるよ。",
            source_language="ja",
        )
        self.assertTrue(removed)
        self.assertGreaterEqual(similarity, 0.82)
        self.assertEqual(cleaned, "Programming plans. I'll have a whole concept ready.")

    def test_preserves_real_chinese_to_english_repetition(self):
        current_translation = "I'll show you tomorrow. I promise."
        cleaned, removed, _ = strip_repeated_subtitle_prefix(
            previous_translation="I'll show you tomorrow.",
            current_translation=current_translation,
            previous_source="我明天会给你看。",
            current_source="我明天会给你看。我保证。",
            source_language="zh",
        )
        self.assertFalse(removed)
        self.assertEqual(cleaned, current_translation)

    def test_preserves_real_spanish_to_english_repetition(self):
        current_translation = "I'll show it to you tomorrow. I promise."
        cleaned, removed, _ = strip_repeated_subtitle_prefix(
            previous_translation="I'll show it to you tomorrow.",
            current_translation=current_translation,
            previous_source="Te lo mostraré mañana.",
            current_source="Te lo mostraré mañana. Lo prometo.",
            source_language="es",
        )
        self.assertFalse(removed)
        self.assertEqual(cleaned, current_translation)


class TestSTTStability(unittest.TestCase):
    def test_always_rejects_blocked_phrase(self):
        service = build_filter_service()
        result = transcription("Дима Торжок")
        self.assertIn("blocked phrase", service._transcription_rejection_reason(result))


    def test_repeated_utterance_is_suppressed(self):
        service = build_filter_service()
        result = transcription("これは本当です")
        self.assertEqual(service._prepare_transcription(result), "これは本当です")
        self.assertEqual(service._prepare_transcription(result), "")

    def test_two_repeated_stt_phrases_are_preserved(self):
        service = build_filter_service()
        result = transcription("本当にありがとう。本当にありがとう。")
        self.assertEqual(service._prepare_transcription(result), result.text)

    def test_large_unspaced_stt_loop_is_capped(self):
        service = build_filter_service()
        result = transcription("押さえたら" * 100)
        self.assertEqual(
            service._prepare_transcription(result),
            "押さえたら" * 3,
        )

    def test_parakeet_caps_repetition_inside_one_utterance(self):
        service = build_filter_service()
        service.sherpa_model_id = "auto"
        service._resolved_sherpa_model_id = (
            "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"
        )
        repeated = "wait wait wait wait wait wait wait wait"
        self.assertEqual(
            service._prepare_transcription(transcription(repeated)),
            "wait wait wait",
        )


class TestSubtitleDisplayHistory(unittest.TestCase):
    def test_keeps_last_published_subtitle_independent_of_context(self):
        history = SubtitleDisplayHistory()

        self.assertEqual(history.advance("Первая строка"), ("", "Первая строка"))
        # Timeouts and rejected chunks do not call advance and must not clear display history.
        self.assertEqual(
            history.advance("Вторая строка"),
            ("Первая строка", "Вторая строка"),
        )

    def test_display_duration_uses_four_words_per_second_with_bounds(self):
        self.assertEqual(subtitle_display_duration("одно слово"), 1.0)
        self.assertEqual(subtitle_display_duration(" ".join(["слово"] * 10)), 2.5)
        self.assertEqual(subtitle_display_duration(" ".join(["слово"] * 30)), 4.0)

    def test_display_duration_catches_up_when_two_subtitles_are_waiting(self):
        long_subtitle = " ".join(["слово"] * 30)

        self.assertEqual(
            subtitle_display_duration(long_subtitle, pending_subtitles=1),
            4.0,
        )
        self.assertEqual(
            subtitle_display_duration(long_subtitle, pending_subtitles=2),
            3.0,
        )
        self.assertEqual(
            subtitle_display_duration("короткий текст", pending_subtitles=2),
            1.0,
        )


class TestSubtitleDisplayManager(unittest.IsolatedAsyncioTestCase):
    async def test_waits_before_showing_the_next_ready_subtitle(self):
        published = []
        sleep_calls = asyncio.Queue()

        async def fake_sleep(seconds):
            release = asyncio.Event()
            await sleep_calls.put((seconds, release))
            await release.wait()

        display_manager = SubtitleDisplayManager(
            lambda previous, current: published.append((previous, current)),
            sleep=fake_sleep,
        )
        display_manager.start()
        display_manager.add("Длинный первый субтитр из шести отдельных слов")
        display_manager.add("Одно")

        first_delay, release_first = await asyncio.wait_for(sleep_calls.get(), 1)
        self.assertEqual(
            published,
            [("", "Длинный первый субтитр из шести отдельных слов")],
        )

        release_first.set()
        second_delay, _ = await asyncio.wait_for(sleep_calls.get(), 1)
        await display_manager.close()

        self.assertEqual(
            published,
            [
                ("", "Длинный первый субтитр из шести отдельных слов"),
                ("Длинный первый субтитр из шести отдельных слов", "Одно"),
            ],
        )
        self.assertEqual([first_delay, second_delay], [1.75, 1.0])

    async def test_shortens_current_display_when_backlog_reaches_two(self):
        published = []
        sleep_calls = asyncio.Queue()

        async def fake_sleep(seconds):
            release = asyncio.Event()
            await sleep_calls.put((seconds, release))
            await release.wait()

        display_manager = SubtitleDisplayManager(
            lambda previous, current: published.append((previous, current)),
            sleep=fake_sleep,
        )
        display_manager.start()
        display_manager.add(" ".join(["длинный"] * 30))

        initial_delay, _ = await asyncio.wait_for(sleep_calls.get(), 1)
        display_manager.add("Второй субтитр")
        display_manager.add("Третий субтитр")
        shortened_delay, release_shortened = await asyncio.wait_for(
            sleep_calls.get(),
            1,
        )

        self.assertEqual(initial_delay, 4.0)
        self.assertLessEqual(shortened_delay, 3.0)
        self.assertGreater(shortened_delay, 2.95)
        self.assertEqual(len(published), 1)

        release_shortened.set()
        await asyncio.sleep(0)
        await display_manager.close()

    async def test_drops_oldest_pending_subtitle_when_display_falls_behind(self):
        display_manager = SubtitleDisplayManager(lambda *_: None, max_pending=2)

        display_manager.add("Первый ожидающий")
        display_manager.add("Второй ожидающий")
        display_manager.add("Самый свежий")

        self.assertEqual(display_manager._queue.qsize(), 2)
        self.assertEqual(display_manager._queue.get_nowait()[0], "Второй ожидающий")
        self.assertEqual(display_manager._queue.get_nowait()[0], "Самый свежий")

    async def test_skips_pending_subtitle_that_is_too_old_to_be_realtime(self):
        published = []
        display_manager = SubtitleDisplayManager(
            lambda previous, current: published.append((previous, current)),
            max_age_s=0.001,
        )
        display_manager.add("Устаревший")
        await asyncio.sleep(0.01)
        display_manager.start()
        await asyncio.sleep(0)
        await display_manager.close()

        self.assertEqual(published, [])

    def test_strips_published_text_without_losing_previous(self):
        history = SubtitleDisplayHistory()
        history.advance("  Previous  ")

        self.assertEqual(history.advance("  Current  "), ("Previous", "Current"))


class TestOverlayStability(unittest.TestCase):
    def test_overlay_queue_keeps_only_latest_messages(self):
        overlay = SubtitleOverlay(SimpleNamespace())
        for index in range(10):
            overlay._put_latest(OverlayMessage(str(index - 1), str(index)))

        messages = []
        while not overlay._queue.empty():
            messages.append(overlay._queue.get_nowait())

        self.assertEqual(len(messages), 4)
        self.assertEqual([message.translation for message in messages], ["6", "7", "8", "9"])


class TestTranslationLogStability(unittest.TestCase):
    def test_translation_log_rotates_at_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "translations_log.txt"
            log_path.write_text("old payload", encoding="utf-8")

            append_translation_log(str(log_path), "new payload", max_bytes=15)

            self.assertEqual(log_path.read_text(encoding="utf-8"), "new payload")
            self.assertEqual(
                (log_path.parent / "translations_log.1.txt").read_text(encoding="utf-8"),
                "old payload",
            )


class TestBrowserBridgeSecurity(unittest.TestCase):
    def test_supported_video_origins_are_allowed(self):
        self.assertTrue(is_allowed_browser_origin("https://www.youtube.com"))
        self.assertTrue(is_allowed_browser_origin("https://www.twitch.tv"))
        self.assertTrue(is_allowed_browser_origin(""))

    def test_unrelated_web_origins_are_rejected(self):
        self.assertFalse(is_allowed_browser_origin("https://example.com"))
        self.assertFalse(is_allowed_browser_origin("null"))

if __name__ == "__main__":
    unittest.main()
