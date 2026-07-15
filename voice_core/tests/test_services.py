import asyncio
import sys
import tempfile
import threading
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from voice_core.config import Settings
from voice_core.services import (
    HotwordDetector,
    LanguageModel,
    SpeechToText,
    TextToSpeech,
    ToolResult,
    _merge_tool_call_deltas,
    _message_content,
    _silero_sentences,
    _stream_content,
)
from voice_core.user_memory import USER_MEMORY_TOOLS


class LanguageModelResponseTests(unittest.TestCase):
    def test_extracts_string_content(self) -> None:
        self.assertEqual(_message_content({"content": "  ответ  "}), "ответ")

    def test_extracts_structured_text_content(self) -> None:
        self.assertEqual(
            _message_content({"content": [{"type": "text", "text": "первая "}, {"text": "часть"}]}),
            "первая часть",
        )

    def test_reasoning_is_not_used_as_spoken_answer(self) -> None:
        self.assertEqual(_message_content({"content": "", "reasoning_content": "секретные мысли"}), "")

    def test_stream_content_preserves_spaces_between_chunks(self) -> None:
        self.assertEqual(_stream_content("часть "), "часть ")
        self.assertEqual(_stream_content([{"text": "ещё "}, {"text": "часть"}]), "ещё часть")

    def test_streamed_tool_call_fragments_are_merged(self) -> None:
        calls: dict[int, dict[str, object]] = {}

        _merge_tool_call_deltas(
            calls,
            [{"index": 0, "id": "call-1", "function": {"name": "remember_", "arguments": "{\"preferred"}}],
        )
        _merge_tool_call_deltas(
            calls,
            [{"index": 0, "function": {"name": "name", "arguments": "_name\":\"Пупсик\"}"}}],
        )

        self.assertEqual(calls[0]["id"], "call-1")
        self.assertEqual(calls[0]["function"], {
            "name": "remember_name",
            "arguments": '{"preferred_name":"Пупсик"}',
        })


class LanguageModelToolChoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_tool_ends_completion_without_second_request(self) -> None:
        model = LanguageModel(Settings())
        payloads: list[dict[str, object]] = []

        async def fake_completion(payload, _request_number):
            payloads.append(dict(payload))
            return (
                {
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [
                        {
                            "id": "stop-1",
                            "type": "function",
                            "function": {
                                "name": "end_conversation",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "tool_calls",
            )

        async def tool_handler(name, arguments):
            self.assertEqual(name, "end_conversation")
            self.assertEqual(arguments, {})
            return ToolResult('{"ok":true}', terminate=True)

        model._stream_completion = fake_completion
        try:
            answer = await model.reply(
                [],
                "На этом закончим",
                USER_MEMORY_TOOLS,
                tool_handler,
            )
        finally:
            await model.close()

        self.assertEqual(answer, "")
        self.assertEqual(len(payloads), 1)

    async def test_terminal_tool_can_supply_backend_response(self) -> None:
        model = LanguageModel(Settings())
        payloads = []

        async def fake_completion(payload, _request_number):
            payloads.append(payload)
            return (
                {
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [
                        {
                            "id": "lookup-1",
                            "type": "function",
                            "function": {
                                "name": "lookup_user_name",
                                "arguments": '{"subject":"current_user"}',
                            },
                        }
                    ],
                },
                "tool_calls",
            )

        async def tool_handler(_name, _arguments):
            return ToolResult(
                '{"found":true,"answer_name":"Валерий"}',
                terminate=True,
                response="Тебя зовут Валерий.",
            )

        model._stream_completion = fake_completion
        try:
            answer = await model.reply([], "Как меня зовут?", USER_MEMORY_TOOLS, tool_handler)
        finally:
            await model.close()

        self.assertEqual(answer, "Тебя зовут Валерий.")
        self.assertEqual(len(payloads), 1)

    async def test_image_is_added_only_to_current_user_message(self) -> None:
        model = LanguageModel(Settings())
        payloads: list[dict[str, object]] = []

        async def fake_completion(payload, _request_number):
            payloads.append(dict(payload))
            return (
                {"content": "На изображении тест.", "reasoning_content": "", "tool_calls": []},
                "stop",
            )

        model._stream_completion = fake_completion
        try:
            answer = await model.reply(
                [{"role": "user", "content": "старый текст"}],
                "Что здесь?",
                image_data=b"image-bytes",
                image_content_type="image/png",
            )
        finally:
            await model.close()

        self.assertEqual(answer, "На изображении тест.")
        messages = payloads[0]["messages"]
        self.assertEqual(messages[1]["content"], "старый текст")
        current_content = messages[-1]["content"]
        self.assertEqual(current_content[0], {"type": "text", "text": "Что здесь?"})
        self.assertTrue(
            current_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    async def test_required_tool_is_forced_only_before_its_result(self) -> None:
        model = LanguageModel(Settings())
        payloads: list[dict[str, object]] = []

        async def fake_completion(payload, request_number):
            payloads.append(dict(payload))
            if request_number == 1:
                return (
                    {
                        "content": "",
                        "reasoning_content": "",
                        "tool_calls": [
                            {
                                "id": "lookup-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_user_name",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "tool_calls",
                )
            return (
                {"content": "Тебя зовут Кэп.", "reasoning_content": "", "tool_calls": []},
                "stop",
            )

        async def tool_handler(name, arguments):
            self.assertEqual(name, "lookup_user_name")
            self.assertEqual(arguments, {})
            return '{"found":true,"preferred_name":"Кэп"}'

        model._stream_completion = fake_completion
        try:
            answer = await model.reply(
                [],
                "Как меня зовут?",
                USER_MEMORY_TOOLS,
                tool_handler,
                required_tool_name="lookup_user_name",
            )
        finally:
            await model.close()

        self.assertEqual(answer, "Тебя зовут Кэп.")
        self.assertEqual(
            payloads[0]["tool_choice"],
            "required",
        )
        self.assertEqual(len(payloads[0]["tools"]), 1)
        self.assertEqual(
            payloads[0]["tools"][0]["function"]["name"], "lookup_user_name"
        )
        self.assertNotIn("tool_choice", payloads[1])


class SpeechToTextTests(unittest.TestCase):
    def test_leading_silence_and_initial_prompt_are_passed_to_whisper(self) -> None:
        captured: dict[str, object] = {}

        class FakeModel:
            def transcribe(self, audio: np.ndarray, **options: object):
                captured["audio"] = audio
                captured["options"] = options
                return [SimpleNamespace(text=" Омни, привет ")], None

        settings = replace(
            Settings(),
            whisper_beam_size=2,
            whisper_speech_gate=False,
            whisper_leading_silence_ms=400,
        )
        stt = SpeechToText(settings)
        stt._model = FakeModel()

        transcript = stt._transcribe_sync(np.ones(160, dtype=np.float32))

        audio = captured["audio"]
        self.assertIsInstance(audio, np.ndarray)
        np.testing.assert_array_equal(audio[:6400], np.zeros(6400, dtype=np.float32))
        np.testing.assert_array_equal(audio[6400:], np.ones(160, dtype=np.float32))
        self.assertIsNone(captured["options"]["initial_prompt"])
        self.assertIsNone(captured["options"]["hotwords"])
        self.assertEqual(captured["options"]["beam_size"], 2)
        self.assertFalse(captured["options"]["vad_filter"])
        self.assertEqual(transcript, "Омни, привет")

    def test_speech_gate_discards_silence_before_whisper(self) -> None:
        class UnexpectedModel:
            def transcribe(self, audio: np.ndarray, **options: object):
                raise AssertionError("Whisper must not run for silence")

        settings = replace(
            Settings(),
            whisper_speech_gate=True,
            whisper_min_speech_ms=180,
        )
        stt = SpeechToText(settings)
        stt._model = UnexpectedModel()

        self.assertEqual(stt._transcribe_sync(np.zeros(16_000, dtype=np.float32)), "")


class HotwordDetectorTests(unittest.TestCase):
    def test_alias_is_detected_in_partial_transcript(self) -> None:
        class FakeModel:
            def transcribe(self, audio: np.ndarray, **options: object):
                return [SimpleNamespace(text=" Вомни, как меня зовут ")], None

        settings = replace(
            Settings(),
            wake_word="омни",
            wake_word_aliases=("помни", "вомни"),
        )
        detector = HotwordDetector(settings)
        detector._model = FakeModel()

        detected, transcript = detector._detect_sync(np.ones(160, dtype=np.float32))

        self.assertTrue(detected)
        self.assertEqual(transcript, "Вомни, как меня зовут")

    def test_spaced_and_misrecognized_wake_aliases_are_detected(self) -> None:
        settings = replace(
            Settings(),
            wake_word="омни",
            wake_word_aliases=(
                "помни",
                "вомни",
                "омли",
                "омне",
                "о мне",
                "о мни",
                "амни",
                "умни",
                "омний",
                "омник",
                "умник",
            ),
        )
        detector = HotwordDetector(settings)

        for transcript in (
            "Помни.",
            "Вомни, привет",
            "Омли, привет",
            "Омне, привет",
            "О мне!",
            "О, мне, привет",
            "О мни, ответь",
            "Амни, ответь",
            "Умни, ответь",
            "Омний, ответь",
            "Омник, ответь",
            "Умник, ответь",
        ):
            class FakeModel:
                def transcribe(self, audio: np.ndarray, **options: object):
                    return [SimpleNamespace(text=transcript)], None

            detector._model = FakeModel()
            detected, _ = detector._detect_sync(np.ones(160, dtype=np.float32))
            self.assertTrue(detected, transcript)


class HotwordDetectorConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_detector_runs_two_final_checks_in_parallel(self) -> None:
        started = threading.Event()
        release = threading.Event()
        counter_lock = threading.Lock()
        started_count = 0
        detector = HotwordDetector(Settings())

        def blocking_detect(_audio):
            nonlocal started_count
            with counter_lock:
                started_count += 1
                if started_count == detector._parallelism:
                    started.set()
            release.wait(timeout=2)
            return False, ""

        detector._detect_sync = blocking_detect
        active = [
            asyncio.create_task(detector.detect(np.ones(160, dtype=np.float32)))
            for _ in range(detector._parallelism)
        ]
        await asyncio.to_thread(started.wait, 2)
        try:
            busy_result = await detector.try_detect(
                np.ones(160, dtype=np.float32)
            )
        finally:
            release.set()
            await asyncio.gather(*active)

        self.assertEqual(started_count, detector._parallelism)
        self.assertIsNone(busy_result)

    async def test_partial_check_yields_to_waiting_final_check(self) -> None:
        detector = HotwordDetector(Settings())
        detector._detect_sync = lambda _audio: (False, "")
        await detector._slots.acquire()
        await detector._slots.acquire()
        waiting = asyncio.create_task(
            detector.detect(np.ones(160, dtype=np.float32))
        )
        await asyncio.sleep(0)
        try:
            self.assertIsNone(
                await detector.try_detect(np.ones(160, dtype=np.float32))
            )
        finally:
            detector._slots.release()
            detector._slots.release()
            await waiting

    async def test_cancelled_requests_hold_slots_until_native_inference_finishes(self) -> None:
        started = threading.Event()
        release = threading.Event()
        counter_lock = threading.Lock()
        started_count = 0
        detector = HotwordDetector(Settings())

        def blocking_detect(_audio):
            nonlocal started_count
            with counter_lock:
                started_count += 1
                if started_count == detector._parallelism:
                    started.set()
            release.wait(timeout=2)
            return False, ""

        detector._detect_sync = blocking_detect
        active = [
            asyncio.create_task(detector.detect(np.ones(160, dtype=np.float32)))
            for _ in range(detector._parallelism)
        ]
        await asyncio.to_thread(started.wait, 2)
        for request in active:
            request.cancel()
        await asyncio.sleep(0)
        try:
            self.assertIsNone(
                await detector.try_detect(np.ones(160, dtype=np.float32))
            )
        finally:
            release.set()
            results = await asyncio.gather(*active, return_exceptions=True)

        self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))


class TextToSpeechTests(unittest.IsolatedAsyncioTestCase):
    def test_silero_text_is_split_only_at_sentence_boundaries(self) -> None:
        self.assertEqual(
            _silero_sentences("Первая фраза, с запятой. Вопрос? Ответ!"),
            ["Первая фраза, с запятой.", "Вопрос?", "Ответ!"],
        )

    async def test_discovers_piper_models_and_switches_silero_per_guild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            configured = models / "ru_RU-dmitri-medium.onnx"
            configured.touch()
            (models / "ru_RU-ruslan-medium.onnx").touch()
            tts = TextToSpeech(
                replace(
                    Settings(),
                    piper_model_path=str(configured),
                    tts_default_voice="",
                )
            )

            class FakeSilero:
                def apply_tts(self, **options):
                    self.options = options
                    return np.array([0.1, -0.1], dtype=np.float32)

            fake_silero = FakeSilero()
            tts._load_silero = lambda: fake_silero

            ids = {voice.id for voice in tts.voices()}
            self.assertIn("piper:ru_RU-dmitri-medium", ids)
            self.assertIn("piper:ru_RU-ruslan-medium", ids)
            self.assertIn("silero:xenia", ids)
            self.assertEqual(
                tts.selected_voice("other"), "piper:ru_RU-dmitri-medium"
            )

            selected = await tts.select_voice("guild", "silero:xenia")
            audio, sample_rate = await tts.synthesize("Привет?", "guild")

            self.assertEqual(selected.engine, "silero")
            self.assertEqual(tts.selected_voice("guild"), "silero:xenia")
            self.assertEqual(sample_rate, 48_000)
            np.testing.assert_array_equal(
                audio, np.array([0.1, -0.1], dtype=np.float32)
            )
            self.assertEqual(fake_silero.options["speaker"], "xenia")
            self.assertEqual(fake_silero.options["sample_rate"], 48_000)
            self.assertTrue(fake_silero.options["put_accent"])
            self.assertTrue(fake_silero.options["put_yo"])

    async def test_silero_adds_configured_silence_between_sentences(self) -> None:
        tts = TextToSpeech(
            replace(
                Settings(),
                tts_default_voice="silero:eugene",
                silero_sentence_silence_ms=100,
            )
        )

        class FakeSilero:
            def __init__(self):
                self.texts = []

            def apply_tts(self, **options):
                self.texts.append(options["text"])
                return np.array([0.25], dtype=np.float32)

        fake_silero = FakeSilero()
        tts._load_silero = lambda: fake_silero

        audio, sample_rate = await tts.synthesize("Первая. Вторая?", "guild")

        self.assertEqual(fake_silero.texts, ["Первая.", "Вторая?"])
        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(len(audio), 4_802)
        self.assertEqual(audio[0], 0.25)
        self.assertEqual(audio[-1], 0.25)

    async def test_silero_receives_pronounceable_cyrillic_instead_of_latin(self) -> None:
        tts = TextToSpeech(replace(Settings(), tts_default_voice="silero:eugene"))

        class FakeSilero:
            text = ""

            def apply_tts(self, **options):
                self.text = options["text"]
                return np.array([0.25], dtype=np.float32)

        fake_silero = FakeSilero()
        tts._load_silero = lambda: fake_silero

        await tts.synthesize("OpenAI API работает в Discord.", "guild")

        self.assertEqual(
            fake_silero.text,
            "оупен эй ай эй пи ай работает в дискорд.",
        )
        self.assertNotRegex(fake_silero.text, r"[A-Za-z]")

    async def test_rejects_unknown_voice(self) -> None:
        tts = TextToSpeech(Settings())
        with self.assertRaises(ValueError):
            await tts.select_voice("guild", "silero:unknown")

    def test_uses_configured_silero_voice_by_default(self) -> None:
        tts = TextToSpeech(
            replace(Settings(), tts_default_voice="silero:kseniya")
        )

        self.assertEqual(tts.selected_voice("guild"), "silero:kseniya")

    def test_rejects_unknown_default_voice_at_startup(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown TTS_DEFAULT_VOICE"):
            TextToSpeech(
                replace(Settings(), tts_default_voice="silero:unknown")
            )

    def test_suppresses_syntax_warning_from_packaged_silero_code(self) -> None:
        class FakeModel:
            def to(self, _device):
                return self

        class FakeImporter:
            def __init__(self, _path):
                pass

            def load_pickle(self, _package, _resource):
                warnings.warn("invalid escape sequence '\\^'", SyntaxWarning)
                return FakeModel()

        fake_torch = SimpleNamespace(
            package=SimpleNamespace(PackageImporter=FakeImporter)
        )
        tts = TextToSpeech(replace(Settings(), tts_default_voice=""))
        tts._download_silero_model = lambda: None

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tts._load_silero()

        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
