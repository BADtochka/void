import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from voice_core import app as app_module
from voice_core.services import ToolResult
from voice_core.turn_queue import PreparedTurn, TurnRequest
from voice_core.user_memory import UserMemoryStore


class _FakeSpeechToText:
    async def transcribe(self, _audio):
        return "ошибочный текст"


class _FakeImageSpeechToText:
    async def transcribe(self, _audio):
        return "что изображено"


class _FakeHotwordDetector:
    async def detect(self, _audio):
        return True, "Вомни."


class _FakeLanguageModel:
    async def reply(self, *_args, **_kwargs):
        return "Готовый ответ."


class _FakeTextToSpeech:
    async def synthesize(self, _text, _guild_id):
        return np.zeros(100, dtype=np.float32), 16_000


class _FakeUserMemory:
    def set(self, *_args):
        return None

    def get(self, *_args):
        return None

    def has_web_search_access(self, *_args):
        return False


class _FakeSelectableTextToSpeech:
    def selected_voice(self, guild_id):
        return f"selected:{guild_id}"

    def voices(self):
        return [
            SimpleNamespace(
                as_dict=lambda: {
                    "id": "silero:xenia",
                    "label": "Silero · xenia",
                    "engine": "silero",
                }
            )
        ]

    def selected_effect(self, guild_id):
        return f"effect:{guild_id}"

    def effects(self):
        return [SimpleNamespace(as_dict=lambda: {"id": "robotic", "label": "Роботизированный"})]

    async def select_voice(self, _guild_id, voice_id):
        if voice_id != "silero:xenia":
            raise ValueError("unknown TTS voice")
        return SimpleNamespace(
            id=voice_id,
            label="Silero · xenia",
            engine="silero",
            as_dict=lambda: {
                "id": voice_id,
                "label": "Silero · xenia",
                "engine": "silero",
            },
        )

    async def select_effect(self, _guild_id, effect_id):
        if effect_id != "robotic":
            raise ValueError("unknown TTS effect")
        return SimpleNamespace(
            id=effect_id,
            label="Роботизированный",
            as_dict=lambda: {"id": effect_id, "label": "Роботизированный"},
        )

class VoiceTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_stop_phrases_bypass_generation_and_stop_guild(self) -> None:
        class StopSpeechToText:
            def __init__(self, transcript):
                self.transcript = transcript

            async def transcribe(self, _audio):
                return self.transcript

        class FakeRecognitionQueue:
            async def cancel_guild(self, guild_id):
                self.cancelled = guild_id
                return 2

        class FakeGenerationQueue:
            async def cancel_guild(self, guild_id):
                self.cancelled = guild_id

        for transcript in ("Омни, хватит.", "Омни, стоп."):
            with self.subTest(transcript=transcript):
                guild_id = f"stop-phrase-{transcript}"
                request = TurnRequest(b"", guild_id, "channel", "user", "User")
                recognition = FakeRecognitionQueue()
                generation = FakeGenerationQueue()
                events = app_module.event_bus.subscribe()
                app_module.store.open_followup(guild_id, "user")
                try:
                    with (
                        patch.object(app_module, "stt", StopSpeechToText(transcript)),
                        patch.object(app_module, "recognition_queue", recognition),
                        patch.object(app_module, "generation_queue", generation),
                    ):
                        result = await app_module.recognize_turn(request)

                    self.assertIsNone(result)
                    self.assertEqual(recognition.cancelled, guild_id)
                    self.assertEqual(generation.cancelled, guild_id)
                    event = events.get_nowait()
                    self.assertEqual(event.type, "followup_stopped")
                    self.assertFalse(
                        app_module.store.followup_active(guild_id, "user")
                    )
                finally:
                    app_module.event_bus.unsubscribe(events)
                    app_module.followups.stop_guild(guild_id)
                    app_module.store.reset(guild_id)

    def test_chat_delivery_request_is_detected(self) -> None:
        self.assertTrue(app_module.requested_chat_delivery("Омни, отправь ответ в чат"))
        self.assertTrue(app_module.requested_chat_delivery("Продублируй это текстом"))
        self.assertFalse(app_module.requested_chat_delivery("Отправь запрос в LM Studio"))

    async def test_chat_tool_queues_selection_full_and_previous_response(self) -> None:
        guild_id = "chat-tool-test"
        request = TurnRequest(b"", guild_id, "channel", "user", "User")
        deliveries: list[str | None] = []
        app_module.store.append_turn(guild_id, "вопрос", "Предыдущий ответ")
        try:
            await app_module.execute_assistant_tool(
                request,
                "продублируй часть в чат",
                "send_message_to_chat",
                {"scope": "selection", "content": "Нужная часть"},
                deliveries,
            )
            await app_module.execute_assistant_tool(
                request,
                "отправь весь ответ в чат",
                "send_message_to_chat",
                {"scope": "full_response"},
                deliveries,
            )
            await app_module.execute_assistant_tool(
                request,
                "отправь предыдущий ответ в чат",
                "send_message_to_chat",
                {"scope": "previous_response"},
                deliveries,
            )

            self.assertEqual(
                deliveries,
                ["Нужная часть", None, "Предыдущий ответ"],
            )
        finally:
            app_module.store.reset(guild_id)

    async def test_full_response_is_published_to_discord_after_generation(self) -> None:
        class ChatToolLanguageModel:
            async def reply(self, _history, _prompt, _tools, tool_handler, **_kwargs):
                await tool_handler(
                    "send_message_to_chat", {"scope": "full_response"}
                )
                return "Полный ответ для Discord."

        guild_id = "chat-event-test"
        request = TurnRequest(b"", guild_id, "channel", "user", "User")
        events = app_module.event_bus.subscribe()
        try:
            with (
                patch.object(app_module, "llm", ChatToolLanguageModel()),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
                patch.object(app_module, "user_memory", _FakeUserMemory()),
            ):
                await app_module.generate_turn(
                    PreparedTurn(request, "отправь ответ", "отправь ответ", True)
                )

            published = []
            while not events.empty():
                published.append(events.get_nowait())
            chat_event = next(event for event in published if event.type == "chat_message")
            self.assertEqual(chat_event.content, "Полный ответ для Discord.")
            self.assertEqual(chat_event.channelId, "channel")
        finally:
            app_module.event_bus.unsubscribe(events)
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            await asyncio.sleep(0)

    async def test_preferred_name_request_is_forced_and_persisted(self) -> None:
        class MemoryToolLanguageModel:
            required_tool_name = None

            async def reply(
                self, _history, _prompt, _tools, tool_handler, **kwargs
            ):
                self.required_tool_name = kwargs.get("required_tool_name")
                await tool_handler(
                    "remember_preferred_name", {"preferred_name": "не доверяем модели"}
                )
                return "Буду называть тебя Рыжий."

        guild_id = "preferred-name-tool-test"
        request = TurnRequest(b"", guild_id, "channel", "user", "User")
        model = MemoryToolLanguageModel()
        with tempfile.TemporaryDirectory() as directory:
            memory = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            memory.prepare()
            try:
                with (
                    patch.object(app_module, "llm", model),
                    patch.object(app_module, "tts", _FakeTextToSpeech()),
                    patch.object(app_module, "user_memory", memory),
                ):
                    await app_module.generate_turn(
                        PreparedTurn(
                            request,
                            "мобой о мне ты меня можешь называть рыжий",
                            "ты меня можешь называть рыжий",
                            True,
                        )
                    )

                self.assertEqual(
                    model.required_tool_name, "remember_preferred_name"
                )
                self.assertEqual(memory.get(guild_id, "user", "preferred_name"), "рыжий")
            finally:
                app_module.followups.stop_guild(guild_id)
                app_module.store.reset(guild_id)
                await asyncio.sleep(0)

    async def test_admin_web_search_request_receives_search_tool(self) -> None:
        class CapturingLanguageModel:
            tools = []
            required_tool_name = None

            async def reply(self, _history, _prompt, tools, _handler, **kwargs):
                self.tools = tools
                self.required_tool_name = kwargs.get("required_tool_name")
                return "Результаты поиска."

        guild_id = "admin-web-search-test"
        model = CapturingLanguageModel()
        request = TurnRequest(
            b"", guild_id, "channel", "admin", "Admin", user_is_admin=True
        )
        try:
            with (
                patch.object(app_module, "llm", model),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
                patch.object(app_module, "user_memory", _FakeUserMemory()),
            ):
                await app_module.generate_turn(
                    PreparedTurn(
                        request,
                        "поищи в сети свежие новости",
                        "поищи в сети свежие новости",
                        True,
                    )
                )

            tool_names = {tool["function"]["name"] for tool in model.tools}
            self.assertIn("search_web", tool_names)
            self.assertEqual(model.required_tool_name, "search_web")
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            await asyncio.sleep(0)

    async def test_non_whitelisted_web_search_skips_language_model(self) -> None:
        class ForbiddenLanguageModel:
            async def reply(self, *_args, **_kwargs):
                raise AssertionError("LM Studio must not receive a denied web search")

        guild_id = "denied-web-search-test"
        request = TurnRequest(b"", guild_id, "channel", "user", "User")
        try:
            with (
                patch.object(app_module, "llm", ForbiddenLanguageModel()),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
                patch.object(app_module, "user_memory", _FakeUserMemory()),
            ):
                reply = await app_module.generate_turn(
                    PreparedTurn(
                        request,
                        "найди в интернете новости",
                        "найди в интернете новости",
                        True,
                    )
                )

            self.assertIsNotNone(reply)
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            await asyncio.sleep(0)

    async def test_web_search_tool_enforces_persistent_whitelist(self) -> None:
        class FakePublicInformation:
            async def execute(self, tool_name, arguments):
                self.call = (tool_name, arguments)
                return '{"found":true}'

        with tempfile.TemporaryDirectory() as directory:
            memory = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            memory.prepare()
            public_information = FakePublicInformation()
            request = TurnRequest(b"", "guild", "channel", "user", "User")
            with (
                patch.object(app_module, "user_memory", memory),
                patch.object(app_module, "public_information", public_information),
            ):
                with self.assertRaises(PermissionError):
                    await app_module.execute_assistant_tool(
                        request,
                        "поищи в сети",
                        "search_web",
                        {"query": "тест"},
                    )
                memory.grant_web_search_access("guild", "user", "User")
                result = await app_module.execute_assistant_tool(
                    request,
                    "поищи в сети",
                    "search_web",
                    {"query": "тест"},
                )

            self.assertEqual(result, '{"found":true}')
            self.assertEqual(public_information.call, ("search_web", {"query": "тест"}))

    def test_end_conversation_tool_stays_compact(self) -> None:
        serialized_tool = json.dumps(
            app_module.END_CONVERSATION_TOOL, ensure_ascii=False
        )

        self.assertIn("end_conversation", serialized_tool)
        self.assertIn("Не проси специальное слово", serialized_tool)
        self.assertNotIn("стопэ", serialized_tool)
        self.assertLess(len(serialized_tool), 700)

    async def test_parallel_users_receive_distinct_identity_context(self) -> None:
        class CapturingLanguageModel:
            def __init__(self):
                self.calls = []

            async def reply(self, history, prompt, *_args, **_kwargs):
                self.calls.append((history, prompt))
                return "Готовый ответ."

        guild_id = "speaker-context-test"
        model = CapturingLanguageModel()
        first_request = TurnRequest(
            b"", guild_id, "channel", "user-1", "Одинаковый ник"
        )
        second_request = TurnRequest(
            b"", guild_id, "channel", "user-2", "Одинаковый ник"
        )
        try:
            with (
                patch.object(app_module, "llm", model),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
                patch.object(app_module, "user_memory", _FakeUserMemory()),
            ):
                await app_module.generate_turn(
                    PreparedTurn(first_request, "первый", "первый", True)
                )
                await app_module.generate_turn(
                    PreparedTurn(second_request, "второй", "второй", True)
                )

            first_history, first_prompt = model.calls[0]
            second_history, second_prompt = model.calls[1]
            self.assertEqual(first_history, [])
            self.assertIn('"identity_key": "speaker_1"', first_prompt)
            self.assertNotIn('"identity_key": "speaker_2"', first_prompt)
            self.assertIn('"identity_key": "speaker_1"', second_prompt)
            self.assertIn('"identity_key": "speaker_2"', second_prompt)
            self.assertIn("current_identity=speaker_2", second_prompt)
            self.assertIn("author_identity=speaker_1", second_history[0]["content"])
            self.assertIn("reply_to_identity=speaker_1", second_history[1]["content"])
            self.assertIn("current_identity", second_prompt)
            self.assertIn("не переноси на других", second_prompt.casefold())

        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            await asyncio.sleep(0)

    async def test_end_conversation_tool_runs_complete_stop_flow(self) -> None:
        class FakeRecognitionQueue:
            async def cancel_guild(self, guild_id):
                self.cancelled = guild_id
                return 2

        class FakeGenerationQueue:
            async def cancel_pending_guild(self, guild_id):
                self.cancelled = guild_id
                return 3

        guild_id = "tool-stop-test"
        request = TurnRequest(b"", guild_id, "channel", "user", "User")
        recognition = FakeRecognitionQueue()
        generation = FakeGenerationQueue()
        events = app_module.event_bus.subscribe()
        app_module.store.open_followup(guild_id, "user")
        app_module.followups.open(guild_id, "channel", "user", 30)
        try:
            with (
                patch.object(app_module, "recognition_queue", recognition),
                patch.object(app_module, "generation_queue", generation),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
            ):
                result = await app_module.execute_assistant_tool(
                    request,
                    "давай на этом закончим",
                    "end_conversation",
                    {},
                )

            self.assertIsInstance(result, ToolResult)
            self.assertTrue(result.terminate)
            self.assertIsNone(result.response)
            self.assertFalse(app_module.store.followup_active(guild_id, "user"))
            self.assertEqual(recognition.cancelled, guild_id)
            self.assertEqual(generation.cancelled, guild_id)
            event = events.get_nowait()
            self.assertEqual(event.type, "followup_stopped")
            self.assertEqual(event.userId, "user")
            self.assertEqual(event.content, app_module.END_CONVERSATION_FAREWELL)
            self.assertTrue(event.audioBase64)
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            app_module.event_bus.unsubscribe(events)
            await asyncio.sleep(0)

    async def test_final_hotword_check_waits_for_detector(self) -> None:
        class Detector:
            async def detect(self, audio):
                self.audio_samples = len(audio)
                return True, "Омни, ответь"

            async def try_detect(self, _audio):
                raise AssertionError("Final check must wait for the detector")

        detector = Detector()
        audio = bytes(48_000 * 2 * 2 * 4)
        with patch.object(app_module, "hotword_detector", detector):
            result = await app_module.detect_hotword(audio, final=True)

        self.assertTrue(result["detected"])
        self.assertFalse(result["busy"])
        self.assertEqual(result["transcript"], "Омни, ответь")
        self.assertEqual(detector.audio_samples, 16_000 * 4)

    async def test_partial_hotword_check_remains_non_blocking(self) -> None:
        class BusyDetector:
            async def detect(self, _audio):
                raise AssertionError("Partial check must not wait for the detector")

            async def try_detect(self, audio):
                self.audio_samples = len(audio)
                return None

        detector = BusyDetector()
        audio = bytes(48_000 * 2 * 2 * 4)
        with patch.object(app_module, "hotword_detector", detector):
            result = await app_module.detect_hotword(audio)

        self.assertFalse(result["detected"])
        self.assertTrue(result["busy"])
        self.assertEqual(detector.audio_samples, 16_000 * 3)

    async def test_tts_selection_api_lists_and_switches_voice(self) -> None:
        fake_tts = _FakeSelectableTextToSpeech()
        with patch.object(app_module, "tts", fake_tts):
            listed = await app_module.get_tts_selection("guild")
            selected = await app_module.set_tts_selection(
                "guild", app_module.TtsSelection(voiceId="silero:xenia")
            )
            selected_effect = await app_module.set_tts_effect(
                "guild", app_module.TtsEffectSelection(effectId="robotic")
            )

        self.assertEqual(listed["selected"], "selected:guild")
        self.assertEqual(listed["voices"][0]["id"], "silero:xenia")
        self.assertEqual(listed["selectedEffect"], "effect:guild")
        self.assertEqual(listed["effects"][0]["id"], "robotic")
        self.assertEqual(selected["id"], "silero:xenia")
        self.assertEqual(selected_effect["id"], "robotic")

    async def test_web_search_access_endpoints_require_admin_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            memory.prepare()
            with patch.object(app_module, "user_memory", memory):
                with self.assertRaises(app_module.HTTPException) as denied:
                    await app_module.grant_web_search_access(
                        "guild",
                        "user",
                        app_module.WebSearchAccessGrant(displayName="User"),
                        x_requester_is_admin=False,
                    )
                await app_module.grant_web_search_access(
                    "guild",
                    "user",
                    app_module.WebSearchAccessGrant(displayName="User"),
                    x_requester_is_admin=True,
                )
                listed = await app_module.list_web_search_access(
                    "guild", x_requester_is_admin=True
                )
                await app_module.revoke_web_search_access(
                    "guild", "user", x_requester_is_admin=True
                )

            self.assertEqual(denied.exception.status_code, 403)
            self.assertEqual(listed["users"], [{"userId": "user", "displayName": "User"}])
            self.assertFalse(memory.has_web_search_access("guild", "user"))

    async def test_turn_endpoint_splits_pcm_and_image(self) -> None:
        captured: list[TurnRequest] = []

        class FakeRecognitionQueue:
            async def submit(self, request):
                captured.append(request)
                return None

        pcm = bytes(48_000 * 2 * 2)
        image = b"png-image"
        with patch.object(app_module, "recognition_queue", FakeRecognitionQueue()):
            response = await app_module.enqueue_turn(
                body=pcm + image,
                x_guild_id="guild",
                x_channel_id="channel",
                x_user_id="user",
                x_display_name="User",
                x_audio_age_ms=0,
                x_early_hotword_detected=False,
                x_audio_byte_length=len(pcm),
                x_image_content_type="image/png",
            )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(captured[0].audio, pcm)
        self.assertEqual(captured[0].image_data, image)
        self.assertEqual(captured[0].image_content_type, "image/png")

    async def test_image_turn_bypasses_wake_word_and_keeps_image(self) -> None:
        guild_id = "image-turn-test"
        events = app_module.event_bus.subscribe()
        request = TurnRequest(
            audio=bytes(48_000 * 2 * 2),
            guild_id=guild_id,
            channel_id="channel",
            user_id="user",
            display_name="User",
            image_content_type="image/png",
            image_data=b"png-data",
        )
        try:
            with patch.object(app_module, "stt", _FakeImageSpeechToText()):
                result = await app_module.recognize_turn(request)

            self.assertIsNotNone(result)
            self.assertTrue(result.direct_wake)
            self.assertEqual(result.request.image_data, b"png-data")
            self.assertIn("что изображено", result.accepted_text)
            self.assertEqual(events.get_nowait().type, "request_recognized")
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            app_module.event_bus.unsubscribe(events)
            await asyncio.sleep(0)

    async def test_image_is_sent_only_with_its_own_turn(self) -> None:
        class CapturingImageLanguageModel:
            def __init__(self):
                self.images = []

            async def reply(self, _history, _prompt, *_args, **kwargs):
                self.images.append(
                    (kwargs.get("image_content_type"), kwargs.get("image_data"))
                )
                return "Готовый ответ."

        guild_id = "one-shot-image-test"
        model = CapturingImageLanguageModel()
        image_request = TurnRequest(
            b"",
            guild_id,
            "channel",
            "user",
            "User",
            image_content_type="image/png",
            image_data=b"png-data",
        )
        followup_request = TurnRequest(
            b"", guild_id, "channel", "user", "User"
        )
        try:
            with (
                patch.object(app_module, "llm", model),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
                patch.object(app_module, "user_memory", _FakeUserMemory()),
            ):
                await app_module.generate_turn(
                    PreparedTurn(image_request, "что здесь", "что здесь", True)
                )
                await app_module.generate_turn(
                    PreparedTurn(followup_request, "продолжим", "продолжим", False)
                )

            self.assertEqual(
                model.images,
                [("image/png", b"png-data"), (None, None)],
            )
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            await asyncio.sleep(0)

    async def test_followup_countdown_starts_only_after_playback_confirmation(self) -> None:
        guild_id = "playback-followup-test"
        events = app_module.event_bus.subscribe()
        request = TurnRequest(b"", guild_id, "channel", "user", "User")
        turn = PreparedTurn(request, "продолжение", "продолжение", False)
        try:
            with (
                patch.object(app_module, "llm", _FakeLanguageModel()),
                patch.object(app_module, "tts", _FakeTextToSpeech()),
                patch.object(app_module, "user_memory", _FakeUserMemory()),
            ):
                result = await app_module.generate_turn(turn)

            self.assertIsNotNone(result)
            self.assertEqual(events.get_nowait().type, "generation_started")
            self.assertTrue(events.empty())
            self.assertFalse(app_module.store.followup_active(guild_id, "user"))

            response = await app_module.start_followup_after_playback(
                guild_id, "user", "channel"
            )

            self.assertEqual(response.status_code, 204)
            reopened = events.get_nowait()
            self.assertEqual(reopened.type, "followup_reopened")
            self.assertEqual(reopened.userId, "user")
            self.assertEqual(
                reopened.followupMs,
                round(app_module.settings.followup_seconds * 1000),
            )
            self.assertTrue(app_module.store.followup_active(guild_id, "user"))
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.store.reset(guild_id)
            app_module.event_bus.unsubscribe(events)
            await asyncio.sleep(0)

    async def test_early_hotword_without_continuation_does_not_generate(self) -> None:
        guild_id = "hotword-only-test"
        events = app_module.event_bus.subscribe()
        request = TurnRequest(
            audio=bytes(48_000 * 2 * 2),
            guild_id=guild_id,
            channel_id="channel",
            user_id="user",
            display_name="User",
            early_hotword_detected=True,
        )
        try:
            with (
                patch.object(app_module, "stt", _FakeSpeechToText()),
                patch.object(app_module, "hotword_detector", _FakeHotwordDetector()),
            ):
                result = await app_module.recognize_turn(request)

            event = events.get_nowait()
            self.assertIsNone(result)
            self.assertEqual(event.type, "request_recognized")
            self.assertEqual(event.userId, "user")
            self.assertTrue(event.awaitingContent)
            self.assertTrue(
                app_module.store.followup_active(guild_id, "user", now=10**12)
            )
        finally:
            app_module.followups.stop_guild(guild_id)
            app_module.event_bus.unsubscribe(events)
            await asyncio.sleep(0)

    async def test_name_lookup_tool_uses_current_and_fuzzy_other_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            memory.prepare()
            memory.set("guild", "current", "preferred_name", "Кэп")
            memory.set("guild", "current", "discord_display_name", "captain")
            memory.set("guild", "other", "preferred_name", "Пупсик")
            memory.set("guild", "other", "discord_display_name", ".formallybad")
            request = TurnRequest(
                audio=b"",
                guild_id="guild",
                channel_id="channel",
                user_id="current",
                display_name="Captain",
            )

            with patch.object(app_module, "user_memory", memory):
                current = json.loads(
                    await app_module.execute_user_memory_tool(
                        request,
                        "как меня зовут",
                        "lookup_user_name",
                        {"subject": "current_user"},
                    )
                )
                overridden = json.loads(
                    await app_module.execute_user_memory_tool(
                        request,
                        "как меня зовут",
                        "lookup_user_name",
                        {"subject": "Пупсик"},
                    )
                )
                other = json.loads(
                    await app_module.execute_user_memory_tool(
                        request,
                        "как зовут формали бэда",
                        "lookup_user_name",
                        {"subject": "дела"},
                    )
                )
                missing = json.loads(
                    await app_module.execute_user_memory_tool(
                        request,
                        "как зовут xyzzy",
                        "lookup_user_name",
                        {"subject": "xyzzy"},
                    )
                )

            self.assertEqual(current["preferred_name"], "Кэп")
            self.assertEqual(current["scope"], "current_user")
            self.assertEqual(current["answer_name"], "Кэп")
            self.assertEqual(overridden["scope"], "current_user")
            self.assertEqual(overridden["answer_name"], "Кэп")
            self.assertEqual(other["preferred_name"], "Пупсик")
            self.assertEqual(other["scope"], "other_user")
            self.assertEqual(other["answer_name"], "Пупсик")
            self.assertIn("третьем лице", other["response_instruction"])
            self.assertNotEqual(other["answer_name"], "Кэп")
            self.assertGreater(other["confidence"], 0.7)
            self.assertFalse(missing["found"])

    async def test_assistant_name_lookup_returns_one_terminal_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            memory.prepare()
            memory.set("guild", "current", "preferred_name", "Валерий")
            request = TurnRequest(
                b"", "guild", "channel", "current", ".formallybad"
            )

            with patch.object(app_module, "user_memory", memory):
                result = await app_module.execute_assistant_tool(
                    request,
                    "как меня зовут",
                    "lookup_user_name",
                    {"subject": "current_user"},
                )

            self.assertIsInstance(result, ToolResult)
            self.assertTrue(result.terminate)
            self.assertEqual(result.response, "Тебя зовут Валерий.")
            self.assertNotIn("formallybad", result.response)


if __name__ == "__main__":
    unittest.main()
