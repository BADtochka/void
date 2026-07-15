from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv()

from fastapi import Body, FastAPI, Header, HTTPException, Response, WebSocket, WebSocketDisconnect  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .audio import DISCORD_CHANNELS, DISCORD_SAMPLE_RATE, discord_pcm_to_whisper, float_mono_to_discord_pcm, limit_discord_pcm  # noqa: E402
from .config import Settings  # noqa: E402
from .dialogue import ConversationStore, prepare_for_speech  # noqa: E402
from .event_bus import FollowupTracker, VoiceEvent, VoiceEventBus  # noqa: E402
from .public_info import (  # noqa: E402
    PUBLIC_INFO_TOOLS,
    PublicInformationService,
    requested_web_search,
)
from .services import HotwordDetector, LanguageModel, SpeechToText, TextToSpeech, ToolResult  # noqa: E402
from .speech_normalization import russianize_address_name  # noqa: E402
from .tooling import (  # noqa: E402
    END_CONVERSATION_FAREWELL,
    END_CONVERSATION_TOOL,
    SEND_MESSAGE_TO_CHAT_TOOL,
    WEB_SEARCH_DENIED_SPEECH,
    build_turn_prompt,
    requested_chat_delivery,
    required_tool_for_turn,
    select_assistant_tools,
)
from .turn_queue import (  # noqa: E402
    GenerationQueue,
    PreparedTurn,
    RecognitionQueue,
    TurnRequest,
)
from .user_memory import (  # noqa: E402
    USER_MEMORY_TOOLS,
    UserMemoryStore,
    name_lookup_is_about_current_user,
    name_lookup_other_subject,
    requested_preferred_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-core")

settings = Settings()
store = ConversationStore(
    settings.wake_word,
    settings.followup_seconds,
    settings.max_history_turns,
    settings.wake_word_aliases,
    settings.stop_phrases,
    settings.followup_min_chars,
    settings.followup_ignore_phrases,
    settings.dialogue_cooldown_seconds,
)
stt = SpeechToText(settings)
hotword_detector = HotwordDetector(settings)
llm = LanguageModel(settings)
tts = TextToSpeech(settings)
user_memory = UserMemoryStore(settings.user_memory_db_path)
public_information = PublicInformationService()
event_bus = VoiceEventBus()


async def expire_dialogue(guild_id: str, channel_id: str, user_id: str) -> None:
    store.finish(guild_id, user_id)
    cancelled_recognition = await recognition_queue.cancel_speaker(guild_id, user_id)
    cancelled_generation = await generation_queue.cancel_pending_speaker(
        guild_id, user_id
    )
    logger.info(
        "Conversation expired guild_id=%s channel_id=%s user_id=%s "
        "cancelled_recognition=%s cancelled_generation=%s cooldown_seconds=%.1f",
        guild_id,
        channel_id,
        user_id,
        cancelled_recognition,
        cancelled_generation,
        settings.dialogue_cooldown_seconds,
    )


followups = FollowupTracker(event_bus, expire_dialogue)


class UserDirectoryEntry(BaseModel):
    userId: str = Field(min_length=1, max_length=32)
    displayName: str = Field(min_length=1, max_length=100)


class TtsSelection(BaseModel):
    voiceId: str = Field(min_length=1, max_length=100)


class TtsEffectSelection(BaseModel):
    effectId: str = Field(min_length=1, max_length=32)


class WebSearchAccessGrant(BaseModel):
    displayName: str = Field(min_length=1, max_length=100)


ASSISTANT_TOOLS = [
    *USER_MEMORY_TOOLS,
    *PUBLIC_INFO_TOOLS,
    SEND_MESSAGE_TO_CHAT_TOOL,
    END_CONVERSATION_TOOL,
]


async def execute_user_memory_tool(
    request: TurnRequest,
    transcript: str,
    tool_name: str,
    _arguments: dict[str, object],
) -> str:
    if tool_name == "remember_preferred_name":
        preferred_name = requested_preferred_name(transcript)
        if preferred_name is None:
            raw = _arguments.get("preferred_name")
            if raw is not None and str(raw).strip():
                preferred_name = str(raw).strip()
        if preferred_name is None:
            raise ValueError("preferred_name is required")
        user_memory.set(
            request.guild_id, request.user_id, "preferred_name", preferred_name
        )
        logger.info(
            "User memory saved guild_id=%s user_id=%s key=preferred_name",
            request.guild_id,
            request.user_id,
        )
        return json.dumps(
            {"ok": True, "preferred_name": preferred_name}, ensure_ascii=False
        )

    if tool_name == "forget_preferred_name":
        deleted = user_memory.delete(
            request.guild_id, request.user_id, "preferred_name"
        )
        logger.info(
            "User memory deleted guild_id=%s user_id=%s key=preferred_name deleted=%s",
            request.guild_id,
            request.user_id,
            deleted,
        )
        return json.dumps({"ok": True, "deleted": deleted}, ensure_ascii=False)

    if tool_name == "lookup_user_name":
        raw_subject = _arguments.get("subject", _arguments.get("query"))
        subject = str(raw_subject).strip() if raw_subject is not None else ""
        if name_lookup_is_about_current_user(transcript):
            subject = "current_user"
        else:
            overheard = name_lookup_other_subject(transcript)
            if overheard:
                subject = overheard
        if subject.casefold() in {
            "current_user",
            "я",
            "меня",
            "мне",
            "моё имя",
            "мое имя",
        }:
            preferred_name = user_memory.get(
                request.guild_id, request.user_id, "preferred_name"
            )
            display_name = user_memory.get(
                request.guild_id, request.user_id, "discord_display_name"
            ) or request.display_name
            logger.info(
                "Current user name lookup guild_id=%s user_id=%s found_preferred=%s",
                request.guild_id,
                request.user_id,
                preferred_name is not None,
            )
            return json.dumps(
                {
                    "found": True,
                    "scope": "current_user",
                    "answer_name": preferred_name or display_name,
                    "preferred_name": preferred_name,
                    "discord_display_name": display_name,
                    "response_instruction": (
                        "Ответь текущему говорящему во втором лице: 'тебя зовут ...'."
                    ),
                },
                ensure_ascii=False,
            )

        if not subject:
            raise ValueError(
                "subject is required: use current_user only for the current speaker, "
                "otherwise pass the other participant's name"
            )

        match = user_memory.find_best_name(
            request.guild_id, subject, exclude_user_id=request.user_id
        )
        logger.info(
            "Fuzzy user name lookup guild_id=%s query=%r found=%s confidence=%.3f",
            request.guild_id,
            subject,
            match is not None,
            match.confidence if match else 0.0,
        )
        if match is None:
            return json.dumps(
                {
                    "found": False,
                    "scope": "other_user",
                    "query": subject,
                    "response_instruction": (
                        "Скажи, что имя другого участника не найдено; не говори о текущем пользователе."
                    ),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "found": True,
                "scope": "other_user",
                "query": subject,
                "answer_name": match.preferred_name or match.display_name,
                "preferred_name": match.preferred_name,
                "discord_display_name": match.display_name,
                "matched_by": match.matched_value,
                "confidence": round(match.confidence, 3),
                "response_instruction": (
                    "Ответь о другом участнике в третьем лице: 'его/этого пользователя зовут ...'. "
                    "Не говори 'тебя зовут' и не приписывай это имя текущему говорящему."
                ),
            },
            ensure_ascii=False,
        )

    raise ValueError(f"unknown tool: {tool_name}")


async def stop_conversation(
    request: TurnRequest,
    *,
    source: str,
    interrupt_active_generation: bool,
    farewell: str | None = None,
) -> tuple[int, int]:
    store.finish(request.guild_id)
    cancelled_recognition = await recognition_queue.cancel_guild(request.guild_id)
    if interrupt_active_generation:
        await generation_queue.cancel_guild(request.guild_id)
        cancelled_generation = 0
    else:
        cancelled_generation = await generation_queue.cancel_pending_guild(
            request.guild_id
        )
    followups.stop_guild(request.guild_id)
    spoken_farewell = prepare_for_speech(farewell or "", limit=120) if farewell else ""
    farewell_audio: str | None = None
    if spoken_farewell:
        try:
            speech, sample_rate = await tts.synthesize(spoken_farewell, request.guild_id)
            farewell_audio = base64.b64encode(
                float_mono_to_discord_pcm(speech, sample_rate)
            ).decode("ascii")
        except Exception:
            logger.exception(
                "Farewell speech synthesis failed guild_id=%s text=%r",
                request.guild_id,
                spoken_farewell,
            )
            spoken_farewell = ""
    event_bus.publish(
        VoiceEvent(
            "followup_stopped",
            request.guild_id,
            request.channel_id,
            request.user_id,
            content=spoken_farewell or None,
            audioBase64=farewell_audio,
        )
    )
    logger.info(
        "Conversation stopped source=%s guild_id=%s user_id=%s display_name=%s cancelled_recognition=%s cancelled_generation=%s farewell=%r",
        source,
        request.guild_id,
        request.user_id,
        request.display_name,
        cancelled_recognition,
        cancelled_generation,
        spoken_farewell or None,
    )
    return cancelled_recognition, cancelled_generation


async def execute_assistant_tool(
    request: TurnRequest,
    transcript: str,
    tool_name: str,
    arguments: dict[str, object],
    chat_deliveries: list[str | None] | None = None,
) -> str | ToolResult:
    if tool_name == "end_conversation":
        raw_farewell = arguments.get("farewell")
        farewell = (
            prepare_for_speech(str(raw_farewell), limit=120)
            if raw_farewell is not None and str(raw_farewell).strip()
            else END_CONVERSATION_FAREWELL
        )
        cancelled_recognition, cancelled_generation = await stop_conversation(
            request,
            source="llm_tool",
            interrupt_active_generation=False,
            farewell=farewell,
        )
        return ToolResult(
            json.dumps(
                {
                    "ok": True,
                    "conversation_ended": True,
                    "cancelled_recognition": cancelled_recognition,
                    "cancelled_generation": cancelled_generation,
                    "farewell": farewell,
                },
                ensure_ascii=False,
            ),
            terminate=True,
        )
    if tool_name == "send_message_to_chat":
        if chat_deliveries is None:
            raise RuntimeError("chat delivery context is unavailable")
        scope = str(arguments.get("scope") or "").strip()
        if scope == "full_response":
            chat_deliveries.append(None)
        elif scope == "selection":
            content = prepare_for_speech(str(arguments.get("content") or ""), limit=1_900)
            if not content:
                raise ValueError("content is required for scope=selection")
            chat_deliveries.append(content)
        elif scope == "previous_response":
            previous = store.last_assistant_message(
                request.guild_id, request.user_id
            )
            if not previous:
                raise ValueError("there is no previous assistant response")
            chat_deliveries.append(previous)
        else:
            raise ValueError(f"unknown chat delivery scope: {scope}")
        response_instruction = (
            "Теперь дай полный содержательный ответ на запрос пользователя. Не ограничивайся "
            "подтверждением: именно этот следующий ответ backend продублирует в чат."
            if scope == "full_response"
            else "Кратко подтверди отправку и продолжи обычный голосовой диалог."
        )
        return json.dumps(
            {
                "ok": True,
                "queued": True,
                "scope": scope,
                "response_instruction": response_instruction,
            },
            ensure_ascii=False,
        )
    if tool_name == "search_web":
        if not (
            request.user_is_admin
            or user_memory.has_web_search_access(request.guild_id, request.user_id)
        ):
            logger.info(
                "Web search denied via tool guild_id=%s user_id=%s",
                request.guild_id,
                request.user_id,
            )
            return ToolResult(
                json.dumps(
                    {
                        "ok": False,
                        "denied": True,
                        "error": "web search is not allowed for the current user",
                    },
                    ensure_ascii=False,
                ),
                terminate=True,
                response=WEB_SEARCH_DENIED_SPEECH,
            )
        return await public_information.execute(tool_name, arguments)
    if tool_name in {"get_current_weather", "lookup_topic", "get_random_joke"}:
        return await public_information.execute(tool_name, arguments)
    result = await execute_user_memory_tool(
        request, transcript, tool_name, arguments
    )
    if tool_name != "lookup_user_name":
        return result

    lookup = json.loads(result)
    if lookup.get("found") and lookup.get("scope") == "current_user":
        response = f"Тебя зовут {lookup['answer_name']}."
    elif lookup.get("found"):
        response = f"Этого пользователя зовут {lookup['answer_name']}."
    else:
        response = "Не удалось найти имя этого пользователя."
    return ToolResult(result, terminate=True, response=response)


async def recognize_turn(request: TurnRequest) -> PreparedTurn | None:
    whisper_audio = discord_pcm_to_whisper(request.audio)
    transcript = await stt.transcribe(whisper_audio)
    logger.info("Heard %s (%s): %s", request.display_name, request.user_id, transcript)

    if store.stop_if_requested(request.guild_id, transcript):
        await stop_conversation(
            request,
            source="local_phrase",
            interrupt_active_generation=True,
        )
        return None

    force_wake = request.early_hotword_detected or bool(request.image_data)
    acceptance_transcript = transcript
    if request.early_hotword_detected and not request.image_data and store.wake_remainder(transcript) is None:
        detected, final_hotword_transcript = await hotword_detector.detect(whisper_audio)
        final_remainder = store.wake_remainder(final_hotword_transcript)
        logger.info(
            "Final hotword verification user_id=%s detected=%s transcript=%r has_continuation=%s",
            request.user_id,
            detected,
            final_hotword_transcript,
            bool(final_remainder),
        )
        if detected and not final_remainder:
            acceptance_transcript = final_hotword_transcript
            force_wake = False

    now = asyncio.get_running_loop().time()
    accepted = store.accept_turn(
        request.guild_id,
        acceptance_transcript,
        speaker_id=request.user_id,
        now=now,
        utterance_started_at=request.started_at or now,
        force_wake=force_wake,
    )
    if accepted is None:
        return None

    event_bus.publish(
        VoiceEvent(
            "request_recognized" if accepted.direct_wake else "followup_recognized",
            request.guild_id,
            request.channel_id,
            request.user_id,
            None,
            not bool(accepted.text) if accepted.direct_wake else False,
        )
    )
    store.hold_followup(request.guild_id, request.user_id)
    followups.stop_user(request.guild_id, request.user_id)

    if not accepted.text:
        logger.info(
            "Hotword-only utterance activated follow-up without generation guild_id=%s user_id=%s",
            request.guild_id,
            request.user_id,
        )
        return None

    return PreparedTurn(
        request=request,
        transcript=transcript,
        accepted_text=accepted.text,
        direct_wake=accepted.direct_wake,
    )


async def publish_status_speech(
    request: TurnRequest,
    text: str,
) -> None:
    spoken = prepare_for_speech(text, limit=180)
    if not spoken:
        return
    try:
        speech, sample_rate = await tts.synthesize(spoken, request.guild_id)
    except Exception:
        logger.exception(
            "Status speech synthesis failed guild_id=%s text=%r",
            request.guild_id,
            spoken,
        )
        return
    pcm = float_mono_to_discord_pcm(speech, sample_rate)
    event_bus.publish(
        VoiceEvent(
            "status_speech",
            request.guild_id,
            request.channel_id,
            request.user_id,
            content=spoken,
            audioBase64=base64.b64encode(pcm).decode("ascii"),
        )
    )
    logger.info(
        "Status speech queued guild_id=%s user_id=%s text=%r bytes=%s",
        request.guild_id,
        request.user_id,
        spoken,
        len(pcm),
    )


async def generate_turn(turn: PreparedTurn) -> bytes | None:
    request = turn.request
    accepted = turn.accepted_text
    event_bus.publish(
        VoiceEvent(
            "generation_started",
            request.guild_id,
            request.channel_id,
            request.user_id,
        )
    )

    user_memory.set(
        request.guild_id,
        request.user_id,
        "discord_display_name",
        request.display_name,
    )
    preferred_name = user_memory.get(
        request.guild_id, request.user_id, "preferred_name"
    )
    speaker_name = preferred_name or russianize_address_name(request.display_name)
    participant = store.register_participant(
        request.guild_id,
        request.user_id,
        request.display_name,
        preferred_name,
    )
    roster = [
        {
            "identity_key": item.identity_key,
            "display_name": item.display_name,
            "preferred_name": item.preferred_name,
            "spoken_name": item.preferred_name
            or russianize_address_name(item.display_name),
        }
        for item in store.participants(request.guild_id, request.user_id)
    ]
    web_search_allowed = request.user_is_admin or user_memory.has_web_search_access(
        request.guild_id, request.user_id
    )
    prompt = build_turn_prompt(
        roster=roster,
        identity_key=participant.identity_key,
        speaker_name=speaker_name,
        accepted_text=accepted,
        web_search_allowed=web_search_allowed,
    )
    chat_deliveries: list[str | None] = []
    if requested_web_search(accepted) and not web_search_allowed:
        answer = WEB_SEARCH_DENIED_SPEECH
        logger.info(
            "Web search denied guild_id=%s user_id=%s",
            request.guild_id,
            request.user_id,
        )
    else:
        assistant_tools = select_assistant_tools(
            accepted, web_search_allowed=web_search_allowed
        )
        answer = await llm.reply(
            store.history(request.guild_id, request.user_id),
            prompt,
            assistant_tools,
            lambda name, arguments: execute_assistant_tool(
                request, accepted, name, arguments, chat_deliveries
            ),
            required_tool_name=required_tool_for_turn(
                accepted, web_search_allowed=web_search_allowed
            ),
            image_data=request.image_data or None,
            image_content_type=request.image_content_type or None,
            on_status_speech=lambda text: publish_status_speech(request, text),
            blocked_status_tools=(
                frozenset({"search_web"}) if not web_search_allowed else frozenset()
            ),
        )
    spoken_answer = prepare_for_speech(answer)
    if not spoken_answer:
        return None

    sent_chat_messages: set[str] = set()
    for queued_content in chat_deliveries:
        content = queued_content or spoken_answer
        if not content or content in sent_chat_messages:
            continue
        sent_chat_messages.add(content)
        event_bus.publish(
            VoiceEvent(
                "chat_message",
                request.guild_id,
                request.channel_id,
                request.user_id,
                content=content,
            )
        )
        logger.info(
            "Discord chat message queued guild_id=%s channel_id=%s user_id=%s chars=%s",
            request.guild_id,
            request.channel_id,
            request.user_id,
            len(content),
        )

    logger.info("Assistant: %s", spoken_answer)
    speech, sample_rate = await tts.synthesize(spoken_answer, request.guild_id)
    store.append_turn(
        request.guild_id,
        accepted,
        spoken_answer,
        speaker_id=request.user_id,
        identity_key=participant.identity_key,
        speaker_name=speaker_name,
    )
    return float_mono_to_discord_pcm(speech, sample_rate)


recognition_queue = RecognitionQueue(
    recognize_turn, settings.recognition_pending_per_speaker
)
generation_queue = GenerationQueue(generate_turn)


@asynccontextmanager
async def lifespan(_: FastAPI):
    user_memory.prepare()
    logger.info("User memory database is ready at %s", settings.user_memory_db_path)
    logger.info("Preparing Whisper model %s", settings.whisper_model)
    await stt.prepare()
    logger.info("Preparing hotword model %s", settings.hotword_model)
    await hotword_detector.prepare()
    await generation_queue.start()
    await recognition_queue.start()
    logger.info("Whisper model is loaded and voice-core is ready")
    try:
        yield
    finally:
        await recognition_queue.stop()
        await generation_queue.stop()
        await followups.close()
        await llm.close()
        await public_information.close()


app = FastAPI(title="Local Discord Voice Core", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str | int | bool]:
    stats = {**recognition_queue.stats(), **generation_queue.stats()}
    stats["active"] = bool(
        stats["recognition_active"] or stats["generation_active"]
    )
    stats["queued"] = int(stats["recognition_queued"]) + int(
        stats["generation_direct_queued"]
    ) + int(stats["generation_followup_queued"])
    return {"status": "ok", "event_subscribers": event_bus.subscriber_count, **stats}


@app.websocket("/v1/events")
async def voice_events(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = event_bus.subscribe()
    logger.info("Voice event subscriber connected subscribers=%s", event_bus.subscriber_count)

    async def send_events() -> None:
        while True:
            event = await queue.get()
            await websocket.send_json(event.as_dict())

    async def wait_for_disconnect() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

    sender = asyncio.create_task(send_events())
    receiver = asyncio.create_task(wait_for_disconnect())
    try:
        _, pending = await asyncio.wait(
            (sender, receiver), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sender.cancel()
        receiver.cancel()
        event_bus.unsubscribe(queue)
        logger.info("Voice event subscriber disconnected subscribers=%s", event_bus.subscriber_count)


@app.delete("/v1/conversations/{guild_id}", status_code=204)
async def reset_conversation(guild_id: str) -> Response:
    store.finish(guild_id)
    await recognition_queue.cancel_guild(guild_id)
    await generation_queue.cancel_guild(guild_id)
    followups.stop_guild(guild_id)
    return Response(status_code=204)


@app.post("/v1/generations/{guild_id}/interrupt", status_code=204)
async def interrupt_generation(guild_id: str) -> Response:
    await generation_queue.cancel_guild(guild_id)
    return Response(status_code=204)


@app.post(
    "/v1/guilds/{guild_id}/users/{user_id}/followup/playback-finished",
    status_code=204,
)
async def start_followup_after_playback(
    guild_id: str,
    user_id: str,
    x_channel_id: str = Header(),
) -> Response:
    if not store.open_followup(guild_id, user_id):
        logger.info(
            "Follow-up reopen ignored during dialogue cooldown guild_id=%s "
            "channel_id=%s user_id=%s",
            guild_id,
            x_channel_id,
            user_id,
        )
        return Response(status_code=204)
    followups.open(guild_id, x_channel_id, user_id, settings.followup_seconds)
    event_bus.publish(
        VoiceEvent(
            "followup_reopened",
            guild_id,
            x_channel_id,
            user_id,
            round(settings.followup_seconds * 1000),
        )
    )
    logger.info(
        "Follow-up countdown started after playback guild_id=%s channel_id=%s user_id=%s seconds=%.1f",
        guild_id,
        x_channel_id,
        user_id,
        settings.followup_seconds,
    )
    return Response(status_code=204)


@app.put("/v1/guilds/{guild_id}/users", status_code=204)
async def sync_user_directory(
    guild_id: str, users: list[UserDirectoryEntry]
) -> Response:
    for user in users[:100]:
        user_memory.set(
            guild_id, user.userId, "discord_display_name", user.displayName
        )
    logger.info("User directory synced guild_id=%s users=%s", guild_id, len(users[:100]))
    return Response(status_code=204)


@app.get("/v1/guilds/{guild_id}/web-search-access")
async def list_web_search_access(
    guild_id: str,
    x_requester_is_admin: bool = Header(default=False),
) -> dict[str, object]:
    if not x_requester_is_admin:
        raise HTTPException(status_code=403, detail="Administrator permission required")
    return {
        "users": [
            {"userId": entry.user_id, "displayName": entry.display_name}
            for entry in user_memory.list_web_search_access(guild_id)
        ]
    }


@app.put("/v1/guilds/{guild_id}/web-search-access/{user_id}", status_code=204)
async def grant_web_search_access(
    guild_id: str,
    user_id: str,
    grant: WebSearchAccessGrant,
    x_requester_is_admin: bool = Header(default=False),
) -> Response:
    if not x_requester_is_admin:
        raise HTTPException(status_code=403, detail="Administrator permission required")
    user_memory.grant_web_search_access(guild_id, user_id, grant.displayName)
    logger.info("Web search access granted guild_id=%s user_id=%s", guild_id, user_id)
    return Response(status_code=204)


@app.delete("/v1/guilds/{guild_id}/web-search-access/{user_id}", status_code=204)
async def revoke_web_search_access(
    guild_id: str,
    user_id: str,
    x_requester_is_admin: bool = Header(default=False),
) -> Response:
    if not x_requester_is_admin:
        raise HTTPException(status_code=403, detail="Administrator permission required")
    deleted = user_memory.revoke_web_search_access(guild_id, user_id)
    logger.info(
        "Web search access revoked guild_id=%s user_id=%s deleted=%s",
        guild_id,
        user_id,
        deleted,
    )
    return Response(status_code=204)


@app.get("/v1/guilds/{guild_id}/tts")
async def get_tts_selection(guild_id: str) -> dict[str, object]:
    return {
        "selected": tts.selected_voice(guild_id),
        "voices": [voice.as_dict() for voice in tts.voices()],
        "selectedEffect": tts.selected_effect(guild_id),
        "effects": [effect.as_dict() for effect in tts.effects()],
    }


@app.put("/v1/guilds/{guild_id}/tts")
async def set_tts_selection(
    guild_id: str, selection: TtsSelection
) -> dict[str, str]:
    try:
        voice = await tts.select_voice(guild_id, selection.voiceId)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    logger.info(
        "TTS voice selected guild_id=%s voice_id=%s engine=%s",
        guild_id,
        voice.id,
        voice.engine,
    )
    return voice.as_dict()


@app.put("/v1/guilds/{guild_id}/tts/effect")
async def set_tts_effect(
    guild_id: str, selection: TtsEffectSelection
) -> dict[str, str]:
    try:
        effect = await tts.select_effect(guild_id, selection.effectId)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    logger.info(
        "TTS effect selected guild_id=%s effect_id=%s",
        guild_id,
        effect.id,
    )
    return effect.as_dict()


@app.post("/v1/hotword")
async def detect_hotword(
    audio: bytes = Body(media_type="application/octet-stream"),
    final: bool = False,
) -> dict[str, str | bool]:
    bytes_per_second = DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * 2
    if final:
        detected, transcript = await hotword_detector.detect(
            discord_pcm_to_whisper(audio)
        )
        logger.info(
            "Hotword transcription mode=final audio_seconds=%.2f detected=%s transcript=%r",
            len(audio) / bytes_per_second,
            detected,
            transcript,
        )
        return {"detected": detected, "transcript": transcript, "busy": False}

    partial_audio = audio[-bytes_per_second * 3 :]
    result = await hotword_detector.try_detect(discord_pcm_to_whisper(partial_audio))
    if result is None:
        logger.info(
            "Hotword transcription mode=partial audio_seconds=%.2f busy=True transcript=''",
            len(partial_audio) / bytes_per_second,
        )
        return {"detected": False, "transcript": "", "busy": True}
    detected, transcript = result
    logger.info(
        "Hotword transcription mode=partial audio_seconds=%.2f detected=%s busy=False transcript=%r",
        len(partial_audio) / bytes_per_second,
        detected,
        transcript,
    )
    return {"detected": detected, "transcript": transcript, "busy": False}


@app.post("/v1/turn")
async def enqueue_turn(
    body: bytes = Body(media_type="application/octet-stream"),
    x_guild_id: str = Header(),
    x_channel_id: str = Header(),
    x_user_id: str = Header(),
    x_display_name: str = Header(),
    x_audio_age_ms: float = Header(default=0),
    x_early_hotword_detected: bool = Header(default=False),
    x_user_is_admin: bool = Header(default=False),
    x_audio_byte_length: int | None = Header(default=None),
    x_image_content_type: str | None = Header(default=None),
) -> Response:
    image_data = b""
    image_content_type = ""
    audio = body
    if x_audio_byte_length is not None:
        if x_audio_byte_length <= 0 or x_audio_byte_length >= len(body):
            raise HTTPException(status_code=400, detail="Invalid audio/image boundary")
        image_content_type = (x_image_content_type or "").casefold()
        if image_content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="Unsupported image type")
        audio = body[:x_audio_byte_length]
        image_data = body[x_audio_byte_length:]
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image is too large")

    bytes_per_second = DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * 2
    original_size = len(audio)
    audio, truncated = limit_discord_pcm(audio, settings.max_input_seconds)
    if truncated:
        logger.warning(
            "Audio utterance truncated guild_id=%s channel_id=%s user_id=%s original_bytes=%s accepted_bytes=%s",
            x_guild_id,
            x_channel_id,
            x_user_id,
            original_size,
            len(audio),
        )
    if len(audio) < bytes_per_second // 4:
        return Response(status_code=204)

    loop = asyncio.get_running_loop()
    audio_age_seconds = min(
        max(x_audio_age_ms, 0.0) / 1000,
        settings.max_input_seconds,
    )
    prepared = await recognition_queue.submit(
        TurnRequest(
            audio=audio,
            guild_id=x_guild_id,
            channel_id=x_channel_id,
            user_id=x_user_id,
            display_name=unquote(x_display_name),
            started_at=loop.time() - audio_age_seconds,
            early_hotword_detected=x_early_hotword_detected,
            user_is_admin=x_user_is_admin,
            image_content_type=image_content_type,
            image_data=image_data,
        )
    )
    if prepared is None:
        return Response(status_code=204)

    pcm = await generation_queue.submit(prepared)
    if pcm is None:
        return Response(status_code=204)
    return Response(content=pcm, media_type="application/octet-stream")
