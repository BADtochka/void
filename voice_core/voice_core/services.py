from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .config import Settings
from .audio_effects import apply_robotic_voice_effect
from .dialogue import normalize_phrase, trim_truncated_completion
from .speech_normalization import normalize_russian_tts_text
from .tooling import is_incomplete_tool_promise, tool_denied_speech, tool_status_speech
from .whisper_models import resolve_model
from .windows_dlls import configure_windows_cuda_dlls

logger = logging.getLogger(__name__)

_SILERO_SENTENCE_PATTERN = re.compile(r"\S(?:.*?\S)?(?:[.!?]+(?=\s|$)|$)")
_MAX_TOOL_ROUNDS = 3
_TOOL_LIMIT_FALLBACK = "Сейчас не получается проверить это. Скажи ещё раз чуть проще."
_LM_REQUEST_FALLBACK = "Сейчас не могу ответить. Попробуй ещё раз."
_VISION_IMAGE_FALLBACK = "Не удалось разобрать изображение. Пришли картинку ещё раз."
_INCOMPLETE_TOOL_NUDGE = (
    "Ты пообещал посмотреть или найти информацию, но не вызвал инструмент. "
    "Сейчас обязательно сделай tool call (например search_web или lookup_topic). "
    "Не отвечай обычным текстом до результата инструмента."
)
_INCOMPLETE_TOOL_FALLBACK = "Не удалось это проверить. Скажи ещё раз."


def _vision_image_url(encoded_image: str, content_type: str, *, style: str) -> str:
    """Build image_url.url for vision models.

    LM Studio rejects OpenAI-style data URIs and expects raw base64 in `url`.
    Keep data_uri as a fallback for other OpenAI-compatible servers.
    """
    if style == "data_uri":
        return f"data:{content_type};base64,{encoded_image}"
    return encoded_image


def _build_vision_user_content(
    user_text: str,
    encoded_image: str,
    content_type: str,
    *,
    style: str,
) -> list[dict[str, object]]:
    return [
        {"type": "text", "text": user_text},
        {
            "type": "image_url",
            "image_url": {
                "url": _vision_image_url(encoded_image, content_type, style=style)
            },
        },
    ]


def _http_error_text(error: httpx.HTTPStatusError) -> str:
    response = error.response
    if response is None:
        return ""
    try:
        return response.text or ""
    except Exception:
        return ""


def _is_vision_image_error(error_text: str) -> bool:
    lowered = error_text.casefold()
    return any(
        marker in lowered
        for marker in (
            "base64 encoded image",
            "'url' field must be",
            "image_url",
        )
    ) or (
        "base64" in lowered
        and any(marker in lowered for marker in ("image", "url"))
    )


def _silero_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = [
        match.group(0).strip()
        for match in _SILERO_SENTENCE_PATTERN.finditer(normalized)
    ]
    return sentences or [normalized]


@dataclass(frozen=True)
class ToolResult:
    content: str
    terminate: bool = False
    response: str | None = None


ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str | ToolResult]]
StatusSpeechHandler = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class TtsVoice:
    id: str
    label: str
    engine: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TtsEffect:
    id: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return ""


def _stream_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text") or "")
            for item in value
            if isinstance(item, (str, dict))
        )
    return ""


def _merge_tool_call_deltas(
    accumulated: dict[int, dict[str, Any]], deltas: object
) -> None:
    if not isinstance(deltas, list):
        return
    for fallback_index, delta in enumerate(deltas):
        if not isinstance(delta, dict):
            continue
        index = delta.get("index", fallback_index)
        if not isinstance(index, int):
            index = fallback_index
        tool_call = accumulated.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if delta.get("id"):
            tool_call["id"] = str(delta["id"])
        if delta.get("type"):
            tool_call["type"] = str(delta["type"])
        function = delta.get("function")
        if not isinstance(function, dict):
            continue
        target_function = tool_call["function"]
        target_function["name"] += str(function.get("name") or "")
        target_function["arguments"] += str(function.get("arguments") or "")


class SpeechToText:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._model is None:
            configure_windows_cuda_dlls()
            from faster_whisper import WhisperModel

            model_path = resolve_model(self._settings.whisper_model)
            self._model = WhisperModel(
                model_path,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
        return self._model

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        configure_windows_cuda_dlls()
        if self._settings.whisper_speech_gate:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            speech = get_speech_timestamps(
                audio,
                VadOptions(
                    threshold=self._settings.whisper_speech_gate_threshold,
                    min_speech_duration_ms=self._settings.whisper_min_speech_ms,
                    min_silence_duration_ms=200,
                    speech_pad_ms=0,
                ),
            )
            if not speech:
                logger.info("Silero speech gate discarded audio without speech")
                return ""

        leading_samples = round(
            16_000 * self._settings.whisper_leading_silence_ms / 1000
        )
        if leading_samples > 0:
            audio = np.pad(audio, (leading_samples, 0))
        segments, _ = self._load().transcribe(
            audio,
            language=self._settings.whisper_language or None,
            beam_size=self._settings.whisper_beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            hotwords=None,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio)

    async def prepare(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load)


class HotwordDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._parallelism = max(1, settings.hotword_parallelism)
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._slots = asyncio.Semaphore(self._parallelism)
        self._active = 0
        self._waiting = 0
        wake_words = (settings.wake_word, *settings.wake_word_aliases)
        self._wake_phrases = tuple(
            normalize_phrase(word)
            for word in dict.fromkeys(wake_words)
            if normalize_phrase(word)
        )

    def _load(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model

            configure_windows_cuda_dlls()
            from faster_whisper import WhisperModel

            model_path = resolve_model(self._settings.hotword_model)
            self._model = WhisperModel(
                model_path,
                device=self._settings.hotword_device,
                compute_type=self._settings.hotword_compute_type,
                num_workers=self._parallelism,
            )
        return self._model

    def _detect_sync(self, audio: np.ndarray) -> tuple[bool, str]:
        segments, _ = self._load().transcribe(
            audio,
            language=self._settings.whisper_language or None,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        normalized = f" {normalize_phrase(transcript)} "
        detected = any(f" {wake_phrase} " in normalized for wake_phrase in self._wake_phrases)
        return detected, transcript

    async def _detect_in_slot(self, audio: np.ndarray) -> tuple[bool, str]:
        inference = asyncio.create_task(asyncio.to_thread(self._detect_sync, audio))
        try:
            return await asyncio.shield(inference)
        except asyncio.CancelledError:
            # asyncio.to_thread cannot stop native CTranslate2 inference. Keep the
            # slot occupied until it really finishes to preserve the worker limit.
            await inference
            raise

    async def detect(self, audio: np.ndarray) -> tuple[bool, str]:
        loop = asyncio.get_running_loop()
        queued_at = loop.time()
        waiting_ahead = self._waiting
        self._waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self._waiting -= 1
        wait_ms = round((loop.time() - queued_at) * 1000)
        self._active += 1
        started_at = loop.time()
        logger.info(
            "Hotword final check started wait_ms=%s waiting_ahead=%s active=%s waiting=%s",
            wait_ms,
            waiting_ahead,
            self._active,
            self._waiting,
        )
        try:
            return await self._detect_in_slot(audio)
        finally:
            processing_ms = round((loop.time() - started_at) * 1000)
            self._active -= 1
            self._slots.release()
            logger.info(
                "Hotword final check finished processing_ms=%s active=%s waiting=%s",
                processing_ms,
                self._active,
                self._waiting,
            )

    async def try_detect(self, audio: np.ndarray) -> tuple[bool, str] | None:
        # Run alongside other speakers while slots are free; finals still wait in detect().
        if self._slots.locked():
            return None
        await self._slots.acquire()
        self._active += 1
        try:
            return await self._detect_in_slot(audio)
        finally:
            self._active -= 1
            self._slots.release()

    async def prepare(self) -> None:
        async with self._slots:
            await asyncio.to_thread(self._load)


class LanguageModel:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.lmstudio_timeout_seconds, connect=5.0)
        )

    async def _stream_completion(
        self, payload: dict[str, Any], request_number: int
    ) -> tuple[dict[str, Any], str | None]:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        generated_chunks = 0
        finish_reason: str | None = None
        max_tokens = int(payload["max_tokens"])

        async def report_progress() -> None:
            while True:
                await asyncio.sleep(self._settings.lmstudio_progress_interval_seconds)
                elapsed = loop.time() - started_at
                approximate_percent = min(99, round(generated_chunks / max_tokens * 100))
                logger.info(
                    "LM Studio progress request=%s elapsed=%.1fs token_chunks~=%s/%s progress~=%s%% speed~=%.1f chunks/s content_chars=%s",
                    request_number,
                    elapsed,
                    generated_chunks,
                    max_tokens,
                    approximate_percent,
                    generated_chunks / elapsed if elapsed else 0.0,
                    sum(map(len, content_parts)),
                )

        progress_task = asyncio.create_task(
            report_progress(), name=f"lmstudio-progress-{request_number}"
        )
        logger.info(
            "LM Studio generation started request=%s max_tokens=%s",
            request_number,
            max_tokens,
        )
        try:
            async with self._client.stream(
                "POST",
                f"{self._settings.lmstudio_base_url}/chat/completions",
                json={**payload, "stream": True},
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    logger.error(
                        "LM Studio request failed status=%s request=%s body=%s",
                        response.status_code,
                        request_number,
                        body[:1_000],
                    )
                    # Rebuild a non-streaming response so callers can read body after aread().
                    error_response = httpx.Response(
                        response.status_code,
                        request=response.request,
                        content=body.encode("utf-8"),
                        headers=response.headers,
                    )
                    raise httpx.HTTPStatusError(
                        f"Client error '{response.status_code}' for url '{response.url}'",
                        request=response.request,
                        response=error_response,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    content = _stream_content(delta.get("content"))
                    reasoning = _stream_content(delta.get("reasoning_content"))
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    _merge_tool_call_deltas(tool_calls, delta.get("tool_calls"))
                    if content or reasoning or delta.get("tool_calls"):
                        generated_chunks += 1
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
        finally:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)

        message = {
            "content": "".join(content_parts).strip(),
            "reasoning_content": "".join(reasoning_parts),
            "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
        }
        elapsed = loop.time() - started_at
        logger.info(
            "LM Studio generation finished request=%s elapsed=%.1fs token_chunks~=%s speed~=%.1f chunks/s",
            request_number,
            elapsed,
            generated_chunks,
            generated_chunks / elapsed if elapsed else 0.0,
        )
        return message, finish_reason

    async def reply(
        self,
        history: list[dict[str, str]],
        user_text: str,
        tools: list[dict[str, object]] | None = None,
        tool_handler: ToolHandler | None = None,
        required_tool_name: str | None = None,
        image_data: bytes | None = None,
        image_content_type: str | None = None,
        on_status_speech: StatusSpeechHandler | None = None,
        blocked_status_tools: frozenset[str] | None = None,
    ) -> str:
        user_content: str | list[dict[str, object]] = user_text
        encoded_image: str | None = None
        image_url_style = "raw"
        if image_data and image_content_type:
            encoded_image = base64.b64encode(image_data).decode("ascii")
            user_content = _build_vision_user_content(
                user_text,
                encoded_image,
                image_content_type,
                style=image_url_style,
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._settings.system_prompt},
            *history,
            {"role": "user", "content": user_content},
        ]
        empty_retries = 0
        tool_rounds = 0
        request_number = 0
        tools_disabled_after_error = False
        image_format_retried = False
        incomplete_promise_retries = 0
        progress_already_announced = False
        status_blocked = blocked_status_tools or frozenset()
        while True:
            request_number += 1
            payload: dict[str, Any] = {
                "model": self._settings.lmstudio_model,
                "messages": messages,
                "temperature": self._settings.lmstudio_temperature if empty_retries == 0 else 0.2,
                "max_tokens": (
                    self._settings.lmstudio_max_tokens
                    if empty_retries == 0
                    else max(512, self._settings.lmstudio_max_tokens * 2)
                ),
            }
            if tools:
                if required_tool_name and tool_rounds == 0:
                    required_tools = [
                        tool
                        for tool in tools
                        if (tool.get("function") or {}).get("name")
                        == required_tool_name
                    ]
                    if not required_tools:
                        raise ValueError(
                            f"required tool is not configured: {required_tool_name}"
                        )
                    payload["tools"] = required_tools
                    payload["tool_choice"] = "required"
                else:
                    payload["tools"] = tools
            if self._settings.lmstudio_reasoning_effort:
                payload["reasoning_effort"] = self._settings.lmstudio_reasoning_effort

            try:
                message, finish_reason = await self._stream_completion(
                    payload, request_number
                )
            except httpx.HTTPStatusError as error:
                status = error.response.status_code if error.response is not None else None
                error_text = _http_error_text(error)
                if (
                    status == 400
                    and encoded_image is not None
                    and image_content_type
                    and not image_format_retried
                    and _is_vision_image_error(error_text)
                ):
                    image_format_retried = True
                    image_url_style = (
                        "data_uri" if image_url_style == "raw" else "raw"
                    )
                    logger.warning(
                        "LM Studio rejected vision image format; retrying with style=%s",
                        image_url_style,
                    )
                    messages[-1] = {
                        "role": "user",
                        "content": _build_vision_user_content(
                            user_text,
                            encoded_image,
                            image_content_type,
                            style=image_url_style,
                        ),
                    }
                    continue
                if (
                    status == 400
                    and tools
                    and not tools_disabled_after_error
                    and "tools" in payload
                    and not _is_vision_image_error(error_text)
                ):
                    logger.warning(
                        "LM Studio rejected tools payload; retrying without tools"
                    )
                    tools = None
                    required_tool_name = None
                    tools_disabled_after_error = True
                    continue
                logger.exception(
                    "LM Studio chat completions failed status=%s",
                    status,
                )
                if encoded_image is not None and _is_vision_image_error(error_text):
                    return _VISION_IMAGE_FALLBACK
                return _LM_REQUEST_FALLBACK
            except httpx.HTTPError:
                logger.exception("LM Studio chat completions request failed")
                return _LM_REQUEST_FALLBACK

            content = _message_content(message)
            tool_calls = message.get("tool_calls") or []
            logger.info(
                "LM Studio completion request=%s finish_reason=%s content_chars=%s reasoning_chars=%s tool_calls=%s",
                request_number,
                finish_reason,
                len(content),
                len(message.get("reasoning_content") or ""),
                len(tool_calls),
            )

            if tool_calls:
                if tool_handler is None:
                    raise RuntimeError("LM Studio requested a tool, but no tool handler is configured")
                if tool_rounds >= _MAX_TOOL_ROUNDS:
                    logger.warning(
                        "LM Studio exceeded the maximum number of tool rounds; finishing without more tools"
                    )
                    spoken = content.strip()
                    if spoken:
                        return spoken
                    if tools is None:
                        return _TOOL_LIMIT_FALLBACK
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": tool_calls,
                        }
                    )
                    for tool_call in tool_calls:
                        function = tool_call.get("function") or {}
                        tool_name = str(function.get("name") or "tool")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": str(tool_call.get("id") or tool_name),
                                "content": json.dumps(
                                    {
                                        "ok": False,
                                        "error": "tool limit reached; answer in plain spoken text now",
                                    }
                                ),
                            }
                        )
                    tools = None
                    required_tool_name = None
                    continue
                tool_rounds += 1
                messages.append(
                    {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
                )
                model_status = content.strip() if content else ""
                announced_model_status = False
                ending_only = all(
                    str((tool_call.get("function") or {}).get("name") or "")
                    == "end_conversation"
                    for tool_call in tool_calls
                )
                tool_names_this_round = {
                    str((tool_call.get("function") or {}).get("name") or "")
                    for tool_call in tool_calls
                }
                skip_status_for_denied_tools = bool(tool_names_this_round) and (
                    tool_names_this_round <= status_blocked
                )
                if (
                    model_status
                    and on_status_speech is not None
                    and not ending_only
                    and not skip_status_for_denied_tools
                    and not progress_already_announced
                ):
                    await on_status_speech(model_status)
                    announced_model_status = True
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    tool_name = str(function.get("name") or "")
                    if (
                        on_status_speech is not None
                        and not announced_model_status
                        and not progress_already_announced
                        and tool_name
                        and tool_name != "end_conversation"
                        and tool_name not in status_blocked
                    ):
                        announcement = tool_status_speech(tool_name)
                        if announcement:
                            await on_status_speech(announcement)
                            announced_model_status = True
                            progress_already_announced = True
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError("tool arguments must be an object")
                        if (
                            tool_name == "end_conversation"
                            and model_status
                            and not str(arguments.get("farewell") or "").strip()
                        ):
                            arguments = {**arguments, "farewell": model_status}
                        raw_tool_result = await tool_handler(tool_name, arguments)
                    except PermissionError as error:
                        logger.warning("Tool %s denied: %s", tool_name, error)
                        denial = tool_denied_speech(tool_name)
                        return denial
                    except Exception as error:
                        logger.warning("Tool %s failed: %s", tool_name, error)
                        tool_result = json.dumps({"ok": False, "error": str(error)})
                        terminate = False
                        raw_tool_result = None
                    else:
                        if isinstance(raw_tool_result, ToolResult):
                            tool_result = raw_tool_result.content
                            terminate = raw_tool_result.terminate
                        else:
                            tool_result = raw_tool_result
                            terminate = False
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_call.get("id") or tool_name),
                            "content": tool_result,
                        }
                    )
                    if terminate:
                        logger.info("Tool %s terminated the current completion", tool_name)
                        if isinstance(raw_tool_result, ToolResult) and raw_tool_result.response:
                            return raw_tool_result.response
                        if model_status and not ending_only:
                            return "" if announced_model_status else model_status
                        return ""
                continue

            if content:
                if (
                    tools
                    and not tool_calls
                    and incomplete_promise_retries < 1
                    and is_incomplete_tool_promise(content)
                ):
                    incomplete_promise_retries += 1
                    logger.warning(
                        "LM Studio promised progress without a tool call; nudging request=%s text=%r",
                        request_number,
                        content,
                    )
                    if on_status_speech is not None:
                        await on_status_speech(content)
                        progress_already_announced = True
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": _INCOMPLETE_TOOL_NUDGE})
                    continue
                if (
                    tools
                    and not tool_calls
                    and incomplete_promise_retries
                    and is_incomplete_tool_promise(content)
                ):
                    logger.warning(
                        "LM Studio still promised progress without tools; using fallback"
                    )
                    return _INCOMPLETE_TOOL_FALLBACK
                if finish_reason == "length":
                    trimmed = trim_truncated_completion(content)
                    if trimmed:
                        logger.warning(
                            "LM Studio hit max_tokens; trimmed spoken answer from %s to %s chars",
                            len(content),
                            len(trimmed),
                        )
                        content = trimmed
                return content

            if tools is None and empty_retries == 0 and tool_rounds >= _MAX_TOOL_ROUNDS:
                logger.warning(
                    "LM Studio returned empty content after tool limit; using fallback reply"
                )
                return _TOOL_LIMIT_FALLBACK

            if empty_retries == 0:
                empty_retries += 1
                logger.warning("LM Studio returned empty content; retrying without reasoning")
                continue

            raise RuntimeError("LM Studio returned empty content after retry")

    async def close(self) -> None:
        await self._client.aclose()


class TextToSpeech:
    _SILERO_MODEL = "v5_5_ru"
    _SILERO_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
    _SILERO_SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")
    _EFFECTS = (
        TtsEffect("none", "Обычный"),
        TtsEffect("robotic", "Роботизированный"),
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        configured_path = Path(settings.piper_model_path).resolve()
        piper_paths = {configured_path}
        if configured_path.parent.is_dir():
            piper_paths.update(configured_path.parent.glob("*.onnx"))
        self._piper_paths = {
            f"piper:{path.stem}": path for path in sorted(piper_paths)
        }
        piper_default_voice_id = f"piper:{configured_path.stem}"
        self._default_voice_id = settings.tts_default_voice or piper_default_voice_id
        available_voice_ids = {
            *self._piper_paths,
            *(f"silero:{speaker}" for speaker in self._SILERO_SPEAKERS),
        }
        if self._default_voice_id not in available_voice_ids:
            available = ", ".join(sorted(available_voice_ids))
            raise ValueError(
                f"unknown TTS_DEFAULT_VOICE: {self._default_voice_id}; "
                f"available voices: {available}"
            )
        available_effect_ids = {effect.id for effect in self._EFFECTS}
        self._default_effect_id = settings.tts_default_effect
        if self._default_effect_id not in available_effect_ids:
            available = ", ".join(sorted(available_effect_ids))
            raise ValueError(
                f"unknown TTS_DEFAULT_EFFECT: {self._default_effect_id}; "
                f"available effects: {available}"
            )
        self._silero_model_path = configured_path.parent / f"{self._SILERO_MODEL}.pt"
        self._piper_models: dict[str, Any] = {}
        self._silero_model: Any = None
        self._guild_voices: dict[str, str] = {}
        self._guild_effects: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def voices(self) -> list[TtsVoice]:
        piper = [
            TtsVoice(voice_id, f"Piper · {path.stem}", "piper")
            for voice_id, path in self._piper_paths.items()
        ]
        silero = [
            TtsVoice(f"silero:{speaker}", f"Silero · {speaker}", "silero")
            for speaker in self._SILERO_SPEAKERS
        ]
        return [*piper, *silero]

    def selected_voice(self, guild_id: str) -> str:
        return self._guild_voices.get(guild_id, self._default_voice_id)

    def effects(self) -> list[TtsEffect]:
        return list(self._EFFECTS)

    def selected_effect(self, guild_id: str) -> str:
        return self._guild_effects.get(guild_id, self._default_effect_id)

    async def select_effect(self, guild_id: str, effect_id: str) -> TtsEffect:
        effect = next((item for item in self._EFFECTS if item.id == effect_id), None)
        if effect is None:
            raise ValueError(f"unknown TTS effect: {effect_id}")
        async with self._lock:
            self._guild_effects[guild_id] = effect_id
        return effect

    async def select_voice(self, guild_id: str, voice_id: str) -> TtsVoice:
        voice = next((item for item in self.voices() if item.id == voice_id), None)
        if voice is None:
            raise ValueError(f"unknown TTS voice: {voice_id}")
        async with self._lock:
            await asyncio.to_thread(self._prepare_voice_sync, voice_id)
            self._guild_voices[guild_id] = voice_id
        return voice

    def _load_piper(self, voice_id: str) -> Any:
        model = self._piper_models.get(voice_id)
        if model is None:
            from piper import PiperVoice

            path = self._piper_paths[voice_id]
            model = PiperVoice.load(str(path))
            self._piper_models[voice_id] = model
        return model

    def _load_silero(self) -> Any:
        if self._silero_model is None:
            import torch

            self._download_silero_model()
            with warnings.catch_warnings():
                # Silero v5_5_ru contains a Python 3.13-invalid escape in packaged code.
                warnings.simplefilter("ignore", SyntaxWarning)
                importer = torch.package.PackageImporter(str(self._silero_model_path))
                model = importer.load_pickle("tts_models", "model")
            model.to("cpu")
            self._silero_model = model
        return self._silero_model

    def _download_silero_model(self) -> None:
        if self._silero_model_path.is_file():
            return
        self._silero_model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._silero_model_path.with_suffix(".pt.part")
        downloaded = 0
        last_percent = -10
        try:
            with httpx.stream(
                "GET",
                self._SILERO_MODEL_URL,
                follow_redirects=True,
                timeout=httpx.Timeout(180.0, connect=20.0),
            ) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                with temporary_path.open("wb") as target:
                    for chunk in response.iter_bytes(1024 * 1024):
                        target.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 // total
                            if percent >= last_percent + 10:
                                last_percent = percent
                                logger.info(
                                    "Silero model download progress: %s%% (%s/%s MB)",
                                    percent,
                                    downloaded // (1024 * 1024),
                                    total // (1024 * 1024),
                                )
            temporary_path.replace(self._silero_model_path)
            logger.info(
                "Silero model downloaded path=%s bytes=%s",
                self._silero_model_path,
                downloaded,
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _prepare_voice_sync(self, voice_id: str) -> None:
        if voice_id.startswith("piper:"):
            self._load_piper(voice_id)
            return
        if voice_id.startswith("silero:"):
            self._load_silero()
            return
        raise ValueError(f"unknown TTS voice: {voice_id}")

    def _synthesize_piper_sync(
        self, text: str, voice_id: str
    ) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig

        audio_parts: list[bytes] = []
        sample_rate: int | None = None
        synthesis_config = SynthesisConfig(length_scale=self._settings.piper_length_scale)
        for chunk in self._load_piper(voice_id).synthesize(
            text, syn_config=synthesis_config
        ):
            if audio_parts and self._settings.piper_sentence_silence_ms > 0:
                silence_samples = round(
                    chunk.sample_rate * self._settings.piper_sentence_silence_ms / 1000
                )
                audio_parts.append(bytes(silence_samples * 2))
            audio_parts.append(chunk.audio_int16_bytes)
            sample_rate = chunk.sample_rate

        if sample_rate is None:
            return np.empty(0, dtype=np.float32), 48_000
        pcm = np.frombuffer(b"".join(audio_parts), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, sample_rate

    def _synthesize_silero_sync(
        self, text: str, voice_id: str
    ) -> tuple[np.ndarray, int]:
        speaker = voice_id.removeprefix("silero:")
        if speaker not in self._SILERO_SPEAKERS:
            raise ValueError(f"unknown Silero speaker: {speaker}")
        model = self._load_silero()
        sentences = _silero_sentences(normalize_russian_tts_text(text))
        audio_parts: list[np.ndarray] = []
        silence_samples = round(
            48_000 * self._settings.silero_sentence_silence_ms / 1000
        )
        for index, sentence in enumerate(sentences):
            audio = model.apply_tts(
                text=sentence,
                speaker=speaker,
                sample_rate=48_000,
                put_accent=self._settings.silero_put_accent,
                put_yo=self._settings.silero_put_yo,
            )
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio_parts.append(np.asarray(audio, dtype=np.float32))
            if index < len(sentences) - 1 and silence_samples > 0:
                audio_parts.append(np.zeros(silence_samples, dtype=np.float32))
        if not audio_parts:
            return np.empty(0, dtype=np.float32), 48_000
        return np.concatenate(audio_parts), 48_000

    def _synthesize_sync(
        self, text: str, voice_id: str, effect_id: str
    ) -> tuple[np.ndarray, int]:
        if voice_id.startswith("silero:"):
            audio, sample_rate = self._synthesize_silero_sync(text, voice_id)
        else:
            audio, sample_rate = self._synthesize_piper_sync(text, voice_id)
        if effect_id == "robotic":
            audio = apply_robotic_voice_effect(
                audio,
                sample_rate,
                pitch_semitones=self._settings.tts_robot_pitch_semitones,
                harmony_volume=self._settings.tts_robot_harmony_volume,
                modulation_hz=self._settings.tts_robot_modulation_hz,
                modulation_depth=self._settings.tts_robot_modulation_depth,
                reverb_amount=self._settings.tts_robot_reverb,
            )
        return audio, sample_rate

    async def synthesize(
        self, text: str, guild_id: str
    ) -> tuple[np.ndarray, int]:
        async with self._lock:
            voice_id = self.selected_voice(guild_id)
            effect_id = self.selected_effect(guild_id)
            return await asyncio.to_thread(
                self._synthesize_sync, text, voice_id, effect_id
            )
