from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger("voice-core.queue")


@dataclass(frozen=True)
class TurnRequest:
    audio: bytes
    guild_id: str
    channel_id: str
    user_id: str
    display_name: str
    started_at: float = 0.0
    early_hotword_detected: bool = False
    image_content_type: str = ""
    image_data: bytes = b""

    @property
    def speaker_key(self) -> tuple[str, str, str]:
        return self.guild_id, self.channel_id, self.user_id


@dataclass(frozen=True)
class PreparedTurn:
    request: TurnRequest
    transcript: str
    accepted_text: str
    direct_wake: bool


@dataclass
class _RecognitionItem:
    request_id: int
    request: TurnRequest
    future: asyncio.Future[PreparedTurn | None]
    queued_at: float


@dataclass
class _GenerationItem:
    request_id: int
    turn: PreparedTurn
    future: asyncio.Future[bytes | None]
    queued_at: float


RecognitionProcessor = Callable[[TurnRequest], Awaitable[PreparedTurn | None]]
GenerationProcessor = Callable[[PreparedTurn], Awaitable[bytes | None]]


class RecognitionQueue:
    """Fair STT queue that retains only the newest utterances per speaker."""

    def __init__(self, processor: RecognitionProcessor, pending_per_speaker: int = 2) -> None:
        if pending_per_speaker <= 0:
            raise ValueError("pending_per_speaker must be positive")
        self._processor = processor
        self._pending_per_speaker = pending_per_speaker
        self._pending: OrderedDict[tuple[str, str, str], deque[_RecognitionItem]] = (
            OrderedDict()
        )
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task[None] | None = None
        self._active_item: _RecognitionItem | None = None
        self._accepting = False
        self._next_request_id = 1
        self._coalesced = 0
        self._cancelled = 0

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._accepting = True
        self._worker_task = asyncio.create_task(
            self._worker(), name="voice-recognition-worker"
        )
        logger.info(
            "Recognition queue started pending_per_speaker=%s",
            self._pending_per_speaker,
        )

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        async with self._condition:
            self._accepting = False
            for items in self._pending.values():
                for item in items:
                    _resolve(item.future, None)
            self._pending.clear()
            self._condition.notify_all()
        await self._worker_task
        self._worker_task = None
        logger.info("Recognition queue stopped")

    async def submit(self, request: TurnRequest) -> PreparedTurn | None:
        if not self._accepting:
            raise RuntimeError("Recognition queue is not running")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PreparedTurn | None] = loop.create_future()
        item = _RecognitionItem(
            self._next_request_id, request, future, loop.time()
        )
        self._next_request_id += 1

        async with self._condition:
            items = self._pending.setdefault(request.speaker_key, deque())
            if len(items) >= self._pending_per_speaker:
                replaced = items.popleft()
                self._coalesced += 1
                _resolve(replaced.future, None)
                logger.info(
                    "Recognition request coalesced old_request_id=%s new_request_id=%s user_id=%s",
                    replaced.request_id,
                    item.request_id,
                    request.user_id,
                )
            items.append(item)
            self._condition.notify()

        logger.info(
            "Recognition request queued request_id=%s queued=%s guild_id=%s channel_id=%s user_id=%s",
            item.request_id,
            self.queued_count,
            request.guild_id,
            request.channel_id,
            request.user_id,
        )
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    @property
    def queued_count(self) -> int:
        return sum(len(items) for items in self._pending.values())

    async def cancel_speaker(self, guild_id: str, user_id: str) -> int:
        cancelled = 0
        async with self._condition:
            for speaker_key, items in list(self._pending.items()):
                if speaker_key[0] != guild_id or speaker_key[2] != user_id:
                    continue
                self._pending.pop(speaker_key)
                for item in items:
                    _resolve(item.future, None)
                    cancelled += 1
            self._cancelled += cancelled

        if cancelled:
            logger.info(
                "Recognition requests cancelled guild_id=%s user_id=%s count=%s queued=%s",
                guild_id,
                user_id,
                cancelled,
                self.queued_count,
            )
        return cancelled

    async def cancel_guild(self, guild_id: str) -> int:
        cancelled = 0
        async with self._condition:
            for speaker_key, items in list(self._pending.items()):
                if speaker_key[0] != guild_id:
                    continue
                self._pending.pop(speaker_key)
                for item in items:
                    _resolve(item.future, None)
                    cancelled += 1
            self._cancelled += cancelled

        if cancelled:
            logger.info(
                "Recognition requests cancelled guild_id=%s count=%s queued=%s",
                guild_id,
                cancelled,
                self.queued_count,
            )
        return cancelled

    def stats(self) -> dict[str, int | bool]:
        return {
            "recognition_active": self._active_item is not None,
            "recognition_queued": self.queued_count,
            "recognition_coalesced": self._coalesced,
            "recognition_cancelled": self._cancelled,
        }

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: bool(self._pending) or not self._accepting
                )
                if not self._pending and not self._accepting:
                    break
                speaker_key, items = self._pending.popitem(last=False)
                item = items.popleft()
                if items:
                    self._pending[speaker_key] = items

            if item.future.cancelled():
                continue
            self._active_item = item
            started_at = loop.time()
            wait_ms = round((started_at - item.queued_at) * 1000)
            logger.info(
                "Recognition started request_id=%s wait_ms=%s queued=%s",
                item.request_id,
                wait_ms,
                self.queued_count,
            )
            try:
                result = await self._processor(item.request)
            except Exception as error:
                logger.exception("Recognition failed request_id=%s", item.request_id)
                _reject(item.future, error)
            else:
                logger.info(
                    "Recognition completed request_id=%s processing_ms=%s accepted=%s direct_wake=%s",
                    item.request_id,
                    round((loop.time() - started_at) * 1000),
                    result is not None,
                    result.direct_wake if result else False,
                )
                _resolve(item.future, result)
            finally:
                self._active_item = None


class GenerationQueue:
    """Serial LM/TTS queue with wake priority and coalesced follow-ups."""

    def __init__(self, processor: GenerationProcessor) -> None:
        self._processor = processor
        self._direct: deque[_GenerationItem] = deque()
        self._followups: OrderedDict[tuple[str, str, str], _GenerationItem] = (
            OrderedDict()
        )
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task[None] | None = None
        self._active_item: _GenerationItem | None = None
        self._active_processor_task: asyncio.Task[bytes | None] | None = None
        self._accepting = False
        self._next_request_id = 1
        self._coalesced = 0
        self._interrupted = 0

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._accepting = True
        self._worker_task = asyncio.create_task(
            self._worker(), name="voice-generation-worker"
        )
        logger.info("Generation queue started")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        async with self._condition:
            self._accepting = False
            self._cancel_all_pending()
            self._interrupt_active()
            self._condition.notify_all()
        await self._worker_task
        self._worker_task = None
        logger.info("Generation queue stopped")

    async def submit(self, turn: PreparedTurn) -> bytes | None:
        if not self._accepting:
            raise RuntimeError("Generation queue is not running")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes | None] = loop.create_future()
        item = _GenerationItem(self._next_request_id, turn, future, loop.time())
        self._next_request_id += 1

        async with self._condition:
            if turn.direct_wake:
                self._interrupt_active(turn.request.guild_id)
                self._cancel_followups(turn.request.guild_id)
                self._direct.append(item)
            else:
                key = turn.request.speaker_key
                replaced = self._followups.pop(key, None)
                if replaced is not None:
                    self._coalesced += 1
                    _resolve(replaced.future, None)
                    logger.info(
                        "Generation follow-up coalesced old_request_id=%s new_request_id=%s user_id=%s",
                        replaced.request_id,
                        item.request_id,
                        turn.request.user_id,
                    )
                self._followups[key] = item
            self._condition.notify()

        logger.info(
            "Generation request queued request_id=%s priority=%s direct_queued=%s followup_queued=%s user_id=%s",
            item.request_id,
            "wake" if turn.direct_wake else "followup",
            len(self._direct),
            len(self._followups),
            turn.request.user_id,
        )
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def cancel_guild(self, guild_id: str) -> None:
        async with self._condition:
            self._interrupt_active(guild_id)
            self._cancel_pending_guild(guild_id)

    async def cancel_pending_guild(self, guild_id: str) -> int:
        async with self._condition:
            cancelled = self._cancel_pending_guild(guild_id)
        if cancelled:
            logger.info(
                "Pending generation requests cancelled guild_id=%s count=%s",
                guild_id,
                cancelled,
            )
        return cancelled

    def stats(self) -> dict[str, int | bool]:
        return {
            "generation_active": self._active_item is not None,
            "generation_direct_queued": len(self._direct),
            "generation_followup_queued": len(self._followups),
            "generation_coalesced": self._coalesced,
            "generation_interrupted": self._interrupted,
        }

    def _interrupt_active(self, guild_id: str | None = None) -> None:
        if self._active_item is None or self._active_processor_task is None:
            return
        if guild_id is not None and self._active_item.turn.request.guild_id != guild_id:
            return
        if self._active_processor_task.done():
            return
        self._interrupted += 1
        logger.info(
            "Generation interrupted request_id=%s guild_id=%s",
            self._active_item.request_id,
            self._active_item.turn.request.guild_id,
        )
        self._active_processor_task.cancel()

    def _cancel_followups(self, guild_id: str) -> None:
        for key, item in list(self._followups.items()):
            if key[0] != guild_id:
                continue
            self._followups.pop(key)
            self._coalesced += 1
            _resolve(item.future, None)

    def _cancel_pending_guild(self, guild_id: str) -> int:
        cancelled_before = self._coalesced
        kept_direct: deque[_GenerationItem] = deque()
        while self._direct:
            item = self._direct.popleft()
            if item.turn.request.guild_id == guild_id:
                self._coalesced += 1
                _resolve(item.future, None)
            else:
                kept_direct.append(item)
        self._direct = kept_direct
        self._cancel_followups(guild_id)
        return self._coalesced - cancelled_before

    def _cancel_all_pending(self) -> None:
        while self._direct:
            _resolve(self._direct.popleft().future, None)
        for item in self._followups.values():
            _resolve(item.future, None)
        self._followups.clear()

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: bool(self._direct or self._followups) or not self._accepting
                )
                if not self._direct and not self._followups and not self._accepting:
                    break
                if self._direct:
                    item = self._direct.popleft()
                else:
                    _, item = self._followups.popitem(last=False)

            if item.future.cancelled():
                continue
            self._active_item = item
            started_at = loop.time()
            wait_ms = round((started_at - item.queued_at) * 1000)
            logger.info(
                "Generation started request_id=%s priority=%s wait_ms=%s direct_queued=%s followup_queued=%s",
                item.request_id,
                "wake" if item.turn.direct_wake else "followup",
                wait_ms,
                len(self._direct),
                len(self._followups),
            )
            try:
                self._active_processor_task = asyncio.create_task(
                    self._processor(item.turn),
                    name=f"voice-generation-{item.request_id}",
                )
                result = await self._active_processor_task
            except asyncio.CancelledError:
                logger.info("Generation cancellation completed request_id=%s", item.request_id)
                _resolve(item.future, None)
            except Exception as error:
                logger.exception("Generation failed request_id=%s", item.request_id)
                _reject(item.future, error)
            else:
                logger.info(
                    "Generation completed request_id=%s processing_ms=%s has_reply=%s",
                    item.request_id,
                    round((loop.time() - started_at) * 1000),
                    result is not None,
                )
                _resolve(item.future, result)
            finally:
                self._active_processor_task = None
                self._active_item = None


def _resolve(future: asyncio.Future, value: object) -> None:
    if not future.done():
        future.set_result(value)


def _reject(future: asyncio.Future, error: Exception) -> None:
    if not future.done():
        future.set_exception(error)
