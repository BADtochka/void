from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Literal

VoiceEventType = Literal[
    "request_recognized",
    "generation_started",
    "followup_recognized",
    "followup_stopped",
    "followup_expired",
    "followup_reopened",
    "chat_message",
    "status_speech",
]


@dataclass(frozen=True)
class VoiceEvent:
    type: VoiceEventType
    guildId: str
    channelId: str
    userId: str
    followupMs: int | None = None
    awaitingContent: bool | None = None
    content: str | None = None
    audioBase64: str | None = None

    def as_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


class VoiceEventBus:
    def __init__(self, subscriber_capacity: int = 32) -> None:
        self._subscriber_capacity = subscriber_capacity
        self._subscribers: set[asyncio.Queue[VoiceEvent]] = set()

    def subscribe(self) -> asyncio.Queue[VoiceEvent]:
        queue: asyncio.Queue[VoiceEvent] = asyncio.Queue(self._subscriber_capacity)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[VoiceEvent]) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: VoiceEvent) -> None:
        for queue in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


class FollowupTracker:
    def __init__(self, event_bus: VoiceEventBus) -> None:
        self._event_bus = event_bus
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    def open(self, guild_id: str, channel_id: str, user_id: str, seconds: float) -> None:
        key = guild_id, user_id
        existing = self._tasks.pop(key, None)
        if existing:
            existing.cancel()
        self._tasks[key] = asyncio.create_task(
            self._expire(key, channel_id, seconds),
            name=f"followup-expiry-{guild_id}-{user_id}",
        )

    def stop_guild(self, guild_id: str) -> None:
        for key, task in list(self._tasks.items()):
            if key[0] != guild_id:
                continue
            self._tasks.pop(key)
            task.cancel()

    def stop_user(self, guild_id: str, user_id: str) -> None:
        task = self._tasks.pop((guild_id, user_id), None)
        if task:
            task.cancel()

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _expire(
        self, key: tuple[str, str], channel_id: str, seconds: float
    ) -> None:
        try:
            await asyncio.sleep(seconds)
            if self._tasks.get(key) is not asyncio.current_task():
                return
            self._tasks.pop(key, None)
            self._event_bus.publish(
                VoiceEvent("followup_expired", key[0], channel_id, key[1])
            )
        except asyncio.CancelledError:
            return
