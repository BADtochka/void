import unittest

import asyncio

from voice_core.event_bus import FollowupTracker, VoiceEvent, VoiceEventBus


class VoiceEventBusTests(unittest.TestCase):
    def test_awaiting_content_is_serialized_for_gateway(self) -> None:
        event = VoiceEvent(
            "request_recognized",
            "guild",
            "channel",
            "owner",
            followupMs=30_000,
            awaitingContent=True,
        )

        self.assertEqual(event.as_dict()["awaitingContent"], True)

    def test_chat_message_content_is_serialized_for_gateway(self) -> None:
        event = VoiceEvent(
            "chat_message",
            "guild",
            "channel",
            "owner",
            content="Текст для Discord",
        )

        self.assertEqual(event.as_dict()["content"], "Текст для Discord")

    def test_events_are_broadcast_and_oldest_is_dropped_at_capacity(self) -> None:
        bus = VoiceEventBus(subscriber_capacity=2)
        first = bus.subscribe()
        second = bus.subscribe()

        bus.publish(VoiceEvent("request_recognized", "guild", "channel", "one"))
        bus.publish(VoiceEvent("generation_started", "guild", "channel", "one"))
        bus.publish(VoiceEvent("followup_recognized", "guild", "channel", "two"))

        self.assertEqual(first.get_nowait().type, "generation_started")
        self.assertEqual(first.get_nowait().type, "followup_recognized")
        self.assertEqual(second.get_nowait().type, "generation_started")
        self.assertEqual(second.get_nowait().type, "followup_recognized")

        bus.unsubscribe(first)
        self.assertEqual(bus.subscriber_count, 1)


class FollowupTrackerTests(unittest.IsolatedAsyncioTestCase):
    async def test_expiry_is_emitted_and_reopen_replaces_old_timer(self) -> None:
        bus = VoiceEventBus()
        events = bus.subscribe()
        tracker = FollowupTracker(bus)
        try:
            tracker.open("guild", "channel", "user", 0.01)
            tracker.open("guild", "channel", "user", 0.03)
            await asyncio.sleep(0.015)
            self.assertTrue(events.empty())
            event = await asyncio.wait_for(events.get(), 0.05)
            self.assertEqual(event.type, "followup_expired")
            self.assertEqual(event.userId, "user")
        finally:
            await tracker.close()


if __name__ == "__main__":
    unittest.main()
