import asyncio
import unittest

from voice_core.turn_queue import (
    GenerationQueue,
    PreparedTurn,
    RecognitionQueue,
    TurnRequest,
)


def request(user_id: str, audio: bytes | None = None) -> TurnRequest:
    return TurnRequest(audio or user_id.encode(), "guild", "channel", user_id, user_id)


def prepared(user_id: str, text: str, direct: bool) -> PreparedTurn:
    return PreparedTurn(request(user_id, text.encode()), text, text, direct)


class RecognitionQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_guild_removes_pending_audio_from_every_speaker(self) -> None:
        blocker_started = asyncio.Event()
        release_blocker = asyncio.Event()

        async def processor(request):
            if request.user_id == "blocker":
                blocker_started.set()
                await release_blocker.wait()
            return PreparedTurn(request, "accepted", "accepted", False)

        queue = RecognitionQueue(processor, pending_per_speaker=2)
        await queue.start()
        try:
            blocker = asyncio.create_task(
                queue.submit(TurnRequest(b"", "guild", "channel", "blocker", "Blocker"))
            )
            await blocker_started.wait()
            pending_one = asyncio.create_task(
                queue.submit(TurnRequest(b"", "guild", "channel", "one", "One"))
            )
            pending_two = asyncio.create_task(
                queue.submit(TurnRequest(b"", "guild", "channel", "two", "Two"))
            )
            other_guild = asyncio.create_task(
                queue.submit(TurnRequest(b"", "other", "channel", "three", "Three"))
            )
            await asyncio.sleep(0)

            cancelled = await queue.cancel_guild("guild")
            release_blocker.set()

            self.assertEqual(cancelled, 2)
            self.assertIsNone(await pending_one)
            self.assertIsNone(await pending_two)
            self.assertIsNotNone(await blocker)
            self.assertIsNotNone(await other_guild)
        finally:
            release_blocker.set()
            await queue.stop()

    async def test_pending_audio_is_bounded_and_coalesced_per_speaker(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        processed: list[str] = []

        async def processor(turn: TurnRequest) -> PreparedTurn:
            processed.append(turn.audio.decode())
            if turn.audio == b"active":
                started.set()
                await release.wait()
            text = turn.audio.decode()
            return PreparedTurn(turn, text, text, False)

        queue = RecognitionQueue(processor, pending_per_speaker=2)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(request("user", b"active")))
            await started.wait()
            old = asyncio.create_task(queue.submit(request("user", b"old")))
            newer = asyncio.create_task(queue.submit(request("user", b"newer")))
            newest = asyncio.create_task(queue.submit(request("user", b"newest")))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(active, old, newer, newest)
        finally:
            await queue.stop()

        self.assertEqual(processed, ["active", "newer", "newest"])
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertIsNotNone(results[3])
        self.assertEqual(queue.stats()["recognition_coalesced"], 1)

    async def test_speakers_are_processed_round_robin(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        order: list[str] = []

        async def processor(turn: TurnRequest) -> PreparedTurn:
            text = turn.audio.decode()
            order.append(text)
            if text == "active":
                started.set()
                await release.wait()
            return PreparedTurn(turn, text, text, False)

        queue = RecognitionQueue(processor, pending_per_speaker=2)
        await queue.start()
        try:
            tasks = [asyncio.create_task(queue.submit(request("one", b"active")))]
            await started.wait()
            tasks.extend(
                [
                    asyncio.create_task(queue.submit(request("one", b"one-2"))),
                    asyncio.create_task(queue.submit(request("one", b"one-3"))),
                    asyncio.create_task(queue.submit(request("two", b"two-1"))),
                ]
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(*tasks)
        finally:
            await queue.stop()

        self.assertEqual(order, ["active", "one-2", "two-1", "one-3"])

    async def test_cancel_speaker_removes_only_that_users_pending_audio(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        processed: list[str] = []

        async def processor(turn: TurnRequest) -> PreparedTurn:
            text = turn.audio.decode()
            processed.append(text)
            if text == "active":
                started.set()
                await release.wait()
            return PreparedTurn(turn, text, text, False)

        queue = RecognitionQueue(processor, pending_per_speaker=2)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(request("blocker", b"active")))
            await started.wait()
            stale_one = asyncio.create_task(queue.submit(request("stopped", b"stale-1")))
            stale_two = asyncio.create_task(queue.submit(request("stopped", b"stale-2")))
            other = asyncio.create_task(queue.submit(request("other", b"other")))
            await asyncio.sleep(0)
            cancelled = await queue.cancel_speaker("guild", "stopped")
            release.set()
            results = await asyncio.gather(active, stale_one, stale_two, other)
        finally:
            await queue.stop()

        self.assertEqual(cancelled, 2)
        self.assertEqual(processed, ["active", "other"])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(queue.stats()["recognition_cancelled"], 2)

    async def test_submit_requires_running_queue(self) -> None:
        queue = RecognitionQueue(lambda _: asyncio.sleep(0, result=None))
        with self.assertRaisesRegex(RuntimeError, "not running"):
            await queue.submit(request("one"))


class GenerationQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_pending_speaker_keeps_other_users_requests(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        processed: list[str] = []

        async def processor(turn: PreparedTurn) -> bytes:
            processed.append(turn.accepted_text)
            if turn.accepted_text == "active":
                started.set()
                await release.wait()
            return turn.accepted_text.encode()

        queue = GenerationQueue(processor)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(prepared("blocker", "active", False)))
            await started.wait()
            stale = asyncio.create_task(queue.submit(prepared("expired", "stale", True)))
            other = asyncio.create_task(queue.submit(prepared("other", "other", True)))
            await asyncio.sleep(0)
            cancelled = await queue.cancel_pending_speaker("guild", "expired")
            release.set()
            results = await asyncio.gather(active, stale, other)
        finally:
            await queue.stop()

        self.assertEqual(cancelled, 1)
        self.assertIsNone(results[1])
        self.assertEqual(results[2], b"other")
        self.assertNotIn("stale", processed)

    async def test_cancel_pending_guild_keeps_active_tool_generation_alive(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def processor(turn: PreparedTurn) -> bytes:
            if turn.accepted_text == "active":
                started.set()
                await release.wait()
            return turn.accepted_text.encode()

        queue = GenerationQueue(processor)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(prepared("one", "active", False)))
            await started.wait()
            pending = asyncio.create_task(queue.submit(prepared("two", "pending", False)))
            await asyncio.sleep(0)
            cancelled = await queue.cancel_pending_guild("guild")
            await asyncio.sleep(0)
            self.assertFalse(active.done())
            release.set()
            results = await asyncio.gather(active, pending)
        finally:
            await queue.stop()

        self.assertEqual(cancelled, 1)
        self.assertEqual(results, [b"active", None])

    async def test_wake_interrupts_active_generation_and_coalesces_followups(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        order: list[str] = []

        async def processor(turn: PreparedTurn) -> bytes:
            order.append(turn.accepted_text)
            if turn.accepted_text == "active":
                started.set()
                await release.wait()
            return turn.accepted_text.encode()

        queue = GenerationQueue(processor)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(prepared("one", "active", False)))
            await started.wait()
            # Same-speaker follow-ups while generating are dropped so they can't
            # chase the still-running answer with another LM turn.
            old = asyncio.create_task(queue.submit(prepared("one", "old", False)))
            latest = asyncio.create_task(queue.submit(prepared("one", "latest", False)))
            wake = asyncio.create_task(queue.submit(prepared("two", "wake", True)))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(active, old, latest, wake)
        finally:
            await queue.stop()

        self.assertEqual(order, ["active", "wake"])
        self.assertEqual(results, [None, None, None, b"wake"])
        self.assertEqual(queue.stats()["generation_interrupted"], 1)

    async def test_followup_from_same_speaker_is_dropped_while_generating(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def processor(turn: PreparedTurn) -> bytes:
            if turn.accepted_text == "active":
                started.set()
                await release.wait()
            return turn.accepted_text.encode()

        queue = GenerationQueue(processor)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(prepared("one", "active", False)))
            await started.wait()
            followup = asyncio.create_task(
                queue.submit(prepared("one", "и где ответ", False))
            )
            await asyncio.sleep(0)
            self.assertTrue(followup.done())
            self.assertIsNone(followup.result())
            release.set()
            result = await active
        finally:
            await queue.stop()

        self.assertEqual(result, b"active")

    async def test_stop_cancels_pending_guild_requests(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def processor(turn: PreparedTurn) -> bytes:
            if turn.accepted_text == "active":
                started.set()
                await release.wait()
            return turn.accepted_text.encode()

        queue = GenerationQueue(processor)
        await queue.start()
        try:
            active = asyncio.create_task(queue.submit(prepared("one", "active", False)))
            await started.wait()
            pending = asyncio.create_task(queue.submit(prepared("two", "wake", True)))
            await asyncio.sleep(0)
            await queue.cancel_guild("guild")
            release.set()
            results = await asyncio.gather(active, pending)
        finally:
            await queue.stop()

        self.assertEqual(results, [None, None])


if __name__ == "__main__":
    unittest.main()
