import tempfile
import unittest
from pathlib import Path

from voice_core.user_memory import (
    UserMemoryStore,
    requested_name_lookup,
    requested_preferred_name,
    requested_preferred_name_forget,
    requested_user_memory_tool,
)


class UserMemoryStoreTests(unittest.TestCase):
    def test_name_lookup_questions_are_detected(self) -> None:
        self.assertTrue(requested_name_lookup("Омни, как меня зовут?"))
        self.assertTrue(requested_name_lookup("Как зовут формали бэда?"))
        self.assertTrue(requested_name_lookup("Как его зовут?"))
        self.assertTrue(requested_name_lookup("Какое имя у этого человека?"))
        self.assertFalse(requested_name_lookup("Как тебя зовут?"))
        self.assertFalse(requested_name_lookup("Назови случайное имя"))

    def test_preferred_name_requires_explicit_request(self) -> None:
        self.assertEqual(requested_preferred_name("называй меня Пупсик"), "Пупсик")
        self.assertEqual(
            requested_preferred_name("Зови меня Кэп, пожалуйста"), "Кэп"
        )
        self.assertEqual(
            requested_preferred_name(
                "мобой о мне ты меня можешь называть рыжий"
            ),
            "рыжий",
        )
        self.assertEqual(
            requested_preferred_name("Ты можешь звать меня Рыжий"), "Рыжий"
        )
        self.assertIsNone(requested_preferred_name("Это уже можно разобраться"))
        self.assertIsNone(requested_preferred_name("Меня зовут aim"))

    def test_forget_requires_explicit_request(self) -> None:
        self.assertTrue(requested_preferred_name_forget("Забудь, как меня называть"))
        self.assertFalse(requested_preferred_name_forget("Я забыл задать вопрос"))

    def test_explicit_memory_requests_force_the_matching_tool(self) -> None:
        self.assertEqual(
            requested_user_memory_tool("ты меня можешь называть рыжий"),
            "remember_preferred_name",
        )
        self.assertEqual(
            requested_user_memory_tool("забудь, как меня называть"),
            "forget_preferred_name",
        )
        self.assertIsNone(requested_user_memory_tool("меня зовут Сергей"))

    def test_memory_persists_and_is_isolated_by_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memory.sqlite3")
            store = UserMemoryStore(path)
            store.prepare()
            store.set("guild", "user-1", "preferred_name", "Пупсик")

            reopened = UserMemoryStore(path)
            self.assertEqual(
                reopened.get("guild", "user-1", "preferred_name"), "Пупсик"
            )
            self.assertIsNone(reopened.get("guild", "user-2", "preferred_name"))

    def test_memory_can_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            store.prepare()
            store.set("guild", "user", "preferred_name", "Пупсик")

            self.assertTrue(store.delete("guild", "user", "preferred_name"))
            self.assertIsNone(store.get("guild", "user", "preferred_name"))

    def test_fuzzy_name_lookup_matches_transliterated_discord_nickname(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            store.prepare()
            store.set("guild", "user-1", "preferred_name", "Пупсик")
            store.set("guild", "user-1", "discord_display_name", ".formallybad")
            store.set("guild", "user-2", "preferred_name", "Александр")
            store.set("guild", "user-2", "discord_display_name", "aim")

            match = store.find_best_name("guild", "формали бэд")

            self.assertIsNotNone(match)
            self.assertEqual(match.preferred_name, "Пупсик")
            self.assertEqual(match.user_id, "user-1")
            self.assertEqual(match.display_name, ".formallybad")
            self.assertEqual(match.matched_value, ".formallybad")
            self.assertGreater(match.confidence, 0.75)

    def test_fuzzy_name_lookup_can_exclude_current_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            store.prepare()
            store.set("guild", "current", "preferred_name", "Пупсик")
            store.set("guild", "current", "discord_display_name", "formalbad")
            store.set("guild", "other", "preferred_name", "Кэп")
            store.set("guild", "other", "discord_display_name", "formallybad")

            match = store.find_best_name(
                "guild", "формали бэд", exclude_user_id="current"
            )

            self.assertIsNotNone(match)
            self.assertEqual(match.user_id, "other")
            self.assertEqual(match.preferred_name, "Кэп")

    def test_fuzzy_name_lookup_is_isolated_by_guild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserMemoryStore(str(Path(directory) / "memory.sqlite3"))
            store.prepare()
            store.set("other-guild", "user", "preferred_name", "Секрет")

            self.assertIsNone(store.find_best_name("guild", "Секрет"))


if __name__ == "__main__":
    unittest.main()
