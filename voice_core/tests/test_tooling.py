import unittest

from voice_core.tooling import (
    DEFAULT_SYSTEM_PROMPT,
    END_CONVERSATION_TOOL,
    requested_end_conversation,
    required_tool_for_turn,
    select_assistant_tools,
    tool_status_speech,
)


class ToolingTests(unittest.TestCase):
    def test_system_prompt_stays_persona_focused(self) -> None:
        self.assertIn("Омни", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("current_identity", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("сам решай", DEFAULT_SYSTEM_PROMPT.casefold())
        self.assertIn("русскую озвучку", DEFAULT_SYSTEM_PROMPT.casefold())
        self.assertIn("нет прав", DEFAULT_SYSTEM_PROMPT.casefold())
        self.assertNotIn("get_current_weather", DEFAULT_SYSTEM_PROMPT)
        self.assertLess(len(DEFAULT_SYSTEM_PROMPT), 1_800)

    def test_end_conversation_detection_covers_free_form(self) -> None:
        self.assertTrue(requested_end_conversation("давай на этом закончим"))
        self.assertTrue(requested_end_conversation("Омни, больше не слушай"))
        self.assertTrue(requested_end_conversation("хватит"))
        self.assertFalse(requested_end_conversation("какая погода в Москве"))

    def test_required_tool_is_never_forced(self) -> None:
        self.assertIsNone(
            required_tool_for_turn("давай завершим", web_search_allowed=False)
        )
        self.assertIsNone(
            required_tool_for_turn("как меня зовут", web_search_allowed=False)
        )
        self.assertIsNone(
            required_tool_for_turn("какая погода", web_search_allowed=True)
        )

    def test_select_tools_returns_full_catalog(self) -> None:
        tools = select_assistant_tools(
            "какая сейчас погода в Казани",
            web_search_allowed=True,
        )
        names = {
            str((tool.get("function") or {}).get("name") or "") for tool in tools
        }
        self.assertEqual(
            names,
            {
                "remember_preferred_name",
                "forget_preferred_name",
                "lookup_user_name",
                "get_current_weather",
                "lookup_topic",
                "get_random_joke",
                "search_web",
                "send_message_to_chat",
                "end_conversation",
            },
        )

    def test_select_tools_always_includes_search_web(self) -> None:
        tools = select_assistant_tools("Как твои дела?", web_search_allowed=False)
        names = {
            str((tool.get("function") or {}).get("name") or "") for tool in tools
        }
        self.assertIn("search_web", names)
        self.assertIn("end_conversation", names)
        self.assertIn("lookup_user_name", names)
        self.assertIn("get_current_weather", names)

    def test_tool_status_speech_is_neutral(self) -> None:
        self.assertEqual(tool_status_speech("search_web"), "Ищу в сети.")
        self.assertIsNone(tool_status_speech("unknown_tool"))

    def test_incomplete_tool_promise_detection(self) -> None:
        from voice_core.tooling import is_incomplete_tool_promise

        self.assertTrue(is_incomplete_tool_promise("tochkablsq, сейчас посмотрю."))
        self.assertTrue(is_incomplete_tool_promise("Секунду."))
        self.assertTrue(is_incomplete_tool_promise("Ищу в сети."))
        self.assertFalse(is_incomplete_tool_promise("В Чебоксарах около пятисот тысяч человек."))
        self.assertFalse(is_incomplete_tool_promise("Привет, как дела?"))

    def test_end_conversation_tool_accepts_optional_farewell(self) -> None:
        parameters = END_CONVERSATION_TOOL["function"]["parameters"]
        self.assertIn("farewell", parameters["properties"])


if __name__ == "__main__":
    unittest.main()
