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
        self.assertIn("не пиши в тексте имена инструментов", DEFAULT_SYSTEM_PROMPT.casefold())
        self.assertNotIn("get_current_weather", DEFAULT_SYSTEM_PROMPT)
        self.assertLess(len(DEFAULT_SYSTEM_PROMPT), 1_600)

    def test_end_conversation_detection_covers_free_form(self) -> None:
        self.assertTrue(requested_end_conversation("давай на этом закончим"))
        self.assertTrue(requested_end_conversation("Омни, больше не слушай"))
        self.assertTrue(requested_end_conversation("хватит"))
        self.assertFalse(requested_end_conversation("какая погода в Москве"))

    def test_required_tool_forces_end_conversation(self) -> None:
        self.assertEqual(
            required_tool_for_turn("давай завершим", web_search_allowed=False),
            "end_conversation",
        )

    def test_select_tools_narrows_on_weather_intent(self) -> None:
        tools = select_assistant_tools(
            "какая сейчас погода в Казани",
            web_search_allowed=True,
        )
        names = {
            str((tool.get("function") or {}).get("name") or "") for tool in tools
        }
        self.assertIn("get_current_weather", names)
        self.assertIn("end_conversation", names)
        self.assertNotIn("search_web", names)
        self.assertNotIn("get_random_joke", names)
        self.assertNotIn("lookup_user_name", names)

    def test_select_tools_omits_name_lookup_for_smalltalk(self) -> None:
        tools = select_assistant_tools("Как твои дела?", web_search_allowed=False)
        names = {
            str((tool.get("function") or {}).get("name") or "") for tool in tools
        }
        self.assertNotIn("lookup_user_name", names)
        self.assertNotIn("remember_preferred_name", names)

    def test_select_tools_includes_name_lookup_only_for_name_questions(self) -> None:
        tools = select_assistant_tools("как меня зовут", web_search_allowed=False)
        names = {
            str((tool.get("function") or {}).get("name") or "") for tool in tools
        }
        self.assertIn("lookup_user_name", names)

    def test_tool_status_speech_is_neutral(self) -> None:
        self.assertEqual(tool_status_speech("search_web"), "Ищу в сети.")
        self.assertIsNone(tool_status_speech("unknown_tool"))

    def test_end_conversation_tool_accepts_optional_farewell(self) -> None:
        parameters = END_CONVERSATION_TOOL["function"]["parameters"]
        self.assertIn("farewell", parameters["properties"])


if __name__ == "__main__":
    unittest.main()
