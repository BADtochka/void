from __future__ import annotations

import json
import re
from typing import Any

from .dialogue import normalize_phrase, normalize_stop_request
from .public_info import (
    PUBLIC_INFO_TOOLS,
    WEB_SEARCH_TOOL,
    requested_public_tool,
)
from .user_memory import (
    USER_MEMORY_TOOLS,
    requested_name_lookup,
    requested_user_memory_tool,
)

DEFAULT_SYSTEM_PROMPT = (
    "Ты Омни — локальный голосовой собеседник в Discord. Отвечай по-русски, кратко и естественно. "
    "Обычно 1–3 коротких предложения. Без Markdown, списков и блоков кода. "
    "Запрещены эмодзи и символы вроде 🚀🙏. Не ставь метки вроде [Омни]. "
    "Не выводи think, reasoning, analysis и внутренние рассуждения. "
    "Ответ озвучивается синтезатором — пиши только то, что можно произнести вслух. "
    "В запросе есть roster участников и current_identity: разные identity_key — разные люди. "
    "Отвечай только current_identity; местоимения «я/меня/мне» в реплике относятся только к нему. "
    "Обращайся по preferred_name или display_name; никогда не произноси identity_key. "
    "Нужные факты и действия выполняй через инструменты, не выдумывай свежие данные. "
    "Никогда не пиши в тексте имена инструментов (end_conversation, lookup_user_name и т.п.) — "
    "только вызывай их. Не давай инструкций интерфейса: «нажми», «напиши продолжить», «если нужно». "
    "Перед вызовом инструмента можешь коротко сказать нейтральную фразу "
    "(«Секунду», «Сейчас посмотрю») — она озвучится сразу. "
    "Чтобы закончить голосовой диалог, вызови end_conversation, не описывая это словами."
)

TOOL_STATUS_SPEECH: dict[str, str] = {
    "get_current_weather": "Секунду, смотрю погоду.",
    "lookup_topic": "Сейчас посмотрю.",
    "get_random_joke": "Сейчас подберу.",
    "search_web": "Ищу в сети.",
    "remember_preferred_name": "Хорошо, запоминаю.",
    "forget_preferred_name": "Хорошо, забываю.",
    "lookup_user_name": "Секунду, уточняю имя.",
    "send_message_to_chat": "Отправляю в чат.",
    "end_conversation": "Хорошо, на связи.",
}

END_CONVERSATION_FAREWELL = "Хорошо, на связи."

END_CONVERSATION_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "end_conversation",
        "description": (
            "Заверши голосовой диалог и перестань слушать follow-up. "
            "Вызывай при явной просьбе закончить, замолчать, отключиться "
            "или перестать слушать. Не проси специальное слово. "
            "Прощание передай только в аргумент farewell, не в тексте ответа. "
            "Не вызывай для остановки музыки, программы или другой задачи."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "farewell": {
                    "type": "string",
                    "description": "Короткая прощальная фраза для озвучки, например «Хорошо, пока».",
                }
            },
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TO_CHAT_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "send_message_to_chat",
        "description": (
            "Отправь текст в Discord-чат по просьбе пользователя. "
            "scope=full_response — весь текущий ответ после генерации; "
            "scope=selection — только переданный content; "
            "scope=previous_response — предыдущий ответ ассистента из истории."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["full_response", "selection", "previous_response"],
                    "description": "Какой текст отправить в Discord-чат.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Полный текст выбранной части только для scope=selection."
                    ),
                },
            },
            "required": ["scope"],
            "additionalProperties": False,
        },
    },
}

_CORE_TOOLS: list[dict[str, object]] = [
    *USER_MEMORY_TOOLS,
    SEND_MESSAGE_TO_CHAT_TOOL,
    END_CONVERSATION_TOOL,
]

_PUBLIC_TOOLS_BY_NAME = {
    str((tool.get("function") or {}).get("name") or ""): tool
    for tool in (*PUBLIC_INFO_TOOLS, WEB_SEARCH_TOOL)
}

_END_CONVERSATION_MARKERS = (
    "на этом закончим",
    "на этом все",
    "на этом всё",
    "закончим разговор",
    "завершим разговор",
    "закончим диалог",
    "завершим диалог",
    "закончи диалог",
    "заверши диалог",
    "закрой диалог",
    "больше не слушай",
    "можешь не слушать",
    "перестань слушать",
    "не слушай меня",
    "можешь отключаться",
    "можешь отключиться",
    "давай закончим",
    "давай завершим",
    "хватит слушать",
    "хватит болтать",
    "все хватит",
    "всё хватит",
)


def requested_chat_delivery(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    action_requested = any(
        marker in normalized
        for marker in (
            "отправь",
            "отправить",
            "скинь",
            "скинуть",
            "продублируй",
            "продублировать",
            "дублируй",
            "напиши",
        )
    )
    destination_requested = any(
        marker in normalized for marker in ("в чат", "в текстовый чат", "текстом")
    )
    return action_requested and destination_requested


def requested_end_conversation(text: str) -> bool:
    normalized = normalize_stop_request(text)
    if not normalized:
        return False
    compact = normalize_phrase(text)
    if any(marker in compact for marker in _END_CONVERSATION_MARKERS):
        return True
    tokens = set(normalized.split())
    end_tokens = {
        "стоп",
        "хватит",
        "замолчи",
        "молчи",
        "отключайся",
        "отключись",
        "остановись",
        "пока",
    }
    if tokens & end_tokens and len(tokens) <= 4:
        return True
    return bool(
        re.search(
            r"\b(?:закончи|закончить|заканчивай|заверши|завершить|прекрати|прекращай)\b",
            normalized,
        )
    )


def tool_status_speech(tool_name: str) -> str | None:
    message = TOOL_STATUS_SPEECH.get(tool_name)
    return message.strip() if message else None


def select_assistant_tools(
    text: str,
    *,
    web_search_allowed: bool,
) -> list[dict[str, object]]:
    """Attach a compact tool set for the turn to reduce LM context size."""
    tools: list[dict[str, object]] = list(_CORE_TOOLS)
    selected_names = {
        str((tool.get("function") or {}).get("name") or "") for tool in tools
    }

    def add_tool(name: str) -> None:
        if name in selected_names:
            return
        if name == "search_web" and not web_search_allowed:
            return
        tool = _PUBLIC_TOOLS_BY_NAME.get(name)
        if tool is None:
            return
        tools.append(tool)
        selected_names.add(name)

    public = requested_public_tool(text)
    if public:
        add_tool(public)
        return tools

    # Soft intents: keep encyclopedic/weather/joke tools, add web only when allowed.
    for name in ("get_current_weather", "lookup_topic", "get_random_joke"):
        add_tool(name)
    if web_search_allowed:
        add_tool("search_web")
    return tools


def required_tool_for_turn(text: str, *, web_search_allowed: bool) -> str | None:
    if requested_end_conversation(text):
        return "end_conversation"
    if requested_name_lookup(text):
        return "lookup_user_name"
    if requested_chat_delivery(text):
        return "send_message_to_chat"
    memory_tool = requested_user_memory_tool(text)
    if memory_tool:
        return memory_tool
    public = requested_public_tool(text)
    if public == "search_web" and not web_search_allowed:
        return None
    return public


def build_turn_prompt(
    *,
    roster: list[dict[str, Any]],
    identity_key: str,
    speaker_name: str,
    accepted_text: str,
    web_search_allowed: bool,
) -> str:
    return (
        "[Контекст участников. Не цитируй identity_key.]\n"
        f"participants={json.dumps(roster, ensure_ascii=False)}\n"
        f"current_identity={identity_key}\n"
        f"current_name={json.dumps(speaker_name, ensure_ascii=False)}\n"
        "Отвечай только current_identity. Его «я/меня/мне» не переноси на других.\n"
        f"web_search={'allowed' if web_search_allowed else 'denied'}\n"
        f"[Реплика]\n{accepted_text}"
    )
