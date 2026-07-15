from __future__ import annotations

import json
import re
from typing import Any

from .dialogue import normalize_phrase, normalize_stop_request
from .public_info import PUBLIC_INFO_TOOLS, WEB_SEARCH_TOOL
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
    "Отвечай только current_identity. current_name — уже готовое русское обращение к говорящему "
    "(preferred name или Discord-ник, переведённый в русскую озвучку): обращайся только так. "
    "Не произноси сырой латинский Discord-ник. "
    "Местоимения «я/меня/мне» в реплике относятся только к current_identity. "
    "Никогда не произноси identity_key. "
    "Сам решай, когда нужен инструмент: вызывай tool call, не пиши имена инструментов в тексте. "
    "На приветствие и обычную болтовню отвечай текстом без инструментов. "
    "Перед вызовом инструмента можешь коротко сказать нейтральную фразу "
    "(«Секунду», «Сейчас посмотрю») — но только вместе с tool call в том же ответе. "
    "Не обещай «посмотреть/найти/проверить», если не вызываешь инструмент. "
    "Если инструмент недоступен из‑за прав — прямо скажи, что нет прав, без обещаний выполнить. "
    "Чтобы закончить голосовой диалог, вызови end_conversation; прощание — только в farewell."
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

TOOL_DENIED_SPEECH: dict[str, str] = {
    "search_web": (
        "У тебя нет прав на поиск в сети. Их могут выдать администраторы сервера."
    ),
}

END_CONVERSATION_FAREWELL = "Хорошо, на связи."
WEB_SEARCH_DENIED_SPEECH = TOOL_DENIED_SPEECH["search_web"]

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


def tool_denied_speech(tool_name: str) -> str:
    return TOOL_DENIED_SPEECH.get(
        tool_name,
        "Эта возможность тебе сейчас недоступна.",
    )


_PROGRESS_PROMISE_MARKERS = (
    "секунд",
    "минутку",
    "момент",
    "подожди",
    "сейчас посм",
    "сейчас провер",
    "сейчас уточн",
    "сейчас найд",
    "сейчас поищ",
    "сейчас глян",
    "сейчас ищ",
    "сейчас смотр",
    "сейчас подбер",
    "ищу в сети",
    "смотрю погоду",
    "уточняю имя",
    "отправляю в чат",
    "запоминаю",
    "забываю",
    "один момент",
    "давай посмотр",
)


def is_incomplete_tool_promise(text: str) -> bool:
    """True when the model only promised to look something up without a tool call."""
    cleaned = re.sub(r"\s+", " ", text.strip()).casefold().rstrip(".!…")
    if not cleaned or len(cleaned) > 100:
        return False
    # Drop a leading nickname: «tochkablsq, сейчас посмотрю»
    cleaned = re.sub(r"^[\w.]{2,32},\s+", "", cleaned, count=1)
    cleaned = cleaned.strip(" ,.!")
    if not cleaned:
        return False
    for speech in TOOL_STATUS_SPEECH.values():
        if cleaned == speech.casefold().rstrip(".!…"):
            return True
    return any(marker in cleaned for marker in _PROGRESS_PROMISE_MARKERS) and len(
        cleaned
    ) <= 60


def select_assistant_tools(
    text: str,
    *,
    web_search_allowed: bool,
) -> list[dict[str, object]]:
    """Return the full tool catalog; the model chooses what to call."""
    _ = (text, web_search_allowed)
    return [
        *USER_MEMORY_TOOLS,
        *PUBLIC_INFO_TOOLS,
        WEB_SEARCH_TOOL,
        SEND_MESSAGE_TO_CHAT_TOOL,
        END_CONVERSATION_TOOL,
    ]


def required_tool_for_turn(text: str, *, web_search_allowed: bool) -> str | None:
    """Force deterministic user-memory operations that must be scoped by user ID."""
    _ = web_search_allowed
    memory_tool = requested_user_memory_tool(text)
    if memory_tool is not None:
        return memory_tool
    if requested_name_lookup(text):
        return "lookup_user_name"
    return None


def build_turn_prompt(
    *,
    roster: list[dict[str, Any]],
    identity_key: str,
    speaker_name: str,
    accepted_text: str,
    web_search_allowed: bool,
) -> str:
    web_search_line = (
        "web_search=allowed"
        if web_search_allowed
        else (
            "web_search=denied — у этого пользователя нет прав на поиск в сети. "
            "Если просят искать в интернете, вызови search_web (backend озвучит отказ) "
            "или сразу скажи, что нет прав. Не обещай, что сейчас поищешь."
        )
    )
    return (
        "[Контекст участников. Не цитируй identity_key и сырые Discord-ники латиницей.]\n"
        f"participants={json.dumps(roster, ensure_ascii=False)}\n"
        f"current_identity={identity_key}\n"
        f"current_name={json.dumps(speaker_name, ensure_ascii=False)}\n"
        "Отвечай только current_identity. Его «я/меня/мне» не переноси на других. "
        "current_name — русское обращение к говорящему; используй его. "
        "В roster поле spoken_name — как произносить других участников.\n"
        f"{web_search_line}\n"
        f"[Реплика]\n{accepted_text}"
    )
