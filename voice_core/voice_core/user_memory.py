from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


_PREFERRED_NAME_PATTERNS = (
    re.compile(r"\b(?:называй|зови)\s+меня\s+(.+)$", re.IGNORECASE),
    re.compile(
        r"\b(?:ты\s+)?меня\s+(?:можешь\s+)?(?:называть|звать)\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ты\s+)?можешь\s+(?:называть|звать)\s+меня\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(r"\bобращайся\s+ко\s+мне\s+(?:как\s+)?(.+)$", re.IGNORECASE),
    re.compile(
        r"\bзапомни[,:]?\s+(?:что\s+)?(?:меня\s+(?:зовут|называть)|мо[её]\s+имя)\s+(.+)$",
        re.IGNORECASE,
    ),
)


def requested_preferred_name(text: str) -> str | None:
    for pattern in _PREFERRED_NAME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group(1)).strip(" \t\r\n\"'.,!?;:")
        name = re.sub(r"\s*,?\s*пожалуйста$", "", name, flags=re.IGNORECASE).strip()
        if 0 < len(name) <= 80:
            return name
    return None


def requested_preferred_name_forget(text: str) -> bool:
    normalized = re.sub(r"[^\w]+", " ", text.casefold()).strip()
    return "забудь" in normalized and any(
        marker in normalized for marker in ("имя", "называть", "обращаться", "обращение")
    )


def requested_user_memory_tool(text: str) -> str | None:
    if requested_preferred_name(text) is not None:
        return "remember_preferred_name"
    if requested_preferred_name_forget(text):
        return "forget_preferred_name"
    return None


def requested_name_lookup(text: str) -> bool:
    normalized = re.sub(r"[^\w]+", " ", text.casefold()).strip()
    return bool(
        re.search(r"\bкак\s+меня\s+зовут\b", normalized)
        or re.search(r"\bкак\s+зовут\b", normalized)
        or re.search(r"\bкак\s+(?:его|её|ее|их)\s+зовут\b", normalized)
        or re.search(r"\bкакое\s+имя\s+у\b", normalized)
        or re.search(r"\bчь[её]\s+имя\b", normalized)
        or re.search(r"\bимя\s+у\b", normalized)
    )


_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
        "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k",
        "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
)


def _name_forms(value: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)
    if not normalized:
        return ()
    transliterated = normalized.translate(_CYRILLIC_TO_LATIN)
    return tuple(dict.fromkeys((normalized, transliterated)))


def _name_similarity(query: str, candidate: str) -> float:
    query_forms = _name_forms(query)
    candidate_forms = _name_forms(candidate)
    if not query_forms or not candidate_forms:
        return 0.0
    return max(
        SequenceMatcher(None, query_form, candidate_form).ratio()
        for query_form in query_forms
        for candidate_form in candidate_forms
    )


@dataclass(frozen=True)
class UserNameMatch:
    user_id: str
    preferred_name: str | None
    display_name: str | None
    matched_value: str
    confidence: float


@dataclass(frozen=True)
class WebSearchAccessEntry:
    user_id: str
    display_name: str | None


class UserMemoryStore:
    def __init__(self, database_path: str) -> None:
        self._path = Path(database_path)

    def prepare(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    guild_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (guild_id, user_id, memory_key)
                )
                """
            )

    def get(self, guild_id: str, user_id: str, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT memory_value
                FROM user_memories
                WHERE guild_id = ? AND user_id = ? AND memory_key = ?
                """,
                (guild_id, user_id, key),
            ).fetchone()
        return str(row[0]) if row else None

    def set(self, guild_id: str, user_id: str, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memories (guild_id, user_id, memory_key, memory_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (guild_id, user_id, memory_key) DO UPDATE SET
                    memory_value = excluded.memory_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (guild_id, user_id, key, value),
            )

    def delete(self, guild_id: str, user_id: str, key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM user_memories
                WHERE guild_id = ? AND user_id = ? AND memory_key = ?
                """,
                (guild_id, user_id, key),
            )
        return cursor.rowcount > 0

    def grant_web_search_access(
        self, guild_id: str, user_id: str, display_name: str
    ) -> None:
        self.set(guild_id, user_id, "discord_display_name", display_name)
        self.set(guild_id, user_id, "web_search_access", "true")

    def revoke_web_search_access(self, guild_id: str, user_id: str) -> bool:
        return self.delete(guild_id, user_id, "web_search_access")

    def has_web_search_access(self, guild_id: str, user_id: str) -> bool:
        return self.get(guild_id, user_id, "web_search_access") == "true"

    def list_web_search_access(self, guild_id: str) -> list[WebSearchAccessEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT access.user_id, display.memory_value
                FROM user_memories AS access
                LEFT JOIN user_memories AS display
                  ON display.guild_id = access.guild_id
                 AND display.user_id = access.user_id
                 AND display.memory_key = 'discord_display_name'
                WHERE access.guild_id = ?
                  AND access.memory_key = 'web_search_access'
                  AND access.memory_value = 'true'
                ORDER BY COALESCE(display.memory_value, access.user_id) COLLATE NOCASE
                """,
                (guild_id,),
            ).fetchall()
        return [
            WebSearchAccessEntry(
                user_id=str(user_id),
                display_name=str(display_name) if display_name else None,
            )
            for user_id, display_name in rows
        ]

    def find_best_name(
        self, guild_id: str, query: str, *, exclude_user_id: str | None = None
    ) -> UserNameMatch | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    user_id,
                    MAX(CASE WHEN memory_key = 'preferred_name' THEN memory_value END),
                    MAX(CASE WHEN memory_key = 'discord_display_name' THEN memory_value END)
                FROM user_memories
                WHERE guild_id = ?
                  AND memory_key IN ('preferred_name', 'discord_display_name')
                GROUP BY user_id
                """,
                (guild_id,),
            ).fetchall()

        best: UserNameMatch | None = None
        for user_id, preferred_name, display_name in rows:
            if exclude_user_id is not None and str(user_id) == exclude_user_id:
                continue
            for candidate in (preferred_name, display_name):
                if not candidate:
                    continue
                confidence = _name_similarity(query, str(candidate))
                if best is None or confidence > best.confidence:
                    best = UserNameMatch(
                        str(user_id),
                        str(preferred_name) if preferred_name else None,
                        str(display_name) if display_name else None,
                        str(candidate),
                        confidence,
                    )
        return best

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


USER_MEMORY_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "remember_preferred_name",
            "description": (
                "Сохрани имя или обращение, которым текущий говорящий явно попросил называть его. "
                "Вызывай только при явной просьбе пользователя запомнить или использовать это обращение."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_name": {
                        "type": "string",
                        "description": "Имя или обращение без кавычек и пояснений.",
                    }
                },
                "required": ["preferred_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_preferred_name",
            "description": (
                "Удали сохранённое обращение текущего говорящего, когда он явно просит забыть его."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_user_name",
            "description": (
                "Обязательно вызови этот инструмент при вопросе, как зовут текущего или другого "
                "участника. Всегда передавай subject: значение current_user только для вопросов "
                "'как меня зовут', а для другого человека — услышанное имя или ник, даже с ошибкой "
                "распознавания. Никогда не подставляй current_user, если спрашивают о другом "
                "участнике. Backend сам найдёт максимально похожую запись в базе. В ответе строго "
                "учитывай scope и response_instruction: при other_user говори о найденном человеке "
                "в третьем лице и не отвечай 'тебя зовут'. Не угадывай имя по истории."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": (
                            "Строго current_user для вопроса о самом говорящем; иначе неточно "
                            "услышанное имя или ник другого участника из вопроса."
                        ),
                    }
                },
                "required": ["subject"],
                "additionalProperties": False,
            },
        },
    },
]
