from __future__ import annotations

import html
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("voice-core.public-info")


def requested_public_tool(text: str) -> str | None:
    normalized = " ".join(text.casefold().split())
    if requested_web_search(text):
        return "search_web"
    if any(marker in normalized for marker in ("анекдот", "пошути", "шутку", "шутка")):
        return "get_random_joke"
    if any(
        marker in normalized
        for marker in ("погода", "температура на улице", "сколько градусов")
    ):
        return "get_current_weather"
    if re.search(r"\b(?:расскажи|расскажите)\s+(?:мне\s+)?(?:про|о|об)\s+\S", normalized):
        return "lookup_topic"
    if re.search(r"\b(?:кто\s+так(?:ой|ая|ое)|что\s+такое)\b", normalized):
        return "lookup_topic"
    return None


def requested_web_search(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    if any(marker in normalized for marker in ("загугли", "погугли")):
        return True
    action = any(
        marker in normalized
        for marker in ("найди", "поищи", "поиск", "поищем", "посмотри")
    )
    source = any(
        marker in normalized
        for marker in ("в интернете", "в сети", "в вебе", "web", "онлайн")
    )
    return action and source


WEB_SEARCH_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Найди актуальную информацию в интернете через поисковую выдачу. Используй только "
            "когда пользователь просит поискать, проверить свежие сведения, новости, текущие "
            "версии или другую информацию, которой может не быть во внутренних знаниях. "
            "Кратко перескажи результаты и явно отмечай неопределённость поисковых сниппетов."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Короткий и точный поисковый запрос без вводных слов.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Количество результатов; обычно достаточно 5.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


PUBLIC_INFO_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Узнай текущую погоду в указанном городе. Обязательно вызывай для вопросов "
                "о погоде сейчас, температуре, осадках или ветре в конкретном месте."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, при необходимости со страной или регионом.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_topic",
            "description": (
                "Найди краткую фактическую справку по теме в русской Википедии. Используй, "
                "когда пользователь просит рассказать о человеке, месте, событии, термине, "
                "произведении или другой энциклопедической теме. Не вызывай для выдуманной истории."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Тема без вводных слов вроде 'расскажи про'.",
                    },
                    "english_query": {
                        "type": "string",
                        "description": (
                            "Короткий перевод темы на английский для резервного источника. "
                            "Для известных людей добавь профессию, чтобы избежать страницы неоднозначности."
                        ),
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_joke",
            "description": (
                "Получи безопасный случайный анекдот. Вызывай, когда пользователь просит "
                "рассказать анекдот или шутку. Если пользователь указал тему, обязательно передай "
                "её в topic и переведи в короткий английский search_query для поиска шутки по теме. "
                "Если темы нет, не выдумывай её. Результат может быть на английском: переведи или "
                "естественно адаптируй его на русский, сохранив смысл шутки."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["any", "programming"],
                        "description": "Тематика шутки.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Тема, которую явно указал пользователь, на русском языке.",
                    },
                    "search_query": {
                        "type": "string",
                        "description": (
                            "Одно-два ключевых слова темы на английском для JokeAPI contains. "
                            "Передавай только вместе с topic."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


WEATHER_CODES = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозевый туман",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "сильная ледяная морось",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "слабый ледяной дождь",
    67: "сильный ледяной дождь",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "слабый ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "слабый снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с сильным градом",
}


def _required_text(arguments: dict[str, object], key: str, max_length: int = 160) -> str:
    value = " ".join(str(arguments.get(key) or "").split())
    if not value:
        raise ValueError(f"{key} is required")
    if len(value) > max_length:
        raise ValueError(f"{key} is too long")
    return value


def _plain_excerpt(value: object) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", str(value or "")))
    return " ".join(text.split())


def _missing_topic_joke(
    topic: str, search_query: str, reason: object
) -> dict[str, Any]:
    return {
        "found": False,
        "topic": topic,
        "search_query": search_query,
        "reason": str(reason or "Подходящая шутка не найдена."),
        "response_instruction": (
            "Кратко скажи, что в источнике не нашлось анекдота по указанной теме. "
            "Не подменяй его несвязанной шуткой."
        ),
    }


def _duckduckgo_result_url(value: str) -> str:
    url = html.unescape(value).strip()
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com"):
        redirected = parse_qs(parsed.query).get("uddg", [])
        if redirected:
            url = unquote(redirected[0])
            parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


class PublicInformationService:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=4.0),
            headers={
                "User-Agent": "OmniDiscordVoiceAssistant/0.1 (local personal assistant)"
            },
            follow_redirects=True,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, tool_name: str, arguments: dict[str, object]) -> str:
        logger.info("Public API tool started tool=%s arguments=%s", tool_name, arguments)
        if tool_name == "get_current_weather":
            result = await self._weather(_required_text(arguments, "city"))
        elif tool_name == "lookup_topic":
            result = await self._topic(
                _required_text(arguments, "topic"),
                " ".join(str(arguments.get("english_query") or "").split()),
            )
        elif tool_name == "get_random_joke":
            topic = " ".join(str(arguments.get("topic") or "").split())
            search_query = " ".join(str(arguments.get("search_query") or "").split())
            if search_query and not topic:
                raise ValueError("search_query requires topic")
            result = await self._joke(
                str(arguments.get("category") or "any"),
                topic,
                search_query,
            )
        elif tool_name == "search_web":
            max_results = int(arguments.get("max_results") or 5)
            result = await self._web_search(
                _required_text(arguments, "query", max_length=240),
                min(max(max_results, 1), 8),
            )
        else:
            raise ValueError(f"unknown public information tool: {tool_name}")
        logger.info("Public API tool finished tool=%s", tool_name)
        return json.dumps(result, ensure_ascii=False)

    async def _web_search(self, query: str, max_results: int) -> dict[str, Any]:
        response = await self._client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )
        response.raise_for_status()
        document = BeautifulSoup(response.text, "html.parser")
        results: list[dict[str, str]] = []
        for item in document.select(".result"):
            link = item.select_one(".result__a")
            if link is None:
                continue
            title = _plain_excerpt(link.get_text(" ", strip=True))
            url = _duckduckgo_result_url(str(link.get("href") or ""))
            if not title or not url:
                continue
            snippet_node = item.select_one(".result__snippet")
            snippet = (
                _plain_excerpt(snippet_node.get_text(" ", strip=True))
                if snippet_node is not None
                else ""
            )
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
        return {"query": query, "found": bool(results), "results": results}

    async def _weather(self, city: str) -> dict[str, Any]:
        geocoding = await self._client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "ru", "format": "json"},
        )
        geocoding.raise_for_status()
        locations = geocoding.json().get("results") or []
        if not locations:
            return {"found": False, "city_query": city}

        location = locations[0]
        forecast = await self._client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m,wind_gusts_10m"
                ),
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        payload = forecast.json()
        current = payload.get("current") or {}
        units = payload.get("current_units") or {}
        code = int(current.get("weather_code", -1))
        return {
            "found": True,
            "location": {
                "city": location.get("name"),
                "region": location.get("admin1"),
                "country": location.get("country"),
                "timezone": location.get("timezone"),
            },
            "observed_at": current.get("time"),
            "conditions": WEATHER_CODES.get(code, f"код погоды {code}"),
            "temperature": current.get("temperature_2m"),
            "temperature_unit": units.get("temperature_2m", "°C"),
            "feels_like": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
            "wind_gusts": current.get("wind_gusts_10m"),
            "source": "Open-Meteo",
        }

    async def _topic(self, topic: str, english_query: str = "") -> dict[str, Any]:
        try:
            search = await self._client.get(
                "https://api.wikimedia.org/core/v1/wikipedia/ru/search/page",
                params={"q": topic, "limit": 1},
            )
            search.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("Wikipedia search unavailable topic=%r error=%s", topic, error)
            return await self._duckduckgo_topic(topic, english_query)
        pages = search.json().get("pages") or []
        if not pages:
            return await self._duckduckgo_topic(topic, english_query)

        page = pages[0]
        title = str(page.get("title") or topic)
        extract = _plain_excerpt(page.get("excerpt"))
        description = _plain_excerpt(page.get("description"))
        try:
            summary_response = await self._client.get(
                f"https://ru.wikipedia.org/api/rest_v1/page/summary/{quote(str(page['key']), safe='')}",
            )
            summary_response.raise_for_status()
            summary = _plain_excerpt(summary_response.json().get("extract"))
        except (httpx.HTTPError, KeyError, ValueError):
            logger.warning("Wikipedia summary unavailable topic=%r", topic)
            summary = ""

        return {
            "found": True,
            "topic_query": topic,
            "title": title,
            "description": description,
            "summary": (summary or extract)[:4_000],
            "url": f"https://ru.wikipedia.org/wiki/{quote(str(page.get('key') or title))}",
            "source": "Википедия",
        }

    async def _duckduckgo_topic(
        self, topic: str, english_query: str
    ) -> dict[str, Any]:
        query = english_query or topic
        response = await self._client.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
        )
        response.raise_for_status()
        payload = response.json()
        summary = _plain_excerpt(payload.get("AbstractText"))
        if not summary:
            return {"found": False, "topic_query": topic}
        return {
            "found": True,
            "topic_query": topic,
            "title": _plain_excerpt(payload.get("Heading")) or topic,
            "description": "",
            "summary": summary[:4_000],
            "url": payload.get("AbstractURL"),
            "source": payload.get("AbstractSource") or "DuckDuckGo",
            "response_instruction": "Ответь по-русски; переведи справку, если она на английском.",
        }

    async def _joke(
        self, category: str, topic: str = "", search_query: str = ""
    ) -> dict[str, Any]:
        api_category = "Programming" if category.casefold() == "programming" else "Any"
        params = {
            "safe-mode": "",
            "blacklistFlags": "nsfw,religious,political,racist,sexist,explicit",
            "lang": "en",
        }
        if topic:
            if not search_query:
                return {
                    "found": False,
                    "topic": topic,
                    "reason": "Для тематического поиска нужен английский search_query.",
                    "response_instruction": (
                        "Кратко скажи, что не удалось подобрать анекдот по этой теме."
                    ),
                }
            params["contains"] = search_query
        response = await self._client.get(
            f"https://v2.jokeapi.dev/joke/{api_category}",
            params=params,
        )
        if response.status_code == 400 and topic:
            try:
                error_payload = response.json()
            except ValueError:
                pass
            else:
                if error_payload.get("error") and error_payload.get("code") == 106:
                    logger.info(
                        "JokeAPI has no topic match topic=%r search_query=%r",
                        topic,
                        search_query,
                    )
                    return _missing_topic_joke(
                        topic, search_query, error_payload.get("message")
                    )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            if topic:
                return _missing_topic_joke(
                    topic, search_query, payload.get("message")
                )
            raise RuntimeError(str(payload.get("message") or "JokeAPI returned an error"))
        text = (
            str(payload.get("joke") or "")
            if payload.get("type") == "single"
            else f"{payload.get('setup', '')}\n{payload.get('delivery', '')}"
        ).strip()
        if not text:
            raise RuntimeError("JokeAPI returned an empty joke")
        return {
            "found": True,
            "joke": text,
            "requested_topic": topic or None,
            "search_query": search_query or None,
            "language": payload.get("lang", "en"),
            "category": payload.get("category"),
            "source": "JokeAPI",
            "response_instruction": "Переведи или адаптируй шутку на русский и расскажи без пояснений.",
        }
