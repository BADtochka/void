import json
import unittest

import httpx

from voice_core.public_info import (
    PUBLIC_INFO_TOOLS,
    WEB_SEARCH_TOOL,
    PublicInformationService,
    requested_public_tool,
    requested_web_search,
)


class PublicInformationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_weather_resolves_city_and_conditions(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "geocoding-api.open-meteo.com":
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "name": "Москва",
                                "admin1": "Москва",
                                "country": "Россия",
                                "timezone": "Europe/Moscow",
                                "latitude": 55.75,
                                "longitude": 37.62,
                            }
                        ]
                    },
                )
            self.assertEqual(request.url.host, "api.open-meteo.com")
            return httpx.Response(
                200,
                json={
                    "current": {
                        "time": "2026-07-14T15:00",
                        "temperature_2m": 24.5,
                        "apparent_temperature": 25.0,
                        "relative_humidity_2m": 60,
                        "precipitation": 0.0,
                        "weather_code": 2,
                        "wind_speed_10m": 8.0,
                        "wind_gusts_10m": 14.0,
                    },
                    "current_units": {
                        "temperature_2m": "°C",
                        "wind_speed_10m": "km/h",
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute("get_current_weather", {"city": "Москва"})
            )

        self.assertTrue(result["found"])
        self.assertEqual(result["location"]["city"], "Москва")
        self.assertEqual(result["conditions"], "переменная облачность")
        self.assertEqual(result["temperature"], 24.5)

    async def test_unknown_city_is_reported_without_forecast_request(self) -> None:
        requests = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute("get_current_weather", {"city": "Неттакогогорода"})
            )

        self.assertEqual(result, {"found": False, "city_query": "Неттакогогорода"})
        self.assertEqual(requests, 1)

    async def test_topic_uses_wikipedia_search_and_summary(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/search/page"):
                return httpx.Response(
                    200,
                    json={
                        "pages": [
                            {
                                "key": "Альберт_Эйнштейн",
                                "title": "Альберт Эйнштейн",
                                "description": "физик-теоретик",
                                "excerpt": "Немецкий физик",
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={"extract": "Альберт Эйнштейн создал общую теорию относительности."},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute("lookup_topic", {"topic": "Эйнштейн"})
            )

        self.assertTrue(result["found"])
        self.assertEqual(result["title"], "Альберт Эйнштейн")
        self.assertIn("теорию относительности", result["summary"])

    async def test_topic_falls_back_when_wikipedia_rejects_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "api.wikimedia.org":
                return httpx.Response(403, text="robot policy")
            self.assertEqual(request.url.host, "api.duckduckgo.com")
            self.assertEqual(request.url.params["q"], "quantum physics")
            return httpx.Response(
                200,
                json={
                    "Heading": "Quantum mechanics",
                    "AbstractText": "Quantum mechanics describes matter at atomic scales.",
                    "AbstractURL": "https://en.wikipedia.org/wiki/Quantum_mechanics",
                    "AbstractSource": "Wikipedia",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute(
                    "lookup_topic",
                    {"topic": "квантовая физика", "english_query": "quantum physics"},
                )
            )

        self.assertTrue(result["found"])
        self.assertEqual(result["title"], "Quantum mechanics")
        self.assertEqual(result["source"], "Wikipedia")

    async def test_random_joke_returns_both_parts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("safe-mode", request.url.params)
            self.assertNotIn("contains", request.url.params)
            return httpx.Response(
                200,
                json={
                    "error": False,
                    "category": "Programming",
                    "type": "twopart",
                    "setup": "Setup",
                    "delivery": "Punchline",
                    "lang": "en",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute("get_random_joke", {"category": "programming"})
            )

        self.assertEqual(result["joke"], "Setup\nPunchline")
        self.assertEqual(result["language"], "en")

    async def test_random_joke_filters_by_optional_topic(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["contains"], "programmer")
            return httpx.Response(
                200,
                json={
                    "error": False,
                    "category": "Programming",
                    "type": "single",
                    "joke": "A programmer walks into a bar.",
                    "lang": "en",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute(
                    "get_random_joke",
                    {
                        "category": "programming",
                        "topic": "программисты",
                        "search_query": "programmer",
                    },
                )
            )

        self.assertTrue(result["found"])
        self.assertEqual(result["requested_topic"], "программисты")
        self.assertEqual(result["search_query"], "programmer")

    async def test_missing_topic_match_does_not_return_unrelated_joke(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": True,
                    "code": 106,
                    "message": "No matching joke found",
                    "causedBy": [
                        "No jokes were found that match your provided filter(s)."
                    ],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute(
                    "get_random_joke",
                    {"topic": "космонавты", "search_query": "astronaut"},
                )
            )

        self.assertFalse(result["found"])
        self.assertEqual(result["topic"], "космонавты")
        self.assertNotIn("joke", result)

    async def test_other_joke_api_bad_requests_remain_errors(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": True,
                    "code": 107,
                    "message": "Invalid filter",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            with self.assertRaises(httpx.HTTPStatusError):
                await service.execute(
                    "get_random_joke",
                    {"topic": "нейросеть", "search_query": "neural network"},
                )

    async def test_web_search_parses_results_and_unwraps_redirects(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "html.duckduckgo.com")
            self.assertEqual(request.url.params["q"], "актуальная версия Python")
            return httpx.Response(
                200,
                html="""
                    <div class="result">
                      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org%2Fdownloads%2F">Python releases</a>
                      <div class="result__snippet">Latest stable Python release.</div>
                    </div>
                    <div class="result">
                      <a class="result__a" href="https://docs.python.org/3/">Python docs</a>
                      <div class="result__snippet">Official documentation.</div>
                    </div>
                """,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = PublicInformationService(client)
            result = json.loads(
                await service.execute(
                    "search_web",
                    {"query": "актуальная версия Python", "max_results": 1},
                )
            )

        self.assertTrue(result["found"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["title"], "Python releases")
        self.assertEqual(
            result["results"][0]["url"], "https://python.org/downloads/"
        )
        self.assertEqual(result["results"][0]["snippet"], "Latest stable Python release.")

    def test_tools_have_distinct_names_and_strict_arguments(self) -> None:
        functions = [tool["function"] for tool in PUBLIC_INFO_TOOLS]
        self.assertEqual(
            {function["name"] for function in functions},
            {"get_current_weather", "lookup_topic", "get_random_joke"},
        )
        for function in functions:
            self.assertFalse(function["parameters"]["additionalProperties"])
        self.assertEqual(WEB_SEARCH_TOOL["function"]["name"], "search_web")
        self.assertFalse(
            WEB_SEARCH_TOOL["function"]["parameters"]["additionalProperties"]
        )

    def test_explicit_requests_force_the_matching_tool(self) -> None:
        self.assertEqual(
            requested_public_tool("Какая сейчас погода в Москве?"),
            "get_current_weather",
        )
        self.assertEqual(
            requested_public_tool("Расскажи анекдот про программистов"),
            "get_random_joke",
        )
        self.assertEqual(
            requested_public_tool("Расскажи мне про квантовую физику"),
            "lookup_topic",
        )
        self.assertEqual(
            requested_public_tool("Кто такой Альберт Эйнштейн?"),
            "lookup_topic",
        )
        self.assertIsNone(requested_public_tool("Расскажи короткую историю"))

    def test_explicit_web_search_requests_are_detected(self) -> None:
        for text in (
            "Найди в интернете свежие новости",
            "Поищи в сети документацию",
            "Загугли последнюю версию Python",
        ):
            self.assertTrue(requested_web_search(text))
            self.assertEqual(requested_public_tool(text), "search_web")
        self.assertFalse(requested_web_search("Найди ошибку в этом коде"))


if __name__ == "__main__":
    unittest.main()
