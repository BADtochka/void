import unittest

from voice_core.speech_normalization import normalize_russian_tts_text


class RussianSpeechNormalizationTests(unittest.TestCase):
    def test_normalizes_numbers_and_units_in_water_answer(self) -> None:
        text = (
            "Формула: 30 мл на каждый килограмм. "
            "При весе 70 кг — около 2 литров."
        )

        self.assertEqual(
            normalize_russian_tts_text(text),
            "Формула: тридцать миллилитров на каждый килограмм. "
            "При весе семьдесят килограммов — около двух литров.",
        )

    def test_selects_russian_unit_forms(self) -> None:
        self.assertEqual(
            normalize_russian_tts_text("1 л, 2 л, 5 л, 21 л, 12 л"),
            "один литр, два литра, пять литров, двадцать один литр, двенадцать литров",
        )

    def test_normalizes_decimals_percentages_and_temperature(self) -> None:
        self.assertEqual(
            normalize_russian_tts_text("1,5 л, 25% и 20 °C"),
            "одна целая пять десятых литра, двадцать пять процентов и двадцать градусов Цельсия",
        )

    def test_supports_genitive_context_and_latin_units(self) -> None:
        self.assertEqual(
            normalize_russian_tts_text("от 1 kg до 5 kg, менее 2 ml"),
            "от одного килограмма до пяти килограммов, менее двух миллилитров",
        )

    def test_normalizes_plain_numbers_and_latin_model_prefix(self) -> None:
        self.assertEqual(
            normalize_russian_tts_text("В 2026 году модель v5_5 стала лучше"),
            "В две тысячи двадцать шесть году модель ви5_5 стала лучше",
        )

    def test_pronounces_english_acronyms_and_known_words(self) -> None:
        self.assertEqual(
            normalize_russian_tts_text("OpenAI API работает в Discord"),
            "оупен эй ай эй пи ай работает в дискорд",
        )
        self.assertEqual(
            normalize_russian_tts_text("CPU, GPU и Python"),
            "си пи ю, джи пи ю и пайтон",
        )

    def test_transliterates_unknown_latin_words_instead_of_dropping_them(self) -> None:
        normalized = normalize_russian_tts_text("Rust and Kotlin")

        self.assertEqual(normalized, "руст анд котлин")
        self.assertNotRegex(normalized, r"[A-Za-z]")

    def test_russianize_address_name_converts_discord_nicks(self) -> None:
        from voice_core.speech_normalization import russianize_address_name

        self.assertEqual(russianize_address_name("tochkablsq"), "точкаблск")
        self.assertEqual(russianize_address_name(".formallybad"), "формаллйбад")
        self.assertEqual(russianize_address_name("Пупсик"), "Пупсик")
        self.assertEqual(russianize_address_name("Маша"), "Маша")


if __name__ == "__main__":
    unittest.main()
