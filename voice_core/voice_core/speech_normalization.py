from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from num2words import num2words


@dataclass(frozen=True)
class UnitForms:
    one: str
    few: str
    many: str
    aliases: tuple[str, ...]


_UNITS = (
    UnitForms(
        "миллилитр",
        "миллилитра",
        "миллилитров",
        ("мл", "ml", "миллилитр", "миллилитра", "миллилитров"),
    ),
    UnitForms(
        "литр", "литра", "литров", ("л", "l", "литр", "литра", "литров")
    ),
    UnitForms(
        "килограмм",
        "килограмма",
        "килограммов",
        ("кг", "kg", "килограмм", "килограмма", "килограммов"),
    ),
    UnitForms(
        "грамм", "грамма", "граммов", ("г", "g", "грамм", "грамма", "граммов")
    ),
    UnitForms(
        "километр",
        "километра",
        "километров",
        ("км", "km", "километр", "километра", "километров"),
    ),
    UnitForms(
        "сантиметр",
        "сантиметра",
        "сантиметров",
        ("см", "cm", "сантиметр", "сантиметра", "сантиметров"),
    ),
    UnitForms(
        "миллиметр",
        "миллиметра",
        "миллиметров",
        ("мм", "mm", "миллиметр", "миллиметра", "миллиметров"),
    ),
    UnitForms(
        "метр", "метра", "метров", ("м", "m", "метр", "метра", "метров")
    ),
    UnitForms("процент", "процента", "процентов", ("%", "процент", "процента", "процентов")),
    UnitForms("градус Цельсия", "градуса Цельсия", "градусов Цельсия", ("°c", "°с")),
)

_ALIAS_TO_UNIT = {alias.casefold(): unit for unit in _UNITS for alias in unit.aliases}
_UNIT_ALTERNATIVES = "|".join(
    re.escape(alias)
    for alias in sorted(_ALIAS_TO_UNIT, key=len, reverse=True)
)
_NUMBER_SOURCE = r"(?<![\w])(?P<number>\d+(?:[.,]\d+)?)"
_NUMBER_WITH_UNIT = re.compile(
    rf"(?P<prefix>\b(?:около|до|от|более|менее|свыше)\s+)?"
    rf"{_NUMBER_SOURCE}\s*(?P<unit>{_UNIT_ALTERNATIVES})(?!\w)",
    flags=re.IGNORECASE,
)
_PLAIN_NUMBER = re.compile(_NUMBER_SOURCE)
_LATIN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

_ENGLISH_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и",
    "f": "эф", "g": "джи", "h": "эйч", "i": "ай", "j": "джей",
    "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу",
    "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти",
    "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс", "y": "уай",
    "z": "зэд",
}

_KNOWN_ENGLISH_WORDS = {
    "backend": "бэкенд",
    "discord": "дискорд",
    "frontend": "фронтенд",
    "github": "гитхаб",
    "hello": "хэллоу",
    "intel": "интел",
    "javascript": "джаваскрипт",
    "linux": "линукс",
    "nvidia": "энвидиа",
    "openai": "оупен эй ай",
    "python": "пайтон",
    "server": "сёрвер",
    "silero": "силеро",
    "studio": "студио",
    "typescript": "тайпскрипт",
    "whisper": "уиспер",
    "windows": "уиндоус",
    "world": "уорлд",
}

_ENGLISH_GROUPS = (
    ("tion", "шн"), ("sion", "жн"), ("tch", "ч"), ("dge", "дж"),
    ("igh", "ай"), ("sh", "ш"), ("ch", "ч"), ("th", "т"),
    ("ph", "ф"), ("ck", "к"), ("qu", "кв"), ("ng", "нг"),
    ("ee", "и"), ("oo", "у"), ("ea", "и"), ("ai", "эй"),
    ("ay", "эй"), ("oy", "ой"), ("ou", "ау"), ("ow", "ау"),
)

_ENGLISH_CHARACTERS = {
    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "q": "к",
    "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
    "w": "в", "x": "кс", "y": "й", "z": "з",
}


def _parse_number(raw: str) -> Decimal:
    return Decimal(raw.replace(",", "."))


def _number_words(value: Decimal, case: str = "n") -> str:
    if value == value.to_integral_value():
        return str(num2words(int(value), lang="ru", case=case))
    return str(num2words(value, lang="ru", case=case))


def _unit_form(value: Decimal, forms: UnitForms) -> str:
    if value != value.to_integral_value():
        return forms.few
    integer = abs(int(value))
    last_two = integer % 100
    if 11 <= last_two <= 14:
        return forms.many
    last = integer % 10
    if last == 1:
        return forms.one
    if 2 <= last <= 4:
        return forms.few
    return forms.many


def _transliterate_english_word(word: str) -> str:
    lowered = word.casefold().replace("'", "")
    known = _KNOWN_ENGLISH_WORDS.get(lowered)
    if known:
        return known
    if len(lowered) == 1 or (word.isupper() and len(lowered) <= 8):
        return " ".join(_ENGLISH_LETTER_NAMES[letter] for letter in lowered)

    result: list[str] = []
    index = 0
    while index < len(lowered):
        group = next(
            (
                (source, replacement)
                for source, replacement in _ENGLISH_GROUPS
                if lowered.startswith(source, index)
            ),
            None,
        )
        if group is not None:
            source, replacement = group
            result.append(replacement)
            index += len(source)
            continue

        letter = lowered[index]
        next_letter = lowered[index + 1] if index + 1 < len(lowered) else ""
        if letter == "c":
            result.append("с" if next_letter in "eiy" else "к")
        elif letter == "g":
            result.append("дж" if next_letter in "eiy" else "г")
        else:
            result.append(_ENGLISH_CHARACTERS.get(letter, ""))
        index += 1
    return "".join(result)


def _normalize_latin_words(text: str) -> str:
    return _LATIN_WORD.sub(
        lambda match: _transliterate_english_word(match.group(0)), text
    )


def normalize_russian_tts_text(text: str) -> str:
    def replace_unit(match: re.Match[str]) -> str:
        try:
            value = _parse_number(match.group("number"))
        except InvalidOperation:
            return match.group(0)
        forms = _ALIAS_TO_UNIT[match.group("unit").casefold()]
        prefix = match.group("prefix") or ""
        genitive = bool(prefix)
        if genitive:
            unit_form = forms.few if abs(value) == 1 else forms.many
        else:
            unit_form = _unit_form(value, forms)
        return f"{prefix}{_number_words(value, 'g' if genitive else 'n')} {unit_form}"

    normalized = _NUMBER_WITH_UNIT.sub(replace_unit, text)

    def replace_number(match: re.Match[str]) -> str:
        try:
            return _number_words(_parse_number(match.group("number")))
        except InvalidOperation:
            return match.group(0)

    normalized = _PLAIN_NUMBER.sub(replace_number, normalized)
    return _normalize_latin_words(normalized)
