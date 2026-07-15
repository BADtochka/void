from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _casefold_csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        item.strip().casefold()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Settings:
    lmstudio_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "local-model")
    lmstudio_temperature: float = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.6"))
    lmstudio_max_tokens: int = _positive_int("LMSTUDIO_MAX_TOKENS", 220)
    lmstudio_timeout_seconds: float = _positive_float("LMSTUDIO_TIMEOUT_SECONDS", 180.0)
    lmstudio_progress_interval_seconds: float = _positive_float(
        "LMSTUDIO_PROGRESS_INTERVAL_SECONDS", 2.0
    )
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "ru")
    whisper_beam_size: int = _positive_int("WHISPER_BEAM_SIZE", 2)
    whisper_speech_gate: bool = _boolean("WHISPER_SPEECH_GATE", True)
    whisper_speech_gate_threshold: float = _positive_float(
        "WHISPER_SPEECH_GATE_THRESHOLD", 0.5
    )
    whisper_min_speech_ms: int = _positive_int("WHISPER_MIN_SPEECH_MS", 180)
    hotword_model: str = os.getenv("HOTWORD_MODEL", "small").strip()
    hotword_device: str = os.getenv("HOTWORD_DEVICE", "cpu").strip()
    hotword_compute_type: str = os.getenv("HOTWORD_COMPUTE_TYPE", "int8").strip()
    whisper_leading_silence_ms: int = _non_negative_int(
        "WHISPER_LEADING_SILENCE_MS", 400
    )
    piper_model_path: str = os.getenv("PIPER_MODEL_PATH", "models/ru_RU-ruslan-medium.onnx")
    tts_default_voice: str = os.getenv("TTS_DEFAULT_VOICE", "").strip()
    piper_length_scale: float = _positive_float("PIPER_LENGTH_SCALE", 1.18)
    piper_sentence_silence_ms: int = _non_negative_int("PIPER_SENTENCE_SILENCE_MS", 320)
    silero_put_accent: bool = _boolean("SILERO_PUT_ACCENT", True)
    silero_put_yo: bool = _boolean("SILERO_PUT_YO", True)
    silero_sentence_silence_ms: int = _non_negative_int(
        "SILERO_SENTENCE_SILENCE_MS", 220
    )
    wake_word: str = os.getenv("WAKE_WORD", "омни").strip().casefold()
    wake_word_aliases: tuple[str, ...] = _casefold_csv(
        "WAKE_WORD_ALIASES",
        "помни,вомни,омли,омне,о мне,о мни,амни,умни,омний,омник,умник",
    )
    stop_phrases: tuple[str, ...] = _casefold_csv(
        "STOP_PHRASES",
        "стоп,стопэ,стопе,хватит,хвати,фатит,достаточно,харе,харэ,заканчивай,"
        "заканчиваем,закончи,закончить,закончим,закончили,заверши,завершить,"
        "заверши диалог,закрой диалог,прекрати,прекращай,прекрати слушать,"
        "перестань слушать,не слушай,замолчи,молчи,можешь отключаться,"
        "отключайся,остановись,останови диалог,до свидания,пока",
    )
    followup_seconds: float = _positive_float("FOLLOWUP_SECONDS", 30.0)
    followup_min_chars: int = _non_negative_int("FOLLOWUP_MIN_CHARS", 4)
    followup_ignore_phrases: tuple[str, ...] = _casefold_csv(
        "FOLLOWUP_IGNORE_PHRASES", "да,ага,угу,нет,ок,окей,ладно,понятно"
    )
    max_history_turns: int = _positive_int("MAX_HISTORY_TURNS", 8)
    max_input_seconds: int = _positive_int("MAX_UTTERANCE_SECONDS", 30)
    recognition_pending_per_speaker: int = _positive_int(
        "RECOGNITION_PENDING_PER_SPEAKER", 2
    )
    user_memory_db_path: str = os.getenv(
        "USER_MEMORY_DB_PATH", "data/user-memory.sqlite3"
    ).strip()
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        (
            "Ты локальный голосовой собеседник Омни в Discord. Отвечай по-русски, кратко и естественно. "
            "Не используй Markdown, списки, эмодзи и блоки кода, если пользователь прямо их не запросил. "
            "Используй короткие предложения и пунктуацию для естественных пауз. "
            "Не добавляй перед ответом имя, роль или метки вроде [Омни]. "
            "Не выводи теги think, reasoning, analysis и внутренние рассуждения. "
            "Каждая реплика содержит служебный roster участников, стабильный identity_key текущего "
            "говорящего и его display/preferred name. Считай разные identity_key разными людьми, "
            "даже если их имена совпадают, и связывай реплики одного identity_key с одним человеком. "
            "Исторические user-сообщения содержат author_identity, а assistant-сообщения — "
            "reply_to_identity. Это служебная принадлежность: не цитируй её. Местоимения первого "
            "лица внутри каждой реплики относятся только к её author_identity. Никогда не переноси "
            "вопрос, мнение, имя, предпочтение или факт от одного identity_key к другому. "
            "Последний current_identity — единственный текущий говорящий и адресат нового ответа. "
            "Отвечай текущему говорящему. Можешь естественно обращаться к любому участнику по его "
            "preferred_name, а если оно не задано — по display_name; эти пользовательские имена можно "
            "произносить. Никогда не произноси, не цитируй и не выводи identity_key или другие "
            "внутренние идентификаторы. "
            "Если пользователь просит запомнить обращение, обязательно используй доступный инструмент. "
            "При любом вопросе о том, как зовут текущего или другого участника, обязательно используй "
            "инструмент lookup_user_name и отвечай только по его результату. Если scope=other_user, "
            "говори о найденном участнике в третьем лице и никогда не отвечай 'тебя зовут'. "
            "Если пользователь просит отправить или продублировать ответ либо его часть в Discord-чат, "
            "обязательно используй send_message_to_chat. Не утверждай, что сообщение отправлено, без инструмента. "
            "Для текущей погоды обязательно используй get_current_weather и не выдумывай свежие данные. "
            "Когда пользователь просит фактически рассказать об энциклопедической теме, используй lookup_topic. "
            "Для просьбы рассказать анекдот или шутку используй get_random_joke. "
            "Если пользователь явно просит закончить голосовой диалог, перестать слушать, замолчать "
            "или отключиться, обязательно вызови end_conversation вместо текстового ответа. Не проси "
            "его произнести специальное слово: распознавай любую равнозначную формулировку. "
            "Твой ответ будет озвучен синтезатором речи."
        ),
    )
