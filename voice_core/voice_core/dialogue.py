from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field


@dataclass
class Conversation:
    messages_by_user: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    active_users: dict[str, float] = field(default_factory=dict)
    awaiting_content_user: str | None = None
    participants: dict[str, Participant] = field(default_factory=dict)
    next_identity_number: int = 1
    cooldown_until: float = 0.0
    cooldown_until_by_user: dict[str, float] = field(default_factory=dict)


@dataclass
class Participant:
    identity_key: str
    display_name: str
    preferred_name: str | None = None


@dataclass(frozen=True)
class AcceptedSpeech:
    text: str
    direct_wake: bool


def normalize_phrase(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


_STOP_FILLER_WORDS = frozenset(
    {"ну", "пожалуйста", "плиз", "прошу", "давай", "ладно", "всё", "все"}
)

_STOP_INTERJECTIONS = frozenset(
    {
        "бля",
        "блин",
        "блять",
        "блядь",
        "ёб",
        "еб",
        "епт",
        "ёпт",
        "черт",
        "чёрт",
        "йоу",
        "эй",
        "алло",
        "слыш",
        "слушай",
    }
)

_STOP_COMMAND_TOKENS = frozenset(
    {
        "стоп",
        "стопэ",
        "стопе",
        "хватит",
        "хвати",
        "фатит",
        "достаточно",
        "харе",
        "харэ",
        "замолчи",
        "молчи",
        "отключайся",
        "отключись",
        "остановись",
        "пока",
    }
)


def normalize_stop_request(text: str) -> str:
    return " ".join(
        word
        for word in normalize_phrase(text).split()
        if word not in _STOP_FILLER_WORDS
    )


def _spoken_phrase_pattern(phrase: str) -> str:
    words = re.findall(r"\w+", phrase.casefold(), flags=re.UNICODE)
    return r"[\W_]*".join(re.escape(word) for word in words)


class ConversationStore:
    def __init__(
        self,
        wake_word: str,
        followup_seconds: float,
        max_turns: int,
        wake_word_aliases: tuple[str, ...] = (),
        stop_phrases: tuple[str, ...] = (),
        followup_min_chars: int = 0,
        followup_ignore_phrases: tuple[str, ...] = (),
        dialogue_cooldown_seconds: float = 0.0,
    ) -> None:
        self._wake_word_label = wake_word.strip().capitalize()
        self._wake_words = tuple(dict.fromkeys((wake_word, *wake_word_aliases))) if wake_word else ()
        self._wake_token_sequences = tuple(
            normalize_phrase(word).split()
            for word in sorted(self._wake_words, key=len, reverse=True)
            if normalize_phrase(word)
        )
        alternatives = "|".join(
            _spoken_phrase_pattern(word)
            for word in sorted(self._wake_words, key=len, reverse=True)
        )
        self._wake_word_pattern = (
            re.compile(
                rf"\b(?P<wake>{alternatives})\b(?P<separator>[,.!?;:\s-]*)",
                flags=re.IGNORECASE,
            )
            if alternatives
            else None
        )
        self._followup_seconds = followup_seconds
        base_stop_phrases = {
            normalize_phrase(phrase) for phrase in stop_phrases
        }
        normalized_wakes = tuple(
            normalize_phrase(wake_word) for wake_word in self._wake_words
        )
        stop_phrases_normalized = set(base_stop_phrases)
        for stop_phrase in base_stop_phrases:
            if any(
                stop_phrase == wake or stop_phrase.startswith(f"{wake} ")
                for wake in normalized_wakes
            ):
                continue
            for wake in normalized_wakes:
                stop_phrases_normalized.add(f"{wake} {stop_phrase}")
                # ASR often glues wake and stop without a separator: «Омнистоп».
                if " " not in wake and " " not in stop_phrase:
                    stop_phrases_normalized.add(f"{wake}{stop_phrase}")
        self._stop_phrases = frozenset(stop_phrases_normalized)
        self._followup_min_chars = followup_min_chars
        self._followup_ignore_phrases = frozenset(
            normalize_phrase(phrase) for phrase in followup_ignore_phrases
        )
        self._dialogue_cooldown_seconds = dialogue_cooldown_seconds
        self._max_messages = max_turns * 2
        self._items: dict[str, Conversation] = {}

    def accept(
        self,
        key: str,
        transcript: str,
        speaker_id: str = "default",
        now: float | None = None,
    ) -> str | None:
        accepted = self.accept_turn(key, transcript, speaker_id, now)
        if accepted is None:
            return None
        return accepted.text or None

    def wake_remainder(self, transcript: str) -> str | None:
        if self._wake_word_pattern is None:
            return None
        match = self._wake_word_pattern.search(transcript)
        if match is None:
            return None
        return (
            transcript[: match.start()] + transcript[match.end() :]
        ).strip(" ,.!?;:-")

    def accept_turn(
        self,
        key: str,
        transcript: str,
        speaker_id: str = "default",
        now: float | None = None,
        utterance_started_at: float | None = None,
        force_wake: bool = False,
    ) -> AcceptedSpeech | None:
        text = transcript.strip()
        if not text:
            return None

        if self.stop_if_requested(key, text, now=now):
            return None

        conversation = self._items.setdefault(key, Conversation())
        current_time = time.monotonic() if now is None else now
        eligibility_time = current_time if utterance_started_at is None else utterance_started_at
        if eligibility_time <= max(
            conversation.cooldown_until,
            conversation.cooldown_until_by_user.get(speaker_id, 0.0),
        ):
            return None
        awaiting_user = conversation.awaiting_content_user
        if (
            awaiting_user is not None
            and eligibility_time > conversation.active_users.get(awaiting_user, 0.0)
        ):
            conversation.awaiting_content_user = None
            awaiting_user = None
        if awaiting_user is not None and awaiting_user != speaker_id:
            return None
        speaker_was_active = eligibility_time <= conversation.active_users.get(
            speaker_id, 0.0
        )
        if self._wake_word_pattern is None:
            return AcceptedSpeech(text, direct_wake=True)

        match = self._wake_word_pattern.search(text)
        if match:
            cleaned = self.wake_remainder(text)
            if awaiting_user == speaker_id or speaker_was_active:
                if not cleaned:
                    return None
                conversation.awaiting_content_user = None
                return AcceptedSpeech(cleaned, direct_wake=False)
            conversation.active_users[speaker_id] = current_time + self._followup_seconds
            if not cleaned:
                conversation.awaiting_content_user = speaker_id
                return AcceptedSpeech("", direct_wake=True)
            conversation.awaiting_content_user = None
            return AcceptedSpeech(
                text=(
                    text[: match.start()]
                    + self._wake_word_label
                    + match.group("separator")
                    + text[match.end() :]
                ),
                direct_wake=True,
            )

        if force_wake:
            if awaiting_user == speaker_id or speaker_was_active:
                conversation.awaiting_content_user = None
                return AcceptedSpeech(text, direct_wake=False)
            conversation.active_users[speaker_id] = current_time + self._followup_seconds
            conversation.awaiting_content_user = None
            return AcceptedSpeech(
                f"{self._wake_word_label}, {text}", direct_wake=True
            )

        if eligibility_time <= conversation.active_users.get(speaker_id, 0.0):
            normalized = normalize_phrase(text)
            compact_length = len(normalized.replace(" ", ""))
            if normalized in self._followup_ignore_phrases:
                return None
            if compact_length < self._followup_min_chars:
                return None
            if conversation.awaiting_content_user == speaker_id:
                conversation.awaiting_content_user = None
            return AcceptedSpeech(text, direct_wake=False)
        return None

    def stop_if_requested(
        self, key: str, transcript: str, now: float | None = None
    ) -> bool:
        if not self._is_stop_request(transcript):
            return False
        self.stop(key, now=now)
        return True

    def _strip_wake_phrases(self, normalized: str) -> str:
        tokens = normalized.split()
        if not tokens or not self._wake_token_sequences:
            return normalized
        kept: list[str] = []
        index = 0
        while index < len(tokens):
            matched = False
            for wake_tokens in self._wake_token_sequences:
                end = index + len(wake_tokens)
                if tokens[index:end] == wake_tokens:
                    index = end
                    matched = True
                    break
            if matched:
                continue
            kept.append(tokens[index])
            index += 1
        return " ".join(kept)

    def _is_stop_request(self, transcript: str) -> bool:
        normalized = normalize_stop_request(transcript)
        if not normalized:
            return False
        if normalized in self._stop_phrases:
            return True

        without_wakes = self._strip_wake_phrases(normalized)
        without_noise = " ".join(
            token
            for token in without_wakes.split()
            if token not in _STOP_INTERJECTIONS
        )
        if not without_noise:
            return False
        if without_noise in self._stop_phrases:
            return True

        tokens = without_noise.split()
        if tokens and set(tokens) <= _STOP_COMMAND_TOKENS and len(tokens) <= 8:
            return True

        for phrase in self._stop_phrases:
            phrase_tokens = phrase.split()
            if len(phrase_tokens) < 2:
                continue
            if without_noise == phrase or without_noise.endswith(f" {phrase}"):
                return True
        return False

    def stop(self, key: str, now: float | None = None) -> None:
        self.finish(key, now=now)

    def finish(
        self,
        key: str,
        speaker_id: str | None = None,
        now: float | None = None,
    ) -> None:
        conversation = self._items.setdefault(key, Conversation())
        current_time = time.monotonic() if now is None else now
        cooldown_until = (
            current_time + self._dialogue_cooldown_seconds
            if self._dialogue_cooldown_seconds > 0
            else 0.0
        )
        if speaker_id is None:
            conversation.active_users.clear()
            conversation.awaiting_content_user = None
            conversation.messages_by_user.clear()
            conversation.participants.clear()
            conversation.cooldown_until = cooldown_until
            return

        conversation.active_users.pop(speaker_id, None)
        if conversation.awaiting_content_user == speaker_id:
            conversation.awaiting_content_user = None
        conversation.messages_by_user.pop(speaker_id, None)
        conversation.participants.pop(speaker_id, None)
        conversation.cooldown_until_by_user[speaker_id] = cooldown_until

    def cooldown_active(
        self, key: str, speaker_id: str, now: float | None = None
    ) -> bool:
        current_time = time.monotonic() if now is None else now
        conversation = self._items.setdefault(key, Conversation())
        return current_time <= max(
            conversation.cooldown_until,
            conversation.cooldown_until_by_user.get(speaker_id, 0.0),
        )

    def followup_active(
        self, key: str, speaker_id: str, now: float | None = None
    ) -> bool:
        current_time = time.monotonic() if now is None else now
        conversation = self._items.setdefault(key, Conversation())
        return current_time <= conversation.active_users.get(speaker_id, 0.0)

    def open_followup(
        self, key: str, speaker_id: str, now: float | None = None
    ) -> bool:
        current_time = time.monotonic() if now is None else now
        conversation = self._items.setdefault(key, Conversation())
        if current_time <= max(
            conversation.cooldown_until,
            conversation.cooldown_until_by_user.get(speaker_id, 0.0),
        ):
            return False
        conversation.active_users[speaker_id] = current_time + self._followup_seconds
        return True

    def hold_followup(self, key: str, speaker_id: str) -> None:
        conversation = self._items.setdefault(key, Conversation())
        conversation.active_users[speaker_id] = float("inf")

    def history(
        self, key: str, speaker_id: str = "default"
    ) -> list[dict[str, str]]:
        conversation = self._items.setdefault(key, Conversation())
        return list(conversation.messages_by_user.get(speaker_id, ()))

    def register_participant(
        self,
        key: str,
        speaker_id: str,
        display_name: str,
        preferred_name: str | None = None,
    ) -> Participant:
        conversation = self._items.setdefault(key, Conversation())
        participant = conversation.participants.get(speaker_id)
        if participant is None:
            participant = Participant(
                identity_key=f"speaker_{conversation.next_identity_number}",
                display_name=display_name,
                preferred_name=preferred_name,
            )
            conversation.next_identity_number += 1
            conversation.participants[speaker_id] = participant
        else:
            participant.display_name = display_name
            participant.preferred_name = preferred_name
        return participant

    def participants(
        self, key: str, speaker_id: str | None = None
    ) -> tuple[Participant, ...]:
        conversation = self._items.setdefault(key, Conversation())
        if speaker_id is not None:
            participant = conversation.participants.get(speaker_id)
            return (participant,) if participant is not None else ()
        return tuple(conversation.participants.values())

    def append_turn(
        self,
        key: str,
        user_text: str,
        assistant_text: str,
        *,
        speaker_id: str = "default",
        identity_key: str | None = None,
        speaker_name: str | None = None,
    ) -> None:
        conversation = self._items.setdefault(key, Conversation())
        messages = conversation.messages_by_user.setdefault(speaker_id, [])
        if identity_key:
            encoded_name = json.dumps(speaker_name or "", ensure_ascii=False)
            user_text = (
                f"author_identity={identity_key}; author_name={encoded_name}\n"
                f"{user_text}"
            )
            assistant_text = (
                f"reply_to_identity={identity_key}; reply_to_name={encoded_name}\n"
                f"{assistant_text}"
            )
        messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        del messages[: max(0, len(messages) - self._max_messages)]

    def last_assistant_message(
        self, key: str, speaker_id: str = "default"
    ) -> str | None:
        conversation = self._items.get(key)
        if conversation is None:
            return None
        for message in reversed(conversation.messages_by_user.get(speaker_id, ())):
            if message["role"] == "assistant":
                content = message["content"]
                if content.startswith("reply_to_identity="):
                    _, separator, answer = content.partition("\n")
                    if separator:
                        return answer
                if content.startswith("[Служебная принадлежность исторического ответа]"):
                    _, separator, answer = content.partition("\nanswer=")
                    if separator:
                        return answer
                return content
        return None

    def reset(self, key: str) -> None:
        self._items.pop(key, None)


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F600-\U0001F64F"
    "\U000024C2-\U0001F251"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)

_TOOL_HALLUCINATION_PATTERN = re.compile(
    r"\b(?:"
    r"end[_\s-]?conversation|endconversation|"
    r"lookup[_\s-]?user[_\s-]?name|lookupusername|"
    r"remember[_\s-]?preferred[_\s-]?name|rememberpreferredname|"
    r"forget[_\s-]?preferred[_\s-]?name|forgetpreferredname|"
    r"send[_\s-]?message[_\s-]?to[_\s-]?chat|sendmessagetochat|"
    r"get[_\s-]?current[_\s-]?weather|getcurrentweather|"
    r"lookup[_\s-]?topic|lookuptopic|"
    r"get[_\s-]?random[_\s-]?joke|getrandomjoke|"
    r"search[_\s-]?web|searchweb|"
    r"rememberwebsearch|rememberallowed"
    r")\b",
    flags=re.IGNORECASE,
)

_UI_INSTRUCTION_PATTERN = re.compile(
    r"\([^)]*(?:нажми|напиши|продолжить|end[_\s-]?conversation)[^)]*\)",
    flags=re.IGNORECASE,
)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+")


def _cut_at_tool_hallucination(text: str) -> str:
    match = _TOOL_HALLUCINATION_PATTERN.search(text)
    if not match:
        return text
    if match.start() == 0:
        return ""
    return text[: match.start()].rstrip(" ,;:—-")


def _trim_repeated_parentheticals(text: str) -> str:
    for match in re.finditer(r"\([^)]{8,}\)", text):
        fragment = match.group(0)
        first = text.find(fragment)
        if first == -1:
            continue
        second = text.find(fragment, first + len(fragment))
        if second != -1:
            return text[:first].rstrip(" ,;:—-")
    return text


def trim_truncated_completion(text: str, *, max_sentences: int = 3) -> str:
    """Drop loops and tool/UI junk common when the model hits max_tokens."""
    cleaned = _trim_repeated_parentheticals(text.strip())
    cleaned = _cut_at_tool_hallucination(cleaned)
    cleaned = _EMOJI_PATTERN.sub("", cleaned)
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_PATTERN.split(cleaned)
        if sentence.strip()
    ]
    if len(sentences) > max_sentences:
        cleaned = " ".join(sentences[:max_sentences])
    else:
        cleaned = " ".join(sentences)
    return cleaned.strip()


def prepare_for_speech(text: str, limit: int = 1_200) -> str:
    text = re.sub(
        r"<(?:think|reasoning|analysis)\b[^>]*>.*?</(?:think|reasoning|analysis)\s*>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<\s*/?\s*(?:think|reasoning|analysis)\s*>?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*\[\s*омни\s*\]\s*[:\-–—]?\s*", "", text, flags=re.IGNORECASE)
    # Drop Discord-like Latin nicknames used as vocatives: «tochkablsq, сейчас…»
    text = re.sub(r"^[a-z0-9._]{2,32},\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```.*?```", " Фрагмент кода опущен. ", text, flags=re.DOTALL)
    text = re.sub(
        r"\[Служебная принадлежность[^\]]*\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:author|reply_to)_(?:identity|name)\s*=\s*(?:\"[^\"]*\"|\S+)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[`*_#>]", "", text)
    text = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", text)
    text = re.sub(
        r"\s*\(?\s*(?:user[_ ]?id|discord\s*id)\s*[:=]?\s*\d{10,20}\s*\)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{17,20}\b", "", text)
    text = re.sub(
        r"\b(?:speaker|participant|identity)[_-]?\d+\b\s*[,;:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _EMOJI_PATTERN.sub("", text)
    text = _UI_INSTRUCTION_PATTERN.sub(" ", text)
    text = trim_truncated_completion(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{shortened}."
