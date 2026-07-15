import unittest

from voice_core.dialogue import ConversationStore, prepare_for_speech


class ConversationStoreTests(unittest.TestCase):
    def test_last_assistant_message_returns_previous_response(self) -> None:
        store = ConversationStore("омни", 30, 8)
        store.append_turn("guild", "первый вопрос", "Первый ответ")
        store.append_turn("guild", "второй вопрос", "Второй ответ")

        self.assertEqual(store.last_assistant_message("guild"), "Второй ответ")
        self.assertIsNone(store.last_assistant_message("other"))

    def test_history_marks_both_speaker_and_reply_recipient(self) -> None:
        store = ConversationStore("омни", 30, 8)
        store.append_turn(
            "guild",
            "Я люблю чай",
            "Запомнил.",
            identity_key="speaker_2",
            speaker_name="Рыжий",
        )

        history = store.history("guild")
        self.assertIn("author_identity=speaker_2", history[0]["content"])
        self.assertIn('author_name="Рыжий"', history[0]["content"])
        self.assertIn("utterance=Я люблю чай", history[0]["content"])
        self.assertIn("reply_to_identity=speaker_2", history[1]["content"])
        self.assertIn("answer=Запомнил.", history[1]["content"])
        self.assertEqual(store.last_assistant_message("guild"), "Запомнил.")

    def test_history_ownership_metadata_is_not_spoken(self) -> None:
        text = (
            "[Служебная принадлежность исторического ответа] "
            'reply_to_identity=speaker_2 reply_to_name="Рыжий" Всё готово.'
        )

        self.assertEqual(prepare_for_speech(text), "Всё готово.")

    def test_hotword_only_reserves_next_content_for_its_owner(self) -> None:
        store = ConversationStore("омни", followup_seconds=30, max_turns=2)

        armed = store.accept_turn(
            "guild", "Омни", speaker_id="owner", now=10
        )
        blocked = store.accept_turn(
            "guild", "Омни, чужой вопрос", speaker_id="other", now=11
        )
        repeated = store.accept_turn(
            "guild", "Омни", speaker_id="owner", now=12
        )
        content = store.accept_turn(
            "guild", "мой вопрос", speaker_id="owner", now=13
        )
        next_user = store.accept_turn(
            "guild", "Омни, теперь мой", speaker_id="other", now=14
        )

        self.assertTrue(armed.direct_wake)
        self.assertEqual(armed.text, "")
        self.assertIsNone(blocked)
        self.assertIsNone(repeated)
        self.assertFalse(content.direct_wake)
        self.assertEqual(content.text, "мой вопрос")
        self.assertTrue(next_user.direct_wake)

    def test_active_user_hotword_is_treated_as_followup(self) -> None:
        store = ConversationStore("омни", followup_seconds=30, max_turns=2)

        first = store.accept_turn(
            "guild", "Омни, первый вопрос", speaker_id="owner", now=10
        )
        repeated = store.accept_turn(
            "guild", "Омни, уточнение", speaker_id="owner", now=11
        )

        self.assertTrue(first.direct_wake)
        self.assertFalse(repeated.direct_wake)
        self.assertEqual(repeated.text, "уточнение")

    def test_awaiting_content_owner_expires_with_followup_window(self) -> None:
        store = ConversationStore("омни", followup_seconds=30, max_turns=2)
        store.accept_turn("guild", "Омни", speaker_id="owner", now=10)

        accepted = store.accept_turn(
            "guild",
            "Омни, вопрос после таймаута",
            speaker_id="other",
            now=50,
            utterance_started_at=41,
        )

        self.assertIsNotNone(accepted)
        self.assertTrue(accepted.direct_wake)

    def test_participant_identity_is_stable_and_distinct_from_display_name(self) -> None:
        store = ConversationStore("омни", followup_seconds=30, max_turns=2)

        first = store.register_participant("guild", "user-1", "Одинаковый ник")
        second = store.register_participant("guild", "user-2", "Одинаковый ник")
        renamed = store.register_participant(
            "guild", "user-1", "Новый ник", "Пупсик"
        )

        self.assertEqual(first.identity_key, "speaker_1")
        self.assertEqual(second.identity_key, "speaker_2")
        self.assertEqual(renamed.identity_key, "speaker_1")
        self.assertEqual(renamed.display_name, "Новый ник")
        self.assertEqual(renamed.preferred_name, "Пупсик")
        self.assertEqual(
            [item.identity_key for item in store.participants("guild")],
            ["speaker_1", "speaker_2"],
        )

    def test_wake_word_opens_followup_window(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            wake_word_aliases=("помни", "умник"),
        )

        self.assertEqual(store.accept("guild", "Омни, привет", now=10), "Омни, привет")
        self.assertEqual(store.accept("other", "Помни, привет", now=10), "Омни, привет")
        self.assertEqual(store.accept("third", "Умник, ты здесь?", now=10), "Омни, ты здесь?")
        self.assertEqual(store.accept("guild", "как дела?", now=20), "как дела?")
        self.assertIsNone(store.accept("guild", "ты здесь?", now=51))

    def test_followup_is_per_user_does_not_slide_and_ignores_fillers(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            followup_min_chars=4,
            followup_ignore_phrases=("да", "ага"),
        )

        self.assertEqual(
            store.accept("guild", "Омни, слушай", speaker_id="user-1", now=10),
            "Омни, слушай",
        )
        self.assertIsNone(
            store.accept("guild", "обычная речь", speaker_id="user-2", now=11)
        )
        self.assertIsNone(store.accept("guild", "Ага", speaker_id="user-1", now=12))
        self.assertEqual(
            store.accept("guild", "продолжай рассказ", speaker_id="user-1", now=20),
            "продолжай рассказ",
        )
        self.assertIsNone(
            store.accept("guild", "ещё одна реплика", speaker_id="user-1", now=41)
        )

    def test_direct_wake_request_is_not_filtered_as_filler(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            followup_min_chars=4,
            followup_ignore_phrases=("да",),
        )

        self.assertEqual(
            store.accept("guild", "Омни, да", speaker_id="user", now=10), "Омни, да"
        )

    def test_hotword_without_continuation_opens_followup_but_has_no_request(self) -> None:
        store = ConversationStore("омни", followup_seconds=30, max_turns=2)

        hotword = store.accept_turn("guild", "Омни.", speaker_id="user", now=10)

        self.assertIsNotNone(hotword)
        self.assertTrue(hotword.direct_wake)
        self.assertEqual(hotword.text, "")
        self.assertIsNone(store.accept("other", "Омни.", speaker_id="user", now=10))
        self.assertEqual(
            store.accept("guild", "теперь вопрос", speaker_id="user", now=11),
            "теперь вопрос",
        )

    def test_wake_remainder_handles_aliases(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            wake_word_aliases=("вомни", "омли", "о мне"),
        )

        self.assertEqual(store.wake_remainder("Вомни, как дела?"), "как дела")
        self.assertEqual(store.wake_remainder("Омли, как дела?"), "как дела")
        self.assertEqual(store.wake_remainder("О, мне, как дела?"), "как дела")
        self.assertEqual(store.wake_remainder("Вомни."), "")
        self.assertIsNone(store.wake_remainder("обычная речь"))

    def test_followup_eligibility_uses_audio_start_time(self) -> None:
        store = ConversationStore("омни", followup_seconds=30, max_turns=2)
        store.accept_turn("guild", "Омни, слушай", speaker_id="user", now=10)

        accepted = store.accept_turn(
            "guild",
            "начал говорить вовремя",
            speaker_id="user",
            now=50,
            utterance_started_at=39,
        )
        rejected = store.accept_turn(
            "guild",
            "начал говорить поздно",
            speaker_id="user",
            now=50,
            utterance_started_at=41,
        )

        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)

    def test_history_is_trimmed_to_configured_turns(self) -> None:
        store = ConversationStore("", followup_seconds=30, max_turns=1)
        store.append_turn("guild", "one", "first")
        store.append_turn("guild", "two", "second")

        self.assertEqual(
            store.history("guild"),
            [{"role": "user", "content": "two"}, {"role": "assistant", "content": "second"}],
        )

    def test_followup_can_be_reopened_after_expiry(self) -> None:
        store = ConversationStore("омни", followup_seconds=10, max_turns=2)
        store.open_followup("guild", "user", now=20)

        self.assertTrue(store.followup_active("guild", "user", now=29))
        self.assertFalse(store.followup_active("guild", "user", now=31))

        store.open_followup("guild", "user", now=31)
        self.assertTrue(store.followup_active("guild", "user", now=40))

    def test_held_followup_does_not_expire_until_opened_after_playback(self) -> None:
        store = ConversationStore("омни", followup_seconds=10, max_turns=2)
        store.hold_followup("guild", "user")

        self.assertTrue(store.followup_active("guild", "user", now=10**12))

        store.open_followup("guild", "user", now=100)
        self.assertTrue(store.followup_active("guild", "user", now=109))
        self.assertFalse(store.followup_active("guild", "user", now=111))

    def test_stop_phrase_closes_followup_window(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            stop_phrases=("омни стоп", "помни стоп"),
        )

        self.assertEqual(store.accept("guild", "Омни, привет", now=10), "Омни, привет")
        self.assertIsNone(store.accept("guild", "Омни, стоп.", now=15))
        self.assertIsNone(store.accept("guild", "это не уйдет в модель", now=16))

    def test_stop_aliases_and_bare_followup_stop_are_recognized(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            wake_word_aliases=("омли", "о мне"),
            stop_phrases=("стоп", "хватит"),
        )

        self.assertEqual(
            store.accept("guild", "Омли, слушай", speaker_id="user", now=10),
            "Омни, слушай",
        )
        self.assertTrue(store.stop_if_requested("guild", "стоп"))
        self.assertIsNone(
            store.accept("guild", "это не уйдет", speaker_id="user", now=11)
        )

        self.assertEqual(
            store.accept("guild", "О, мне, слушай", speaker_id="user", now=20),
            "Омни, слушай",
        )
        self.assertTrue(store.stop_if_requested("guild", "О, мне, стоп"))

    def test_every_stop_alias_is_combined_with_every_wake_alias(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            wake_word_aliases=("помни", "о мне"),
            stop_phrases=("фатит", "отключайся"),
        )

        self.assertTrue(store.stop_if_requested("guild", "Помни, фатит"))
        self.assertTrue(store.stop_if_requested("guild", "О, мне, отключайся"))

    def test_stop_phrase_ignores_polite_fillers(self) -> None:
        store = ConversationStore(
            "омни",
            followup_seconds=30,
            max_turns=2,
            wake_word_aliases=("о мне",),
            stop_phrases=("хватит",),
        )

        self.assertTrue(
            store.stop_if_requested("guild", "О, мне, ну пожалуйста, хватит")
        )

    def test_markdown_is_removed_before_speech(self) -> None:
        self.assertEqual(prepare_for_speech("**Ответ:** [ссылка](https://example.com)"), "Ответ: ссылка")

    def test_internal_user_identifiers_are_removed_before_speech(self) -> None:
        self.assertEqual(
            prepare_for_speech(
                "Привет, formallybad (userid: 441931502234632193). Слушай."
            ),
            "Привет, formallybad. Слушай.",
        )

    def test_internal_speaker_key_is_removed_before_speech(self) -> None:
        self.assertEqual(
            prepare_for_speech("speaker_2, отвечаю на твой вопрос."),
            "отвечаю на твой вопрос.",
        )

    def test_assistant_name_prefix_is_removed_before_speech(self) -> None:
        self.assertEqual(
            prepare_for_speech("[Омни] Привет. Всё отлично. А ты как?"),
            "Привет. Всё отлично. А ты как?",
        )
        self.assertEqual(prepare_for_speech("[ОМНИ]: Слушаю."), "Слушаю.")

    def test_reasoning_tags_are_removed_before_speech(self) -> None:
        self.assertEqual(
            prepare_for_speech(
                "<think>Скрытое рассуждение.</think> Вот готовый ответ."
            ),
            "Вот готовый ответ.",
        )
        self.assertEqual(
            prepare_for_speech("Хочешь пример плана? </think"),
            "Хочешь пример плана?",
        )
        self.assertEqual(
            prepare_for_speech("<analysis>служебный текст</analysis>Продолжаем."),
            "Продолжаем.",
        )


if __name__ == "__main__":
    unittest.main()
