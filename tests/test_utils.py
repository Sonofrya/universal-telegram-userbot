import pytest
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Устанавливаем переменные окружения до импорта config
os.environ.setdefault('TELEGRAM_API_ID', 'test')
os.environ.setdefault('TELEGRAM_API_HASH', 'test')
os.environ.setdefault('TELEGRAM_PHONE', '+70000000000')
os.environ.setdefault('BUSINESS_KEYWORDS', 'тест,разработка,дизайн')
os.environ.setdefault('TARGET_USER_IDS', '123456')
os.environ.setdefault('FULL_CYCLE_PHRASES', 'полный цикл,под ключ,комплексный')

from utils import (
    clean_text,
    is_too_short,
    contains_full_cycle_phrases,
    extract_keywords_from_text,
    calculate_text_complexity,
    format_message_info,
    validate_config,
    contains_blacklisted_words,
    is_forward_notification,
)


class TestCleanText:
    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_removes_hashtags(self):
        result = clean_text("#тест привет")
        assert "#" not in result

    def test_removes_mentions(self):
        result = clean_text("@user привет")
        assert "@" not in result

    def test_removes_urls(self):
        result = clean_text("посмотри http://example.com тут")
        assert "http" not in result

    def test_removes_special_chars(self):
        result = clean_text("Привет, мир!")
        assert result == "Привет мир"

    def test_collapses_whitespace(self):
        result = clean_text("слово    слово")
        assert "    " not in result

    def test_preserves_cyrillic(self):
        result = clean_text("Привет мир")
        assert result == "Привет мир"

    def test_hello_world(self):
        assert clean_text("Hello world!") == "Hello world"


class TestIsTooShort:
    def test_empty_string(self):
        assert is_too_short("") is True

    def test_none_input(self):
        assert is_too_short(None) is True

    def test_short_text(self):
        assert is_too_short("два слова") is True

    def test_long_enough_text(self):
        assert is_too_short("это достаточно длинное сообщение для теста") is False


class TestContainsFullCyclePhrases:
    def test_empty_string(self):
        assert contains_full_cycle_phrases("") is False

    def test_none_input(self):
        assert contains_full_cycle_phrases(None) is False

    def test_contains_phrase(self):
        assert contains_full_cycle_phrases("Мы делаем полный цикл производства") is True

    def test_contains_pod_kluch(self):
        assert contains_full_cycle_phrases("Производство под ключ") is True

    def test_no_matching_phrase(self):
        assert contains_full_cycle_phrases("просто обычный текст без фраз") is False


class TestExtractKeywords:
    def test_empty_string(self):
        assert extract_keywords_from_text("") == []

    def test_extracts_words(self):
        keywords = extract_keywords_from_text("разработка приложений для бизнеса")
        assert "разработка" in keywords
        assert "приложений" in keywords
        assert "бизнеса" in keywords

    def test_filters_stop_words(self):
        keywords = extract_keywords_from_text("для разработки")
        assert "для" not in keywords
        assert "разработки" in keywords

    def test_hello_world(self):
        keywords = extract_keywords_from_text("Hello world test")
        assert "hello" in keywords
        assert "world" in keywords
        assert "test" in keywords


class TestCalculateTextComplexity:
    def test_empty_string(self):
        result = calculate_text_complexity("")
        assert result['complexity'] == 0
        assert result['word_count'] == 0

    def test_simple_text(self):
        result = calculate_text_complexity("Hello world")
        assert result['word_count'] == 2
        assert result['sentence_count'] == 1

    def test_multi_sentence(self):
        result = calculate_text_complexity("Первое предложение. Второе предложение.")
        assert result['sentence_count'] == 2

    def test_returns_all_keys(self):
        result = calculate_text_complexity("Тестовый текст.")
        expected_keys = {'complexity', 'word_count', 'sentence_count',
                         'avg_word_length', 'avg_sentence_length'}
        assert expected_keys == set(result.keys())


class TestFormatMessageInfo:
    def test_basic_format(self):
        data = {
            'message_date': '01.01.2025 12:00',
            'sender_info': 'Иван',
            'chat_title': 'Тестовый чат',
            'message_id': 123,
        }
        result = format_message_info(data)
        assert '01.01.2025 12:00' in result
        assert 'Иван' in result
        assert 'Тестовый чат' in result
        assert '123' in result

    def test_empty_data(self):
        result = format_message_info({})
        assert isinstance(result, str)

    def test_with_ml_probability(self):
        data = {'ml_probability': 0.85}
        result = format_message_info(data)
        assert '0.850' in result


class TestValidateConfig:
    def test_valid_config_returns_empty(self):
        errors = validate_config()
        assert isinstance(errors, list)
        assert len(errors) == 0


class TestContainsBlacklistedWords:
    def test_empty_string(self):
        assert contains_blacklisted_words("") is False

    def test_none_input(self):
        assert contains_blacklisted_words(None) is False

    def test_contains_blacklisted(self):
        assert contains_blacklisted_words("это спам сообщение") is True

    def test_no_blacklisted(self):
        assert contains_blacklisted_words("обычное сообщение") is False


class TestIsForwardNotification:
    def test_empty_string(self):
        assert is_forward_notification("") is False

    def test_none_input(self):
        assert is_forward_notification(None) is False

    def test_forward_pattern_ru(self):
        assert is_forward_notification("это пересланное сообщение") is True

    def test_forward_pattern_en(self):
        assert is_forward_notification("forwarded message from user") is True

    def test_regular_message(self):
        assert is_forward_notification("обычное сообщение") is False
