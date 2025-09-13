import pytest
from utils import clean_text, calculate_text_complexity, extract_keywords_from_text

class TestUtils:
    
    def test_clean_text(self):
        assert clean_text("") == ""
        assert clean_text("Hello world!") == "Hello world"
        assert clean_text("@user #hashtag http://example.com") == "user hashtag"
        assert clean_text("Привет, мир!") == "Привет мир"
    
    def test_calculate_text_complexity(self):
        result = calculate_text_complexity("Hello world")
        assert 'complexity' in result
        assert 'word_count' in result
        assert 'sentence_count' in result
        assert result['word_count'] == 2
        assert result['sentence_count'] == 1
    
    def test_extract_keywords_from_text(self):
        keywords = extract_keywords_from_text("Hello world test")
        assert len(keywords) > 0
        assert "hello" in keywords
        assert "world" in keywords
        assert "test" in keywords
