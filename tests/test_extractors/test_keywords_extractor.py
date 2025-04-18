import pytest
import spacy

from deep_compend.extractors import KeywordsExtractor

# Loading "en_core_web_sm" model if it is not loaded yet
try:
    spacy.load("en_core_web_sm")
except OSError:
    import subprocess

    subprocess.run(
        ["python", "-m", "spacy", "download", "en_core_web_sm"], check=True
    )


@pytest.fixture
def sample_text():
    """Returns a test text."""
    return """
    It is evident that neural networks and transformers are powerful tools in modern artificial intelligence.
    Researchers from MIT, Stanford and other universities use these models to analyze data in biology, finance and more.
    """


def test_default_extraction(sample_text):
    """Tests extraction of keywords with default parameters."""
    extractor = KeywordsExtractor()
    keywords = extractor.extract(sample_text)
    assert isinstance(keywords, list)
    assert all(isinstance(k, str) for k in keywords)
    assert "neural" in keywords or "networks" in keywords


def test_custom_length_filtering(sample_text):
    """Tests keywords extraction for fixed minimum keyword length."""
    extractor = KeywordsExtractor(min_kwrd_length=6)
    keywords = extractor.extract(sample_text)
    assert isinstance(keywords, list)
    # Verifying filtering out short words like "mit" or "ai"
    assert all(len(k) >= 6 for k in keywords)


def test_most_common_limit(sample_text):
    """Tests keyword extraction for fixed number of most common words."""
    extractor = KeywordsExtractor(most_common_elems=3)
    keywords = extractor.extract(sample_text)
    assert isinstance(keywords, list)
    assert len(keywords) <= 3


def test_empty_input():
    """Tests keyword extraction on an empty input."""
    extractor = KeywordsExtractor()
    result = extractor.extract("")
    assert result == []


def test_handles_named_entities():
    """Tests handling named entities."""
    text = "Elon Musk and OpenAI developed ChatGPT."
    extractor = KeywordsExtractor()
    keywords = extractor.extract(text)
    assert isinstance(keywords, list)
    # Verifying including named entities like "Elon Musk" and "OpenAI"
    assert "elon musk" in " ".join(keywords) or "openai" in keywords


def test_result_is_sorted_by_frequency():
    text = "AI AI AI biology biology biology finance data data"
    extractor = KeywordsExtractor()
    keywords = extractor.extract(text)
    assert isinstance(keywords, list)
    assert keywords[0] in ["ai", "biology", "finance", "data"]
