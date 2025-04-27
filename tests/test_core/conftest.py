import pytest

from deep_compend.core.summarizer import ArticleSummarizer


@pytest.fixture(scope="package")
def summarizer(test_pdf_path):
    """Returns an instance of ArticleSummarizer with 't5-small' model."""
    model_name = "google-t5/t5-small"
    summarizer = ArticleSummarizer(model_path=model_name)
    _ = summarizer.summarize(pdf_path=str(test_pdf_path))
    return summarizer
