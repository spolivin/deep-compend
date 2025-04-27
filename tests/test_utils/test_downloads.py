import os

import nltk
import pytest


def test_download_arxiv_paper(test_pdf_path):
    """Tests if the test article has been downloaded correctly."""
    assert os.path.getsize(test_pdf_path) > 0
    assert test_pdf_path.exists()


def test_nltk_patch():
    """Tests if NLTK resources have been downloaded correctly."""
    try:
        nltk.data.find(resource_name="tokenizers/punkt")
        nltk.data.find(resource_name="tokenizers/punkt_tab")
    except LookupError as e:
        pytest.fail(f"Required NLTK resource missing: {e}")
