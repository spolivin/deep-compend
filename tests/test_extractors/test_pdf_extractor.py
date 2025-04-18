import os

import pytest

from deep_compend.extractors import PDFExtractor
from deep_compend.utils.downloads import download_arxiv_paper


@pytest.fixture(scope="module")
def downloaded_pdf(tmp_path_factory):
    """Downloads a sample ArXiv paper."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "sample.pdf"

    # Downloading paper from Arxiv
    download_arxiv_paper(arxiv_id="1512.03385", save_path=pdf_path)

    # Embedded test of `download_arxiv_paper`
    assert pdf_path.exists(), "PDF file was not created"
    assert os.path.getsize(pdf_path) > 0

    return pdf_path


def test_pdf_extractor_returns_text(downloaded_pdf):
    """Tests the correctness of the text extracted from PDF."""
    extractor = PDFExtractor(str(downloaded_pdf))
    text = extractor.retrieve_processed_text()

    assert isinstance(text, str)
    assert len(text) > 200
