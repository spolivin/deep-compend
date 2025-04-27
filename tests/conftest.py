import pytest

from deep_compend.utils.downloads import download_arxiv_paper


@pytest.fixture(scope="session")
def test_pdf_path(tmp_path_factory):
    """Downloads a sample ArXiv paper."""
    test_pdf = tmp_path_factory.mktemp("test_data") / "test_paper.pdf"

    download_arxiv_paper(arxiv_id="1512.03385", save_path=str(test_pdf))

    return test_pdf
