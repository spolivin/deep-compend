import pytest

from deep_compend.summarizer import ArticleSummarizer
from deep_compend.utils.downloads import download_arxiv_paper


@pytest.fixture(scope="module")
def test_pdf_path(tmp_path_factory):
    """Downloads a sample ArXiv paper."""
    test_pdf = tmp_path_factory.mktemp("data") / "paper.pdf"

    download_arxiv_paper(arxiv_id="1512.03385", save_path=test_pdf)

    return str(test_pdf)


@pytest.fixture(scope="module")
def summarizer():
    """Returns an instance of ArticleSummarizer with 't5-small' model."""
    model_name = "google-t5/t5-small"
    return ArticleSummarizer(model_path=model_name)


def test_summarization_output(summarizer, test_pdf_path):
    """Tests the output of `summarize` method and class attributes."""
    summary = summarizer.summarize(pdf_path=test_pdf_path)
    assert isinstance(summary, str)
    assert len(summary.split()) > 5
    assert hasattr(summarizer, "summary")
    assert hasattr(summarizer, "input_token_count")
    assert hasattr(summarizer, "output_token_count")


def test_summary_stats_structure(summarizer, test_pdf_path):
    """Tests the summarization statistics."""
    _ = summarizer.summarize(test_pdf_path)
    stats = summarizer._get_stats()
    assert stats.word_count_full > 0
    assert stats.word_count_summary > 0
    assert "%" in stats.compression_rate


def test_generate_summary_report_creates_file(
    summarizer, test_pdf_path, tmp_path
):
    """Tests report generation."""
    _ = summarizer.summarize(test_pdf_path)
    filename = "test_report.txt"
    summarizer.generate_summary_report(
        filename=filename, save_folder=str(tmp_path), kwrds_num=3
    )
    assert (tmp_path / filename).exists()


def test_invalid_report_extension_raises(summarizer, test_pdf_path):
    """Tests incorrect naming of the report to be generated."""
    _ = summarizer.summarize(test_pdf_path)
    with pytest.raises(
        ValueError, match="Summary report should have 'txt' extension"
    ):
        summarizer.generate_summary_report(filename="invalid_report.pdf")


def test_max_context_window(summarizer):
    """Tests the maximum context window for the loaded model."""
    max_context_window = summarizer._get_max_context_window()
    assert max_context_window == 512
