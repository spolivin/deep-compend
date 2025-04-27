import os

import pytest


def test_summarization_output(summarizer):
    """Tests the output of `summarize` method and class attributes."""
    summary = summarizer.summary
    assert isinstance(summary, str)
    assert len(summary.split()) > 5
    assert hasattr(summarizer, "summary")
    assert hasattr(summarizer, "input_token_count")
    assert hasattr(summarizer, "output_token_count")
    assert hasattr(summarizer, "summary")
    assert hasattr(summarizer, "summarization_config")


def test_summary_stats_structure(summarizer):
    """Tests the summarization statistics."""
    stats = summarizer._get_stats()
    assert stats.word_count_full > 0
    assert stats.word_count_summary > 0
    assert "%" in stats.compression_rate


def test_generate_summary_report_creates_file(summarizer, tmp_path):
    """Tests report generation."""
    filename = "test_report.txt"
    summarizer.generate_summary_report(
        filename=filename, save_folder=str(tmp_path), kwrds_num=3
    )
    generated_report_path = tmp_path / filename
    assert os.path.getsize(generated_report_path) > 0
    assert generated_report_path.exists()


def test_invalid_report_extension_raises(summarizer):
    """Tests incorrect naming of the report to be generated."""
    with pytest.raises(
        ValueError, match="Summary report should have 'txt' extension"
    ):
        summarizer.generate_summary_report(filename="invalid_report.pdf")


def test_max_context_window(summarizer):
    """Tests the maximum context window for the loaded model."""
    max_context_window = summarizer._get_max_context_window()
    assert max_context_window == 512
