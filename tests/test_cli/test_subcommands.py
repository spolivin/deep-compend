import os
from dataclasses import asdict

import pytest

from deep_compend.cli.subcommands import (
    run_keyword_extraction,
    run_summarization,
    run_text_extraction,
)


def test_run_text_extraction(test_pdf_path):
    """Tests text extraction for `extract-text` subcommand."""
    extracted_text = run_text_extraction(pdf_path=str(test_pdf_path))
    assert extracted_text is not None
    assert isinstance(extracted_text, str)
    assert len(extracted_text) > 200


@pytest.mark.parametrize(
    "lm,min_kwrd_length,max_keywords_num",
    [
        ("en_core_web_sm", 3, 2),
        ("en_core_web_sm", 5, 4),
        ("en_core_web_md", 5, 7),
        ("en_core_web_lg", 4, 10),
    ],
)
def test_run_keyword_extraction(
    test_pdf_path, lm, min_kwrd_length, max_keywords_num
):
    """Tests keyword extraction for `extract-keywords` subcommand."""
    extracted_keywords = run_keyword_extraction(
        pdf_path=str(test_pdf_path),
        min_kwrd_length=min_kwrd_length,
        lm=lm,
        max_keywords_num=max_keywords_num,
    )
    assert isinstance(extracted_keywords, list)
    assert len(extracted_keywords) <= max_keywords_num
    assert all(len(k) >= min_kwrd_length for k in extracted_keywords)


def test_run_summarization_no_report(default_config, test_pdf_path):
    """Tests summary generation for `summarize` subcommand without report generation."""
    default_config.filepath = str(test_pdf_path)
    config = asdict(default_config)
    generated_summary = run_summarization(config=config)
    assert len(generated_summary) > 10


def test_run_summarization_with_report(
    default_config, test_pdf_path, tmp_path
):
    """Tests summary generation for `summarize` subcommand with report generation."""
    # Overriding params for test article and save folder for summary report
    default_config.filepath = str(test_pdf_path)
    default_config.save_folder = str(tmp_path)
    report_name = default_config.report_name
    config = asdict(default_config)
    run_summarization(config=config, generate_report=True)
    # Verifying the existence and consistency of a summary report
    generated_report_path = tmp_path / report_name
    assert (generated_report_path).exists()
    assert os.path.getsize(generated_report_path) > 0
