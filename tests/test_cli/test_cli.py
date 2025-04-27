import os
import sys

import pytest

from deep_compend.cli.cli import main


@pytest.mark.parametrize(
    "cli_args, expected_help_output",
    [
        (["deep-compend", "--help"], "usage"),
        (["deep-compend", "extract-text", "--help"], "usage"),
        (["deep-compend", "extract-keywords", "--help"], "usage"),
        (["deep-compend", "summarize", "--help"], "usage"),
    ],
)
def test_main_cli_help_message(
    monkeypatch, capsys, cli_args, expected_help_output
):
    """Tests the help message of CLI."""
    # Mocking running the CLI
    monkeypatch.setattr(sys, "argv", cli_args)

    try:
        main()
    except SystemExit:
        pass

    # Checking if the output contains the expected words
    captured = capsys.readouterr()
    assert expected_help_output in captured.out


@pytest.mark.parametrize(
    "cli_args",
    [
        ["deep-compend", "extract-text"],
    ],
)
def test_main_cli_extract_text(monkeypatch, capsys, test_pdf_path, cli_args):
    """Tests `extract-text` subcommand of the CLI."""
    # Adding path to article to the CLI
    final_cli_args = cli_args + [str(test_pdf_path)]
    monkeypatch.setattr(sys, "argv", final_cli_args)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Extracted text" in captured.out


@pytest.mark.parametrize(
    "cli_args",
    [
        ["deep-compend", "extract-keywords", "--max-keywords-num", "5"],
        ["deep-compend", "extract-keywords", "--min-keywords-length", "5"],
    ],
)
def test_main_cli_extract_keywords(
    monkeypatch, capsys, test_pdf_path, cli_args
):
    """Tests `extract-keywords` subcommand of the CLI."""
    final_cli_args = cli_args + [str(test_pdf_path)]
    monkeypatch.setattr(sys, "argv", final_cli_args)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Extracted keywords" in captured.out


@pytest.mark.parametrize(
    "cli_args,expected_config_msg",
    [
        (["deep-compend", "summarize"], "not specified"),
        (
            ["deep-compend", "summarize", "--config", "configs/config.json"],
            "not found",
        ),
        (
            [
                "deep-compend",
                "summarize",
                "--config",
                "configs/t5_small_config.json",
            ],
            "Loading config",
        ),
        (["deep-compend", "summarize", "--num-beams", "3"], "not specified"),
    ],
)
def test_main_cli_summarize(
    monkeypatch, capsys, test_pdf_path, cli_args, expected_config_msg
):
    """Tests `summarize` subcommand of the CLI (without report generation).

    Checks if the correct message is shown when loading config from file and generating summary.
    """
    final_cli_args = cli_args + [str(test_pdf_path)]
    monkeypatch.setattr(sys, "argv", final_cli_args)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert expected_config_msg in captured.out
    assert "Generated summary" in captured.out


@pytest.mark.parametrize(
    "cli_args,expected_config_msg",
    [
        (["deep-compend", "summarize"], "not specified"),
        (
            [
                "deep-compend",
                "summarize",
                "--num-beams",
                "2",
            ],
            "not specified",
        ),
        (
            [
                "deep-compend",
                "summarize",
                "--config",
                "configs/t5_small_config.json",
                "--report-name",
                "summary_report.txt",
            ],
            "Loading config",
        ),
        (
            ["deep-compend", "summarize", "--config", "configs/config.json"],
            "not found",
        ),
    ],
)
def test_main_cli_summarize_report(
    monkeypatch, capsys, test_pdf_path, tmp_path, cli_args, expected_config_msg
):
    """Tests `summarize` subcommand of the CLI (with report generation).

    Checks if the correct message is shown when loading config from file and generating summary.
    """
    # Adding path to article and flag to generate report to the CLI
    final_cli_args = cli_args + [str(test_pdf_path)]
    final_cli_args += ["--generate-summary-report"] + ["True"]
    final_cli_args += ["--save-folder"] + [str(tmp_path)]
    monkeypatch.setattr(sys, "argv", final_cli_args)

    exit_code = main()

    captured = capsys.readouterr()

    generated_report_path = tmp_path / "summary_report.txt"

    assert exit_code == 0
    assert expected_config_msg in captured.out
    assert "Summary saved to" in captured.out
    assert generated_report_path.exists()
    assert os.path.getsize(generated_report_path) > 0
