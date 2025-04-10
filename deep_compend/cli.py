"""CLI for running summarization and generating reports."""

import argparse
import json
from typing import Any

from .configs import SummaryGenerationConfig
from .summarizer import ArticleSummarizer

# Defaults to be automatically used if no CLI args provided
BUILT_IN_DEFAULTS = {
    "min_output_tokens": 30,
    "max_output_tokens": 250,
    "num_beams": 4,
    "length_penalty": 1.0,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "model_path": "google-t5/t5-small",
    "tokenizer_path": None,
    "lora_adapters_path": None,
    "line_width": 100,
    "max_keywords_num": 5,
    "report_name": "summary_report.txt",
    "save_folder": "summaries",
    "min_keywords_length": 3,
}


def load_config(config_path: str) -> dict:
    """Loads JSON file.

    Args:
        config_path (str): Path to a JSON file to be loaded.

    Returns:
        dict: Dict-representation of JSON file or empty dict in case of unspecified/non-existent JSON file.
    """
    # Returning empty dict in case no JSON file is given
    if not config_path:
        print("Warning: Config file not specified. Using built-in defaults.\n")
        return {}
    try:
        # Loading JSON contents if file exists
        with open(config_path) as f:
            print(f"Loading config from '{config_path}'...\n")
            return json.load(f)
    except FileNotFoundError:
        # Returning empty dict in case of non-existent JSON file
        print(
            "Warning: Config file '{config_path}' not found. Using built-in defaults.\n"
        )
        return {}


def run_summarization(config: dict[str, Any]) -> None:
    """Generates summary and creates a summary report.

    Args:
        config (dict[str, Any]): Configuration for summarization task.
    """

    # Displaying the configuration used for summarization
    print("Using configuration:")
    for key, value in config.items():
        print(f"{key} => {value}")
    print()

    # Defining the summary generation config
    summ_config = SummaryGenerationConfig(
        min_length=config["min_output_tokens"],
        max_length=config["max_output_tokens"],
        num_beams=config["num_beams"],
        length_penalty=config["length_penalty"],
        repetition_penalty=config["repetition_penalty"],
        no_repeat_ngram_size=config["no_repeat_ngram_size"],
    )

    # Instantiating an object for summarization
    print("Loading tokenizer and model for summarization...")
    article_summarizer = ArticleSummarizer(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"],
    )

    # Optionally attaching LoRA adapters to the model specified
    print("Tokenizer and model loaded")
    if config.get("lora_adapters_path"):
        print("Attaching LoRA adapters...")
        article_summarizer.load_lora_adapters(
            lora_adapters_path=config["lora_adapters_path"]
        )
        print("LoRA adapters attached")

    # Generating summary of the text
    print("Running summarization...")
    summary = article_summarizer.summarize(  # noqa: F841
        pdf_path=config["filepath"], config=summ_config
    )

    # Generating a summary report
    print("Generating report...")
    article_summarizer.generate_summary_report(
        filename=config["report_name"],
        linewidth=config["line_width"],
        kwrds_num=config["max_keywords_num"],
        save_folder=config["save_folder"],
        min_kwrd_length=config["min_keywords_length"],
    )


def main():
    """Launches summary generation and report creation task after `deep-compend` command."""

    # Defining an Arguments Parser
    parser = argparse.ArgumentParser(
        description="Summarize a PDF article using a Hugging Face model."
    )

    # Positional argument: PDF file path
    parser.add_argument(
        "filepath", type=str, help="Path to the PDF article to be summarized"
    )

    # Other optional arguments
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config JSON file"
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        help="Path to summarization model",
    )
    parser.add_argument(
        "-tp",
        "--tokenizer-path",
        type=str,
        help="Path to summarization model tokenizer",
    )
    parser.add_argument(
        "-mxot",
        "--max-output-tokens",
        type=int,
        help="Maximum number of output tokens",
    )
    parser.add_argument(
        "-mnot",
        "--min-output-tokens",
        type=int,
        help="Minimum number of output tokens",
    )
    parser.add_argument(
        "-nb", "--num-beams", type=int, help="Number of beams for beam search"
    )
    parser.add_argument(
        "-lp",
        "--length-penalty",
        type=float,
        help="Penalty for the summary length",
    )
    parser.add_argument(
        "-rp",
        "--repetition-penalty",
        type=float,
        help="Penalty for repetitive words",
    )
    parser.add_argument(
        "-nrns",
        "--no-repeat-ngram-size",
        type=int,
        help="Avoid repetitive phrases",
    )
    parser.add_argument(
        "-lap", "--lora-adapters-path", type=str, help="Path to LoRA adapters"
    )
    parser.add_argument(
        "-lw",
        "--line-width",
        type=int,
        help="Maximum line width for report formatting",
    )
    parser.add_argument(
        "-mkn",
        "--max-keywords-num",
        type=int,
        help="Maximum number of keywords in the summary report",
    )
    parser.add_argument(
        "-mkl",
        "--min-keywords-length",
        type=int,
        help="Minimum length of keywords to consider in the summary report",
    )
    parser.add_argument(
        "-rn",
        "--report-name",
        type=str,
        help="Name of the output summary report",
    )
    parser.add_argument(
        "-sf",
        "--save-folder",
        type=str,
        help="Folder to save the generated summary",
    )

    # Parsing the arguments
    args = parser.parse_args()

    # Loading config
    config = load_config(config_path=args.config)

    # Retrieving entered CLI arguments
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    # Collecting the final configuration to be used for summary generation
    final_config = (
        {**BUILT_IN_DEFAULTS, **config, **cli_args}
        if args.config and len(cli_args) >= 2
        else {**BUILT_IN_DEFAULTS, **cli_args}
    )

    # Running summarization and generating report
    run_summarization(config=final_config)
