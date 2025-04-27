"""CLI for running summarization and generating reports."""

import argparse
import sys
from dataclasses import asdict

from .config import DefaultCLIParametersConfig, load_config, merge_configs
from .subcommands import (
    run_keyword_extraction,
    run_summarization,
    run_text_extraction,
)


def main():
    """CLI for `deep-compend` command with subcommands."""

    # Defining an Arguments Parser and sub-parsers
    parser = argparse.ArgumentParser(description="Article summarization tool")
    subparsers = parser.add_subparsers(dest="command")

    # ---------------- Summarization/Report generation sub-parser ---------------------------#

    summ_parser = subparsers.add_parser(
        "summarize",
        description="Summarizes a PDF article using a Hugging Face model",
        help="Summarizes a PDF article using a Hugging Face model",
    )

    # Positional argument: PDF file path
    summ_parser.add_argument(
        "filepath", type=str, help="Path to the PDF article to be summarized"
    )

    # Other optional arguments
    summ_parser.add_argument(
        "-c", "--config", type=str, help="Path to the config JSON file"
    )
    summ_parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        help="Path to summarization model",
    )
    summ_parser.add_argument(
        "-tp",
        "--tokenizer-path",
        type=str,
        help="Path to summarization model tokenizer",
    )
    summ_parser.add_argument(
        "-mxot",
        "--max-output-tokens",
        type=int,
        help="Maximum number of output tokens",
    )
    summ_parser.add_argument(
        "-mnot",
        "--min-output-tokens",
        type=int,
        help="Minimum number of output tokens",
    )
    summ_parser.add_argument(
        "-nb", "--num-beams", type=int, help="Number of beams for beam search"
    )
    summ_parser.add_argument(
        "-lp",
        "--length-penalty",
        type=float,
        help="Penalty for the summary length",
    )
    summ_parser.add_argument(
        "-rp",
        "--repetition-penalty",
        type=float,
        help="Penalty for repetitive words",
    )
    summ_parser.add_argument(
        "-nrns",
        "--no-repeat-ngram-size",
        type=int,
        help="Avoid repetitive phrases",
    )
    summ_parser.add_argument(
        "-lap", "--lora-adapters-path", type=str, help="Path to LoRA adapters"
    )
    summ_parser.add_argument(
        "-lw",
        "--line-width",
        type=int,
        help="Maximum line width for report formatting",
    )
    summ_parser.add_argument(
        "-mkn",
        "--max-keywords-num",
        type=int,
        help="Maximum number of keywords in the summary report",
    )
    summ_parser.add_argument(
        "-mkl",
        "--min-keywords-length",
        type=int,
        help="Minimum length of keywords to consider in the summary report",
    )
    summ_parser.add_argument(
        "-rn",
        "--report-name",
        type=str,
        help="Name of the output summary report",
    )
    summ_parser.add_argument(
        "-sf",
        "--save-folder",
        type=str,
        help="Folder to save the generated summary",
    )
    summ_parser.add_argument(
        "-slm",
        "--spacy-lang-model",
        type=str,
        help="Name of Spacy language model to be used for keyword extraction",
    )
    summ_parser.add_argument(
        "-gsr",
        "--generate-summary-report",
        type=bool,
        help="Trigger for summary report generation",
    )

    # ---------------- Text retrieval sub-parser ---------------------------#

    text_parser = subparsers.add_parser(
        "extract-text",
        description="Extracts text from article",
        help="Extracts text from article",
    )
    text_parser.add_argument(
        "filepath", type=str, help="Path to the PDF article"
    )

    # ---------------- Keywords retrieval sub-parser ---------------------------#

    kwrds_parser = subparsers.add_parser(
        "extract-keywords",
        description="Extracts keywords from article",
        help="Extracts keywords from article",
    )
    kwrds_parser.add_argument(
        "filepath", type=str, help="Path to the PDF article to be summarized"
    )
    kwrds_parser.add_argument(
        "-mxkn",
        "--max-keywords-num",
        type=int,
        help="Maximum number of keywords to include",
        default=5,
    )
    kwrds_parser.add_argument(
        "-mnkl",
        "--min-keywords-length",
        type=int,
        help="Minimum length of keywords to consider",
        default=5,
    )
    kwrds_parser.add_argument(
        "-slm",
        "--spacy-lang-model",
        type=str,
        help="Name of Spacy language model to be used for keyword extraction",
        default="en_core_web_sm",
    )

    # -----------------------------------------------------------------------------#

    # Parsing the arguments
    args = parser.parse_args()
    try:
        # Sub-command to extract text from article
        if args.command == "extract-text":
            extracted_text = run_text_extraction(pdf_path=args.filepath)
            print(f"Extracted text: {extracted_text}")

        # Sub-command to extract keywords from article text
        elif args.command == "extract-keywords":
            extracted_keywords = run_keyword_extraction(
                pdf_path=args.filepath,
                lm=args.spacy_lang_model,
                min_kwrd_length=args.min_keywords_length,
                max_keywords_num=args.max_keywords_num,
            )
            print(f"Extracted keywords: {extracted_keywords}")

        # Sub-command to run summarization and summary report generation
        elif args.command == "summarize":
            # Loading config if present
            config = load_config(config_path=args.config)

            # Retrieving keys/values from CLI arguments
            cli_args = {k: v for k, v in vars(args).items() if v is not None}

            # Retrieving default parameters for CLI
            default_params_config = asdict(
                DefaultCLIParametersConfig(filepath=args.filepath)
            )

            # Collecting the final configuration to be used for summary generation
            final_config = merge_configs(
                default_config=default_params_config,
                file_config=config,
                cli_config=cli_args,
            )

            # Running summarization and generating report
            if args.generate_summary_report:
                run_summarization(config=final_config, generate_report=True)
            else:
                # Running summarization and displaying summary without report generation
                generated_summary = run_summarization(config=final_config)
                print(f"Generated summary: {generated_summary}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
