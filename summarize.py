"""
Scientific Articles Summarizer Script
=====================================

This script processes a PDF file, summarizes its content using a Hugging Face model, and generates a report.

In case of both given config and cli arguments, cli arguments override those given in a config file.

Usage:
    python summarize.py input.pdf --config=config.json --report-name=summ_report_lora.txt

Arguments:
    filename (str): Path to the PDF article to be summarized.
    --config (str, optional): Path to the config JSON file.
    --model-path (str, optional): Path to a Hugging Face summarization model.
    --tokenizer-path (str, optional): Path to a Hugging Face summarization model tokenizer.
    --max-output-tokens (int, optional): Maximum number of output tokens.
    --min-output-tokens (int, optional): Minimum number of output tokens.
    --num-beams (int, optional): Number of beams for beam search algorithm.
    --length-penalty (int, optional): Penalty for the summary length.
    --repetition-penalty (int, optional): Penalty for repetitive words.
    --no-repeat-ngram-size (int, optional): Avoiding repetitive phrases.
    --lora-adapters-path (str, optional): Path to LoRA adapters
    --line-width (int, optional): Maximum line width for report formatting.
    --max-keywords-num (int, optional): Maximum number of keywords in the summary report.
    --min-keywords-length (int, optional): Minimum length of keywords to consider in the summary report.
    --report-name (str, optional): Name of the output summary report.
    --save-folder (str, optional): Folder to save the generated summary.
"""

import argparse
import json

from deep_compend import ArticleSummarizer, SummaryGenerationConfig

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

# Load config file
config = load_config(config_path=args.config)

# Retrieving entered CLI arguments
cli_args = {k: v for k, v in vars(args).items() if v is not None}

# Collecting the final configuration to be used for summary generation
final_config = (
    {**BUILT_IN_DEFAULTS, **config, **cli_args}
    if args.config and len(cli_args) >= 2
    else {**BUILT_IN_DEFAULTS, **cli_args}
)


if __name__ == "__main__":
    # Echoing the arguments/values to be used for summarization
    print("Using configuration:")
    for key, value in final_config.items():
        print(f"{key} => {value}")
    print()

    # Defining a summarization generation configuration
    summ_config = SummaryGenerationConfig(
        min_length=final_config["min_output_tokens"],
        max_length=final_config["max_output_tokens"],
        num_beams=final_config["num_beams"],
        length_penalty=final_config["length_penalty"],
        repetition_penalty=final_config["repetition_penalty"],
        no_repeat_ngram_size=final_config["no_repeat_ngram_size"],
    )

    # Instantiating an object for summarization
    print("Loading tokenizer and model for summarization...")
    article_summarizer = ArticleSummarizer(
        model_path=final_config["model_path"],
        tokenizer_path=final_config["tokenizer_path"],
    )

    # Optionally attaching LoRA adapters to the model specified
    print("Tokenizer and model loaded")
    if final_config.get("lora_adapters_path"):
        print("Attaching LoRA adapters...")
        article_summarizer.load_lora_adapters(
            lora_adapters_path=final_config["lora_adapters_path"]
        )
        print("LoRA adapters attached")

    # Generating summary of the text
    print("Running summarization...")
    summary = article_summarizer.summarize(
        pdf_path=final_config["filepath"], config=summ_config
    )

    # Generating a summary report
    print("Generating report...")
    article_summarizer.generate_summary_report(
        filename=final_config["report_name"],
        linewidth=final_config["line_width"],
        kwrds_num=final_config["max_keywords_num"],
        save_folder=final_config["save_folder"],
        min_kwrd_length=final_config["min_keywords_length"],
    )
