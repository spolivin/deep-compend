import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class DefaultCLIParametersConfig:
    """Default configuration settings for running CLI.

    Attributes:
        filepath (str): Path to an article to be summarized.
        min_output_tokens (int): Minimum number of tokens to generate. Defaults to 30.
        max_output_tokens (int): Maximum number of tokens to generate. Defaults to 250.
        num_beams (int): Number of different options to consider during generation. Defaults to 4.
        length_penalty (float): Penalty value for a summary length. Defaults to 1.0.
        repetition_penalty (float): Penalty value for repetitions. Defaults to 1.2.
        no_repeat_ngram_size (int): Ngrams to consider to avoid repetitive phrases. Defaults to 3.
        model_path (str): Path to the summarization model. Defaults to "google-t5/t5-small".
        tokenizer_path (Optional[str]): Path to the summarization model tokenizer. Defaults to None.
        line_width (int): Maximum line width in a summary report. Defaults to 100.
        max_keywords_num (int): Maximum number of keywords to show in a summary report. Defaults to 5.
        report_name (str): Name of a summary report. Defaults to "summary_report.txt".
        save_folder (str): Name of a directory to sace a summary report. Defaults to "summaries".
        min_keywords_length (int): Minimum length of a keyword to consider. Defaults to 3.
        spacy_lang_model (str): Name of a SpaCy model to use for keywords retrieval. Defaults to "en_core_web_sm".
        config (Optional[str]): Name of a config file for summary generation. Defaults to None.
    """

    filepath: str
    min_output_tokens: int = 30
    max_output_tokens: int = 250
    num_beams: int = 4
    lora_adapters_path: Optional[str] = None
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    model_path: str = "google-t5/t5-small"
    tokenizer_path: Optional[str] = None
    line_width: int = 100
    max_keywords_num: int = 5
    report_name: str = "summary_report.txt"
    save_folder: str = "summaries"
    min_keywords_length: int = 3
    spacy_lang_model: str = "en_core_web_sm"
    config: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "DefaultCLIParametersConfig":
        """Filters and overrides only known keys from input dict.

        Args:
            d (dict): Dictionary with values needed to be overridden.

        Returns:
            DefaultCLIParametersConfig: Object with the overridden values.
        """
        return DefaultCLIParametersConfig(
            **{
                k: v
                for k, v in d.items()
                if k in DefaultCLIParametersConfig.__annotations__
            }
        )


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
            f"Warning: Config file '{config_path}' not found. Using built-in defaults.\n"
        )
        return {}


def merge_configs(
    default_config: dict,
    file_config: dict,
    cli_config: dict,
) -> dict:
    """Merges dictionaries with summary generation parameters.

    Args:
        default_config (dict): Collection of default parameters.
        file_config (dict): Collection of parameters retrieved from config file.
        cli_config (dict): Collection of parameters retrieved from CLI.

    Returns:
        dict: Merged dictionary.
    """
    merged = default_config.copy()
    merged.update(file_config)
    merged.update(cli_config)

    return asdict(DefaultCLIParametersConfig.from_dict(merged))
