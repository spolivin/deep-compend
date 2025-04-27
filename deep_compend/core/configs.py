"""Parameter configurations for summary generation."""

from dataclasses import dataclass


@dataclass
class SummaryGenerationConfig:
    """Configuration settings for summarization task.

    Identical to arguments of `generate` method for Transformers.

    Attributes:
        min_length (int): Minimum number of tokens to generate. Defaults to 30.
        max_length (int): Maximum number of tokens to generate. Defaults to 250.
        num_beams (int): Number of different options to consider during generation. Defaults to 4.
        length_penalty (float): Penalty value for a summary length. Defaults to 1.0.
        repetition_penalty (float): Penalty value for repetitions. Defaults to 1.2.
        no_repeat_ngram_size (int): Ngrams to consider to avoid repetitive phrases. Defaults to 3.
        early_stopping (bool): Indicator to stop generation at good point. Defaults to True.
    """

    min_length: int = 30
    max_length: int = 250
    num_beams: int = 4
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
