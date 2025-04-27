"""Text retrieval and summary generation logic."""

import textwrap
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Optional

import nltk
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    logging,
)

from ..extractors import KeywordsExtractor, PDFExtractor
from ..text_preprocessing import prettify_summary
from ..utils.downloads import ensure_nltk_resource
from ..utils.metrics import compression_ratio
from .configs import SummaryGenerationConfig

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


class ArticleSummarizer:
    """Generates a summary of an input PDF-article.

    Attributes:
        device (torch.device): Device to run summarization model on.
        model_path (str): Path to the HF's Transformer model.
        tokenizer_path (Optional[str]): Path to HF's Transformer tokenizer.
        model (PreTrainedModel): Transformers' pretrained model.
        config (PretrainedConfig): Loaded model configuration.
        tokenizer (PreTrainedTokenizer): Transformers' pretrained tokenizer.
        lora_adapters_path (Optional[str]): Path to LoRA adapters to attach.
        pdf_path (str): Path to an article to be summarized.
        clean_text (str): Article's relevant text that has been processed and cleaned.
        word_count_summary (int): Number of words in a summary generated.
        word_count_fully (int): Number of words in input article summarized.
        sentence_count_summary (int): Number of sentences in a summary generated.
        sentence_count_fully (int): Number of sentences in input article summarized.
        input_token_count (int): Number of tokens in input article text.
        output_token_count (int): Number of tokens in the generated summary.
        context_window (int): Maximum context window allowed for the model.
        summary (str): Text of the generated summary.
        summarization_config (dict[str, Any]): Config of summary generation params.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        run_on: str = "auto",
    ):
        """Initializes an ArticleSummarizer instance.

        Args:
            model_path (str): Path to the Transformer model.
            tokenizer_path (Optional[str], optional): Path to Transformer tokenizer. Defaults to None.
            run_on (str): Type of device to run summarization model on. Defaults to "auto".
        """
        # Defining the device to run summarization model on
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if run_on == "auto"
            else torch.device(run_on)
        )
        self.model_path = model_path
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path
        ).to(self.device)
        self.config: PretrainedConfig = self.model.config
        # If tokenizer path is not specified, loading specified model's tokenizer
        self.tokenizer_path = (
            self.model_path if not tokenizer_path else tokenizer_path
        )
        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(self.tokenizer_path)
        )
        # Computes maximum context window for the used model
        self.context_window = self._get_max_context_window()

        self.lora_adapters_path: Optional[str] = None

    def load_lora_adapters(self, lora_adapters_path: str) -> None:
        """Attaches LoRA adapters to the model.

        Args:
            lora_adapters_path (str): Path to LoRA adapters.
        """
        self.lora_adapters_path = lora_adapters_path
        self.model = PeftModel.from_pretrained(self.model, lora_adapters_path)

    def _get_max_context_window(self, safe_default_value: int = 1024) -> int:
        """Retrieves the maximum context window that a model can use without truncation.

        Args:
            safe_default_value (int): Safe default value to return. Defaults to 1024.

        Returns:
            int: Largest context window for a model.
        """
        # Getting values from tokenizer and model config
        tokenizer_max = self.tokenizer.model_max_length
        config_max = getattr(self.config, "max_position_embeddings", None)

        # Setting a reasonable limit and ignoring huge default values
        max_length = min(
            (tokenizer_max if tokenizer_max < 1e6 else float("inf")),
            config_max if config_max else float("inf"),
        )

        # In case everything fails, returning a safe default value
        return (
            int(max_length)
            if max_length != float("inf")
            else safe_default_value
        )

    def summarize(
        self, pdf_path: str, config: Optional[SummaryGenerationConfig] = None
    ) -> str:
        """Summarizes the text from PDF-article.

        Args:
            pdf_path (str): Path to an article to be summarized.
            config (SummaryGenerationConfig): Configuration settings for summarization task.

        Returns:
            str: Generated formatted summary of an article.
        """
        ensure_nltk_resource(resource_id="tokenizers/punkt")
        ensure_nltk_resource(resource_id="tokenizers/punkt_tab")

        # Setting generation config to default params if config not specified
        config = config or SummaryGenerationConfig()
        self.summarization_config: dict[str, Any] = asdict(config)

        # Retrieving and cleaning article text from PDF
        self.pdf_path = pdf_path
        pdf_extractor = PDFExtractor(pdf_path=pdf_path)
        text = pdf_extractor.retrieve_processed_text()
        self.clean_text = text

        # Computing the number of word and sentences in an article text
        self.word_count_full = len(nltk.tokenize.word_tokenize(text))
        self.sentence_count_full = len(nltk.tokenize.sent_tokenize(text))

        # Adding a prefix in case of T5-models
        if "t5" in self.model_path or self.tokenizer_path:
            text = "summarize: " + text

        # Tokenizing input sequence in accordance with max context window
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window,
        ).to(self.device)

        # Computing number of tokens for the input tokenized sequence
        self.input_token_count = len(inputs["input_ids"][0])

        # Generating tokens as output
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs, **self.summarization_config
            )

        # Computing number of tokens in the generated summary
        self.output_token_count = len(summary_ids[0])

        # Decoding tokens
        summary = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
        summary = prettify_summary(summary)
        self.summary = summary

        # Computing number of words and sentences in the generated summary
        self.word_count_summary = len(nltk.tokenize.word_tokenize(summary))
        self.sentence_count_summary = len(nltk.tokenize.sent_tokenize(summary))

        return summary

    @dataclass
    class SummaryStatisticsConfig:
        """Configuration for statistics of the summarization.

        Attributes:
            word_count_summary (int): Number of words in a summary generated.
            word_count_fully (int): Number of words in input article summarized.
            sentence_count_summary (int): Number of sentences in a summary generated.
            sentence_count_fully (int): Number of sentences in input article summarized.
            input_token_count (int): Number of tokens in input article text.
            output_token_count (int): Number of tokens in the generated summary.
            compression_rate (str): Value of compression rate between summary and article (in percent).
        """

        word_count_summary: int
        word_count_full: int
        sentence_count_summary: int
        sentence_count_full: int
        input_token_count: int
        output_token_count: int
        compression_rate: str

    def _get_stats(self) -> SummaryStatisticsConfig:
        """Collects statistics after summary generation.

        Args:
            summary (str): Text of summary generated.
            full_text (str): Full text of an article.

        Returns:
            SummaryStatisticsConfig: Object of SummaryStatisticsConfig class.
        """
        return self.SummaryStatisticsConfig(
            word_count_summary=self.word_count_summary,
            word_count_full=self.word_count_full,
            sentence_count_summary=self.sentence_count_summary,
            sentence_count_full=self.sentence_count_full,
            input_token_count=self.input_token_count,
            output_token_count=self.output_token_count,
            compression_rate=f"{compression_ratio(self.summary, self.clean_text):.2%}",
        )

    class SummaryReportGenerator:
        """Generates a report for the generated summary.

        Attributes:
            summarizer (ArticleSummarizer): Instance of ArxivSummarizer class.
            save_folder (str): Name of a folder where to save the report.
            kwrds_num (int): Number of keywords to include into the report.
            linewidth (int): Max line width in the report.
            statistics (dict[str, Any]): Statistics to include into the report.
            keywords_extractor (KeywordsExtractor): Instance of a KeywordsExtractor class.
        """

        def __init__(
            self,
            summarizer: "ArticleSummarizer",
            save_folder: str = "summaries",
            kwrds_num: int = 5,
            linewidth: int = 100,
            lm: str = "en_core_web_sm",
            min_kwrd_length: int = 3,
            most_common_elems: int = 20,
        ):
            """Initializes a SummaryReportGenerator instance.

            Args:
                summarizer (ArticleSummarizer): Instance of ArticleSummarizer class.
                save_folder (str, optional): Name of a folder where to save the report. Defaults to "summaries".
                kwrds_num (int, optional): Number of keywords to include into the report. Defaults to 5.
                linewidth (int, optional): Max line width in the report. Defaults to 100.
                lm (str, optional): Name of a language model to be used for keyword extraction. Defaults to "en_core_web_sm".
                min_kwrds_length (int, optional): Minimal length of keyword to include. Defaults to 3.
                most_common_elems (int, optional): Number of the most frequent words to consider. Defaults to 20.
            """
            self.summarizer: ArticleSummarizer = summarizer
            self.save_folder = save_folder
            self.kwrds_num = kwrds_num
            self.linewidth = linewidth
            self.statistics: dict[str, Any] = asdict(summarizer._get_stats())
            self.keywords_extractor = KeywordsExtractor(
                lm=lm,
                min_kwrd_length=min_kwrd_length,
                most_common_elems=most_common_elems,
            )

        def _generate_filepath(self, filename: str) -> Path:
            """Creates a Path object to the file for the report.

            Args:
                filename (str): Name of a summary report file.

            Returns:
                Path: Path to the summary report.
            """
            p = Path(self.save_folder)
            p.mkdir(exist_ok=True)
            filepath = p / filename

            return filepath

        def _format_statistics(self) -> str:
            """Formats the statistics for the summary report.

            Returns:
                str: Formatted string with statistics.
            """

            return "\n".join(
                [f"{key}: {value}" for key, value in self.statistics.items()]
            )

        def generate_txt_report(self, filename: Optional[str] = None) -> None:
            """Generates a summary report in TXT-format.

            Args:
                filename (Optional[str], optional): Name of file for the summary report. Defaults to None.
            """
            # Generating Summarization Report ID
            report_id = uuid.uuid4().hex[:6]
            # Using Report ID as filename if it is not specified
            filename = filename or f"{report_id}.txt"

            # Creating a filepath for the summary report
            filepath = self._generate_filepath(filename=filename)
            # Computing keywords for the input article
            kwrds = self.keywords_extractor.extract(self.summarizer.clean_text)
            # Writing to file with a report
            with filepath.open("w", encoding="utf-8") as file:
                file.write(
                    f"=== Summarization Report {report_id.upper()} ===\n\n"
                )
                file.write(f"Article path: '{self.summarizer.pdf_path}'\n")
                file.write(f"Model: {self.summarizer.model_path}\n")
                file.write(f"Tokenizer: {self.summarizer.tokenizer_path}\n")
                file.write(
                    f"Context window: {self.summarizer.context_window}\n"
                )
                file.write(
                    f"LoRA: {'None' if not self.summarizer.lora_adapters_path else self.summarizer.lora_adapters_path}\n"
                )
                file.write(
                    f"Gen time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}\n"
                )
                file.write(f"Keywords: {kwrds[:self.kwrds_num]}\n\n")
                file.write("-" * (self.linewidth + 3) + "\n")
                file.write(
                    "\n".join(
                        textwrap.wrap(
                            self.summarizer.summary, width=self.linewidth
                        )
                    )
                    + "\n"
                )
                file.write("-" * (self.linewidth + 3))
                file.write(
                    f"\n\nStatistics:\n{'-' * 10}\n{self._format_statistics()}\n"
                )
            print(f"Summary saved to '{str(filepath)}'")

    def generate_summary_report(
        self,
        filename: Optional[str] = None,
        save_folder: str = "summaries",
        linewidth: int = 100,
        kwrds_num: int = 5,
        lm: str = "en_core_web_sm",
        min_kwrd_length: int = 3,
        most_common_elems: int = 20,
    ) -> None:
        """Generates a summary report.

        Args:
            filename (Optional[str], optional): Report name. Defaults to None.
            save_folder (str, optional): Folder where to save a report. Defaults to "summaries".
            linewidth (int, optional): Max width of a line in a report. Defaults to 100.
            kwrds_num (int, optional): Number of keywords to show in report. Defaults to 5.
            lm (str, optional): Name of a language model to be used for keyword extraction. Defaults to "en_core_web_sm".
            min_kwrds_length (int, optional): Minimal length of keyword to include. Defaults to 3.
            most_common_elems (int, optional): Number of the most frequent words to consider. Defaults to 20.

        Raises:
            ValueError: Exception raised if extension file in `filename` is not "txt".
        """
        # Validating the filename of a report
        if (filename is not None) and (".txt" not in filename):
            raise ValueError("Summary report should have 'txt' extension.")

        # Collecting all statistics and creating a report
        report_generator = self.SummaryReportGenerator(
            summarizer=self,
            linewidth=linewidth,
            kwrds_num=kwrds_num,
            save_folder=save_folder,
            lm=lm,
            min_kwrd_length=min_kwrd_length,
            most_common_elems=most_common_elems,
        )
        report_generator.generate_txt_report(filename=filename)
