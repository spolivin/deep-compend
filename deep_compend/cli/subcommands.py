from typing import Any, Optional

from ..core.configs import SummaryGenerationConfig
from ..core.summarizer import ArticleSummarizer
from ..extractors import KeywordsExtractor, PDFExtractor


def run_summarization(
    config: dict[str, Any], generate_report: bool = False
) -> Optional[str]:
    """Generates summary and/or creates a summary report.

    Args:
        config (dict[str, Any]): Configuration for summarization task.
        generate_report (bool, optional): Flag to additionally generate summary report. Defaults to False.

    Returns:
        Optional[str]: None or text of the generated summary.
    """
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
    article_summarizer = ArticleSummarizer(
        model_path=config["model_path"],
        tokenizer_path=config.get("tokenizer_path"),
    )

    # Optionally attaching LoRA adapters to the model specified
    if config.get("lora_adapters_path"):
        article_summarizer.load_lora_adapters(
            lora_adapters_path=config["lora_adapters_path"]
        )

    # Generating summary of the text
    summary = article_summarizer.summarize(
        pdf_path=config["filepath"], config=summ_config
    )

    # Generating a summary report
    if generate_report:
        article_summarizer.generate_summary_report(
            filename=config["report_name"],
            linewidth=config["line_width"],
            kwrds_num=config["max_keywords_num"],
            save_folder=config["save_folder"],
            min_kwrd_length=config["min_keywords_length"],
            lm=config["spacy_lang_model"],
        )
        return

    return summary


def run_text_extraction(pdf_path: str) -> str:
    """Retrieves preprocessed text from an article that goes as input to the model.

    Args:
        pdf_path (str): Path to PDF-article.
    """
    pdf_extractor = PDFExtractor(pdf_path=pdf_path)
    extracted_text = pdf_extractor.retrieve_processed_text()

    return extracted_text


def run_keyword_extraction(
    pdf_path: str, lm: str, min_kwrd_length: int, max_keywords_num: int
) -> list[str]:
    """Retrieves keywords from an article.

    Args:
        pdf_path (str): Path to PDF-article.
        lm (str): Name of a language model to be used for keyword extraction.
        min_kwrd_length (int): Minimum length of a keyword to consider.
        max_keywords_num (int): Maximum number of keywords to show.
    """
    pdf_extractor = PDFExtractor(pdf_path=pdf_path)
    text = pdf_extractor.retrieve_processed_text()

    kwrds_extractor = KeywordsExtractor(lm=lm, min_kwrd_length=min_kwrd_length)
    kwrds = kwrds_extractor.extract(text)

    return kwrds[:max_keywords_num]
