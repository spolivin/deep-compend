"""Script for generating summaries of PDF-articles from Arxiv."""

import argparse
import re
import textwrap
import warnings
from pathlib import Path

import fitz
import nltk
import torch
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer, logging

# Selecting GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Path to the base model on HuggingFace
CKPT_PATH = "facebook/bart-large-cnn"
# Path to LoRA adapters of the fine-tuned model
HF_REPO_PATH = "spolivin/bart-arxiv-lora"

# Suppressing warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
# Downloading additional NLTK-packages
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

# Parsing scripts arguments
parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="Path to the PDF-article to be summarized")
parser.add_argument(
    "-mot",
    "--max-output-tokens",
    help="Maximum number of output tokens",
    type=int,
    default=250,
)
parser.add_argument(
    "-stt",
    "--save-to-txt",
    help="Save summary to txt-file",
    type=bool,
    default=False,
)
args = parser.parse_args()


def extract_main_text(pdf_path):
    """
    Extracts the main body text from an article, starting from the 'Introduction'
    section (or its variations) and stopping before the 'References' section.

    :param text: The full text of the article
    :type text: str
    :raises ValueError: Exception raised if the input file is not a PDF
    :return: Extracted main body text
    :rtype: str
    """
    # Validating the input file
    if ".pdf" not in pdf_path:
        raise ValueError("Input file should have 'pdf' extension.")
    # Retrieving the article text
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    # Defining regex pattern for Introduction-like sections
    intro_pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(Introduction|Background|Overview|Intro|The Trends)\b.*?\n",
        re.IGNORECASE,
    )

    # Defining regex pattern for Reference-like section
    references_pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(References|Bibliography|Works Cited)\b",
        re.IGNORECASE,
    )

    # Searching for the introduction section
    intro_match = intro_pattern.search(text)
    start_index = intro_match.start() if intro_match else 0

    # Searching for the references section
    references_match = references_pattern.search(text, start_index)
    end_index = references_match.start() if references_match else len(text)

    # Extracting main content of the article
    main_text = text[start_index:end_index].strip()

    return main_text


def clean_text(text):
    """
    Removes numeric/author-year citations and cleans the input text

    :param text: The full text of the article
    :type text: str
    :return: Cleaned main body text
    :rtype: str
    """
    # Removing numeric citations: e.g. [4], [3,5,8]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # Removing author-year citations: e.g (Doe et al., 2020)
    text = re.sub(r"\(\s*[A-Z][a-z]+(?:\s+et al\.)?,\s*\d{4}\s*\)", "", text)

    # Removing extra spaces left after cleanup
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Removing non-word characters
    text = re.sub(r"\s+", " ", text)
    # Removing spaces left before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    return text.strip()


def summarize_text(
    model,
    tokenizer,
    text,
    device=DEVICE,
    max_input_length=1024,
    max_output_length=args.max_output_tokens,
):
    """
    Tokenizes input sequence and generates summary as output.

    :param model: Transformers Model object
    :param tokenizer: Transformers Tokenizer
    :param text: Article text
    :type text: str
    :param device: Device to be used for summary generation, defaults to DEVICE
    :type device: torch.device, optional
    :param max_input_length: Maximum context window, defaults to 1024
    :type max_input_length: int, optional
    :param max_output_length: Maximum number of generated tokens as summary, defaults
        to args.max_output_tokens
    :type max_output_length: int, optional
    :return: Formatted generated summary of the article
    :rtype: str
    """
    # Tokenizing input sequence
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_input_length
    ).to(device)
    # Moving model to device
    model = model.to(device)
    # Generating tokens as output
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_length=max_output_length, num_beams=4, early_stopping=True
        )
    # Decoding and prettifying output
    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return prettify_summary(raw_summary)


def prettify_summary(summary):
    """
    Fixes the format of the generated summary.

    :param summary: The summary generated by the model.
    :type summary: str
    :return: Prettified generated summary.
    :rtype: str
    """
    # Spliting input into sentences and capitalizing each one
    sentences = nltk.tokenize.sent_tokenize(summary)
    prettified_summary = " ".join(s.capitalize() for s in sentences)

    # Removing unwanted spaces before punctuation
    prettified_summary = re.sub(r"\s+([.,!?])", r"\1", prettified_summary)

    return prettified_summary


if __name__ == "__main__":
    # Retrieving and preprocessing the article text
    pdf_path = args.filepath
    print(f"Extracting article text from '{pdf_path}'...")
    article_text = extract_main_text(pdf_path=pdf_path)
    article_text_cleaned = clean_text(article_text)
    print("Text extracted and preprocessed")

    # Loading tokenizer and model for summarization
    print(f"Loading tokenizer from '{CKPT_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
    print(f"Loading model from '{HF_REPO_PATH}'...")
    lora_model = AutoPeftModelForSeq2SeqLM.from_pretrained(HF_REPO_PATH)
    print("Tokenizer and model loaded")

    # Running summarization
    print(f"Running summarization of the article from '{pdf_path}'...")
    lora_summary = summarize_text(
        model=lora_model, tokenizer=tokenizer, text=article_text_cleaned
    )
    print("Summarization finished")

    # Saving summary in txt-file
    if args.save_to_txt:
        # Wrapping lines of the output summary
        wrapped_lines = textwrap.wrap(lora_summary, width=130)
        # Building filename of the output summary
        match = re.search(r"([^/]+)\.pdf$", pdf_path)
        filename = f"{match.group(1)}_summarized.txt"
        # Defining filepath to the summary
        p = Path("summaries")
        p.mkdir(exist_ok=True)
        filepath = p / filename
        # Writing summary line by line to file
        with filepath.open("w", encoding="utf-8") as file:
            for line in wrapped_lines:
                file.write(line.ljust(130) + "\n")
        print(f"Summary saved to '{str(filepath)}'")
    else:
        print(lora_summary)
