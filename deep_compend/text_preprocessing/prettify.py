import re

import nltk

from ..utils.downloads import ensure_nltk_resource

# Checking the presence of auxiliary NLTK packages
ensure_nltk_resource(resource_id="tokenizers/punkt")
ensure_nltk_resource(resource_id="tokenizers/punkt_tab")


def prettify_summary(summary: str) -> str:
    """
    Corrects the format problems of an input summary text.

    Args:
        summary (str): Text of a summary.

    Returns:
        str: Prettified text.
    """
    # Splitting input into sentences and capitalizing each one
    sentences = nltk.tokenize.sent_tokenize(summary)
    prettified_summary = " ".join(s.capitalize() for s in sentences)
    # Removing unwanted spaces before punctuation
    prettified_summary = re.sub(r"\s+([.,!?])", r"\1", prettified_summary)
    # Fixing word shift problems: e.g. regul- ation
    prettified_summary = re.sub(
        r"(\w+)-\s+\n*\s*(\w+)", r"\1\2", prettified_summary
    )
    # Removing numbers after words: e.g. regularities1
    prettified_summary = re.sub(r"(\w+)\d+", r"\1", prettified_summary)

    return prettified_summary
