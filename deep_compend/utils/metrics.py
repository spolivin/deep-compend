"""Module for storing metrics."""


def compression_ratio(summary: str, full_text: str) -> float:
    """Computes the compression between summary text and full text.

    Args:
        summary (str): Summary of an article.
        full_text (str): Full article text.

    Returns:
        float: Compression rate.
    """
    return len(summary) / len(full_text) if len(full_text) != 0 else 0.0
