from deep_compend.utils.metrics import compression_ratio


def test_compression_ratio_standard_case():
    """Tests standard way of computing compression ratio."""
    full = "This is the full article text with a reasonable length."
    summary = "Short summary."
    ratio = compression_ratio(summary, full)
    expected = len(summary) / len(full)
    assert abs(ratio - expected) < 1e-6


def test_compression_ratio_empty_summary():
    """Tests compression ratio computation in case summary is empty."""
    full = "This is the full article text with a reasonable length."
    summary = ""
    ratio = compression_ratio(summary, full)
    assert ratio == 0.0


def test_compression_ratio_empty_full_text():
    """Tests compression ratio computation in case full text is empty."""
    full = ""
    summary = "Short summary"
    ratio = compression_ratio(summary, full)
    assert ratio == 0.0


def test_compression_ratio_empty_both():
    """Tests compression ratio computation in case summary and full text are empty."""
    full = ""
    summary = ""
    ratio = compression_ratio(summary, full)
    assert ratio == 0.0
