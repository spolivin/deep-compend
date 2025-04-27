import pytest

from deep_compend.utils.metrics import compression_ratio


def test_compression_ratio_standard_case():
    """Tests standard way of computing compression ratio."""
    full = "This is the full article text with a reasonable length."
    summary = "Short summary."
    ratio = compression_ratio(summary, full)
    expected = len(summary) / len(full)
    assert abs(ratio - expected) < 1e-6


@pytest.mark.parametrize(
    "full,summary,expected",
    [
        ("This is the full article text with a reasonable length.", "", 0.0),
        ("", "Short summary", 0.0),
        ("", "", 0.0),
    ],
)
def test_compression_ratio_edge_cases(full, summary, expected):
    """Tests compression ratio computation for edge cases."""
    ratio = compression_ratio(summary, full)
    assert ratio == expected
