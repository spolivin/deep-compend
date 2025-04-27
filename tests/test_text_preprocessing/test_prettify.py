import pytest

from deep_compend.text_preprocessing import prettify_summary


@pytest.mark.parametrize(
    "input,output",
    [
        (
            "this is a test. it should be capitalized.",
            "This is a test. It should be capitalized.",
        ),
        (
            "This is a test , with extra space .",
            "This is a test, with extra space.",
        ),
        (
            "This is a broken word: summar- ization.",
            "This is a broken word: summarization.",
        ),
        (
            "This method1 is better2 than before3.",
            "This method is better than before.",
        ),
        ("", ""),
        ("This is already perfect.", "This is already perfect."),
    ],
)
def test_prettify_summary(input, output):
    """Tests the summary prettification function."""
    prettified = prettify_summary(input)
    assert prettified == output
