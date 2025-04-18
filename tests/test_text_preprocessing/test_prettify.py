import nltk

from deep_compend.text_preprocessing import prettify_summary

nltk.download("punkt", quiet=True)


def test_sentence_capitalization():
    """Tests sentences capitalization."""
    text = "this is a test. it should be capitalized."
    result = prettify_summary(text)
    assert result == "This is a test. It should be capitalized."


def test_removes_space_before_punctuation():
    """Tests the removal of spaces before punctuation."""
    text = "This is a test , with extra space ."
    result = prettify_summary(text)
    assert " ," not in result and " ." not in result
    assert result.endswith("space.")
    assert result == "This is a test, with extra space."


def test_hyphenated_line_breaks_fixed():
    """Tests if the line breaks get fixed."""
    text = "This is a broken word: summar- ization."
    result = prettify_summary(text)
    assert "summarization" in result
    assert "-" not in result


def test_trailing_numbers_removal():
    """Tests the removal of trailing numbers after words."""
    text = "This method1 is better2 than before3."
    result = prettify_summary(text)
    assert "method" in result and "method1" not in result
    assert "before3" not in result


def test_empty_input():
    """Tests empty input."""
    assert prettify_summary("") == ""


def test_normal_text():
    """Tests if normal texts gets changed."""
    text = "This is already perfect."
    result = prettify_summary(text)
    assert result == "This is already perfect."
