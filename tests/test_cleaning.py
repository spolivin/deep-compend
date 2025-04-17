from deep_compend.text_preprocessing import clean_text


def test_numeric_citations_removal():
    """Tests if the numeric citations are removed."""
    text = "This method is used frequently [1]. It is based on earlier studies [2, 3]."
    cleaned = clean_text(text)
    assert "[" not in cleaned and "]" not in cleaned
    assert "frequently" in cleaned
    assert (
        cleaned
        == "This method is used frequently. It is based on earlier studies."
    )


def test_author_year_citations_removal():
    """Tests if the text citations are removed."""
    text = "This has been shown previously (Smith et al., 2021)."
    cleaned = clean_text(text)
    assert "(Smith et al., 2021)" not in cleaned
    assert "previously" in cleaned
    assert cleaned == "This has been shown previously."


def test_extra_spaces_removal():
    """Tests if extra spaces are removed."""
    text = "This    is   a test.  "
    cleaned = clean_text(text)
    assert cleaned == "This is a test."


def test_space_before_punctuation_removal():
    """Tests if the spaces before punctuation are removed."""
    text = "This is a sentence . And another one , too !"
    cleaned = clean_text(text)
    assert cleaned == "This is a sentence. And another one, too!"


def test_mixed_cases_handling():
    """Tests multiple cases of removal."""
    text = "Studies [2] show (Doe, 2020) results are valid ."
    cleaned = clean_text(text)
    assert cleaned == "Studies show results are valid."


def test_empty_input():
    """Tests empty input."""
    assert clean_text("") == ""


def test_citations_only():
    """Tests the removal of only citations in a sentence."""
    text = "[1] (Smith et al., 2020)"
    cleaned = clean_text(text)
    assert cleaned == ""


def test_normal_text():
    """Tests if normal texts gets changed."""
    text = "No citations here. This is just plain text."
    cleaned = clean_text(text)
    assert cleaned == "No citations here. This is just plain text."
