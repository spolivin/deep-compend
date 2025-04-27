"""Extraction of keywords."""

import sys
from collections import Counter

import spacy


class KeywordsExtractor:
    """
    Extractor of relevant words from an article text.

    Attributes:
        lm (str): Name of a language model to be used for extraction.
        min_kwrd_length (int): Minimal length of keyword to include.
        most_common_elems (int): Number of the most frequent words to consider.
    """

    def __init__(
        self,
        lm: str = "en_core_web_sm",
        min_kwrd_length: int = 3,
        most_common_elems: int = 20,
    ):
        """
        Initializes a KeywordsExtractor instance.

        Args:
            lm (str, optional): Name of a language model to be used for extraction. Defaults to "en_core_web_sm".
            min_kwrd_length (int, optional): Minimal length of keyword to include. Defaults to 3.
            most_common_elems (int, optional): Number of the most frequent words to consider. Defaults to 20.
        """
        self.lm = lm
        self.min_kwrd_length = min_kwrd_length
        self.most_common_elems = most_common_elems

        # Downloading a SpaCy language model if not present
        try:
            spacy.load(self.lm)
        except OSError:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "spacy", "download", f"{self.lm}"],
                check=True,
            )

    def extract(self, text: str) -> list[str]:
        """
        Extracts keywords from an input text.

        Args:
            text (str): Text from which to extract keywords.

        Returns:
            list[str]: Collection of extracted keywords.
        """
        # Loading the language model and using it on input text
        nlp = spacy.load(self.lm)
        doc = nlp(text)

        # Extracting noun-based keywords
        candidates = [
            token.text.lower()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop
        ]

        # Extending the keywords candidates with named entities
        candidates.extend([ent.text.lower() for ent in doc.ents])

        # Counting the occurrences
        word_freq = Counter(candidates)
        top_keywords = [
            word
            for word, _ in word_freq.most_common(self.most_common_elems)
            if len(word) >= self.min_kwrd_length and word.isalpha()
        ]

        return top_keywords
