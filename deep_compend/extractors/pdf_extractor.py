"""Extraction and processing of PDF-text."""

import re
from re import Pattern

import fitz

from ..text_preprocessing import clean_text


class PDFExtractor:
    """
    Extractor and processor of relevant PDF-article text.

    Attributes:
        pdf_path (str): Path to the PDF-file.
        intro_pattern (Pattern[str]): RegEx pattern for searching the beginning of Introduction-like section.
        references_pattern (Pattern[str]): RegEx pattern for searching the beginning of References-like section.
    """

    def __init__(self, pdf_path: str):
        """
        Initializes a PDFExtractor instances.

        Args:
            pdf_path (str): Path to the PDF-file.

        Additional Attributes:
            intro_pattern (Pattern[str]): RegEx pattern for searching the beginning of Introduction-like section.
            references_pattern (Pattern[str]): RegEx pattern for searching the beginning of References-like section.
        """
        self.pdf_path = pdf_path
        # Pattern for searching Introduction-like section
        self.intro_pattern: Pattern[str] = re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(Introduction|Background|Overview|Intro|The Trends)\b.*?\n",
            re.IGNORECASE,
        )
        # Pattern for searching References-like section
        self.references_pattern: Pattern[str] = re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(References|Bibliography|Works Cited)\b",
            re.IGNORECASE,
        )

    def _extract_raw_text_from_pdf(self) -> str:
        """
        Extracts the raw text from the PDF-file of an article.

        Raises:
            ValueError: Exception raised if an input file has extension other than PDF.

        Returns:
            str: Retrieved article text in its raw form.
        """
        # Validating the input file
        if ".pdf" not in self.pdf_path:
            raise ValueError("Input file should have 'pdf' extension.")

        # Retrieving the article text
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"

        return text

    def _extract_body_text(self, text: str) -> str:
        """
        Selects the text between the beginning of Introduction and References section.

        Args:
            text (str): Raw text of an article.

        Returns:
            str: Text selected between Introduction and References.
        """
        # Searching for the Introduction-like section
        intro_match = self.intro_pattern.search(text)
        start_index = intro_match.start() if intro_match else 0

        # Searching for the References-like section
        references_match = self.references_pattern.search(text, start_index)
        end_index = references_match.start() if references_match else len(text)

        # Extracting main content of the article
        main_text = text[start_index:end_index].strip()

        return main_text

    def retrieve_processed_text(self) -> str:
        """
        Retrieves the processed and cleaned text.

        Returns:
            str: Processed and cleaned text.
        """
        # Extracting the raw text from PDF
        text = self._extract_raw_text_from_pdf()
        # Extracting the relevant article part
        text = self._extract_body_text(text)
        # Cleaning the text
        text = clean_text(text)

        return text
