from deep_compend.extractors import PDFExtractor


def test_pdf_extractor_returns_text(test_pdf_path):
    """Tests the correctness of the text extracted from PDF."""
    extractor = PDFExtractor(str(test_pdf_path))
    text = extractor.retrieve_processed_text()

    assert isinstance(text, str)
    assert len(text) > 200
