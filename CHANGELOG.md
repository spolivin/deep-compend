# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.2](https://github.com/spolivin/deep-compend/compare/v0.1.1...v0.1.2) - 2025-04-27

### Features
- Re-organize several modules inside the libraries into sub-packages to improve clarity and understanding. New sub-packages that appeared: `cli` (logic behind CLI), `core` (logic behind the way that summaries are generated).
- Add automatic loading of *SpaCy* language models within the framework of `KeywordsExtractor` class from `extractors` sub-package.
- Extend CLI with `--generate-summary-report` flag for controlling report generation.
- Introduce a new function to look for NLTK resources and automatically download them if necessary.

### Fixes
- Fix a problem with CLI changing words like "high-quality" into "highquality" when running the summary prettification routine by correcting the RegEx pattern.

### Continuous Integration
- Switch publishing workflow to use Trusted Publisher for PyPI instead of token.

### Tests
- Extend the testing framework by adding new tests and covering more test cases and re-organizing the existing ones into sub-packages in `tests`.

## [v0.1.1](https://github.com/spolivin/deep-compend/compare/v0.1.0...v0.1.1) - 2025-04-19

### Fixes
- Fix runtime error due to missing NLTK resource `punkt_tab`, which caused failures during summarization.

## v0.1.0 - 2025-04-19

- Initial release with CLI for summarization, PDF parsing, and keyword extraction.
