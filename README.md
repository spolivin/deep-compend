# ***DeepCompend***

[![PyPI](https://img.shields.io/pypi/v/deep-compend)](https://pypi.org/project/deep-compend/)
[![Tests](https://github.com/spolivin/deep-compend/actions/workflows/publish.yml/badge.svg)](https://github.com/spolivin/deep-compend/actions/workflows/publish.yml)
[![License](https://img.shields.io/github/license/spolivin/deep-compend)](https://github.com/spolivin/deep-compend/blob/master/LICENSE.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

 *DeepCompend* is a Python library capable of using any *Hugging Face* summarization model for quick generation of summaries of scientific articles in TXT-format. Using a CLI integrated into library, one can generate summaries and package them into summary reports alongside other helpful information quite easily.

## Installation

In order to use the library one can download the latest version of the library from [PyPi](https://pypi.org/project/deep-compend/) into the virtual environment using the following command:

```bash
pip install deep-compend
```

One can also install the library together with `pytest`, build packages and linters for code prettification:

```bash
pip install deep-compend[test,build,linters]
```

### Development mode

In case one wants to install the library in development mode we need to firstly clone the repository:

```bash
git clone https://github.com/spolivin/deep-compend.git
cd deep-compend
```

Next, we need to create and activate the virtual environment:

* ***Windows***

```bash
python -m venv ./.venv
.venv\Scripts\activate.bat
```

* ***Linux***
```bash
sudo apt-get update
sudo apt-get install python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
```

Lastly, we install the library in editable mode:

```bash
pip install -e .[test,build,linters]
```

## Python API

Suppose, we have a test article located on `articles/test1.pdf`. Hence, we do the following. Firstly, we import necessary classes for summarization:

```python
from deep_compend import ArticleSummarizer, SummaryGenerationConfig
```

* `ArticleSummarizer` - core class for conducting summarization, loading models and generating reports.
* `SummaryGenerationConfig` - configuration for storing the parameters of the summary generation (min/max length of summary, penalties for repetition and length, etc.).

Next, we instantiate objects for these two classes. For instance, if we want to use `facebook/bart-large-cnn` model for summarization:

```python
# Specifying the config for summary generation (given with default values)
summ_config = SummaryGenerationConfig()

# Instantiating a Summarizer object with specifying the device
summarizer = ArticleSummarizer(model_path="facebook/bart-large-cnn", run_on="cuda")
```

We can optionally attach LoRA adapters compatible with the model we used in `model_path`:

```python
# Attaching compatible LoRA adapters if needed
summarizer.load_lora_adapters(lora_adapters_path="spolivin/bart-arxiv-lora")
```

We can now specify the path to the article we need to summarize and can easily generate the summary:

```python
# Generating summary
generated_summary = summarizer.summarize(pdf_path="articles/test1.pdf", config=summ_config)
```

The text in `generated_summary` now contains the summary of the article from `articles/test1.pdf`. Lastly, we generate the report:

```python
# Generating summary report
summarizer.generate_summary_report("summary_report.txt")
```
After successful generation, one will see a message mentioning where summary has been saved (by default summary is saved in a txt-file in `summaries` folder created if non-existent).


## Command Line Interface (CLI)

In order to make the library useful, after library installation a user has access to `deep-compend` command for launching summarization. This CLI command is equipped with the following sub-commands that extends the analysis of an article to be summarized:

```bash
$ deep-compend --help

usage: deep-compend [-h] {summarize,extract-text,extract-keywords} ...

Article summarization tool

positional arguments:
  {summarize,extract-text,extract-keywords}
    summarize           Summarizes a PDF article using a Hugging Face model
    extract-text        Extracts text from article
    extract-keywords    Extracts keywords from article

options:
  -h, --help            show this help message and exit
```

* `summarize`

This subcommand launches the process of summarization like so for instance and displays the generated summary:

```bash
deep-compend summarize articles/test1.pdf --config=configs/config.json
```

In case one wants not to just display the summary but create a report for it, it is necessary to add extra `--generate-summary-report` flag:

```bash
deep-compend summarize articles/test1.pdf --config=configs/config.json --generate-summary-report=True
```
> More examples of using this subcommand can be consulted [here](./scripts/).

Other CLI arguments for this command are as follows:

```bash
$ deep-compend summarize --help

usage: deep-compend summarize [-h] [-c CONFIG] [-mp MODEL_PATH] [-tp TOKENIZER_PATH] [-mxot MAX_OUTPUT_TOKENS] [-mnot MIN_OUTPUT_TOKENS] [-nb NUM_BEAMS]
                              [-lp LENGTH_PENALTY] [-rp REPETITION_PENALTY] [-nrns NO_REPEAT_NGRAM_SIZE] [-lap LORA_ADAPTERS_PATH] [-lw LINE_WIDTH]
                              [-mkn MAX_KEYWORDS_NUM] [-mkl MIN_KEYWORDS_LENGTH] [-rn REPORT_NAME] [-sf SAVE_FOLDER] [-slm SPACY_LANG_MODEL]
                              [-gsr GENERATE_SUMMARY_REPORT]
                              filepath

Summarizes a PDF article using a Hugging Face model

positional arguments:
  filepath              Path to the PDF article to be summarized

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to the config JSON file
  -mp MODEL_PATH, --model-path MODEL_PATH
                        Path to summarization model
  -tp TOKENIZER_PATH, --tokenizer-path TOKENIZER_PATH
                        Path to summarization model tokenizer
  -mxot MAX_OUTPUT_TOKENS, --max-output-tokens MAX_OUTPUT_TOKENS
                        Maximum number of output tokens
  -mnot MIN_OUTPUT_TOKENS, --min-output-tokens MIN_OUTPUT_TOKENS
                        Minimum number of output tokens
  -nb NUM_BEAMS, --num-beams NUM_BEAMS
                        Number of beams for beam search
  -lp LENGTH_PENALTY, --length-penalty LENGTH_PENALTY
                        Penalty for the summary length
  -rp REPETITION_PENALTY, --repetition-penalty REPETITION_PENALTY
                        Penalty for repetitive words
  -nrns NO_REPEAT_NGRAM_SIZE, --no-repeat-ngram-size NO_REPEAT_NGRAM_SIZE
                        Avoid repetitive phrases
  -lap LORA_ADAPTERS_PATH, --lora-adapters-path LORA_ADAPTERS_PATH
                        Path to LoRA adapters
  -lw LINE_WIDTH, --line-width LINE_WIDTH
                        Maximum line width for report formatting
  -mkn MAX_KEYWORDS_NUM, --max-keywords-num MAX_KEYWORDS_NUM
                        Maximum number of keywords in the summary report
  -mkl MIN_KEYWORDS_LENGTH, --min-keywords-length MIN_KEYWORDS_LENGTH
                        Minimum length of keywords to consider in the summary report
  -rn REPORT_NAME, --report-name REPORT_NAME
                        Name of the output summary report
  -sf SAVE_FOLDER, --save-folder SAVE_FOLDER
                        Folder to save the generated summary
  -slm SPACY_LANG_MODEL, --spacy-lang-model SPACY_LANG_MODEL
                        Name of Spacy language model to be used for keyword extraction
  -gsr GENERATE_SUMMARY_REPORT, --generate-summary-report GENERATE_SUMMARY_REPORT
                        Trigger for summary report generation
```

* `extract-text`

This subcommand enables seeing before running the summarization the preprocessed input article text that goes as input to the model specified for the summarization. In other words, the retrieved article text starting from the Introduction and ending before References:

```bash
deep-compend extract-text articles/test1.pdf
```
> Command can be useful for understanding whether the text was retrieved correctly and allows for analyzing the input before actually running any models

* `extract-keywords`

This subcommand retrieved the keywords from the article using *Spacy*'s language models and can be useful for getting the general insight into what the paper is about:

```bash
deep-compend extract-keywords articles/test1.pdf --spacy-lang-model=en_core_web_lg --max-keywords-num=10 --min-keywords-length=7
```
> Command allows specifying the language model to use for extraction (`--spacy-lang-model`), maximum number of keywords to show (`--max-keywords-num`) and what minimum keyword length to consider (`--min-keywords-length`).


## Overriding arguments
There are two ways that one can specify arguments for the script:

* Configuration file (`--config` flag) => examples of configs can be found [here](./configs/).

* CLI arguments.

The script is programmed in such a way that when specifying both config and CLI arguments, argument with the same name in config and CLI will be overridden with the value specified in CLI. For instance, after using this command, the `--num-beams` argument will be overridden with the value of 5:

```bash
deep-compend summarize articles/test1.pdf --config=configs/t5_small_config.json --num-beams=5
```

## Example scripts

I have prepared a few shell-scripts with [examples](./scripts/) of using the script for summarization in order to demonstrate how it can be used. One can run them in the following way for some test article. I have prepared a script for automatic downloading of an article from ArXiv given its ID. For instance, we can load a famous [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) paper which has the ArXiv ID of `1512.03385`:

```bash
# Loading paper from ArXiv and saving it in 'articles' folder
python pull_arxiv_paper.py 1512.03385
```

Now we can run each of the below scripts one by one to test the CLI and different configurations:
```bash
# Using "facebook/bart-large-cnn"
bash scripts/run_bart_large.sh articles/1512.03385.pdf

# Using "facebook/bart-large-cnn" with LoRA adapters
bash scripts/run_bart_lora.sh articles/1512.03385.pdf

# Using default settings
bash scripts/run_default.sh articles/1512.03385.pdf

# Using "google-t5/t5-base"
bash scripts/run_t5_base.sh articles/1512.03385.pdf

# Using "google-t5/t5-small"
bash scripts/run_t5_small.sh articles/1512.03385.pdf

# Using "google-t5/t5-base" with overridden arguments
bash scripts/run_t5_small_override.sh articles/1512.03385.pdf
```

After running these commands, the respective summary reports with additional information and statistics will be generated and saved in `summaries` folder (by default).

## Tests

The library can be tested using the tests present in this repo but first one needs to make sure that the following command has been run:

```bash
pip install deep-compend[test]
```
Or this one (if the library is to be in development mode):

```bash
pip install -e .[test]
```

After that we can easily launch automatic tests:

```bash
pytest
```

## Library limitations

The main limitation consists in the way article sections are named. The library is written to retrieve text starting from "Introduction-like" sections until "References-like" sections to use the result as input to summary generation models. While the library is able to track the most common ways Introduction and References sections are usually named and thus retrieve text accordingly, sometimes these sections can have other names that can pose a problem for retrieving the text correctly.
