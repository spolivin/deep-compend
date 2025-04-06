# ***DeepCompend***

The purpose of this repository is to provide access to Python API for easy generation of summaries of scientific articles for quicker
and easier understanding. Initially planned as a repo solely for summary generation using [the model I fune-tuned](https://huggingface.co/spolivin/bart-arxiv-lora), I have re-purposed the repository and thus written a small Python library called `deep-compend` capable of using any *Hugging Face* summarization model for quick generation of summaries of articles in TXT-format (for now).

Using [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) as a base model, I have fine-tuned it with *LoRA* on [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) dataset with multiple examples of article texts taken from *Arxiv*. Hence, this fine-tuned model can serve as a good way to generate summaries of articles from *Arxiv*.

## Preparing virtual environment

Before being able to run the summarization, one need to firstly create virtual environment and load [necessary packages](./requirements.txt):

* ***Windows***

```bash
python -m venv ./.venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

* ***Linux***
```bash
sudo apt-get update
sudo apt-get install python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Since the library makes use of *Spacy*'s language models, it is also necessary to install some model, for instance, `en_core_web_sm`:
```bash
python -m spacy download en_core_web_sm
```

## Python API

Currently the library exists as [a folder](./deep_compend/) on GitHub so one should first clone the repository:

```bash
git clone https://github.com/spolivin/deep-compend.git
```

Next, we can get right on to summarization. Suppose, we have a test article located on `articles/test1.pdf`. Hence, we do the following. Firstly, we import necessary classes for summarization:

```python
from deep_compend import ArticleSummarizer, SummaryGenerationConfig
```

* `ArticleSummarizer` - core class for conducting summarization, loading models and generating reports.
* `SummaryGenerationConfig` - configuration for storing the parameters of the summary generation (min/max length of summary, penalties for repetition and length, etc.).

Next, we instantiate objects for these two classes. For instance, if we want to use `facebook/bart-large-cnn` model for summarization:

```python
# Specifying the config for summary generation (given with default values)
summ_config = SummaryGenerationConfig()

# Instantiating a Summarizer object
summarizer = ArticleSummarizer(model_path="facebook/bart-large-cnn")
```

We can optionally attach LoRA adapters compatible with the model we used in `model_path`:

```python
# Attaching compatible LoRA adapters if needed
summarizer.load_lora_adapters(lora_adapters_path="spolivin/bart-arxiv-lora")
```

We can now specify the path to the article we need to summarize and can easily generate the summary:

```python
# Generating summary
generated_summary = summarizer.summarize_text(
  pdf_path="articles/test1.pdf", config=summ_config,
)
```

The text in `generated_summary` now contains the summary of the article from `articles/test1.pdf`. Lastly, we generate the report:

```python
# Generating summary report
summarizer.generate_summary_report("summary_report.txt")
```
After successful generation, one will see a message mentioning where summary has been saved (by default summary is saved in a txt-file in `summaries` folder created if non-existent).


## Summarization using one command

In order to make the library useful, I came up with [the script](./summarize.py) that can be used to generate summaries using one command. It has the following arguments:

```
$ python summarize.py --help

usage: summarize.py [-h] [-c CONFIG] [-mp MODEL_PATH] [-tp TOKENIZER_PATH] [-mxot MAX_OUTPUT_TOKENS]
                    [-mnot MIN_OUTPUT_TOKENS] [-nb NUM_BEAMS] [-lp LENGTH_PENALTY] [-rp REPETITION_PENALTY]
                    [-nrns NO_REPEAT_NGRAM_SIZE] [-lap LORA_ADAPTERS_PATH] [-lw LINE_WIDTH] [-mkn MAX_KEYWORDS_NUM]
                    [-rn REPORT_NAME] [-sf SAVE_FOLDER]
                    filepath

Summarize a PDF article using a Hugging Face model.

positional arguments:
  filepath              Path to the PDF article to be summarized

optional arguments:
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
  -rn REPORT_NAME, --report-name REPORT_NAME
                        Name of the output summary report
  -sf SAVE_FOLDER, --save-folder SAVE_FOLDER
                        Folder to save the generated summary
```

## Overriding arguments
There are two ways that one can specify arguments for the script:

* Configuration file (`--config` flag) => examples of configs can be found [here](./configs/).

* CLI arguments.

The script is programmed in such a way that when specifying both config and CLI arguments, argument with the same name in config and cli will be overridden with the value specified in CLI. For instance, after using this command, the `num_beams` argument will be overridden with the value of 5:

```bash
python summarize.py articles/test1.pdf --config=configs/t5_small_config.json --num-beams=5
```

## Example scripts

I have prepared a few shell-scripts with [examples](./scripts/) of using the script in order to demonstrate how it can be used. One can run them in the following way for some test article. I have prepared a script for automatic downloading of an article from ArXiv given its ID. For instance, we can load a paper with ID of `1301.3781`:

```bash
# Loading paper from ArXiv and saving it in 'articles' folder
python pull_arxiv_paper.py 1301.3781
```

Now we can run each of the below scripts one by one to test the library and different configurations:
```bash
# Using "facebook/bart-large-cnn"
bash scripts/run_bart_large.sh articles/1301.3781.pdf

# Using "facebook/bart-large-cnn" with LoRA adapters
bash scripts/run_bart_lora.sh articles/1301.3781.pdf

# Using default settings
bash scripts/run_default.sh articles/1301.3781.pdf

# Using "google-t5/t5-base"
bash scripts/run_t5_base.sh articles/1301.3781.pdf

# Using "google-t5/t5-small"
bash scripts/run_t5_small.sh articles/1301.3781.pdf

# Using "google-t5/t5-base" with overridden arguments
bash scripts/run_t5_small_override.sh articles/1301.3781.pdf
```

After running these commands, the respective summary reports with additional information and statistics will be generated and saved in `summaries` folder (by default).

## Library limitations

The main drawback of the library is that it works currently only with articles written in one column (rather than in two which is the case with many articles). The summary generation results can be drastically different (and potentially incoherent) if attempting to use the library on two-column articles.

Another limitation consists in the way article sections are named. The library is written to retrieve text starting from "Introduction-like" sections until "References-like" sections to use the result as input to summary generation models. While the library is able to track the most common ways Introduction and References sections are usually named and thus retrieve text accordingly, sometimes these sections can have other names that can pose a problem for retrieving the text correctly.
