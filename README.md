# Fine-tuned BART-LoRA model for article summarization

This repository is intended for storing the code of the script which can
be used to generate summaries of scientific articles. Using [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) as base model, I employed *LoRA* to fine-tune the model on [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) dataset with multiple examples of article texts taken from *Arxiv*.

## Preparing virtual environment

Before being able to run the summarization, one need to firstly create virtual environment and load [necessary packages](./requirements.txt):

1. **Windows**

```bash
python -m venv ./.venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

2. **Linux**
```bash
sudo apt-get update
sudo apt-get install python3.10-venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Summarization script

The [script](./summarize.py) used for running summarization has the following arguments:

```
$ python summarize.py --help

usage: summarize.py [-h] [-mot MAX_OUTPUT_TOKENS] [-stt SAVE_TO_TXT] filepath

positional arguments:
  filepath              Path to the PDF-article to be summarized

optional arguments:
  -h, --help            show this help message and exit
  -mot MAX_OUTPUT_TOKENS, --max-output-tokens MAX_OUTPUT_TOKENS
                        Maximum number of output tokens
  -stt SAVE_TO_TXT, --save-to-txt SAVE_TO_TXT
                        Save summary to txt-file
```

The script is yet in the development stage and currently it works as follows:

1. Script accepts the path to the PDF-article to be summarized as a positional argument (`filepath`)
2. Optionally one can specify the maximum number of output tokens that should be generated as a summary (`--max-output-tokens`)
3. Optionally one can choose to save the generated summary in a text file named as `article_name_summarized.txt` saved in `summaries` directory (`--save-to-txt`) in case the article is named as `article_name.pdf`. If left unspecified, the summary will be outputed to console.

### Example usage

Suppose we have a article saved in `articles/some_article.pdf` which we need to summarize. Running the following command will load the model, preprocess the article text and save the generated summary in `some_article_summarized.txt` file with maximum allowed number of output tokens equal to 300:

```bash
python summarize.py articles/some_article.pdf --max-output-tokens=300 --save-to-txt=True
```
