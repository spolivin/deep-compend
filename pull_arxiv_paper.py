"""
Script for loading a paper from ArXiv given its Arxiv ID number.
================================================================

The script downloads a paper from ArXiv given the specified ID of a paper.

User can optionally change the save path of the downloaded paper.

Usage:
    python pull_arxiv_paper.py <arxiv-paper-id> --save-path=some_folder/paper.pdf

Arguments:
    arxivid (str): Paper's ArXiv ID.
    --save-path (str, optional): Path to save the paper.
"""

import argparse

from deep_compend.utils.downloads import download_arxiv_paper

# Defining Arguments parser
parser = argparse.ArgumentParser(description="ArXiv paper loader.")

parser.add_argument("arxivid", type=str, help="Paper's ArXiv ID")

parser.add_argument(
    "-sf",
    "--save-path",
    type=str,
    help="Path to save the downloaded paper",
)

args = parser.parse_args()

if __name__ == "__main__":
    # Defining save path: in 'articles' folder with input ID or specified path in --save-path
    save_path = args.save_path or f"articles/{args.arxivid}.pdf"
    # Downloading the paper from ArXiv
    download_arxiv_paper(arxiv_id=args.arxivid, save_path=save_path)
