ARTICLE_PATH="${1}"
python summarize.py "$ARTICLE_PATH" --config=configs/t5_small_config.json
