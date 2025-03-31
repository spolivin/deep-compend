ARTICLE_PATH="${1}"
python summarize.py "$ARTICLE_PATH" --config=configs/bart_lora_config.json
