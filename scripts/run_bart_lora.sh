ARTICLE_PATH="${1}"
deep-compend "$ARTICLE_PATH" \
    --config=configs/bart_lora_config.json \
    --min-keywords-length=5 \
    --max-keywords-num=7
