ARTICLE_PATH="${1}"
deep-compend summarize "$ARTICLE_PATH" \
    --config=configs/bart_lora_config.json \
    --min-keywords-length=5 \
    --max-keywords-num=7 \
    --spacy-lang-model=en_core_web_lg \
    --generate-summary-report=True
