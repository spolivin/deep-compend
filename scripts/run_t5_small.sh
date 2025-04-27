ARTICLE_PATH="${1}"
deep-compend summarize "$ARTICLE_PATH" \
    --config=configs/t5_small_config.json \
    --max-keywords-num=15 \
    --min-keywords-length=6 \
    --spacy-lang-model=en_core_web_lg \
    --generate-summary-report=True
