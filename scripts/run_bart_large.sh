ARTICLE_PATH="${1}"
deep-compend summarize "$ARTICLE_PATH" \
    --config=configs/bart_large_config.json \
    --spacy-lang-model=en_core_web_lg \
    --generate-summary-report=True
