ARTICLE_PATH="${1}"
deep-compend summarize "$ARTICLE_PATH" \
    --min-keywords-length=7 \
    --max-keywords-num=10 \
    --spacy-lang-model=en_core_web_md
