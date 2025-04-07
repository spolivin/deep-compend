ARTICLE_PATH="${1}"
python summarize.py "$ARTICLE_PATH" \
    --min-keywords-length=7 \
    --max-keywords-num=10
