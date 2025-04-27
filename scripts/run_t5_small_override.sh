ARTICLE_PATH="${1}"
deep-compend summarize "$ARTICLE_PATH" \
    --config=configs/t5_small_config.json \
    --max-keywords-num=10 \
    --repetition-penalty=1.45 \
    --length-penalty=1.3 \
    --report-name=summary_report_t5_small_overriden.txt \
    --spacy-lang-model=en_core_web_lg \
    --generate-summary-report=True
