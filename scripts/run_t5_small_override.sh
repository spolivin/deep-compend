ARTICLE_PATH="${1}"
python summarize.py "$ARTICLE_PATH" \
    --config=configs/t5_small_config.json \
    --max-keywords-num=10 \
    --repetition-penalty=1.45 \
    --length-penalty=1.3 \
    --report-name=summary_report_t5_small_overriden.txt
