#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
VENV_DIR="venv"
PY="$VENV_DIR/bin/python"
LOG_DIR="logs"
RESULTS_DIR="data/results"
STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/daily_$STAMP.log"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# rotate old logs (keep last 10)
ls -1t "$LOG_DIR"/daily_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f

echo "▶️  Triton Daily Refresh started @ $(date -Is)" | tee -a "$LOG_FILE"

# --- activate venv (if needed) ---
if [ ! -x "$PY" ]; then
  echo "⚠️  venv not found at $VENV_DIR — using system python"
  PY="python"
fi

run() {
  echo ""; echo "——— $1 ———" | tee -a "$LOG_FILE"
  shift
  "$PY" "$@" 2>&1 | tee -a "$LOG_FILE"
}

# --- PIPELINE (EOD order) ---
run "Fetch OHLC data"                  services/fetch_data.py
run "Fetch fundamentals"               services/fetch_fundamentals.py
run "Score stocks"                     services/score_stocks.py
run "Train models + predictions"       services/train_models.py
run "Combine predictions (if used)"    services/combine_predictions.py
run "Generate signals + rationale"     services/generate_signals.py
run "News sentiment (RSS, 14d)"        services/fetch_news_sentiment_rss.py --tickers all --window 14
run "Economic calendar (RSS)"          services/generate_economic_calendar_rss.py
run "Smart alerts"                     services/generate_smart_alerts.py
# Optional:
# run "Portfolio summary"                services/portfolio_summary.py

echo ""; echo "✅ Done @ $(date -Is)" | tee -a "$LOG_FILE"
