PY?=venv/Scripts/python.exe

.PHONY: refresh
refresh:
	$(PY) services/fetch_data.py
	$(PY) services/fetch_fundamentals.py
	$(PY) services/score_stocks.py
	$(PY) services/train_models.py
	$(PY) services/combine_predictions.py
	$(PY) services/generate_signals.py
	$(PY) services/fetch_news_sentiment_rss.py --tickers all --window 14
	$(PY) services/generate_economic_calendar_rss.py
	$(PY) services/generate_smart_alerts.py
