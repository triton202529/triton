import unittest

import pandas as pd

from services.feature_generator import add_technical_indicators, combine_ticker_frames


def _ohlcv(ticker, n=40, start="2020-01-02"):
    dates = pd.bdate_range(start, periods=n)
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
            "ticker": ticker,
        }
    )


class FeatureDropnaTests(unittest.TestCase):
    def test_stock_rows_survive_concat_with_etf_capital_gains(self):
        stock = _ohlcv("AAPL")
        etf = _ohlcv("SPY")
        etf["capital_gains"] = 0.0

        combined = pd.concat([stock, etf], ignore_index=True)
        aapl_gains = combined.loc[combined["ticker"] == "AAPL", "capital_gains"]
        self.assertTrue(aapl_gains.isna().all())

        spy_df = combined[combined["ticker"] == "SPY"]
        aapl = combined[combined["ticker"] == "AAPL"].copy()
        out = add_technical_indicators(aapl, spy_df)

        self.assertGreater(len(out), 0, "stock rows were wiped by dropna after ETF concat")
        self.assertTrue(out["close"].notna().all())
        self.assertTrue((out["capital_gains"] == 0).all())

    def test_combine_ticker_frames_keeps_stocks_in_merged_dataset(self):
        stock = add_technical_indicators(_ohlcv("AAPL"))
        etf = add_technical_indicators(_ohlcv("SPY").assign(capital_gains=0.0))
        merged = combine_ticker_frames([stock, etf])

        self.assertGreater(len(merged[merged["ticker"] == "AAPL"]), 0)
        self.assertGreater(len(merged[merged["ticker"] == "SPY"]), 0)


if __name__ == "__main__":
    unittest.main()
