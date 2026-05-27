import os
import tempfile
import unittest

from cleanup_failed_tickers import clean_failed_tickers
from scripts.failed_ticker_utils import load_failed_tickers, parse_failed_ticker_line
from scripts.fetch_and_prepare import clear_old_ticker_results


class PipelineSafetyTests(unittest.TestCase):
    def test_result_cleanup_preserves_csv_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "AAPL.parquet")
            csv_path = os.path.join(tmpdir, "signals_with_rationale.csv")
            log_path = os.path.join(tmpdir, "trade_log.csv")
            subdir_path = os.path.join(tmpdir, "nested")

            with open(parquet_path, "w") as f:
                f.write("old ticker data")
            with open(csv_path, "w") as f:
                f.write("critical signal output")
            with open(log_path, "w") as f:
                f.write("trade history")
            os.mkdir(subdir_path)

            removed = clear_old_ticker_results(tmpdir)

            self.assertEqual(removed, 1)
            self.assertFalse(os.path.exists(parquet_path))
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(log_path))
            self.assertTrue(os.path.isdir(subdir_path))

    def test_failed_ticker_parser_strips_reason(self):
        self.assertEqual(parse_failed_ticker_line("AAPL (fetch error)\n"), "AAPL")
        self.assertEqual(parse_failed_ticker_line("^GSPC (indicator error)"), "^GSPC")
        self.assertEqual(parse_failed_ticker_line("BRK-B"), "BRK-B")
        self.assertIsNone(parse_failed_ticker_line("   "))

    def test_clean_failed_tickers_dedupes_by_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "failed_tickers.txt")
            output_path = os.path.join(tmpdir, "failed_tickers_unique.txt")
            with open(source_path, "w") as f:
                f.write("AAPL (fetch error)\n")
                f.write("AAPL (indicator error)\n")
                f.write("SPY\n")

            tickers = clean_failed_tickers(source_path, output_path)

            self.assertEqual(tickers, ["AAPL", "SPY"])
            self.assertEqual(load_failed_tickers(output_path), ["AAPL", "SPY"])


if __name__ == "__main__":
    unittest.main()
