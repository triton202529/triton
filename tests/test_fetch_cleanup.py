import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FetchAndPrepareCleanupTests(unittest.TestCase):
    def test_cleanup_only_removes_generated_parquet_outputs(self):
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda values: values
        fake_feature_generator = types.ModuleType("services.feature_generator")
        fake_feature_generator.add_technical_indicators = lambda df, spy_df=None: df

        import_stubs = {
            "pandas": types.ModuleType("pandas"),
            "tqdm": fake_tqdm,
            "yfinance": types.ModuleType("yfinance"),
            "services.feature_generator": fake_feature_generator,
        }

        sys.modules.pop("scripts.fetch_and_prepare", None)
        with patch.dict(sys.modules, import_stubs):
            fetch_and_prepare = importlib.import_module("scripts.fetch_and_prepare")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            keep_csv = results_dir / "portfolio_history.csv"
            keep_summary = results_dir / "backtest_summary.csv"
            remove_parquet = results_dir / "AAPL.parquet"
            keep_csv.write_text("date,total_value\n2024-01-01,100\n")
            keep_summary.write_text("metric,value\nreturn,1\n")
            remove_parquet.write_bytes(b"parquet")

            removed = fetch_and_prepare.clear_old_ticker_results(str(results_dir))

            self.assertEqual(removed, 1)
            self.assertTrue(keep_csv.exists())
            self.assertTrue(keep_summary.exists())
            self.assertFalse(remove_parquet.exists())


if __name__ == "__main__":
    unittest.main()
