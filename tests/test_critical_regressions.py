import importlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path


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
            ticker_output = results_dir / "AAPL.parquet"
            report_output = results_dir / "signals_with_rationale.csv"
            summary_output = results_dir / "backtest_summary.csv"

            ticker_output.write_text("stale parquet placeholder")
            report_output.write_text("existing signals")
            summary_output.write_text("existing summary")

            fetch_and_prepare.clear_generated_ticker_files(str(results_dir))

            self.assertFalse(ticker_output.exists())
            self.assertTrue(report_output.exists())
            self.assertTrue(summary_output.exists())


class DashboardRegressionTests(unittest.TestCase):
    def test_dashboard_keeps_existing_tabs_when_adding_learning_lab(self):
        source = Path("view_results.py").read_text()

        self.assertNotIn("Tabs 0 to 19 omitted", source)
        self.assertIn("with tabs[0]:", source)
        self.assertIn("with tabs[19]:", source)
        self.assertIn("with tabs[20]:", source)
        self.assertNotIn('with st.tabs(["AI Learning Lab"])[0]:', source)


if __name__ == "__main__":
    unittest.main()
