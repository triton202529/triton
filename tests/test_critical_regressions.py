import importlib
import tempfile
import unittest
from pathlib import Path


class FetchAndPrepareCleanupTests(unittest.TestCase):
    def test_cleanup_only_removes_generated_parquet_outputs(self):
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
