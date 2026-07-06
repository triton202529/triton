"""Regression tests for dynamic 197Z approved-scope forbidden symbol derivation."""

from __future__ import annotations

import unittest

import pandas as pd

from services.paper_execution_handoff_scope import (
    filter_handoff_lifecycle_dataframe,
    symbols_outside_approved_scope,
    synthetic_contamination_symbols,
)


MONDAY_SCOPE = [
    "CVX",
    "PEP",
    "PFE",
    "SLV",
    "T",
    "TSLA",
    "USO",
    "WMT",
    "XLE",
    "XOM",
]


class Phase197ZDynamicScopeTests(unittest.TestCase):
    def test_tsla_not_forbidden_when_in_current_approved_scope(self) -> None:
        final = ["CVX", "TSLA", "XOM"]
        forbidden = symbols_outside_approved_scope(final, MONDAY_SCOPE)
        self.assertNotIn("TSLA", forbidden)
        self.assertEqual(forbidden, [])

    def test_gld_fails_when_outside_current_approved_scope(self) -> None:
        final = ["CVX", "GLD", "TSLA"]
        forbidden = symbols_outside_approved_scope(final, MONDAY_SCOPE)
        self.assertEqual(forbidden, ["GLD"])

    def test_synthetic_contamination_excludes_currently_approved_symbols(self) -> None:
        injected = synthetic_contamination_symbols(MONDAY_SCOPE)
        self.assertNotIn("TSLA", injected)
        self.assertIn("GLD", injected)
        self.assertTrue(set(injected).isdisjoint(set(MONDAY_SCOPE)))

    def test_excluded_from_pre_is_outside_approved_scope(self) -> None:
        pre = ["CVX", "GLD", "MA", "TSLA", "XOM"]
        outside = symbols_outside_approved_scope(pre, MONDAY_SCOPE)
        self.assertEqual(outside, ["GLD", "MA"])

    def test_handoff_filter_keeps_tsla_excludes_gld(self) -> None:
        rows = []
        for sym in MONDAY_SCOPE + ["GLD", "MA"]:
            rows.append({"ticker": sym, "qty": 1, "limit_price": 100.0})
        df = pd.DataFrame(rows)

        class _ScopeLoader:
            @staticmethod
            def load_paper_approved_execution_scope():
                return MONDAY_SCOPE, "phase_197x_projected_paper_orders", True

        import services.defensive_exit_dispatch_interlock as interlock
        import services.paper_execution_handoff_scope as handoff_scope

        original = interlock.load_paper_approved_execution_scope
        original_should = handoff_scope.should_enforce_approved_197x_handoff_scope
        try:
            interlock.load_paper_approved_execution_scope = (
                _ScopeLoader.load_paper_approved_execution_scope
            )
            handoff_scope.should_enforce_approved_197x_handoff_scope = lambda **_: True
            _filtered, diag = filter_handoff_lifecycle_dataframe(df, mode="paper")
        finally:
            interlock.load_paper_approved_execution_scope = original
            handoff_scope.should_enforce_approved_197x_handoff_scope = original_should

        final_symbols = list(diag.get("final_handoff_symbols") or [])
        excluded = list(diag.get("excluded_from_handoff_symbols") or [])
        self.assertIn("TSLA", final_symbols)
        self.assertNotIn("GLD", final_symbols)
        self.assertIn("GLD", excluded)
        self.assertEqual(diag.get("forbidden_symbols_in_final_handoff"), [])
        self.assertFalse(diag.get("outside_scope_rows_reached_placement"))


if __name__ == "__main__":
    unittest.main()
