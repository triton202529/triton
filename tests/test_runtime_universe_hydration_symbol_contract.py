"""
E0 universe-hydration symbol-collection contract regression.

Covers the TypeError from cycles 3/4 where missing_*_symbols were bools
because loop locals shadowed the symbol lists. No broker calls; no runtime mutation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.runtime_universe_hydration_preflight import (
    HydrationSymbolContractError,
    detect_hydration_gaps,
    require_symbol_collection,
    run_runtime_hydration_preflight,
    symbol_collection_as_set,
)


def test_require_symbol_collection_empty_list():
    assert require_symbol_collection([], field="missing_lifecycle_symbols") == []
    assert symbol_collection_as_set([], field="missing_lifecycle_symbols") == set()


def test_require_symbol_collection_populated():
    assert require_symbol_collection(["aapl", "MSFT"], field="missing_lifecycle_symbols") == [
        "AAPL",
        "MSFT",
    ]
    assert symbol_collection_as_set(("TSLA", "GLD"), field="missing_signals_symbols") == {
        "TSLA",
        "GLD",
    }


def test_require_symbol_collection_false_fails_closed():
    """False is not a valid list; buggy producer emitted last-iteration membership."""
    with pytest.raises(HydrationSymbolContractError, match="bool"):
        require_symbol_collection(False, field="missing_lifecycle_symbols")


def test_require_symbol_collection_true_fails_closed():
    """True must never be treated as an empty symbol list."""
    with pytest.raises(HydrationSymbolContractError, match="bool"):
        require_symbol_collection(True, field="missing_signals_symbols")


@pytest.mark.parametrize(
    "bad",
    ["SPCX", {"SPCX": True}, 1, 0.5, object()],
)
def test_require_symbol_collection_malformed_fails_closed(bad):
    with pytest.raises(HydrationSymbolContractError):
        require_symbol_collection(bad, field="missing_lifecycle_symbols")


def test_require_symbol_collection_none_as_empty():
    assert require_symbol_collection(None, field="missing_lifecycle_symbols") == []


def test_detect_hydration_gaps_emits_lists_not_bools():
    """Producer must keep missing_*_symbols as lists after the critical scan loop."""
    tradable = ["AAA", "BBB", "CCC"]
    lifecycle = {"AAA", "BBB"}  # CCC missing lifecycle
    signals = {"AAA", "BBB", "CCC"}
    processed = {"AAA", "BBB", "CCC"}
    allocation = {"AAA"}

    with (
        patch(
            "services.universe_pipeline_hydration.load_tradable_universe_tickers",
            return_value=tradable,
        ),
        patch(
            "services.runtime_universe_hydration_preflight._processed_ticker_set",
            return_value=processed,
        ),
        patch(
            "services.runtime_universe_hydration_preflight._load_csv_ticker_set",
            side_effect=lambda path: {
                "signal_lifecycle_effective.csv": lifecycle,
                "portfolio_allocation_recommendations.csv": allocation,
                "signals_with_rationale.csv": signals,
                "signals.csv": set(),
            }.get(path.name, set()),
        ),
    ):
        gaps = detect_hydration_gaps(tradable=tradable, apply_quarantine=False)

    assert isinstance(gaps["missing_lifecycle_symbols"], list)
    assert isinstance(gaps["missing_signals_symbols"], list)
    assert isinstance(gaps["missing_processed_symbols"], list)
    assert gaps["missing_lifecycle_symbols"] == ["CCC"]
    assert gaps["missing_signals_symbols"] == []
    assert "CCC" in gaps["stranded_symbols"]
    assert gaps["hydration_needed"] is True
    # Membership must not TypeError (cycle 3/4 failure mode)
    assert "CCC" in gaps["missing_lifecycle_symbols"]
    assert "AAA" not in gaps["missing_lifecycle_symbols"]


def test_detect_hydration_gaps_no_missing_allows_empty_lists():
    tradable = ["AAA", "BBB"]
    present = {"AAA", "BBB"}

    with (
        patch(
            "services.runtime_universe_hydration_preflight._processed_ticker_set",
            return_value=present,
        ),
        patch(
            "services.runtime_universe_hydration_preflight._load_csv_ticker_set",
            return_value=present,
        ),
    ):
        gaps = detect_hydration_gaps(tradable=tradable, apply_quarantine=False)

    assert gaps["missing_lifecycle_symbols"] == []
    assert gaps["missing_signals_symbols"] == []
    assert gaps["stranded_symbols"] == []
    assert gaps["hydration_needed"] is False


def test_cycle_bool_input_no_longer_typeerrors_in_preflight():
    """Exact bool shape from cycles 3/4 becomes controlled CONTRACT_ERROR, not TypeError."""
    base_gaps = {
        "configured_symbols": ["SPCX"],
        "configured_symbol_count": 1,
        "processed_symbol_count": 0,
        "lifecycle_symbol_count": 0,
        "allocation_symbol_count": 0,
        "missing_processed_symbols": False,  # buggy shape
        "missing_lifecycle_symbols": False,
        "missing_allocation_symbols": ["SPCX"],
        "missing_signals_symbols": False,
        "raw_stranded_symbols": ["SPCX"],
        "stranded_symbols": ["SPCX"],
        "artifact_regression_symbols": [],
        "pipeline_artifact_missing_symbols": [],
        "allocation_refresh_only_symbols": [],
        "quarantined_symbols": [],
        "quarantine_reasons_by_symbol": {},
        "actionable_stranded_symbols": ["SPCX"],
        "no_actionable_hydration_needed": False,
        "hydration_needed": True,
    }

    with (
        patch(
            "services.runtime_universe_hydration_preflight.detect_hydration_gaps",
            return_value=dict(base_gaps),
        ),
        patch("services.runtime_universe_hydration_preflight._market_open", return_value=True),
        patch(
            "services.runtime_universe_hydration_preflight._live_execution_permitted",
            return_value=False,
        ),
        patch(
            "services.portfolio_history_sync.load_live_execution_flag",
            return_value=False,
        ),
        patch(
            "services.runtime_universe_hydration_preflight._atomic_write_json",
        ) as write_mock,
        patch(
            "services.universe_pipeline_hydration.ensure_raw_data",
        ) as ensure_raw,
        patch(
            "services.universe_pipeline_hydration.hydrate_universe_symbols",
        ) as hydrate,
        patch(
            "services.universe_pipeline_hydration.preprocess_symbols",
        ) as preprocess,
    ):
        audit = run_runtime_hydration_preflight(verbose=False, check_only=False)

    assert audit["status"] == "CONTRACT_ERROR"
    assert "bool" in str(audit.get("reason") or "").lower()
    assert audit.get("broker_orders_submitted") is False
    assert audit.get("hydration_attempted") is False
    ensure_raw.assert_not_called()
    hydrate.assert_not_called()
    preprocess.assert_not_called()
    write_mock.assert_called()


def test_true_bool_fails_closed_without_broker():
    base_gaps = {
        "configured_symbols": ["ZZZ"],
        "configured_symbol_count": 1,
        "processed_symbol_count": 0,
        "lifecycle_symbol_count": 0,
        "allocation_symbol_count": 0,
        "missing_processed_symbols": [],
        "missing_lifecycle_symbols": True,
        "missing_allocation_symbols": [],
        "missing_signals_symbols": [],
        "raw_stranded_symbols": ["ZZZ"],
        "stranded_symbols": ["ZZZ"],
        "artifact_regression_symbols": [],
        "pipeline_artifact_missing_symbols": [],
        "allocation_refresh_only_symbols": [],
        "quarantined_symbols": [],
        "quarantine_reasons_by_symbol": {},
        "actionable_stranded_symbols": ["ZZZ"],
        "no_actionable_hydration_needed": False,
        "hydration_needed": True,
    }

    with (
        patch(
            "services.runtime_universe_hydration_preflight.detect_hydration_gaps",
            return_value=dict(base_gaps),
        ),
        patch("services.runtime_universe_hydration_preflight._market_open", return_value=False),
        patch(
            "services.runtime_universe_hydration_preflight._live_execution_permitted",
            return_value=False,
        ),
        patch(
            "services.portfolio_history_sync.load_live_execution_flag",
            return_value=False,
        ),
        patch("services.runtime_universe_hydration_preflight._atomic_write_json"),
        patch("services.universe_pipeline_hydration.ensure_raw_data") as ensure_raw,
        patch("services.universe_pipeline_hydration.hydrate_universe_symbols") as hydrate,
        patch("services.universe_pipeline_hydration.preprocess_symbols") as preprocess,
    ):
        audit = run_runtime_hydration_preflight(check_only=False)

    assert audit["status"] == "CONTRACT_ERROR"
    ensure_raw.assert_not_called()
    hydrate.assert_not_called()
    preprocess.assert_not_called()


def test_missing_symbols_still_block_via_check_only():
    tradable = ["MISS"]
    with (
        patch(
            "services.runtime_universe_hydration_preflight._processed_ticker_set",
            return_value=set(),
        ),
        patch(
            "services.runtime_universe_hydration_preflight._load_csv_ticker_set",
            return_value=set(),
        ),
        patch("services.runtime_universe_hydration_preflight._market_open", return_value=True),
        patch(
            "services.runtime_universe_hydration_preflight._live_execution_permitted",
            return_value=False,
        ),
        patch(
            "services.portfolio_history_sync.load_live_execution_flag",
            return_value=False,
        ),
        patch("services.runtime_universe_hydration_preflight._atomic_write_json"),
        patch(
            "services.symbol_hydration_quarantine.apply_quarantine_to_gaps",
            side_effect=lambda g: g,
        ),
    ):
        gaps = detect_hydration_gaps(tradable=tradable, apply_quarantine=False)
        assert gaps["hydration_needed"] is True
        assert "MISS" in gaps["missing_lifecycle_symbols"]
        assert isinstance(gaps["missing_lifecycle_symbols"], list)

        with patch(
            "services.runtime_universe_hydration_preflight.detect_hydration_gaps",
            return_value=gaps,
        ):
            audit = run_runtime_hydration_preflight(check_only=True)

    assert audit["status"] == "GAPS_DETECTED"
    assert audit.get("broker_orders_submitted") is False


def test_no_missing_symbols_e0_passes_skip():
    present = {"OK1", "OK2"}
    with (
        patch(
            "services.runtime_universe_hydration_preflight._processed_ticker_set",
            return_value=present,
        ),
        patch(
            "services.runtime_universe_hydration_preflight._load_csv_ticker_set",
            return_value=present,
        ),
        patch("services.runtime_universe_hydration_preflight._market_open", return_value=True),
        patch(
            "services.runtime_universe_hydration_preflight._live_execution_permitted",
            return_value=False,
        ),
        patch(
            "services.portfolio_history_sync.load_live_execution_flag",
            return_value=False,
        ),
        patch("services.runtime_universe_hydration_preflight._atomic_write_json"),
        patch("services.universe_pipeline_hydration.ensure_raw_data") as ensure_raw,
        patch("services.universe_pipeline_hydration.hydrate_universe_symbols") as hydrate,
    ):
        gaps = detect_hydration_gaps(tradable=["OK1", "OK2"], apply_quarantine=False)
        with patch(
            "services.runtime_universe_hydration_preflight.detect_hydration_gaps",
            return_value=gaps,
        ):
            audit = run_runtime_hydration_preflight(check_only=False)

    assert audit["status"] == "SKIPPED_NO_GAPS"
    assert audit.get("broker_orders_submitted") is False
    ensure_raw.assert_not_called()
    hydrate.assert_not_called()


def test_no_broker_order_methods_on_hydration_module():
    import services.runtime_universe_hydration_preflight as mod

    for name in dir(mod):
        assert "submit" not in name.lower() or name.startswith("_")
    src = open(mod.__file__, encoding="utf-8").read()
    assert "submit_order" not in src
    assert "create_order" not in src
