"""SF30: planner quality uses the non-RF composite score and does not zero a missing delta."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from services.execute_trades import (
    REASON_QUALITY_EVIDENCE_MALFORMED,
    REASON_QUALITY_EVIDENCE_MISSING,
    REASON_QUALITY_EVIDENCE_STALE,
    AccountSnapshot,
    PlannedOrder,
    PositionSnapshot,
    _annotate_operational_quality,
    _normalize_exec_final_score_percentile_rank,
    _operational_quality_binding,
    _parse_operational_composite_score,
    _row_final_quality_score,
    build_execution_plan,
    load_execute_trades_config,
    maybe_execute_plan,
    require_execute_dispatch_contract,
)
import services.execute_trades as et


def _temp_execute_config(tmp: Path) -> Path:
    """TEMP config only. The committed test never reads production files."""
    payload = {
        "min_final_score_threshold": 0.0020,
        "block_negative_final_score": True,
        "enforce_min_final_score": True,
        "exec_score_percentile_normalize": True,
        "diversification_ranking_enabled": False,
        "positions_snapshot_path": "",
        "snapshot_hygiene_enabled": False,
        "sector_caps_enabled": False,
        "capital_engine_enabled": False,
        "delayed_entry_soft_penalty_enabled": False,
        "score_sizing_enabled": False,
        "min_order_notional": 1.0,
        "reserve_cash_pct": 0.0,
    }
    path = tmp / "execute_trades.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _lifecycle(tmp: Path, session: str = "2026-09-18") -> Path:
    path = tmp / "signal_lifecycle_effective.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "effective_stance": "BUY", "session_as_of_date": session},
            {"ticker": "MSFT", "effective_stance": "HOLD", "session_as_of_date": session},
        ]
    ).to_csv(path, index=False)
    return path


def _row(symbol: str, score, *, session, sha, provenance="VALID", **extra):
    data = {
        "ticker": symbol,
        "opportunity_type": "ENTRY",
        "effective_stance": "BUY",
        "confidence": 0.7,
        "delta_pct": "",
        "score": score,
        "session_date": session,
        "provenance_status": provenance,
        "lifecycle_sha256": sha,
        "rf_delta_pct": -0.5,
        "edge_score": 0.2,
        "spread_bucket": "NORMAL",
    }
    data.update(extra)
    return data


@pytest.fixture
def binding(tmp_path, monkeypatch):
    path = _lifecycle(tmp_path)
    cfg_path = _temp_execute_config(tmp_path)
    monkeypatch.setattr(et, "RESULTS", tmp_path)
    monkeypatch.setattr(et, "CONFIG_PATH", cfg_path)
    bound = _operational_quality_binding(path)
    assert bound["ok"] is True
    bound["config_path"] = cfg_path
    return bound


def test_missing_blank_nan_and_garbage_are_not_zero():
    assert _parse_operational_composite_score(None)[1] == REASON_QUALITY_EVIDENCE_MISSING
    assert _parse_operational_composite_score("")[1] == REASON_QUALITY_EVIDENCE_MISSING
    assert _parse_operational_composite_score("nan")[1] == REASON_QUALITY_EVIDENCE_MISSING
    assert _parse_operational_composite_score(float("nan"))[1] == REASON_QUALITY_EVIDENCE_MISSING
    assert _parse_operational_composite_score(float("inf"))[1] == REASON_QUALITY_EVIDENCE_MALFORMED
    assert _parse_operational_composite_score(float("-inf"))[1] == REASON_QUALITY_EVIDENCE_MALFORMED
    assert _parse_operational_composite_score("not-a-score")[1] == REASON_QUALITY_EVIDENCE_MALFORMED
    assert _parse_operational_composite_score(-0.1)[1] == REASON_QUALITY_EVIDENCE_MALFORMED
    assert _parse_operational_composite_score(1.2)[1] == REASON_QUALITY_EVIDENCE_MALFORMED
    value, err = _parse_operational_composite_score(0.72)
    assert err is None and value == pytest.approx(0.72)


def test_stale_session_and_hash_mismatch(binding):
    sha = binding["sha"]
    session = binding["session"]
    frame = pd.DataFrame(
        [
            _row("AAPL", 0.8, session="2020-01-01", sha=sha),
            _row("AMD", 0.8, session=session, sha="0" * 64),
            _row("GOOGL", 0.8, session=session, sha=sha, provenance="STALE"),
            _row("XLK", 0.8, session=session, sha=""),
        ]
    )
    out = _annotate_operational_quality(frame, binding)
    reasons = dict(zip(out["ticker"], out["_quality_evidence_reject"]))
    assert reasons["AAPL"] == REASON_QUALITY_EVIDENCE_STALE
    assert reasons["AMD"] == REASON_QUALITY_EVIDENCE_STALE
    assert reasons["GOOGL"] == REASON_QUALITY_EVIDENCE_STALE
    assert reasons["XLK"] == REASON_QUALITY_EVIDENCE_MISSING


def test_duplicate_symbol_is_malformed_and_hold_is_not_scored(binding):
    sha = binding["sha"]
    session = binding["session"]
    frame = pd.DataFrame(
        [
            _row("AAPL", 0.8, session=session, sha=sha),
            _row("AAPL", 0.4, session=session, sha=sha),
            {
                "ticker": "MSFT",
                "opportunity_type": "HOLD",
                "effective_stance": "HOLD",
                "score": "",
                "delta_pct": "",
            },
        ]
    )
    out = _annotate_operational_quality(frame, binding)
    assert set(out.loc[out["ticker"] == "AAPL", "_quality_evidence_reject"]) == {
        REASON_QUALITY_EVIDENCE_MALFORMED
    }
    hold = out.loc[out["ticker"] == "MSFT"].iloc[0]
    assert str(hold["_quality_evidence_reject"]) == ""


def test_rf_delta_does_not_change_operational_score(binding):
    sha = binding["sha"]
    session = binding["session"]
    left = _annotate_operational_quality(
        pd.DataFrame([_row("AAPL", 0.81, session=session, sha=sha, rf_delta_pct=-0.2)]),
        binding,
    )
    right = _annotate_operational_quality(
        pd.DataFrame([_row("AAPL", 0.81, session=session, sha=sha, rf_delta_pct=0.9)]),
        binding,
    )
    assert float(left.iloc[0]["_operational_quality"]) == float(
        right.iloc[0]["_operational_quality"]
    )
    assert left.iloc[0]["_quality_evidence_reject"] == ""
    scored = _row_final_quality_score(left.iloc[0])
    assert scored == pytest.approx(0.81)
    missing = left.iloc[0].copy()
    missing["score"] = ""
    missing["_operational_quality"] = float("nan")
    missing["_quality_evidence_reject"] = REASON_QUALITY_EVIDENCE_MISSING
    assert math_is_nan(_row_final_quality_score(missing))


def math_is_nan(value: float) -> bool:
    return value != value


def test_percentile_tie_and_single_candidate_keep_documented_rank(binding):
    sha = binding["sha"]
    session = binding["session"]
    tied = _annotate_operational_quality(
        pd.DataFrame(
            [
                _row("AAPL", 0.7, session=session, sha=sha),
                _row("AMD", 0.7, session=session, sha=sha),
            ]
        ),
        binding,
    )
    tied["_exec_final_score"] = tied["_operational_quality"]
    tied["final_score"] = tied["_operational_quality"]
    norm = _normalize_exec_final_score_percentile_rank(tied)
    assert list(norm["_exec_final_score"]) == [0.0, 0.0]
    one = tied.iloc[[0]].copy()
    one_norm = _normalize_exec_final_score_percentile_rank(one)
    assert float(one_norm.iloc[0]["_exec_final_score"]) == 1.0


def test_missing_score_is_not_percentile_zero(binding):
    sha = binding["sha"]
    session = binding["session"]
    frame = _annotate_operational_quality(
        pd.DataFrame(
            [
                _row("AAPL", 0.8, session=session, sha=sha),
                _row("AMD", "", session=session, sha=sha),
            ]
        ),
        binding,
    )
    assert frame.loc[frame["ticker"] == "AMD", "_quality_evidence_reject"].iloc[0] == (
        REASON_QUALITY_EVIDENCE_MISSING
    )
    frame["_exec_final_score"] = frame["_operational_quality"]
    frame["final_score"] = frame["_operational_quality"]
    norm = _normalize_exec_final_score_percentile_rank(frame)
    amd = norm.loc[norm["ticker"] == "AMD"].iloc[0]
    assert amd["_quality_evidence_reject"] == REASON_QUALITY_EVIDENCE_MISSING
    assert math_is_nan(float(amd["_exec_final_score"]))
    assert float(norm.loc[norm["ticker"] == "AAPL", "_exec_final_score"].iloc[0]) == 1.0


def test_planner_rejects_missing_quality_before_any_order(binding, monkeypatch):
    sha = binding["sha"]
    session = binding["session"]
    monkeypatch.setattr(et, "_ref_price", lambda *_a, **_k: 100.0)
    frame = pd.DataFrame([_row("AAPL", "", session=session, sha=sha)])
    cfg = load_execute_trades_config()
    cfg["positions_snapshot_path"] = ""
    planned, _lines, skip, *_rest = build_execution_plan(
        frame,
        cfg,
        {},
        AccountSnapshot(equity=100000, buying_power=100000, cash=100000),
        None,
        "paper",
        verbose=False,
    )
    assert planned == []
    assert skip.get(REASON_QUALITY_EVIDENCE_MISSING, 0) == 1
    assert skip.get("BELOW_MIN_SCORE", 0) == 0


def test_two_distinct_scores_are_not_collapsed_to_zero(binding, monkeypatch):
    sha = binding["sha"]
    session = binding["session"]
    monkeypatch.setattr(et, "_ref_price", lambda *_a, **_k: 100.0)
    frame = pd.DataFrame(
        [
            _row("AAPL", 0.90, session=session, sha=sha),
            _row("AMD", 0.70, session=session, sha=sha, spread_bucket="TIGHT"),
        ]
    )
    cfg = load_execute_trades_config()
    cfg["positions_snapshot_path"] = ""
    cfg["diversification_ranking_enabled"] = False
    planned, lines, skip, *_rest = build_execution_plan(
        frame,
        cfg,
        {},
        AccountSnapshot(equity=100000, buying_power=100000, cash=100000),
        None,
        "paper",
        verbose=False,
    )
    reasons = {ln.symbol: ln.skip_reason for ln in lines}
    assert REASON_QUALITY_EVIDENCE_MISSING not in reasons.values()
    assert skip.get("BELOW_MIN_SCORE", 0) <= 1
    assert any(p.symbol == "AAPL" for p in planned) or reasons.get("AAPL") not in {
        REASON_QUALITY_EVIDENCE_MISSING,
        "BELOW_MIN_SCORE",
    }


def test_dry_run_and_missing_contract_do_not_submit():
    order = PlannedOrder(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="limit",
        time_in_force="day",
        limit_price=100.0,
        stance="BUY",
        rationale="",
        confidence=0.8,
        delta_pct=0.0,
        planned_notional=100.0,
        current_position_qty=0.0,
        mode="paper",
        generated_at="2026-09-21T00:00:00Z",
    )
    code, submitted, reasons = maybe_execute_plan(
        [order],
        "paper",
        True,
        {},
        None,
        False,
        False,
        Path("unused.json"),
    )
    assert (code, submitted, reasons) == (0, 0, [])
    code, submitted, reasons = maybe_execute_plan(
        [order],
        "paper",
        False,
        {},
        None,
        False,
        False,
        Path("unused.json"),
        dispatch_contract=None,
    )
    assert code == 2 and submitted == 0
    assert reasons == ["DISPATCH_CONTRACT_REQUIRED"]
    _doc, reason, _meta = require_execute_dispatch_contract(None, mode="paper")
    assert reason == "DISPATCH_CONTRACT_REQUIRED"


def test_supplied_threshold_is_honored_and_unmodified(binding, monkeypatch):
    cfg_path = Path(binding["config_path"])
    before = cfg_path.read_bytes()
    monkeypatch.setattr(et, "CONFIG_PATH", cfg_path)
    cfg = load_execute_trades_config()
    assert float(cfg["min_final_score_threshold"]) == pytest.approx(0.0020)
    assert cfg["block_negative_final_score"] is True
    assert cfg["enforce_min_final_score"] is True
    assert cfg.get("exec_score_percentile_normalize", True) is True
    assert cfg_path.read_bytes() == before

    sha = binding["sha"]
    session = binding["session"]
    monkeypatch.setattr(et, "_ref_price", lambda *_a, **_k: 100.0)
    boundary = dict(cfg)
    boundary["exec_score_percentile_normalize"] = False
    boundary["diversification_ranking_enabled"] = False
    account = AccountSnapshot(equity=100000, buying_power=100000, cash=100000)

    low, _lines, low_skip, *_rest = build_execution_plan(
        pd.DataFrame([_row("AAPL", 0.0010, session=session, sha=sha)]),
        boundary,
        {},
        account,
        None,
        "paper",
        verbose=False,
    )
    assert low == []
    assert low_skip.get("BELOW_MIN_SCORE", 0) == 1
    assert low_skip.get(REASON_QUALITY_EVIDENCE_MISSING, 0) == 0

    high, _high_lines, high_skip, *_rest = build_execution_plan(
        pd.DataFrame([_row("AAPL", 0.10, session=session, sha=sha)]),
        boundary,
        {},
        account,
        None,
        "paper",
        verbose=False,
    )
    assert high_skip.get("BELOW_MIN_SCORE", 0) == 0
    assert high_skip.get(REASON_QUALITY_EVIDENCE_MISSING, 0) == 0
    assert any(order.symbol == "AAPL" for order in high)
    assert cfg_path.read_bytes() == before
