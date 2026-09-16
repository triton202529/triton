"""Phase 198T-RF2 — Random Forest shadow-only repair.

Deterministic. Disposable paths only. No broker, network, accounting,
authority, runtime, or production artifact mutation.
"""

from __future__ import annotations

import builtins
import hashlib
import os
import shutil
import socket
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-RF2 tests must not run in the production repository.")

_RF1_FROZEN = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_RF1_20260915_234016" r"\frozen_evidence\selected"
)

FEATURE_BUYS_BLOCKED_BY_RF = (
    "AAPL",
    "AMD",
    "C",
    "CVX",
    "INTC",
    "SPCX",
    "T",
    "USO",
    "WFC",
    "XLE",
    "XOM",
    "^VIX",
)
HELD_LONG = ("ARKK", "BITO", "DBA", "DIA", "MA", "TSLA")


def _guard_write(path: Any) -> None:
    try:
        text = os.path.abspath(os.fspath(path))
    except TypeError:
        return
    norm = text.replace("/", "\\").lower()
    prod = str(_PROD_ROOT.resolve()).replace("/", "\\").lower()
    if norm == prod or norm.startswith(prod + "\\"):
        raise RuntimeError(f"production write blocked: {path}")


@pytest.fixture(autouse=True)
def _phase_198t_rf2_safety(monkeypatch):
    real_open = builtins.open
    real_replace = os.replace
    real_remove = os.remove
    real_rename = os.rename

    def guarded_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(flag in mode_s for flag in ("w", "a", "x", "+")):
            _guard_write(file)
        return real_open(file, mode, *args, **kwargs)

    def guarded_replace(src, dst):
        _guard_write(dst)
        return real_replace(src, dst)

    def guarded_remove(path):
        _guard_write(path)
        return real_remove(path)

    def guarded_rename(src, dst):
        _guard_write(dst)
        return real_rename(src, dst)

    class BlockedSocket:
        def __init__(self, *a, **k):
            raise RuntimeError("network/broker calls are blocked in Phase 198T-RF2 tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198T-RF2 tests")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "remove", guarded_remove)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked_connect)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.setenv("PAPER_ONLY", "1")
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield


def _cfg():
    from services.signal_lifecycle import LifecycleConfig, lifecycle_logic_from_dict

    cfg = LifecycleConfig(
        min_hold_days=1,
        cooldown_days_after_exit=1,
        buy_delta_pct=0.0015,
        add_delta_pct=0.0020,
        exit_delta_pct=-0.0015,
        trim_delta_pct=-0.0030,
        hold_means_keep_position=True,
    )
    logic = lifecycle_logic_from_dict(
        {
            "enabled": True,
            "add_confidence_min": 0.52,
            "add_delta_pct_min": 0.008,
            "hold_delta_floor": -0.003,
            "hold_delta_ceiling": 0.006,
            "trim_delta_pct_threshold": -0.001,
            "exit_delta_pct_threshold": -0.0025,
            "exit_confidence_min": 0.50,
        }
    )
    return cfg, logic


def _decide(signal: str, delta: float, confidence: float, pos: str, **kwargs):
    from services.signal_lifecycle import decide_action

    cfg, logic = _cfg()
    return decide_action(
        cfg=cfg,
        logic=logic,
        signal=signal,
        delta_pct=delta,
        confidence=confidence,
        pos_state=pos,
        last_change_date=kwargs.get("last_change_date", date(2026, 9, 1)),
        cooldown_until=kwargs.get("cooldown_until"),
        as_of_date=kwargs.get("as_of_date", date(2026, 9, 15)),
    )


def test_import_generate_signals_makes_no_broker_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import services.generate_signals as gs

    assert gs.RF_SHADOW_ONLY is True
    assert callable(gs.classify_rf_shadow_evidence)


def test_import_lifecycle_allocation_manage_no_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import services.signal_lifecycle as sl
    import services.build_effective_lifecycle as bel
    import services.portfolio_allocation_engine as pae
    import services.manage_positions as mp

    assert callable(sl.decide_action)
    assert callable(bel._deterministic_effective_stance)
    assert callable(pae._decide_action)
    assert callable(mp._compute_weak_signal_score)


def test_feature_buy_not_vetoed_by_negative_rf_residual():
    action, reason = _decide("BUY", -0.0156, 0.72, "FLAT")
    assert action == "BUY"
    assert reason == "flat_buy_to_buy"
    assert reason != "flat_buy_weak_to_wait"


def test_hold_not_promoted_to_exit_or_trim_by_rf_residual():
    exit_action, exit_reason = _decide("HOLD", -0.00465, 0.55, "LONG")
    trim_action, trim_reason = _decide("HOLD", -0.00150, 0.55, "LONG")
    assert exit_action == "HOLD"
    assert exit_reason == "long_hold_signal_hold"
    assert trim_action == "HOLD"
    assert trim_reason == "long_hold_signal_hold"


def test_genuine_risk_exit_from_sell_conviction_still_occurs():
    action, reason = _decide("SELL", float("nan"), 0.80, "LONG")
    assert action == "EXIT"
    assert reason == "long_bearish_reversal_to_exit"


def test_genuine_risk_exit_and_trim_from_allocation_flags():
    from services.portfolio_allocation_engine import _decide_action

    exit_action, exit_reason = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.8,
        delta_pct=-0.05,
    )
    trim_action, trim_reason = _decide_action(
        is_held=True,
        risk_components=["TRIM_PRIORITY"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.8,
        delta_pct=-0.05,
    )
    assert exit_action == "EXIT"
    assert "FORCE_EXIT" in exit_reason
    assert trim_action == "TRIM"
    assert "TRIM_PRIORITY" in trim_reason


def test_rf_prediction_retained_as_shadow_evidence():
    from services.generate_signals import classify_rf_shadow_evidence

    shadow = classify_rf_shadow_evidence(
        predicted_close_raw=327.090393371582,
        close=332.29,
        row_date="2026-09-14",
        session_asof=date(2026, 9, 14),
        column_present=True,
    )
    assert shadow["rf_shadow_only"] is True
    assert shadow["rf_evidence_status"] == "valid"
    assert shadow["rf_predicted_close"] == pytest.approx(327.090393371582)
    assert shadow["rf_delta_pct"] < 0
    assert math_isfinite(shadow["rf_delta_pct"])


def math_isfinite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def test_stale_rf_prediction_is_diagnostic_only():
    from services.generate_signals import classify_rf_shadow_evidence

    shadow = classify_rf_shadow_evidence(
        predicted_close_raw=100.0,
        close=101.0,
        row_date="2026-09-11",
        session_asof=date(2026, 9, 14),
        column_present=True,
    )
    assert shadow["rf_evidence_status"] == "stale"
    assert math_isfinite(shadow["rf_delta_pct"])
    action, _reason = _decide("BUY", shadow["rf_delta_pct"], 0.70, "FLAT")
    assert action == "BUY"


def test_missing_rf_prediction_is_diagnostic_only():
    from services.generate_signals import classify_rf_shadow_evidence

    shadow = classify_rf_shadow_evidence(
        predicted_close_raw=None,
        close=101.0,
        row_date="2026-09-14",
        session_asof=date(2026, 9, 14),
        column_present=False,
    )
    assert shadow["rf_evidence_status"] == "unavailable"
    assert not math_isfinite(shadow["rf_delta_pct"])
    action, _reason = _decide("BUY", float("nan"), 0.70, "FLAT")
    assert action == "BUY"


def test_malformed_rf_evidence_fails_closed_without_changing_decision():
    from services.generate_signals import classify_rf_shadow_evidence

    shadow = classify_rf_shadow_evidence(
        predicted_close_raw="not-a-number",
        close=101.0,
        row_date="2026-09-14",
        session_asof=date(2026, 9, 14),
        column_present=True,
    )
    assert shadow["rf_evidence_status"] == "malformed"
    buy_a, _ = _decide("BUY", float("nan"), 0.70, "FLAT")
    buy_b, _ = _decide("BUY", 0.0, 0.70, "FLAT")
    hold_a, _ = _decide("HOLD", float("nan"), 0.55, "LONG")
    hold_b, _ = _decide("HOLD", -0.05, 0.55, "LONG")
    assert buy_a == buy_b == "BUY"
    assert hold_a == hold_b == "HOLD"


def test_ensure_delta_pct_does_not_recover_rf_from_predicted_close():
    from services.signal_lifecycle import _ensure_delta_pct

    df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "close": [100.0],
            "predicted_close": [90.0],
            "delta_pct": [np.nan],
        }
    )
    out = _ensure_delta_pct(df)
    assert not math_isfinite(float(out.loc[0, "delta_pct"]))


def test_allocation_does_not_depend_on_rf_residual():
    from services.portfolio_allocation_engine import _allocation_score, _decide_action

    neg = _decide_action(
        is_held=False,
        risk_components=[],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="WAIT",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.72,
        delta_pct=-0.02,
    )
    pos = _decide_action(
        is_held=False,
        risk_components=[],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="WAIT",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.72,
        delta_pct=0.02,
    )
    missing = _decide_action(
        is_held=False,
        risk_components=[],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="WAIT",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.72,
        delta_pct=None,
    )
    assert neg == pos == missing
    assert neg[0] == "OPEN_NEW"
    score_neg = _allocation_score(
        edge_score=0.8,
        confidence=0.7,
        delta_pct=-0.04,
        perf_bucket="OK",
        risk_components=[],
        weight_pct=5.0,
    )
    score_pos = _allocation_score(
        edge_score=0.8,
        confidence=0.7,
        delta_pct=0.04,
        perf_bucket="OK",
        risk_components=[],
        weight_pct=5.0,
    )
    assert score_neg == score_pos


def test_position_management_ranking_does_not_depend_on_rf_residual():
    from services.manage_positions import _compute_rotation_score, _compute_weak_signal_score

    assert _compute_weak_signal_score(-0.05, 0.8) == _compute_weak_signal_score(0.05, 0.8)
    assert _compute_weak_signal_score(float("nan"), 0.8) == _compute_weak_signal_score(0.01, 0.8)
    a = _compute_rotation_score(0.10, -0.05, profit_weight=0.6, signal_weight=0.4)
    b = _compute_rotation_score(0.10, 0.05, profit_weight=0.6, signal_weight=0.4)
    c = _compute_rotation_score(0.10, float("nan"), profit_weight=0.6, signal_weight=0.4)
    assert a == b == c


def test_changing_only_rf_output_does_not_change_action_or_quantity():
    from services.manage_positions import _optional_delta

    a, ra = _decide("BUY", -0.20, 0.71, "FLAT")
    b, rb = _decide("BUY", 0.20, 0.71, "FLAT")
    assert (a, ra) == (b, rb)
    c, _ = _decide("HOLD", -0.20, 0.55, "LONG")
    d, _ = _decide("HOLD", 0.20, 0.55, "LONG")
    assert c == d == "HOLD"
    assert not math_isfinite(_optional_delta(None))
    assert not math_isfinite(_optional_delta("bad"))
    assert _optional_delta(-0.01) == pytest.approx(-0.01)


def test_effective_lifecycle_ignores_rf_residual():
    from services.build_effective_lifecycle import _deterministic_effective_stance

    pos, stance, tag = _deterministic_effective_stance(
        "LONG",
        "HOLD",
        delta_pct=-0.05,
        confidence=0.7,
    )
    assert pos == "LONG"
    assert stance == "HOLD"
    buy_pos, buy_stance, _tag = _deterministic_effective_stance(
        "LONG",
        "BUY",
        delta_pct=-0.05,
        confidence=0.9,
    )
    assert buy_pos == "LONG"
    assert buy_stance == "HOLD"


def _bind_generate(monkeypatch, root: Path):
    import services.generate_signals as gs

    results = root / "data" / "results"
    preds = root / "data" / "predictions"
    processed = root / "data" / "processed"
    for path in (results, preds, processed):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(processed))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.chdir(root)
    return gs, results, preds


def _pred_frame(ticker: str, last: date, rows: int = 30, include_pred: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    close = np.linspace(50.0, 55.0, rows)
    data: Dict[str, Any] = {
        "date": dates,
        "close": close,
        "ticker": [ticker] * rows,
    }
    if include_pred:
        data["predicted_close"] = close * 0.98
    return pd.DataFrame(data)


def test_generate_signals_emits_shadow_fields_and_nan_operational_delta(tmp_path, monkeypatch):
    gs, results, preds = _bind_generate(monkeypatch, tmp_path)
    last = date(2026, 8, 18)
    _pred_frame("AAA", last).to_parquet(preds / "AAA_predictions.parquet", index=False)
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: last)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-08-19T12:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    gs.main()
    df = pd.read_csv(results / "signals_with_rationale.csv")
    assert set(
        [
            "rf_predicted_close",
            "rf_delta_pct",
            "rf_evidence_timestamp",
            "rf_evidence_status",
            "rf_shadow_only",
        ]
    ).issubset(df.columns)
    assert bool(df.iloc[0]["rf_shadow_only"]) is True
    assert not math_isfinite(float(df.iloc[0]["delta_pct"]))
    assert math_isfinite(float(df.iloc[0]["rf_delta_pct"]))
    assert str(df.iloc[0]["signal"]) in {"BUY", "HOLD", "SELL"}
    rationale = str(df.iloc[0]["rationale"])
    assert "model " not in rationale.lower() or "shadow" in rationale.lower()


def test_generate_signals_missing_predicted_close_does_not_skip_ticker(tmp_path, monkeypatch):
    gs, results, preds = _bind_generate(monkeypatch, tmp_path)
    last = date(2026, 8, 18)
    _pred_frame("BBB", last, include_pred=False).to_parquet(
        preds / "BBB_predictions.parquet", index=False
    )
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: last)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-08-19T12:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    gs.main()
    df = pd.read_csv(results / "signals_with_rationale.csv")
    assert list(df["ticker"].astype(str)) == ["BBB"]
    assert str(df.iloc[0]["rf_evidence_status"]) == "unavailable"
    assert not math_isfinite(float(df.iloc[0]["delta_pct"]))


def test_confirmation_phrases_callback_keys_and_clipping_unchanged():
    from services.paper_sell_arm_console import CONFIRM_PAPER_SELLS_PHRASE

    assert CONFIRM_PAPER_SELLS_PHRASE == "CONFIRM PAPER SELLS"
    mp_src = (_REPO_ROOT / "services" / "manage_positions.py").read_text(encoding="utf-8")
    assert "qty_available" in mp_src
    assert "ZERO_QTY_AFTER_ROUNDING" in mp_src
    gs_src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    assert "CONFIRM PAPER SELLS" not in gs_src
    sl_src = (_REPO_ROOT / "services" / "signal_lifecycle.py").read_text(encoding="utf-8")
    assert "CONFIRM PAPER SELLS" not in sl_src


def _copy_frozen(tmp_path: Path) -> Path:
    if not _RF1_FROZEN.is_dir():
        pytest.skip(f"frozen RF1 evidence missing: {_RF1_FROZEN}")
    dest = tmp_path / "rf1_frozen"
    shutil.copytree(_RF1_FROZEN, dest)
    return dest


def _repaired_signals(frozen: Path) -> pd.DataFrame:
    from services.generate_signals import classify_rf_shadow_evidence

    src = frozen / "signals_with_rationale.csv"
    df = pd.read_csv(src)
    rows: List[Dict[str, Any]] = []
    session_asof = date(2026, 9, 14)
    for _, row in df.iterrows():
        rec = dict(row)
        shadow = classify_rf_shadow_evidence(
            predicted_close_raw=row.get("predicted_close"),
            close=row.get("close"),
            row_date=row.get("date"),
            session_asof=session_asof,
            column_present=True,
        )
        rec["rf_predicted_close"] = shadow["rf_predicted_close"]
        rec["rf_delta_pct"] = shadow["rf_delta_pct"]
        rec["rf_evidence_timestamp"] = shadow["rf_evidence_timestamp"]
        rec["rf_evidence_status"] = shadow["rf_evidence_status"]
        rec["rf_shadow_only"] = True
        rec["delta_pct"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def replay_sep15_lifecycle(tmp_path: Path) -> pd.DataFrame:
    from services.signal_lifecycle import apply_lifecycle

    frozen = _copy_frozen(tmp_path)
    signals = _repaired_signals(frozen)
    state_src = frozen / "signal_state.json"
    state_path = tmp_path / "signal_state.json"
    state_path.write_text(state_src.read_text(encoding="utf-8"), encoding="utf-8")
    cfg, logic = _cfg()
    out, _state = apply_lifecycle(
        signals,
        state_path=state_path,
        cfg=cfg,
        lifecycle_logic=logic,
        decision_state_path=tmp_path / "decision_prior_state.json",
    )
    old = pd.read_csv(frozen / "signal_lifecycle.csv")
    merged = out.merge(
        old[["ticker", "lifecycle_action", "lifecycle_decision_reason", "signal", "delta_pct"]],
        on="ticker",
        how="left",
        suffixes=("", "_old"),
    )
    if "signal_old" not in merged.columns and "signal" in old.columns:
        merged["signal_old"] = merged["signal"]
    return merged


def test_frozen_sep15_replay_symbol_decisions(tmp_path):
    merged = replay_sep15_lifecycle(tmp_path)
    by = merged.set_index(merged["ticker"].astype(str))

    for ticker in FEATURE_BUYS_BLOCKED_BY_RF:
        row = by.loc[ticker]
        assert str(row["signal"]) == "BUY"
        assert str(row["lifecycle_action_old"]) == "WAIT"
        assert str(row["lifecycle_decision_reason_old"]) == "flat_buy_weak_to_wait"
        assert str(row["lifecycle_action"]) == "BUY"
        assert str(row["lifecycle_decision_reason"]) == "flat_buy_to_buy"
        assert bool(row["rf_would_have_changed"]) is True
        assert bool(row["rf_shadow_only"]) is True
        assert math_isfinite(float(row["rf_delta_pct"]))
        assert float(row["rf_delta_pct"]) <= 0

    meta = by.loc["META"]
    assert str(meta["signal"]) == "BUY"
    assert str(meta["lifecycle_action_old"]) == "BUY"
    assert str(meta["lifecycle_action"]) == "BUY"
    assert str(meta["lifecycle_decision_reason"]) == "flat_buy_to_buy"

    for ticker in ("DIA", "TSLA"):
        row = by.loc[ticker]
        assert str(row["signal"]) == "HOLD"
        assert str(row["lifecycle_decision_reason_old"]) == "long_exit_delta_threshold"
        assert str(row["lifecycle_action"]) == "HOLD"
        assert str(row["lifecycle_action"]) != "EXIT"
        assert str(row["lifecycle_decision_reason"]) == "long_hold_signal_hold"

    ma = by.loc["MA"]
    assert str(ma["signal"]) == "HOLD"
    assert str(ma["lifecycle_decision_reason_old"]) == "long_mild_deterioration_to_trim"
    assert str(ma["lifecycle_action"]) == "HOLD"
    assert str(ma["lifecycle_action"]) != "TRIM"

    for ticker in ("ARKK", "BITO", "DBA"):
        row = by.loc[ticker]
        assert str(row["lifecycle_action"]) == "HOLD"
        assert str(row["lifecycle_action_old"]) == "HOLD"

    stale_or_valid = set(merged["rf_evidence_status"].astype(str))
    assert stale_or_valid <= {"stale", "valid", "unavailable", "malformed"}
    assert "stale" in stale_or_valid


def test_frozen_sep15_allocation_and_manage_ignore_rf_only_changes(tmp_path):
    from services.portfolio_allocation_engine import _decide_action
    from services.manage_positions import _compute_rotation_score, _compute_weak_signal_score

    merged = replay_sep15_lifecycle(tmp_path)
    by = merged.set_index(merged["ticker"].astype(str))
    for ticker in FEATURE_BUYS_BLOCKED_BY_RF:
        row = by.loc[ticker]
        a = _decide_action(
            is_held=False,
            risk_components=[],
            edge_score=0.2,
            sizing_tier="BASE",
            effective_stance=str(row["lifecycle_action"]),
            lifecycle_action=str(row["lifecycle_action"]),
            signal=str(row["signal"]),
            confidence=float(row["confidence"]),
            delta_pct=float(row["rf_delta_pct"]),
        )
        b = _decide_action(
            is_held=False,
            risk_components=[],
            edge_score=0.2,
            sizing_tier="BASE",
            effective_stance=str(row["lifecycle_action"]),
            lifecycle_action=str(row["lifecycle_action"]),
            signal=str(row["signal"]),
            confidence=float(row["confidence"]),
            delta_pct=0.05,
        )
        assert a == b
    for ticker in HELD_LONG:
        row = by.loc[ticker]
        rf = float(row["rf_delta_pct"])
        conf = float(row["confidence"])
        assert _compute_weak_signal_score(rf, conf) == _compute_weak_signal_score(-rf, conf)
        assert _compute_rotation_score(0.02, rf, profit_weight=0.6, signal_weight=0.4) == (
            _compute_rotation_score(0.02, -rf, profit_weight=0.6, signal_weight=0.4)
        )


def test_no_model_retrain_or_new_estimator_in_allowlist():
    files = [
        _REPO_ROOT / "services" / "generate_signals.py",
        _REPO_ROOT / "services" / "signal_lifecycle.py",
        _REPO_ROOT / "services" / "build_effective_lifecycle.py",
        _REPO_ROOT / "services" / "portfolio_allocation_engine.py",
        _REPO_ROOT / "services" / "manage_positions.py",
    ]
    for path in files:
        src = path.read_text(encoding="utf-8")
        assert "RandomForestRegressor(" not in src
        assert "joblib.dump" not in src
        assert "sklearn.ensemble" not in src


def test_candidate_sha256_stable_algorithm():
    digest = hashlib.sha256(
        (_REPO_ROOT / "services" / "signal_lifecycle.py").read_bytes()
    ).hexdigest()
    assert len(digest) == 64
