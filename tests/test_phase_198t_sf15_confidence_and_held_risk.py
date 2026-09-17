"""Phase 198T-SF15 — confidence contract and held-risk operator alert.

Isolated worktree only. Disposable paths. No broker, network, production
write, Stop request, or trading-control invocation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-SF15 tests must not run in the production repository.")
if "isolated" not in str(_REPO_ROOT).lower():
    raise RuntimeError("Phase 198T-SF15 tests must run from an isolated worktree.")

from dashboard import operator_console as oc
from services.portfolio_allocation_engine import (
    MIN_OPEN_NEW_CONFIDENCE,
    _decide_action,
    _load_signals_map,
)
from services.signal_lifecycle import (
    CONFIDENCE_REASON_CONFLICT,
    CONFIDENCE_REASON_MISSING,
    CONFIDENCE_REASON_OK,
    CONFIDENCE_REASON_SCORE_ALIAS,
    LifecycleConfig,
    apply_lifecycle,
    decide_action,
    lifecycle_logic_from_dict,
    resolve_row_confidence,
)


def _cfg() -> LifecycleConfig:
    return LifecycleConfig(
        min_hold_days=1,
        cooldown_days_after_exit=1,
        buy_delta_pct=0.0015,
        add_delta_pct=0.0020,
        exit_delta_pct=-0.0015,
        trim_delta_pct=-0.0030,
        hold_means_keep_position=True,
    )


def _logic():
    return lifecycle_logic_from_dict({"enabled": True})


def test_resolve_confidence_prefers_finite_primary():
    val, reason = resolve_row_confidence({"confidence": 0.820268, "confidence_score": 0.820268})
    assert val == pytest.approx(0.820268)
    assert reason == CONFIDENCE_REASON_OK


def test_resolve_confidence_legacy_score_alias():
    val, reason = resolve_row_confidence({"confidence": "", "confidence_score": 0.820268})
    assert val == pytest.approx(0.820268)
    assert reason == CONFIDENCE_REASON_SCORE_ALIAS


def test_resolve_confidence_missing_and_nonfinite():
    assert resolve_row_confidence({})[1] == CONFIDENCE_REASON_MISSING
    assert resolve_row_confidence({"confidence": float("nan")})[1] == CONFIDENCE_REASON_MISSING
    assert resolve_row_confidence({"confidence": "inf"})[1] == CONFIDENCE_REASON_MISSING
    assert resolve_row_confidence({"confidence_score": None})[0] is None


def test_resolve_confidence_conflict_rejected():
    val, reason = resolve_row_confidence({"confidence": 0.10, "confidence_score": 0.820268})
    assert val is None
    assert reason == CONFIDENCE_REASON_CONFLICT


def test_decide_action_rejects_missing_confidence_before_buy():
    action, reason = decide_action(
        cfg=_cfg(),
        logic=_logic(),
        signal="BUY",
        delta_pct=float("nan"),
        confidence=float("nan"),
        pos_state="FLAT",
        last_change_date=None,
        cooldown_until=None,
        as_of_date=None,
    )
    assert action == "WAIT"
    assert reason == "flat_buy_missing_confidence"


def test_decide_action_finite_confidence_still_buys():
    action, reason = decide_action(
        cfg=_cfg(),
        logic=_logic(),
        signal="BUY",
        delta_pct=float("nan"),
        confidence=0.820268,
        pos_state="FLAT",
        last_change_date=None,
        cooldown_until=None,
        as_of_date=None,
    )
    assert action == "BUY"
    assert reason == "flat_buy_to_buy"


def test_apply_lifecycle_preserves_legacy_score(tmp_path):
    state = tmp_path / "signal_state.json"
    prior = tmp_path / "decision_prior_state.json"
    df = pd.DataFrame(
        [
            {
                "ticker": "META",
                "signal": "BUY",
                "confidence_score": 0.820268,
                "score": 0.820268,
                "as_of_date": "2026-09-16",
            }
        ]
    )
    out, _state = apply_lifecycle(df, state_path=state, cfg=_cfg(), decision_state_path=prior)
    row = out.iloc[0]
    assert float(row["confidence"]) == pytest.approx(0.820268)
    assert row["lifecycle_action"] == "BUY"
    assert row["lifecycle_decision_reason"] == "flat_buy_to_buy"


def test_apply_lifecycle_conflict_not_actionable_buy(tmp_path):
    state = tmp_path / "signal_state.json"
    prior = tmp_path / "decision_prior_state.json"
    df = pd.DataFrame(
        [
            {
                "ticker": "META",
                "signal": "BUY",
                "confidence": 0.10,
                "confidence_score": 0.820268,
                "score": 0.820268,
                "as_of_date": "2026-09-16",
            }
        ]
    )
    out, _state = apply_lifecycle(df, state_path=state, cfg=_cfg(), decision_state_path=prior)
    row = out.iloc[0]
    assert not math.isfinite(float(row["confidence"])) or pd.isna(row["confidence"])
    assert row["lifecycle_action"] == "WAIT"
    assert row["lifecycle_decision_reason"] == "flat_buy_missing_confidence"


def test_allocation_uses_resolved_score_and_rejects_conflict():
    ok = _load_signals_map(
        pd.DataFrame([{"ticker": "META", "signal": "BUY", "confidence_score": 0.820268}])
    )
    assert ok["META"]["confidence"] == pytest.approx(0.820268)
    bad = _load_signals_map(
        pd.DataFrame(
            [{"ticker": "META", "signal": "BUY", "confidence": 0.10, "confidence_score": 0.82}]
        )
    )
    assert bad["META"]["confidence"] is None
    action, reason = _decide_action(
        is_held=False,
        risk_components=[],
        edge_score=0.92,
        sizing_tier="LARGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=None,
        delta_pct=None,
    )
    assert action != "OPEN_NEW"
    assert "conf<" in reason or action == "WATCH"
    assert MIN_OPEN_NEW_CONFIDENCE == 0.50


def _write_held_risk_fixture(
    root: Path,
    *,
    symbol: str = "XLE",
    skip_reason: str = "INVALID_PROCESSED_SESSION_BAR",
    evidence_date: str = "2026-09-16",
    risk_flag: str = "FORCE_EXIT",
    plan_code: str = "STALE_LIFECYCLE_NO_POSITION",
    qty: float = 10.0,
) -> Path:
    results = root / "data" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "signals_asof_diagnostics.json").write_text(
        json.dumps(
            {
                "session_asof_date": evidence_date,
                "per_symbol_skips": [
                    {
                        "symbol": symbol,
                        "expected_session_date": evidence_date,
                        "actual_latest_bar_date": evidence_date,
                        "reason": skip_reason,
                        "class": "tradable",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with (results / "positions_snapshot.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "symbol", "qty"])
        w.writeheader()
        w.writerow({"ticker": symbol, "symbol": symbol, "qty": qty})
    with (results / "performance_risk_overlay.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "risk_flag"])
        w.writeheader()
        w.writerow({"ticker": symbol, "risk_flag": risk_flag})
    with (results / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["symbol", "reason_code", "final_action", "selected_for_execution"]
        )
        w.writeheader()
        w.writerow(
            {
                "symbol": symbol,
                "reason_code": plan_code,
                "final_action": "",
                "selected_for_execution": "False",
            }
        )
    return root


def test_held_risk_alert_current_invalid_evidence(tmp_path):
    root = _write_held_risk_fixture(tmp_path)
    alerts = oc.collect_held_risk_plan_blocked_alerts(root=root)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["symbol"] == "XLE"
    assert alert["evidence_date"] == "2026-09-16"
    assert alert["blocked_action"] == "FORCE_EXIT"
    assert alert["sell_placed"] is False
    assert "Operator review required" in alert["message"]
    assert "No SELL was placed" in alert["message"]


def test_held_risk_alert_stale_evidence(tmp_path):
    root = _write_held_risk_fixture(
        tmp_path,
        skip_reason="STALE_PROCESSED_SESSION_BAR",
        evidence_date="2026-09-10",
    )
    alerts = oc.collect_held_risk_plan_blocked_alerts(root=root)
    assert len(alerts) == 1
    assert alerts[0]["evidence_date"] == "2026-09-10"
    assert "stale" in alerts[0]["message"]


def test_held_risk_alert_absent_when_not_held_or_no_risk(tmp_path):
    root = _write_held_risk_fixture(tmp_path, qty=0.0)
    assert oc.collect_held_risk_plan_blocked_alerts(root=root) == []
    root2 = _write_held_risk_fixture(tmp_path / "ok", risk_flag="OK")
    assert oc.collect_held_risk_plan_blocked_alerts(root=root2) == []


def _patch_streamlit(monkeypatch, session: Dict[str, Any]):
    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getattr__(self, name):
            return lambda *a, **k: None

    class _FakeSession(dict):
        def get(self, k, default=None):
            return super().get(k, default)

    fake = type("ST", (), {})()
    fake.session_state = _FakeSession(session)
    fake.button = lambda *a, **k: False
    fake.caption = lambda *a, **k: None
    fake.info = lambda *a, **k: None
    fake.warning = lambda *a, **k: None
    fake.error = lambda *a, **k: None
    fake.success = lambda *a, **k: None
    fake.markdown = lambda *a, **k: None
    fake.write = lambda *a, **k: None
    fake.text_input = lambda *a, **k: ""
    fake.checkbox = lambda *a, **k: False
    fake.subheader = lambda *a, **k: None
    fake.columns = lambda n: (_Col(), _Col())
    fake.container = lambda *a, **k: _Col()
    fake.expander = lambda *a, **k: _Col()
    monkeypatch.setattr(oc, "st", fake)
    return oc


def test_stop_panel_running_and_stopped_do_not_request_stop(monkeypatch):
    import services.operator_stop_action as osa

    monkeypatch.setattr(
        osa,
        "execute_stop_triton",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not request Stop")),
    )
    monkeypatch.setattr(
        osa,
        "issue_stop_preflight",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not issue preflight")),
    )
    _patch_streamlit(
        monkeypatch,
        {
            "stop_confirm_pending": True,
            "stop_in_progress": False,
            "stop_preflight_expires_at": "2026-09-17T16:40:00Z",
            "stop_preflight_decision": "STOP_ELIGIBLE_BOTH",
            "stop_preflight_targets": ["continuous", "watchdog"],
            "stop_preflight_token": "token",
            "stop_request_nonce": "nonce",
        },
    )
    oc._render_stop_control_column(
        {"decision": "STOP_ELIGIBLE_BOTH", "stop_button_enabled": True},
        start_busy=False,
    )
    _patch_streamlit(monkeypatch, {"stop_confirm_pending": False, "stop_in_progress": False})
    oc._render_stop_control_column(
        {"decision": "STOP_ALREADY_STOPPED", "stop_button_enabled": False},
        start_busy=False,
    )


def test_held_risk_render_does_not_invoke_stop(tmp_path, monkeypatch):
    import services.operator_stop_action as osa

    monkeypatch.setattr(
        osa,
        "execute_stop_triton",
        lambda **k: (_ for _ in ()).throw(AssertionError("alert render must not Stop")),
    )
    _write_held_risk_fixture(tmp_path)
    _patch_streamlit(monkeypatch, {})
    oc._render_held_risk_plan_blocked_alerts(root=tmp_path)
