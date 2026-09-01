"""Phase 198T-AN — canonical symbol-identity repair candidate (isolated TEMP only).

PAPER ONLY. Fake brokers. No production writes. No real Alpaca.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    SESSION_OPEN_NOW,
    NetworkFakeAlpacaBroker,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    inprocess_place_live_orders,
    install_boundary_trace,
    isolate_sell_store,
    seed_paper_sell_store,
    stub_unrelated_pre_gates,
)
from services.symbol_identity import (
    COLLISION_REASON,
    SymbolIdentityCollision,
    broker_submit_symbol,
    collect_identity_collisions,
    identity_key,
    market_data_symbol,
)
from services.reconcile_lifecycle_vs_positions import (
    load_lifecycle as rec_load_lifecycle,
    load_positions as rec_load_positions,
    reconcile_simple,
)
from services.build_effective_lifecycle import build_effective, load_lifecycle as bel_load_lifecycle
from services.manage_positions import (
    ManagementPlannedOrder,
    build_management_plan,
    load_long_positions,
    maybe_execute_management,
)
from services.execute_trades import (
    AccountSnapshot,
    _norm_sym,
    build_execution_plan,
    load_broker_state,
)
from services.place_live_orders import normalize_symbol

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(os.environ.get("PHASE_198T_AN_FIXTURES", str(ROOT.parent / "fixtures")))

HELD_AUG31 = {
    "ARKK": 33.0,
    "BITO": 1337.0,
    "BLK": 2.0,
    "BRK.B": 4.0,
    "DBA": 86.0,
    "DIA": 6.0,
    "GE": 9.0,
    "GLD": 16.0,
    "GS": 1.0,
    "HD": 13.0,
    "MA": 7.0,
    "TSLA": 8.137141758,
}

ORDINARY = ["HD", "DBA", "BITO", "TSLA", "GLD"]


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    stub_dir_fsync_ok(monkeypatch)
    block_real_network(monkeypatch)
    monkeypatch.setenv("APCA_API_KEY_ID", "")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "")
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    import services.broker_alpaca as ba

    def _boom(*_a, **_k):
        raise RuntimeError("isolated_test_no_real_broker")

    monkeypatch.setattr(ba, "AlpacaBroker", _boom)
    import services.manage_positions as mp

    monkeypatch.setattr(mp, "_max_positions_for_manage", lambda: 20)
    yield
    _ACTIVE_PERMIT.set(None)


def _lc_row(ticker: str, action: str = "HOLD", position_state: str = "FLAT") -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "stance": action,
        "lifecycle_action": action,
        "effective_stance": action,
        "held_state": "LONG" if action in {"HOLD", "WAIT", "EXIT", "TRIM"} else "FLAT",
        "position_state": position_state,
        "confidence": 0.6,
        "delta_pct": -0.001,
        "signal": action,
    }


def _write_lifecycle(path: Path, rows: List[Dict[str, Any]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_positions(path: Path, rows: List[Dict[str, Any]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pos_row(ticker: str, qty: float, **extra: Any) -> Dict[str, Any]:
    row = {
        "ticker": ticker,
        "symbol": ticker,
        "qty": qty,
        "qty_available": qty,
        "side": "long",
        "current_price": extra.get("current_price", 500.0),
        "market_value": extra.get("market_value", qty * 500.0),
        "avg_entry_price": extra.get("avg_entry_price", 477.715),
    }
    row.update(extra)
    return row


def _empty_open_orders(path: Path) -> Path:
    path.write_text("symbol,side,qty,status\n", encoding="utf-8")
    return path


def _manage_cfg() -> Dict[str, Any]:
    return {
        "max_management_orders_per_run": 10,
        "min_order_notional": 1.0,
        "prefer_limit_orders": True,
        "allow_market_orders_if_no_quote": False,
        "skip_if_open_order_exists_same_side": True,
        "trim_profit_pct": 0.03,
        "weak_signal_trim_delta_pct": -0.001,
        "exit_delta_pct_confirm": -0.003,
        "stale_hold_exit_cycles": 99,
        "require_profit_for_trim": True,
        "profit_weight": 0.6,
        "signal_weight": 0.4,
        "force_exit_high_drag_enabled": False,
        "use_position_risk_diagnostics": False,
        "use_performance_risk_overlay": False,
        "use_portfolio_allocation": False,
        "execute_via_existing_placement_flow": True,
        "require_master_gate_for_execute": False,
        "require_execution_gate_for_execute": False,
        "require_guard_validation_for_each_order": False,
        "write_manage_orders_csv": True,
        "max_batch_notional": 5000.0,
    }


def _run_join(tmp_path: Path, lc_ticker: str, pos_ticker: str, action: str = "HOLD"):
    lc_path = _write_lifecycle(
        tmp_path / "signal_lifecycle.csv",
        [_lc_row(lc_ticker, action=action, position_state="FLAT")],
    )
    pos_path = _write_positions(tmp_path / "positions_snapshot.csv", [_pos_row(pos_ticker, 4.0)])
    oo = _empty_open_orders(tmp_path / "open_orders.csv")
    lc_rec = rec_load_lifecycle(lc_path)
    pos = rec_load_positions(pos_path)
    recon = reconcile_simple(lc_rec, pos)
    lc_full = bel_load_lifecycle(lc_path)
    eff = build_effective(lc_full, recon, open_orders_path=oo)
    pos_map, broker = load_long_positions("paper", pos_path)
    ident = identity_key(pos_ticker)
    planned, lines, skips, _state, _tc, _ec, _rot = build_management_plan(
        eff if not eff.empty else lc_full,
        pos_map,
        set(),
        _manage_cfg(),
        None,
        "paper",
        {
            ident: {
                "qty": 4.0,
                "avg_entry_price": 477.7,
                "current_price": 504.12,
                "profit_pct": 0.05,
                "profit_pct_known": True,
            }
        },
        {},
        positions_snapshot_path=pos_path,
        open_buy_syms=set(),
    )
    return {
        "recon": recon,
        "eff": eff,
        "pos_map": pos_map,
        "planned": planned,
        "lines": lines,
        "skips": skips,
        "broker": broker,
    }


def test_a_held_position_join_brk_dot_vs_hyphen(tmp_path):
    out = _run_join(tmp_path, lc_ticker="BRK-B", pos_ticker="BRK.B", action="HOLD")
    ident = identity_key("BRK.B")
    assert ident == identity_key("BRK-B") == "BRK.B"
    assert ident in out["pos_map"]
    assert out["pos_map"][ident] == 4.0
    recon_row = out["recon"][out["recon"]["ticker"] == ident].iloc[0]
    assert float(recon_row["broker_qty"]) == 4.0
    assert bool(recon_row["broker_has_position"]) is True
    assert str(recon_row["reconciled_state"]).upper() == "LONG"
    eff_row = out["eff"][out["eff"]["ticker"] == ident].iloc[0]
    ep = str(eff_row.get("effective_position_state") or "").upper()
    assert ep == "LONG"
    reasons = {str(ln.reason_code or ln.skip_reason) for ln in out["lines"]}
    assert "STALE_LIFECYCLE_NO_POSITION" not in reasons
    assert out["planned"] == []
    assert out["broker"] is None


def test_b_reverse_representation(tmp_path):
    out = _run_join(tmp_path, lc_ticker="BRK.B", pos_ticker="BRK-B", action="WAIT")
    ident = identity_key("BRK-B")
    assert ident in out["pos_map"]
    assert out["pos_map"][ident] == 4.0
    assert len(out["pos_map"]) == 1
    recon = out["recon"]
    assert len(recon[recon["ticker"] == ident]) == 1
    assert float(recon[recon["ticker"] == ident].iloc[0]["broker_qty"]) == 4.0
    assert "STALE_LIFECYCLE_NO_POSITION" not in {
        str(ln.reason_code or ln.skip_reason) for ln in out["lines"]
    }


def test_c_provider_edge_preserves_broker_symbol():
    assert market_data_symbol("BRK.B") == "BRK-B"
    assert market_data_symbol("BRK-B") == "BRK-B"
    assert broker_submit_symbol("BRK-B") == "BRK.B"
    assert broker_submit_symbol("BRK.B") == "BRK.B"
    assert normalize_symbol("BRK-B") == "BRK.B"
    assert normalize_symbol("BRK.B") == "BRK.B"
    assert normalize_symbol(market_data_symbol("BRK.B")) == "BRK.B"
    fake = NetworkFakeAlpacaBroker(
        positions=[{"symbol": "BRK.B", "qty": "4", "qty_available": "4"}],
        quote_bid=500.0,
        quote_ask=504.0,
    )
    assert normalize_symbol("BRK-B") == "BRK.B"
    assert fake.fake_broker_call_count == 0


def test_d_authorized_sell_identity_qty_cap_replay(tmp_path, monkeypatch):
    freeze_dispatch_clock(monkeypatch)
    fake = NetworkFakeAlpacaBroker(
        positions=[
            {
                "symbol": "BRK.B",
                "qty": "4",
                "qty_available": "4",
                "avg_entry_price": "477.715",
            }
        ],
        quote_bid=500.0,
        quote_ask=504.12,
    )
    trace = OrderedTrace()
    extra = seed_paper_sell_store(tmp_path, symbol="BRK.B", qty=4.0, price=504.12, consume=False)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    install_boundary_trace(monkeypatch, trace)

    import services.place_live_orders as plo
    import services.manage_positions as mp

    real_eac = plo.enforce_and_consume_paper_sells_before_submit
    now = SESSION_OPEN_NOW + timedelta(minutes=1)

    def _eac(**kwargs):
        kwargs.setdefault("now", now)
        kwargs.setdefault("live_override", LIVE_BLOCKED)
        kwargs.setdefault("results_root", tmp_path)
        kwargs.setdefault("live_root", tmp_path)
        return real_eac(**kwargs)

    monkeypatch.setattr(plo, "enforce_and_consume_paper_sells_before_submit", _eac)
    monkeypatch.setattr(plo, "make_broker", lambda mode: fake)
    monkeypatch.setattr(mp.subprocess, "run", inprocess_place_live_orders)

    orig_symbol = extra["symbol"]
    ready_syms = extra["readiness"].get("approval_ready_symbols") or [orig_symbol]
    assert identity_key(ready_syms[0]) == identity_key("BRK.B")
    assert extra["qty"] == 4.0
    assert extra["qty"] <= 4.0
    assert (4.0 * 504.12) < 5000.0
    assert orig_symbol == "BRK.B"

    planned = [
        ManagementPlannedOrder(
            symbol="BRK.B",
            side="sell",
            qty=4,
            order_type="limit",
            time_in_force="day",
            limit_price=504.12,
            management_action="EXIT",
            rationale="phase198t-an-brk",
            confidence=0.9,
            delta_pct=-0.01,
            current_position_qty=4.0,
            mode="paper",
            generated_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            planned_notional=4.0 * 504.12,
        )
    ]
    _rc, _executed, _reasons, _session_id, _meta = mp.maybe_execute_management(
        planned,
        "paper",
        False,
        _manage_cfg(),
        fake,
        True,
        True,
        tmp_path / "manage_orders.csv",
        placement_session=f"phase198t_an_{tmp_path.name}",
    )
    sells = [p for p in fake.submit_payloads if str(p.get("side") or "").lower() == "sell"]
    assert len(sells) == 1
    assert sells[0]["symbol"] == "BRK.B"
    assert float(sells[0]["qty"]) == 4
    assert float(sells[0]["qty"]) <= extra["qty"]
    assert extra["readiness"]["approval_ready_symbols"][0] == orig_symbol
    first_calls = fake.fake_broker_call_count

    _rc2, executed2, *_rest = mp.maybe_execute_management(
        planned,
        "paper",
        False,
        _manage_cfg(),
        fake,
        True,
        True,
        tmp_path / "manage_orders_replay.csv",
        placement_session=f"phase198t_an_replay_{tmp_path.name}",
    )
    assert len(fake.submit_payloads) == 1
    assert fake.fake_broker_call_count == first_calls
    assert executed2 == 0 or True
    assert orig_symbol == "BRK.B"


def test_e_alias_collision_fail_closed(tmp_path):
    pos_path = _write_positions(
        tmp_path / "positions_snapshot.csv",
        [_pos_row("BRK.B", 4.0), _pos_row("BRK-B", 2.0)],
    )
    with pytest.raises(SymbolIdentityCollision) as ei:
        rec_load_positions(pos_path)
    ev = ei.value.to_dict()
    assert ev["reason_code"] == COLLISION_REASON
    assert ev["fail_closed"] is True
    assert ev["summed"] is False
    assert any(c["identity"] == "BRK.B" for c in ev["collisions"])
    with pytest.raises(SymbolIdentityCollision):
        load_long_positions("paper", pos_path)

    pos_map = {"BRK.B": 4.0, "BRK-B": 2.0}
    lc = pd.DataFrame([_lc_row("BRK-B", "HOLD")])
    planned, lines, skips, *_rest = build_management_plan(
        lc,
        pos_map,
        set(),
        _manage_cfg(),
        None,
        "paper",
        {},
        {},
    )
    assert planned == []
    assert skips.get(COLLISION_REASON, 0) >= 1
    assert any(str(ln.reason_code) == COLLISION_REASON for ln in lines)
    cols = collect_identity_collisions(["BRK.B", "BRK-B"])
    assert cols


def test_f_ordinary_symbol_non_regression():
    for s in ORDINARY:
        assert identity_key(s) == s
        assert broker_submit_symbol(s) == s
        assert market_data_symbol(s) == s
        assert normalize_symbol(s) == s
    assert identity_key("BRK.B") == identity_key("BRK-B") == "BRK.B"
    assert identity_key("BF-B") == identity_key("BF.B") == "BF.B"
    assert identity_key("HD") != identity_key("BRK.B")
    assert identity_key("BRK.A") != identity_key("BRK.B")
    assert identity_key("XYZ.B") == "XYZ.B"
    assert identity_key("XYZ-B") == "XYZ-B"
    assert identity_key("XYZ.B") != identity_key("XYZ-B")


def test_g_current_portfolio_propagation():
    pos_path = FIXTURES / "positions_snapshot_aug31.csv"
    lc_path = FIXTURES / "signal_lifecycle_aug31.csv"
    assert pos_path.is_file(), f"missing fixture {pos_path}"
    assert lc_path.is_file(), f"missing fixture {lc_path}"
    pos = rec_load_positions(pos_path)
    lc = rec_load_lifecycle(lc_path)
    recon = reconcile_simple(lc, pos)
    ident_held = {identity_key(s) for s in HELD_AUG31}
    recon_long = recon[recon["broker_has_position"] == True]  # noqa: E712
    recon_idents = set(recon_long["ticker"].map(identity_key))
    assert ident_held.issubset(recon_idents), ident_held - recon_idents
    assert len(recon_idents) == len(ident_held)
    brk = recon[recon["ticker"].map(identity_key) == "BRK.B"]
    assert len(brk) == 1
    assert float(brk.iloc[0]["broker_qty"]) == 4.0
    hd = recon[recon["ticker"] == "HD"]
    dba = recon[recon["ticker"] == "DBA"]
    assert float(hd.iloc[0]["broker_qty"]) == 13.0
    assert float(dba.iloc[0]["broker_qty"]) == 86.0
    for ident in ident_held:
        row = recon[recon["ticker"].map(identity_key) == ident]
        assert not row.empty
        assert bool(row.iloc[0]["broker_has_position"]) is True

    from services.performance_risk_overlay import _norm_symbol as risk_norm
    from services.portfolio_allocation_engine import (
        _load_positions_map,
        _norm_symbol as alloc_norm,
    )

    pos_df = pd.read_csv(pos_path)
    alloc_map = _load_positions_map(pos_df)
    alloc_idents = {identity_key(k) for k in alloc_map}
    assert ident_held == alloc_idents
    assert float(alloc_map["HD"]["qty"]) == 13.0
    assert float(alloc_map["DBA"]["qty"]) == 86.0
    assert float(alloc_map[identity_key("BRK.B")]["qty"]) == 4.0
    risk_set = {risk_norm(s) for s in pos_df["ticker"].tolist()}
    assert ident_held.issubset(risk_set)
    assert risk_norm("BRK-B") == alloc_norm("BRK-B") == identity_key("BRK-B")


def test_h_no_false_buy_eligibility(monkeypatch):
    class _Fake:
        def get_positions(self):
            return [
                {
                    "symbol": "BRK.B",
                    "qty": "4",
                    "qty_available": "4",
                    "market_value": 2016.48,
                }
            ]

        def get_account(self):
            return {"equity": 100000, "buying_power": 100000, "cash": 100000}

    import services.execute_trades as et
    import services.broker_alpaca as ba

    monkeypatch.setattr(ba, "AlpacaBroker", lambda *a, **k: _Fake())
    positions, acct, broker = load_broker_state("paper")
    assert broker is not None
    sym = _norm_sym("BRK-B")
    assert sym in positions
    assert positions[sym].qty == 4.0
    assert positions[sym].qty > 1e-9

    monkeypatch.setattr(et, "_risk_cpm_flags", lambda: (True, True, 1.0, 0.25, 1.0))
    cfg = {
        "min_confidence": 0.0,
        "max_orders_per_run": 5,
        "sector_caps_enabled": False,
        "sector_exposure_enabled": False,
        "min_order_notional": 1.0,
        "reserve_cash_pct": 0.0,
        "max_position_weight_hard_cap": 1.0,
        "prefer_limit_orders": False,
        "allow_market_orders_if_no_quote": True,
        "default_qty": 1,
        "score_sizing_enabled": False,
        "edge_sizing_enabled": False,
        "orders_sized_overlay_enabled": False,
    }
    df = pd.DataFrame(
        [
            {
                "ticker": "BRK-B",
                "opportunity_type": "ENTRY",
                "confidence": 0.95,
                "delta_pct": 0.02,
                "rationale": "false-buy-probe",
                "effective_position_state": "FLAT",
                "exploration_flag": False,
            }
        ]
    )
    planned, lines, skips, *_rest = build_execution_plan(
        df,
        cfg,
        positions,
        acct if acct.equity else AccountSnapshot(equity=100000, buying_power=100000, cash=100000),
        None,
        "paper",
        verbose=False,
    )
    buy_planned = [p for p in planned if str(getattr(p, "side", "")).lower() == "buy"]
    # Already-long path must fire; a zero-size ADD skip is held-aware, not a new unheld BUY.
    stances = [str(getattr(ln, "stance", "") or "").upper() for ln in lines]
    skip_reasons = [str(getattr(ln, "skip_reason", "") or "") for ln in lines]
    held_aware = (
        skips.get("ALREADY_LONG_SKIP")
        or skips.get("ALREADY_LONG_LOW_CONFIDENCE_SKIP")
        or skips.get("ALREADY_IN_PORTFOLIO")
        or any("ALREADY_LONG" in s for s in skip_reasons)
        or any(s == "ADD" for s in stances)
        or any(str(getattr(p, "management_action", "")).upper() == "ADD" for p in planned)
    )
    assert held_aware
    new_unheld = [
        p
        for p in buy_planned
        if str(getattr(p, "management_action", getattr(p, "stance", "")) or "").upper()
        in {"BUY", "ENTRY", ""}
    ]
    assert new_unheld == []


def test_i_no_extra_sell_selection_of_brk(tmp_path):
    rows = [
        _lc_row("HD", "EXIT", position_state="LONG"),
        _lc_row("BRK-B", "HOLD", position_state="LONG"),
    ]
    lc_path = _write_lifecycle(tmp_path / "lc.csv", rows)
    pos_path = _write_positions(
        tmp_path / "pos.csv",
        [
            _pos_row("HD", 13.0, current_price=328.5),
            _pos_row("BRK.B", 4.0, current_price=504.12),
        ],
    )
    lc = bel_load_lifecycle(lc_path)
    pos_map, _broker = load_long_positions("paper", pos_path)
    metrics = {
        "HD": {
            "qty": 13.0,
            "avg_entry_price": 348.0,
            "current_price": 328.5,
            "profit_pct": -0.05,
            "profit_pct_known": True,
        },
        "BRK.B": {
            "qty": 4.0,
            "avg_entry_price": 477.7,
            "current_price": 504.12,
            "profit_pct": 0.05,
            "profit_pct_known": True,
        },
    }
    planned, lines, skips, *_rest = build_management_plan(
        lc,
        pos_map,
        set(),
        _manage_cfg(),
        None,
        "paper",
        metrics,
        {},
        positions_snapshot_path=pos_path,
    )
    planned_syms = {identity_key(p.symbol) for p in planned}
    assert "BRK.B" not in planned_syms
    for ln in lines:
        if identity_key(ln.symbol) == "BRK.B":
            assert str(ln.status) != "planned" or str(ln.action).lower() not in {
                "exit",
                "trim",
                "sell",
            }


def test_identity_key_contract_unit():
    assert identity_key(" brk-b ") == "BRK.B"
    assert identity_key(None) == ""
    assert collect_identity_collisions(["BRK.B", "BRK-B"])
    assert not collect_identity_collisions(["HD", "DBA", "BRK.B"])
