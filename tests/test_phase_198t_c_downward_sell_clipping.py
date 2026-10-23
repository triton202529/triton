"""Phase 198T-C — downward-only paper SELL clipping production boundary.

PAPER ONLY. Isolated tmp_path SQLite and fake brokers. No production writes,
no real network, no TRITON start, no Streamlit.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from services.manage_positions_cycle_binding import selected_row_hash, sha256_file_bytes
from services.paper_sell_arm_authority import APPROVAL_CLI_CAP, readiness_packet_hash
from services.paper_sell_arm_cap_slice import money_round
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import (
    _ACTIVE_PERMIT,
    _authorized_clip_audit,
    _downward_runtime_qty,
    _price_unavailable,
    attempt_pre_plan_authorized_sell_dispatch,
    begin_paper_sell_dispatch,
    build_pre_plan_planned_orders,
    select_authorized_planned,
    validate_pre_plan_authority_bound_artifacts,
)
from services.paper_sell_arm_authority import (
    revalidate_before_dispatch,
    validate_runtime_clip_audit,
)
from services.paper_sell_arm_store import inspect_sqlite_arm_snapshot
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    block_real_network,
    freeze_dispatch_clock,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198n_c_sell_cap_slice import (
    NOW,
    FakePlanned,
    PacketQuotedFakeBroker,
    _broker,
    _db,
    _pres,
    _write_buy_arm,
    _write_market_session_evidence,
    _write_packet,
)

import pytest

AUG24_ACTIONS = [
    {
        "symbol": "TSLA",
        "action": "EXIT",
        "qty": 8.0,
        "est_price": 353.87,
        "est_notional": 2830.96,
    },
    {
        "symbol": "BLK",
        "action": "EXIT",
        "qty": 1.0,
        "est_price": 1168.50,
        "est_notional": 1168.50,
    },
    {
        "symbol": "HD",
        "action": "EXIT",
        "qty": 2.0,
        "est_price": 339.30,
        "est_notional": 678.60,
    },
    {
        "symbol": "ARKK",
        "action": "EXIT",
        "qty": 3.0,
        "est_price": 84.25,
        "est_notional": 252.75,
    },
    {
        "symbol": "BITO",
        "action": "EXIT",
        "qty": 6.0,
        "est_price": 10.67,
        "est_notional": 64.02,
    },
]
AUG24_TOTAL = 4994.83
CEILING_PRICES = {
    "TSLA": 353.87,
    "BLK": 1168.50,
    "HD": 339.30,
    "ARKK": 84.25,
    "BITO": 10.67,
}


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    stub_dir_fsync_ok(monkeypatch)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def _aug24_packet() -> Dict[str, Any]:
    return {
        "schema_version": "198I-ZA-SELL-READINESS-1",
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "generation_status": "ok",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "generated_at": NOW.isoformat(),
        "cycle_id": "mo-198t-c-aug24",
        "run_id": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_id": "manage-mo-198t-c",
        "plan_id": f"manage_positions_plan@{NOW.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "plan_hash": "planhash198tc000000000000000000000000000000000000000000000000000001",
        "approval_ready_symbols": [a["symbol"] for a in AUG24_ACTIONS],
        "approval_ready_actions": copy.deepcopy(AUG24_ACTIONS),
        "total_proposed_notional": AUG24_TOTAL,
        "market_open_invocation_id": "mo-198t-c-aug24",
        "invocation_id": "mo-198t-c-aug24",
        "source_plan_sha256": "",
        "selected_row_hash": "",
    }


def _write_aug24_artifacts(tmp_path: Path) -> Dict[str, Any]:
    packet = _aug24_packet()
    rows = [
        {
            "symbol": a["symbol"],
            "final_action": a["action"],
            "qty": str(int(a["qty"])),
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        }
        for a in AUG24_ACTIONS
    ]
    csv_path = tmp_path / "manage_positions_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    packet["source_plan_sha256"] = sha256_file_bytes(csv_path)
    packet["selected_row_hash"] = selected_row_hash(rows)
    packet["plan_hash"] = readiness_packet_hash(packet)
    _write_packet(tmp_path, packet)
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"plan_hash": packet["plan_hash"], "timestamp": NOW.isoformat()}),
        encoding="utf-8",
    )
    pos_lines = ["symbol,qty,qty_available,current_price"]
    for a in AUG24_ACTIONS:
        held = 33 if a["symbol"] == "ARKK" else int(a["qty"])
        pos_lines.append(f"{a['symbol']},{held},{held},{a['est_price']}")
    (tmp_path / "positions_snapshot.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=1))
    return packet


def _arm_aug24(tmp_path: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
    _write_buy_arm(tmp_path, now=NOW)
    pres = _pres(tmp_path, packet)
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert armed["ok"] is True, armed
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert confirmed["ok"] is True, confirmed
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    return snap["latest_authority"]["payload"]


class OverlayQuoteBroker(PacketQuotedFakeBroker):
    def __init__(self, packet: Dict[str, Any], overlay: Dict[str, Any], **kwargs: Any):
        super().__init__(packet, **kwargs)
        self._overlay = {str(k).upper(): v for k, v in overlay.items()}

    def _px(self, symbol: str) -> float:
        key = str(symbol).upper()
        if key in self._overlay:
            return float(self._overlay[key])
        return super()._px(symbol)


def _positions() -> List[Dict[str, str]]:
    out = []
    for a in AUG24_ACTIONS:
        held = "33" if a["symbol"] == "ARKK" else str(int(a["qty"]))
        out.append({"symbol": a["symbol"], "qty": held, "qty_available": held})
    return out


def test_01_arkk_exact_boundary_unchanged(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    fake = OverlayQuoteBroker(packet, CEILING_PRICES, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    arkk = next(p for p in planned if p.symbol == "ARKK")
    assert int(arkk.qty) == 3
    assert money_round(3 * 84.25) == 252.75
    audit = {r["symbol"]: r for r in _authorized_clip_audit(authority, planned)}
    assert audit["ARKK"]["outcome"] == "UNCHANGED"


def test_02_arkk_one_cent_clips_and_others_continue(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.26
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    by_sym = {p.symbol: int(p.qty) for p in planned}
    assert by_sym == {"TSLA": 8, "BLK": 1, "HD": 2, "ARKK": 2, "BITO": 6}
    arkk = next(p for p in planned if p.symbol == "ARKK")
    assert money_round(float(arkk.qty) * 84.26) == 168.52
    audit = {r["symbol"]: r for r in _authorized_clip_audit(authority, planned)}
    assert audit["ARKK"]["outcome"] == "CLIPPED_DOWN"
    assert audit["ARKK"]["deferred_quantity"] == 1.0
    json.dumps(list(audit.values()))


def test_03_downward_price_does_not_expand_qty(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.24
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    arkk = next(p for p in planned if p.symbol == "ARKK")
    assert int(arkk.qty) == 3


def test_04_arkk_above_line_max_retains_one_share(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 252.76
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    by_sym = {p.symbol: int(p.qty) for p in planned if int(p.qty) > 0}
    assert by_sym == {"TSLA": 8, "BLK": 1, "HD": 2, "ARKK": 1, "BITO": 6}
    arkk = next(p for p in planned if p.symbol == "ARKK")
    assert int(arkk.qty) == 1
    assert int(arkk.qty) <= 3
    audit = {r["symbol"]: r for r in _authorized_clip_audit(authority, planned)}
    assert audit["ARKK"]["outcome"] == "CLIPPED_DOWN"
    assert audit["ARKK"]["runtime_quantity"] == 1.0
    assert audit["ARKK"]["deferred_quantity"] == 2.0
    assert audit["ARKK"]["reason"] == "authorized_whole_share_retained_above_stale_notional"


def test_05_every_line_above_stale_notional_retains_one_share(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_aug24_artifacts(tmp_path)
    authority = copy.deepcopy(_arm_aug24(tmp_path, packet))
    overlay = {a["symbol"]: float(a["est_notional"]) + 0.01 for a in AUG24_ACTIONS}
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert result.get("dispatched"), result
    submitted = [
        (str(p.get("symbol") or "").upper(), int(float(p.get("qty") or 0)))
        for p in fake.submit_payloads
    ]
    assert submitted == [("TSLA", 1), ("BLK", 1), ("HD", 1), ("ARKK", 1), ("BITO", 1)]
    for symbol, qty in submitted:
        auth_qty = next(int(a["qty"]) for a in AUG24_ACTIONS if a["symbol"] == symbol)
        assert 1 <= qty <= auth_qty


def test_06_multiple_clips_preserve_source_order(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = {"TSLA": 400.0, "BLK": 1168.50, "HD": 400.0, "ARKK": 84.26, "BITO": 10.67}
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    got = [(p.symbol, int(p.qty)) for p in planned]
    assert [g[0] for g in got] == ["TSLA", "BLK", "HD", "ARKK", "BITO"]
    assert got[0][1] < 8 and got[2][1] < 2 and got[3][1] == 2
    for p in planned:
        item = next(a for a in authority["authorized_actions"] if a["symbol"] == p.symbol)
        assert int(p.qty) <= int(item["max_qty"])
        assert (
            money_round(float(p.qty) * float(p.ref_price))
            <= money_round(item["max_notional"]) + 1e-9
        )


def test_07_invalid_prices_fail_closed(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)

    class BadPriceBroker(PacketQuotedFakeBroker):
        def __init__(self, packet, bad, **kwargs):
            super().__init__(packet, **kwargs)
            self._bad = bad

        def get_latest_quote(self, symbol: str):
            if str(symbol).upper() == "ARKK":
                if self._bad == "missing":
                    return {}
                if self._bad == "malformed":
                    return {"bid": "not-a-price", "ask": "not-a-price"}
                return {"bid": self._bad, "ask": self._bad}
            return super().get_latest_quote(symbol)

        def get_latest_trade(self, symbol: str):
            if str(symbol).upper() == "ARKK":
                if self._bad in {"missing", "malformed"}:
                    return {}
                return {"last": self._bad}
            return super().get_latest_trade(symbol)

    for bad in ["missing", 0, -1, float("nan"), float("inf"), "malformed"]:
        fake = BadPriceBroker(packet, bad, positions=_positions())
        ok, reason, planned = build_pre_plan_planned_orders(
            authority, fake, now=NOW + timedelta(minutes=1)
        )
        assert not ok, f"expected fail-closed for {bad!r} got {reason}"
        assert reason.startswith("price_unavailable")
        assert planned == ()
        assert fake.submit_payloads == []
        snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
        assert snap["latest_authority"]["state"] == "UNUSED"


def test_08_position_insufficiency_unchanged(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    positions = _positions()
    for row in positions:
        if row["symbol"] == "ARKK":
            row["qty"] = "2"
            row["qty_available"] = "2"
    fake = OverlayQuoteBroker(packet, CEILING_PRICES, positions=positions)
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert not ok
    assert reason == "insufficient_current_position:ARKK"
    assert planned == ()


def test_09_exact_action_set_and_unrelated_omission(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    assert len(authority.get("authorized_actions") or []) == 5
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    bad = json.loads(
        (tmp_path / "defensive_exit_approval_readiness.json").read_text(encoding="utf-8")
    )
    bad["approval_ready_actions"] = [
        a for a in bad["approval_ready_actions"] if a["symbol"] != "HD"
    ]
    _write_packet(tmp_path, bad)
    ok_bad, _reason_bad, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok_bad


def test_10_11_12_cap_and_qty_properties():
    auth_qty = 3
    max_n = 252.75
    for cents in list(range(1, 30000, 41)) + [8424, 8425, 8426, 25275, 25276]:
        px = cents / 100.0
        qty, _reason = _downward_runtime_qty(auth_qty, max_n, px)
        assert qty <= auth_qty
        if qty > 0:
            assert money_round(qty * px) <= max_n + 1e-9
    qty, _reason = _downward_runtime_qty(3, 252.75, 84.26)
    derived = money_round(8 * 353.87 + 1 * 1168.50 + 2 * 339.30 + qty * 84.26 + 6 * 10.67)
    assert derived <= AUG24_TOTAL <= APPROVAL_CLI_CAP


def test_13_source_immutability(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    before = copy.deepcopy(authority)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.26
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    build_pre_plan_planned_orders(authority, fake, now=NOW + timedelta(minutes=1))
    assert authority == before


def test_14_single_use_and_residual_not_reusable(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.26
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    first = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert first.get("consumed"), first
    assert first.get("dispatched"), first
    audit = {r["symbol"]: r for r in (first.get("details") or {}).get("clip_audit") or []}
    if not audit:
        ok_p, _reason, planned = build_pre_plan_planned_orders(
            authority,
            OverlayQuoteBroker(packet, overlay, positions=_positions()),
            now=NOW + timedelta(minutes=1),
        )
        assert ok_p
        audit = {r["symbol"]: r for r in _authorized_clip_audit(authority, planned)}
    assert audit["ARKK"]["deferred_quantity"] == 1.0
    assert len(fake.submit_payloads) == 5
    got = [
        (str(p.get("symbol") or "").upper(), int(float(p.get("qty") or 0)))
        for p in fake.submit_payloads
    ]
    assert ("ARKK", 2) in got
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["latest_authority"]["state"] == "CONSUMED"
    second_fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    second = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=second_fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=2),
    )
    assert not second.get("dispatched")
    assert second_fake.submit_payloads == []


def test_15_packet_hash_still_fail_closed(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    bad = json.loads(
        (tmp_path / "defensive_exit_approval_readiness.json").read_text(encoding="utf-8")
    )
    bad["approval_ready_actions"][0]["qty"] = 1
    _write_packet(tmp_path, bad)
    ok, _reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok


def test_16_august24_replay_fake_dispatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.26
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    got = [(p.symbol, int(p.qty)) for p in planned]
    assert got == [("TSLA", 8), ("BLK", 1), ("HD", 2), ("ARKK", 2), ("BITO", 6)]
    total = money_round(sum(float(p.qty) * float(p.ref_price) for p in planned))
    assert total == 4910.60
    assert total <= AUG24_TOTAL <= APPROVAL_CLI_CAP
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert result.get("consumed"), result
    assert result.get("dispatched"), result
    assert len(fake.submit_payloads) == 5
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["latest_authority"]["state"] == "CONSUMED"
    assert snap["dispatch_count"] == 1


def test_helpers_invalid_price_and_no_float_compare():
    assert _price_unavailable(None)
    assert _price_unavailable(0)
    assert _price_unavailable(-1)
    assert _price_unavailable(float("nan"))
    assert _price_unavailable(float("inf"))
    assert _price_unavailable("nope")
    assert not _price_unavailable(84.26)
    qty, reason = _downward_runtime_qty(3, 252.75, 84.26)
    assert qty == 2
    assert "clip" in reason or "downward" in reason
    assert math.isfinite(qty)


def _planned_from_auth(
    authority: Dict[str, Any],
    prices: Dict[str, float],
    *,
    omit=None,
    qty_overrides=None,
    action_overrides=None,
    extra=None,
) -> List[FakePlanned]:
    omit_set = {str(s).upper() for s in (omit or [])}
    qty_overrides = qty_overrides or {}
    action_overrides = action_overrides or {}
    rows: List[FakePlanned] = []
    for item in authority.get("authorized_actions") or []:
        sym = str(item.get("symbol") or "").upper()
        if sym in omit_set:
            continue
        px = float(prices[sym])
        qty = float(qty_overrides.get(sym, item.get("max_qty") or 0))
        action = str(action_overrides.get(sym, item.get("action") or "EXIT"))
        rows.append(
            FakePlanned(
                sym,
                "sell",
                qty,
                px,
                stance=action,
                ref_price=px,
                client_order_id=f"coid-{sym}",
            )
        )
    if extra:
        rows.extend(extra)
    return rows


def test_case_a_legitimate_zero_derived_omission(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 252.76
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    sel_ok, sel_reason, selected = select_authorized_planned(planned, authority)
    assert sel_ok, sel_reason
    assert [p.symbol for p in selected] == ["TSLA", "BLK", "HD", "ARKK", "BITO"]
    assert int(next(p.qty for p in selected if p.symbol == "ARKK")) == 1
    audit = {r["symbol"]: r for r in _authorized_clip_audit(authority, planned)}
    assert audit["ARKK"]["outcome"] == "CLIPPED_DOWN"
    assert audit["ARKK"]["runtime_quantity"] == 1
    assert audit["ARKK"]["deferred_quantity"] == 2.0
    assert audit["ARKK"]["current_reference_price"] == 252.76
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert result.get("consumed"), result
    assert [str(p.get("symbol") or "").upper() for p in fake.submit_payloads] == [
        "TSLA",
        "BLK",
        "HD",
        "ARKK",
        "BITO",
    ]
    assert ("ARKK", 1) in [
        (str(p.get("symbol") or "").upper(), int(float(p.get("qty") or 0)))
        for p in fake.submit_payloads
    ]


def test_case_b_unauthorized_positive_omission_rejected(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    planned = _planned_from_auth(authority, CEILING_PRICES, omit=["ARKK"])
    ok, reason, selected = select_authorized_planned(planned, authority)
    assert ok is False
    assert reason == "incomplete_authorized_qty:ARKK"
    assert selected == ()
    fake = OverlayQuoteBroker(packet, CEILING_PRICES, positions=_positions())
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    from services.paper_sell_arm_runtime_enforcement import begin_paper_sell_dispatch
    from tests.test_phase_198n_c_sell_cap_slice import _broker

    positions = {
        a["symbol"]: {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    positions["ARKK"] = {"qty": 33.0, "qty_available": 33.0}
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(authority, positions),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=authority.get("readiness_packet_hash"),
        explicit_runtime_write=True,
    )
    assert not guard.consumed
    assert guard.reason == "incomplete_authorized_qty:ARKK"
    assert fake.submit_payloads == []
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["dispatch_count"] == 0


def test_case_c_clipped_nonzero_omission_rejected(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    prices = dict(CEILING_PRICES)
    prices["ARKK"] = 84.26
    planned = _planned_from_auth(authority, prices, omit=["ARKK"])
    ok, reason, selected = select_authorized_planned(planned, authority)
    assert ok is False
    assert reason == "incomplete_authorized_qty:ARKK"
    assert selected == ()
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    from services.paper_sell_arm_runtime_enforcement import begin_paper_sell_dispatch
    from tests.test_phase_198n_c_sell_cap_slice import _broker

    positions = {
        a["symbol"]: {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    positions["ARKK"] = {"qty": 33.0, "qty_available": 33.0}
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(authority, positions),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=authority.get("readiness_packet_hash"),
        explicit_runtime_write=True,
    )
    assert not guard.consumed
    assert guard.reason == "incomplete_authorized_qty:ARKK"
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["dispatch_count"] == 0


def test_case_d_symbol_substitution_rejected(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    planned = _planned_from_auth(
        authority,
        CEILING_PRICES,
        omit=["ARKK"],
        extra=[FakePlanned("MSFT", "sell", 3, 84.25, stance="EXIT", ref_price=84.25)],
    )
    ok, reason, selected = select_authorized_planned(planned, authority)
    assert ok is False
    assert reason == "incomplete_authorized_qty:ARKK"
    assert selected == ()


def test_case_e_action_substitution_rejected(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    planned = _planned_from_auth(authority, CEILING_PRICES)
    for row in planned:
        if row.symbol == "ARKK":
            row.side = "buy"
            row.stance = "BUY"
    ok, reason, selected = select_authorized_planned(planned, authority)
    assert ok is False
    assert reason in {
        "incomplete_authorized_qty:ARKK",
        "action_mismatch_vs_authorized_slice:ARKK",
    }
    assert selected == ()


def test_case_f_extra_symbol_rejected(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    planned = _planned_from_auth(
        authority,
        CEILING_PRICES,
        extra=[FakePlanned("QQQ", "sell", 1, 10.0, stance="EXIT", ref_price=10.0)],
    )
    ok, reason, selected = select_authorized_planned(
        planned, authority, reject_unapproved_planned=True
    )
    assert ok is False
    assert reason == "unauthorized_symbol_vs_authorized_slice:QQQ"
    assert selected == ()


def test_case_g_quantity_above_derived_is_clipped_not_expanded(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    prices = dict(CEILING_PRICES)
    prices["ARKK"] = 84.26
    planned = _planned_from_auth(authority, prices, qty_overrides={"ARKK": 3})
    ok, reason, selected = select_authorized_planned(planned, authority)
    assert ok, reason
    by_sym = {p.symbol: int(p.qty) for p in selected}
    assert by_sym["ARKK"] == 2
    assert by_sym["ARKK"] < 3
    assert money_round(2 * 84.26) == 168.52


def _pos_map(packet: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out = {}
    for a in packet["approval_ready_actions"]:
        held = 33.0 if a["symbol"] == "ARKK" else float(a["qty"])
        out[str(a["symbol"])] = {"qty": held, "qty_available": held}
    return out


def _req_from_planned(planned) -> List[Dict[str, Any]]:
    rows = []
    for p in planned:
        qty = int(p.qty)
        if qty <= 0:
            continue
        price = float(getattr(p, "limit_price", None) or getattr(p, "ref_price", None) or 0)
        action = str(getattr(p, "stance", None) or getattr(p, "management_action", None) or "EXIT")
        rows.append(
            {
                "symbol": p.symbol,
                "action": action,
                "qty": qty,
                "price": price,
                "notional": money_round(qty * price),
            }
        )
    return rows


def test_d3_legacy_exact_set_still_passes_without_clip_audit(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    planned = _planned_from_auth(authority, CEILING_PRICES)
    req = _req_from_planned(planned)
    ok, reason, _ = revalidate_before_dispatch(
        authority,
        broker=_broker(authority, _pos_map(packet)),
        requested_actions=req,
        readiness_packet_hash_current=authority.get("readiness_packet_hash"),
        now=NOW + timedelta(minutes=1),
    )
    assert ok, reason


def test_d3_legacy_subset_without_clip_audit_fails(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    planned = _planned_from_auth(authority, CEILING_PRICES, omit=["ARKK"])
    req = _req_from_planned(planned)
    ok, reason, _ = revalidate_before_dispatch(
        authority,
        broker=_broker(authority, _pos_map(packet)),
        requested_actions=req,
        readiness_packet_hash_current=authority.get("readiness_packet_hash"),
        now=NOW + timedelta(minutes=1),
    )
    assert not ok
    assert reason == "batch_symbol_set_mismatch_whole_batch_required"


def test_d3_legacy_zero_qty_without_clip_audit_fails(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    req = _req_from_planned(_planned_from_auth(authority, CEILING_PRICES))
    req[3]["qty"] = 0
    ok, reason, _ = revalidate_before_dispatch(
        authority,
        broker=_broker(authority, _pos_map(packet)),
        requested_actions=req,
        readiness_packet_hash_current=authority.get("readiness_packet_hash"),
        now=NOW + timedelta(minutes=1),
    )
    assert not ok
    assert "non_positive_qty" in reason


def test_d3_legacy_extra_symbol_without_clip_audit_fails(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    req = _req_from_planned(_planned_from_auth(authority, CEILING_PRICES))
    req.append({"symbol": "QQQ", "action": "EXIT", "qty": 1, "price": 10.0})
    ok, reason, _ = revalidate_before_dispatch(
        authority,
        broker=_broker(authority, _pos_map(packet)),
        requested_actions=req,
        readiness_packet_hash_current=authority.get("readiness_packet_hash"),
        now=NOW + timedelta(minutes=1),
    )
    assert not ok
    assert reason == "batch_symbol_set_mismatch_whole_batch_required"


def test_d3_forged_clip_audit_fields_rejected(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.26
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason
    audit = _authorized_clip_audit(authority, planned)
    prices = {p.symbol: float(p.ref_price) for p in planned}
    req = _req_from_planned(planned)

    def _check(mutated, expect_substr):
        ok_c, reason_c, _ = validate_runtime_clip_audit(authority, mutated, req, prices)
        assert not ok_c, reason_c
        assert expect_substr in reason_c

    forged_price = copy.deepcopy(audit)
    forged_price[3]["current_reference_price"] = 1.0
    _check(forged_price, "forged_current_reference_price")
    wrong_qty = copy.deepcopy(audit)
    wrong_qty[3]["runtime_quantity"] = 3
    _check(wrong_qty, "runtime_quantity")
    wrong_def = copy.deepcopy(audit)
    wrong_def[3]["deferred_quantity"] = 0
    _check(wrong_def, "deferred")
    wrong_n = copy.deepcopy(audit)
    wrong_n[3]["runtime_estimated_notional"] = 999.0
    _check(wrong_n, "runtime_notional")
    wrong_out = copy.deepcopy(audit)
    wrong_out[3]["outcome"] = "UNCHANGED"
    _check(wrong_out, "outcome")
    missing = copy.deepcopy(audit)[:4]
    ok_m, reason_m, _ = validate_runtime_clip_audit(authority, missing, req, prices)
    assert not ok_m and "row_count" in reason_m
    dup = copy.deepcopy(audit)
    dup.append(copy.deepcopy(audit[0]))
    ok_d, reason_d, _ = validate_runtime_clip_audit(authority, dup, req, prices)
    assert not ok_d and ("duplicate" in reason_d or "row_count" in reason_d)
    extra = copy.deepcopy(audit)
    extra[3] = dict(extra[3], symbol="QQQ")
    ok_e, reason_e, _ = validate_runtime_clip_audit(authority, extra, req, prices)
    assert not ok_e
    auth_qty = copy.deepcopy(audit)
    auth_qty[3]["authorized_quantity"] = 99
    _check(auth_qty, "authorized_quantity")
    max_n = copy.deepcopy(audit)
    max_n[3]["authorized_max_notional"] = 1.0
    _check(max_n, "max_notional")
    act = copy.deepcopy(audit)
    act[3]["action"] = "TRIM"
    _check(act, "action")
    for bad_px in [0, -1, float("nan"), float("inf"), "nope"]:
        bad = copy.deepcopy(audit)
        bad[3]["current_reference_price"] = bad_px
        ok_b, reason_b, _ = validate_runtime_clip_audit(authority, bad, req, prices)
        assert not ok_b
        assert "price" in reason_b or "forged" in reason_b


def test_d3_extra_and_duplicate_broker_requests_fail(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_aug24_artifacts(tmp_path)
    authority = _arm_aug24(tmp_path, packet)
    overlay = dict(CEILING_PRICES)
    overlay["ARKK"] = 84.26
    fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    audit = _authorized_clip_audit(authority, planned)
    prices = {p.symbol: float(p.ref_price) for p in planned}
    req = _req_from_planned(planned)
    extra = list(req) + [{"symbol": "QQQ", "action": "EXIT", "qty": 1, "price": 10.0}]
    ok_e, reason_e, _ = validate_runtime_clip_audit(authority, audit, extra, prices)
    assert not ok_e
    assert "unauthorized_symbol" in reason_e or "extra" in reason_e
    dup = list(req) + [dict(req[0])]
    ok_d, reason_d, _ = validate_runtime_clip_audit(authority, audit, dup, prices)
    assert not ok_d
    assert "duplicate" in reason_d
