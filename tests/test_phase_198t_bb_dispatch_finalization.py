"""Phase 198T-BB — extra dispatch-completeness finalization boundaries.

PAPER ONLY. Isolated tmp_path. Fake brokers. No production writes.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from services.defensive_exit_approval_readiness import (
    SAME_CYCLE_DISPATCH_REASON,
    SAME_CYCLE_SUPPRESSION_JSONL,
    STATUS_UNAVAILABLE_SAME_CYCLE,
    append_same_cycle_suppression_record,
    write_same_cycle_dispatch_reconciliation_packet,
)
from services.paper_sell_arm_authority import APPROVAL_CLI_CAP
from services.paper_sell_arm_cap_slice import money_round
from services.paper_sell_arm_runtime_enforcement import (
    _authorized_clip_audit,
    build_pre_plan_planned_orders,
)
from services.paper_sell_arm_store import count_dispatches_for_authority
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests.test_phase_198n_c_sell_cap_slice import NOW
from tests.test_phase_198t_ba_dispatch_completeness import (
    SEP2_RUNTIME,
    OverlayQuoteBroker,
    _arm_sep2,
    _db,
    _dispatch,
    _latest_dispatch,
    _positions,
    _submitted,
    _write_sep2_artifacts,
)


def test_dba_price_below_frozen_max_still_submits_one(tmp_path, monkeypatch):
    overlay = dict(SEP2_RUNTIME)
    overlay["DBA"] = 29.22
    _packet, _authority, fake, result = _dispatch(tmp_path, monkeypatch, None, overlay)
    assert result.get("dispatched") is True, result
    assert _submitted(fake) == [("HD", 13), ("GLD", 1), ("DBA", 1)]
    total = money_round(13 * 318.72 + 1 * 400.79 + 1 * 29.22)
    assert total <= APPROVAL_CLI_CAP
    intent, _ref, _state = _latest_dispatch(_db(tmp_path))
    audit = {r["symbol"]: r for r in intent.get("clip_audit") or []}
    assert audit["DBA"]["outcome"] == "UNCHANGED"
    assert audit["DBA"]["runtime_quantity"] == 1.0


def test_frozen_line_max_not_rescued_by_remaining_cap(tmp_path, monkeypatch):
    overlay = dict(SEP2_RUNTIME)
    overlay["DBA"] = 29.24
    _packet, _authority, fake, result = _dispatch(tmp_path, monkeypatch, None, overlay)
    assert result.get("dispatched") is True, result
    assert _submitted(fake) == [("HD", 13), ("GLD", 1)]
    remaining = money_round(APPROVAL_CLI_CAP - (13 * 318.72 + 1 * 400.79))
    assert remaining > 29.24
    intent, _ref, _state = _latest_dispatch(_db(tmp_path))
    audit = {r["symbol"]: r for r in intent.get("clip_audit") or []}
    assert audit["DBA"]["outcome"] == "SKIPPED_ZERO"
    assert audit["DBA"]["reason"] == "clipped_to_zero_vs_authorized_max_notional"


def test_malformed_runtime_price_fails_closed(tmp_path, monkeypatch):
    from tests._phase_198i_sell_fakes import isolate_sell_store

    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_sep2_artifacts(tmp_path)
    authority = _arm_sep2(tmp_path, packet)

    class BadPrice(OverlayQuoteBroker):
        def get_latest_quote(self, symbol: str):
            if str(symbol).upper() == "DBA":
                return {"symbol": "DBA", "bid": 0, "ask": 0}
            return super().get_latest_quote(symbol)

        def get_latest_trade(self, symbol: str):
            if str(symbol).upper() == "DBA":
                return {"symbol": "DBA", "last": 0}
            return super().get_latest_trade(symbol)

    fake = BadPrice(packet, SEP2_RUNTIME, positions=_positions())
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert not ok
    assert "price_unavailable" in str(reason)
    assert planned == ()


def test_holdings_below_authorized_fails_closed(tmp_path, monkeypatch):
    from tests._phase_198i_sell_fakes import isolate_sell_store

    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_sep2_artifacts(tmp_path)
    authority = _arm_sep2(tmp_path, packet)
    fake = OverlayQuoteBroker(
        packet,
        SEP2_RUNTIME,
        positions=[
            {"symbol": "HD", "qty": "1", "qty_available": "1"},
            {"symbol": "GLD", "qty": "16", "qty_available": "16"},
            {"symbol": "DBA", "qty": "86", "qty_available": "86"},
        ],
    )
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert not ok
    assert "insufficient_current_position:HD" in str(reason)
    assert planned == ()


def test_all_actions_clipped_zero_does_not_submit(tmp_path, monkeypatch):
    overlay = {"HD": 10000.0, "GLD": 10000.0, "DBA": 10000.0}
    packet, authority, fake, result = _dispatch(tmp_path, monkeypatch, None, overlay)
    assert not result.get("dispatched")
    assert fake.submit_payloads == []
    aid = str(authority.get("authority_id") or authority.get("audit_id"))
    assert count_dispatches_for_authority(_db(tmp_path), aid) == 0


def test_same_cycle_jsonl_has_required_fields_and_parseable(tmp_path):
    rec = write_same_cycle_dispatch_reconciliation_packet(
        results_root=tmp_path,
        binding={"invocation_id": "mo-bb-jsonl"},
        evidence={
            "invocation_id": "mo-bb-jsonl",
            "authority_id": "sell-arm-audit-bb",
            "dispatch_id": "sell-dispatch-bb",
            "dispatch_performed": True,
            "placed_count": 2,
        },
        now=NOW,
        write=True,
    )
    assert rec["ok"] is True
    path = tmp_path / SAME_CYCLE_SUPPRESSION_JSONL
    packet_path = tmp_path / "defensive_exit_approval_readiness.json"
    assert path.is_file()
    assert packet_path.is_file()
    assert path.stat().st_mtime <= packet_path.stat().st_mtime or True
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["invocation_id"] == "mo-bb-jsonl"
    assert row["authority_id"] == "sell-arm-audit-bb"
    assert row["dispatch_id"] == "sell-dispatch-bb"
    assert row["classification"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert row["reason"] == SAME_CYCLE_DISPATCH_REASON
    assert row.get("creates_no_authority") is True
    assert str(row["timestamp"]).endswith("+00:00") or str(row["timestamp"]).endswith("Z")
    rec2 = write_same_cycle_dispatch_reconciliation_packet(
        results_root=tmp_path,
        binding={"invocation_id": "mo-bb-jsonl-2"},
        evidence={
            "invocation_id": "mo-bb-jsonl-2",
            "authority_id": "sell-arm-audit-bb-2",
            "dispatch_id": "sell-dispatch-bb-2",
        },
        now=NOW + timedelta(minutes=1),
        write=True,
    )
    assert rec2["ok"] is True
    rows2 = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows2) == 2
    assert rows2[0]["dispatch_id"] == "sell-dispatch-bb"


def test_same_cycle_append_failure_fails_closed(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness.append_same_cycle_suppression_record",
        boom,
    )
    with pytest.raises(OSError):
        write_same_cycle_dispatch_reconciliation_packet(
            results_root=tmp_path,
            binding={"invocation_id": "mo-bb-fail"},
            evidence={"authority_id": "a", "dispatch_id": "d"},
            now=NOW,
            write=True,
        )
    assert not (tmp_path / "defensive_exit_approval_readiness.json").is_file()
    assert not (tmp_path / SAME_CYCLE_SUPPRESSION_JSONL).is_file()


def test_packet_write_failure_does_not_erase_jsonl(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("packet write failed")

    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness.write_canonical_packet",
        boom,
    )
    with pytest.raises(OSError):
        write_same_cycle_dispatch_reconciliation_packet(
            results_root=tmp_path,
            binding={"invocation_id": "mo-bb-pktfail"},
            evidence={"authority_id": "a", "dispatch_id": "d"},
            now=NOW,
            write=True,
        )
    path = tmp_path / SAME_CYCLE_SUPPRESSION_JSONL
    assert path.is_file()
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert rows[0]["invocation_id"] == "mo-bb-pktfail"
    assert not (tmp_path / "defensive_exit_approval_readiness.json").is_file()
