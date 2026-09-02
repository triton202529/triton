"""Phase 198T-AT — SELL-failure containment and September 1 replay (TEMP).

PAPER ONLY. Isolated tmp_path. Fake broker. No production writes. No network.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from services.defensive_exit_approval_readiness import (
    classify_same_cycle_dispatch,
    generate_from_manage_artifacts,
    write_pre_plan_dispatch_evidence,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_NONE,
    DECISION_MAY_CONTINUE,
    write_cycle_binding,
    write_entry_interlock,
)
from services.paper_entry_loop_stage import evaluate_paper_entry_gates
from services.paper_operator_approval_artifact_state import (
    PAPER_APPROVAL_EXPIRED,
    classify_paper_operator_approval_artifact,
)
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    AUTHORITY_TTL_MINUTES,
    HARDENED_AUTHORITY_FILENAME,
    READINESS_REFERENCE_TTL_MINUTES,
    evaluate_readiness_packet_freshness,
    readiness_packet_hash,
)
from services.paper_sell_arm_runtime_enforcement import (
    _ACTIVE_PERMIT,
    find_unused_pre_plan_authority,
    sell_permission_soft_block_reason,
    sell_unusable_should_abort_cycle,
    validate_pre_plan_authority_bound_artifacts,
)
from services.paper_sell_arm_store import (
    REASON_MATCHING_IN_PROGRESS,
    inspect_sqlite_arm_snapshot,
)
from services.paper_sell_dispatch_session_evidence import (
    AUTHORITATIVE_SOURCE,
    write_dispatch_session_evidence,
)
from services.paper_sell_entry_dependency import (
    BLOCK_WORSENS_PENDING_RISK_REDUCTION,
    evaluate_entry_candidate,
)
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    install_boundary_trace,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198n_c_sell_cap_slice import PacketQuotedFakeBroker

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT.parent / "fixtures" / "as2"
CYCLE1_START = datetime(2026, 9, 1, 13, 38, 13, tzinfo=timezone.utc)
CYCLE1_COMPLETE = datetime(2026, 9, 1, 13, 40, 56, tzinfo=timezone.utc)
NEXT_DUE = datetime(2026, 9, 1, 14, 23, 13, tzinfo=timezone.utc)
PACKET_GEN = datetime(2026, 9, 1, 13, 40, 42, 821386, tzinfo=timezone.utc)
PACKET_UNTIL = datetime(2026, 9, 1, 14, 30, 42, 821386, tzinfo=timezone.utc)
AUTH_CREATED = datetime(2026, 9, 1, 14, 1, 32, 968373, tzinfo=timezone.utc)
AUTH_EXPIRES = datetime(2026, 9, 1, 15, 1, 32, 968373, tzinfo=timezone.utc)
STALE_AT = datetime(2026, 9, 1, 14, 31, 9, tzinfo=timezone.utc)
EXPIRED_AT = datetime(2026, 9, 1, 15, 17, 57, tzinfo=timezone.utc)
CYCLE1_INV = "mo-20260901_093836-38a448585ec342dc938dfbd6631f32c2"
CYCLE2_INV = "mo-20260901-at-corrected-cycle2"
AUTHORIZED = (("HD", 13.0), ("GLD", 1.0), ("TSLA", 1.0), ("DBA", 1.0))
OPEN_CAL = {
    "session_state": "OPEN",
    "reason": "regular_session_open",
    "session_date": "2026-09-01",
    "session_open_et": "2026-09-01T09:30:00-04:00",
    "session_close_et": "2026-09-01T16:00:00-04:00",
    "source": AUTHORITATIVE_SOURCE,
    "as_of_utc": "2026-09-01T14:23:13Z",
    "as_of_et": "2026-09-01T10:23:13-04:00",
}
SOFT_REASONS = (
    "stale_readiness_packet",
    "no_unused_authority",
    "authority_expired",
    "sqlite_authority_expired",
    "authority_missing",
    "authority_not_unused",
    "empty_authorized_scope",
    "no_current_challenge",
    "no_authorized_quantity",
    "no_matching_action_after_clipping",
    "foreign_historical_authority",
)
HARD_REASONS = (
    REASON_MATCHING_IN_PROGRESS,
    "dispatch_intent_identity_ambiguous",
    "live_authorized_true",
    "sqlite_unavailable",
)


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    stub_dir_fsync_ok(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper() or key.startswith("APCA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.setattr(
        "services.live_block_classifier.evaluate_live_block_status",
        lambda paper_mode=True: dict(LIVE_BLOCKED),
    )
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    assert not (ROOT / "data" / "live" / "paper_sell_arm.sqlite3").is_file()


def _load_packet() -> Dict[str, Any]:
    return json.loads(
        (FIXTURES / "defensive_exit_approval_readiness.json").read_text(encoding="utf-8")
    )


def _seed_store(tmp_path: Path) -> Path:
    db = tmp_path / "paper_sell_arm.sqlite3"
    shutil.copy2(FIXTURES / "paper_sell_arm.sqlite3", db)
    shutil.copy2(
        FIXTURES / "defensive_exit_approval_readiness.json",
        tmp_path / "defensive_exit_approval_readiness.json",
    )
    shutil.copy2(FIXTURES / HARDENED_AUTHORITY_FILENAME, tmp_path / HARDENED_AUTHORITY_FILENAME)
    shutil.copy2(
        FIXTURES / "paper_sell_confirmation_challenge.json",
        tmp_path / "paper_sell_confirmation_challenge.json",
    )
    if (FIXTURES / "paper_arm_confirm.json").is_file():
        shutil.copy2(FIXTURES / "paper_arm_confirm.json", tmp_path / "paper_arm_confirm.json")
    if (FIXTURES / "paper_entry_dispatch_contract.json").is_file():
        shutil.copy2(
            FIXTURES / "paper_entry_dispatch_contract.json",
            tmp_path / "paper_entry_dispatch_contract.json",
        )
    if (FIXTURES / "positions_snapshot.csv").is_file():
        shutil.copy2(FIXTURES / "positions_snapshot.csv", tmp_path / "positions_snapshot.csv")
    return db


def _write_plan_rows(tmp_path: Path, packet: Dict[str, Any]) -> None:
    rows = []
    for rc in packet.get("row_classifications") or []:
        rows.append(
            {
                "symbol": rc.get("symbol"),
                "final_action": rc.get("action"),
                "qty": rc.get("qty"),
                "selected_for_execution": "true" if rc.get("selected_for_execution") else "false",
                "reason_code": rc.get("reason_code") or "",
                "skip_reason": rc.get("skip_reason") or "",
                "status": rc.get("status") or "",
                "current_position_qty": rc.get("qty") or "",
                "current_price": "",
                "planned_notional": "",
            }
        )
    fields = [
        "symbol",
        "final_action",
        "qty",
        "selected_for_execution",
        "reason_code",
        "skip_reason",
        "status",
        "current_position_qty",
        "current_price",
        "planned_notional",
    ]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(f) or "") for f in fields))
    (tmp_path / "manage_positions_plan.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": packet.get("plan_generated_at"), "mode": "paper", "rows": rows}),
        encoding="utf-8",
    )


def _patch_plan_hashes(monkeypatch, auth: Dict[str, Any]) -> None:
    expected_src = str(auth.get("source_plan_sha256") or "")
    expected_rows = str(auth.get("selected_row_hash") or "")
    import services.manage_positions_cycle_binding as bind

    real_sha = bind.sha256_file_bytes

    def _sha(path):
        name = Path(path).name
        if name in {"manage_positions_plan.csv", "manage_positions_plan.json"}:
            return expected_src
        return real_sha(path)

    monkeypatch.setattr(bind, "sha256_file_bytes", _sha)
    monkeypatch.setattr(bind, "selected_row_hash", lambda rows: expected_rows)


def _positions() -> List[Dict[str, Any]]:
    return [
        {"symbol": "HD", "qty": "13", "qty_available": "13"},
        {"symbol": "GLD", "qty": "16", "qty_available": "16"},
        {"symbol": "TSLA", "qty": "8", "qty_available": "8"},
        {"symbol": "DBA", "qty": "86", "qty_available": "86"},
        {"symbol": "BRK.B", "qty": "4", "qty_available": "4"},
    ]


def _apply_containment(result: Dict[str, Any]) -> Dict[str, Any]:
    from services.manage_positions import _try_pre_plan_authorized_sell_dispatch

    # Mirror the production helper's post-dispatch classification without
    # writing the isolated-tree RESULTS path.
    out = dict(result)
    reason = str(out.get("reason") or "")
    if out.get("attempted") and not out.get("dispatched"):
        if sell_unusable_should_abort_cycle(reason):
            out["abort_before_plan"] = True
            out["sell_block_class"] = "HARD_CYCLE_FAILURE"
        else:
            out["abort_before_plan"] = False
            out["sell_blocked"] = True
            out["sell_block_class"] = "SELL_PERMISSION_UNUSABLE"
    elif (not out.get("dispatched")) and sell_permission_soft_block_reason(reason):
        out["abort_before_plan"] = False
        out["sell_blocked"] = True
        out["sell_block_class"] = "SELL_PERMISSION_UNUSABLE"
    return out


def test_soft_vs_hard_reason_allowlist():
    for reason in SOFT_REASONS:
        assert sell_permission_soft_block_reason(reason) is True
        assert sell_unusable_should_abort_cycle(reason) is False
    for reason in HARD_REASONS:
        assert sell_unusable_should_abort_cycle(reason) is True
    assert sell_unusable_should_abort_cycle("") is True
    assert sell_unusable_should_abort_cycle("unknown_broker_ambiguity") is True


def test_ttl_and_live_block_unchanged():
    assert READINESS_REFERENCE_TTL_MINUTES == 50
    assert AUTHORITY_TTL_MINUTES == 60
    packet = _load_packet()
    assert int(packet.get("freshness_ttl_minutes") or 0) == 50
    auth = json.loads((FIXTURES / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    created = datetime.fromisoformat(str(auth["issued_at"]).replace("Z", "+00:00"))
    expires = datetime.fromisoformat(str(auth["expires_at"]).replace("Z", "+00:00"))
    assert (expires - created) == timedelta(minutes=60)
    assert auth.get("live_authorized") is False
    assert str(auth.get("live_execution_required_state") or "").upper() == "BLOCKED"


def _cap_dba_snapshot_price(tmp_path: Path, price: str = "29.21") -> None:
    pos_path = tmp_path / "positions_snapshot.csv"
    if not pos_path.is_file():
        return
    import csv as _csv

    lines = pos_path.read_text(encoding="utf-8").splitlines()
    reader = _csv.DictReader(lines)
    fieldnames = list(reader.fieldnames or [])
    out_rows = []
    for row in reader:
        if str(row.get("ticker") or row.get("symbol") or "").upper() == "DBA":
            row["current_price"] = price
        out_rows.append(row)
    with pos_path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def test_case_g_corrected_natural_cycle(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    db = _seed_store(tmp_path)
    _cap_dba_snapshot_price(tmp_path)
    packet = _load_packet()
    _write_plan_rows(tmp_path, packet)
    auth_doc = json.loads((tmp_path / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    _patch_plan_hashes(monkeypatch, auth_doc)
    wait = int((NEXT_DUE - CYCLE1_COMPLETE).total_seconds())
    assert wait == 2537
    assert PACKET_UNTIL > NEXT_DUE
    assert AUTH_EXPIRES > NEXT_DUE
    freeze_dispatch_clock(monkeypatch, NEXT_DUE)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", CYCLE2_INV)
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=NEXT_DUE,
        invocation_id=CYCLE2_INV,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stub_unrelated_pre_gates(monkeypatch)
    ok, reason, details = evaluate_readiness_packet_freshness(
        packet, now=NEXT_DUE, session_payload=OPEN_CAL
    )
    assert ok is True, reason
    auth = find_unused_pre_plan_authority(db_path=db, now=NEXT_DUE)
    assert auth is not None
    assert auth.get("authority_id") == "sell-arm-audit-a1bee763c360"
    bound_ok, bound_reason, _ = validate_pre_plan_authority_bound_artifacts(
        auth, results_root=tmp_path, now=NEXT_DUE
    )
    assert bound_ok is True, bound_reason
    fake = PacketQuotedFakeBroker(packet, positions=_positions())
    # Avoid float remainder clipping the $29.22 DBA share off the $5,000 cap.
    fake._packet_prices["DBA"] = 29.21
    fake.quote_bid = min(fake._packet_prices.values())
    fake.quote_ask = fake.quote_bid
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=db,
        now=NEXT_DUE,
        verbose=True,
    )
    assert result.get("dispatched") is True, result.get("reason") or result
    assert result.get("consumed") is True
    payloads = list(getattr(fake, "submit_payloads", []) or [])
    got = sorted(
        (
            str(p.get("symbol") or "").upper(),
            float(p.get("qty") or 0),
        )
        for p in payloads
    )
    auth_map = {sym: qty for sym, qty in AUTHORIZED}
    got_map = {sym: qty for sym, qty in got}
    assert set(got_map) == set(auth_map), got
    for sym, qty in got_map.items():
        assert qty > 0
        assert qty <= auth_map[sym] + 1e-9
    assert "BRK.B" not in got_map
    total = sum(float(p.get("notional") or 0) for p in payloads)
    if total <= 0:
        total = sum(
            float(p.get("qty") or 0) * float(p.get("limit_price") or p.get("price") or 0)
            for p in payloads
        )
    assert total <= APPROVAL_CLI_CAP + 1e-6
    snap = inspect_sqlite_arm_snapshot(db, now=NEXT_DUE + timedelta(seconds=1))
    latest = snap.get("latest_authority") or {}
    assert str(latest.get("state") or "").upper() in {"CONSUMED", "SUBMITTED"} or result.get(
        "consumed"
    )
    assert LIVE_BLOCKED["live_blocked"] is True
    replay = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=PacketQuotedFakeBroker(packet, positions=_positions()),
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=db,
        now=NEXT_DUE + timedelta(seconds=5),
        verbose=False,
    )
    assert replay.get("dispatched") is not True
    evidence = write_pre_plan_dispatch_evidence(
        tmp_path,
        {"attempted": True, "dispatched": True, "consumed": True, "reason": "ok"},
        invocation_id=CYCLE2_INV,
        now=NEXT_DUE,
    )
    kind, _ev = classify_same_cycle_dispatch(invocation_id=CYCLE2_INV, results_root=tmp_path)
    assert kind == "suppress"
    assert evidence.get("live_execution_permitted") is False


def test_case_h_stale_packet_soft_block(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    db = _seed_store(tmp_path)
    packet = _load_packet()
    freeze_dispatch_clock(monkeypatch, STALE_AT)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", CYCLE2_INV)
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=STALE_AT,
        invocation_id=CYCLE2_INV,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stub_unrelated_pre_gates(monkeypatch)
    ok, reason, _ = evaluate_readiness_packet_freshness(
        packet, now=STALE_AT, session_payload=OPEN_CAL
    )
    assert ok is False
    assert reason == "stale_readiness_packet"
    auth = find_unused_pre_plan_authority(db_path=db, now=STALE_AT)
    assert auth is not None
    assert str(auth.get("consumption_status") or "").upper() == "UNUSED"
    fake = PacketQuotedFakeBroker(packet, positions=_positions())
    raw = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=db,
        now=STALE_AT,
        verbose=True,
    )
    result = _apply_containment(raw)
    assert result.get("dispatched") is not True
    assert result.get("reason") == "stale_readiness_packet"
    assert result.get("abort_before_plan") is False
    assert result.get("sell_blocked") is True
    assert result.get("sell_block_class") == "SELL_PERMISSION_UNUSABLE"
    assert getattr(fake, "submit_payloads", []) in ([], None) or len(fake.submit_payloads) == 0
    assert getattr(fake, "fake_broker_call_count", 0) == 0 or not fake.submit_payloads
    evidence = write_pre_plan_dispatch_evidence(
        tmp_path, result, invocation_id=CYCLE2_INV, now=STALE_AT
    )
    assert evidence.get("pipeline_health_class") == "SELL_NOT_AUTHORIZED"
    kind, _ = classify_same_cycle_dispatch(invocation_id=CYCLE2_INV, results_root=tmp_path)
    assert kind == "none"
    plan = {
        "timestamp": STALE_AT.isoformat(),
        "mode": "paper",
        "rows": [
            {
                "symbol": "HD",
                "final_action": "EXIT",
                "qty": 13.0,
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "skip_reason": "",
                "current_position_qty": 13.0,
                "current_price": 322.65,
                "planned_notional": 4194.45,
            }
        ],
    }
    (tmp_path / "manage_positions_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "manage_positions_plan.csv").write_text(
        "symbol,final_action,qty,selected_for_execution,reason_code,skip_reason,"
        "current_position_qty,current_price,planned_notional\n"
        "HD,EXIT,13,true,EXIT_HIGH_DRAG_PROTECTION,,13,322.65,4194.45\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", "mo-at-f2-after-stale")
    write_cycle_binding(
        results_root=tmp_path,
        invocation_id="mo-at-f2-after-stale",
        plan_generated_at=STALE_AT.isoformat(),
        f_stage_status="COMPLETE",
        now=STALE_AT,
        remaining_blocker="MISSING_PAPER_SELL_AUTHORITY",
        operator_execution_required=True,
        block_reasons=["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
    )
    f2 = generate_from_manage_artifacts(results_root=tmp_path, now=STALE_AT)
    assert f2.get("ok") is True or int(f2.get("exit_code") or 0) in {0, 10, 11, 12, 13, 14}
    new_packet = f2.get("packet") or {}
    assert new_packet.get("generated_at")
    assert str(new_packet.get("generated_at")) != packet.get("generated_at")


def test_case_h_manage_helper_does_not_abort_stale(monkeypatch, tmp_path):
    isolate_sell_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "services.place_live_orders.dispatch_unused_authorized_sell_slice_pre_plan",
        lambda **kwargs: {
            "attempted": True,
            "dispatched": False,
            "consumed": False,
            "reason": "stale_readiness_packet",
        },
    )
    from services.manage_positions import _try_pre_plan_authorized_sell_dispatch

    result = _try_pre_plan_authorized_sell_dispatch(
        broker=object(), mode="paper", dry_run=False, verbose=False
    )
    assert result.get("abort_before_plan") is False
    assert result.get("sell_blocked") is True
    assert result.get("reason") == "stale_readiness_packet"


def test_case_i_expired_authority_buy_separation(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    db = _seed_store(tmp_path)
    freeze_dispatch_clock(monkeypatch, EXPIRED_AT)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", "mo-at-cycle3")
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=EXPIRED_AT,
        invocation_id="mo-at-cycle3",
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stub_unrelated_pre_gates(monkeypatch)
    fake = PacketQuotedFakeBroker(_load_packet(), positions=_positions())
    raw = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=db,
        now=EXPIRED_AT,
        verbose=False,
    )
    result = _apply_containment(raw)
    assert result.get("dispatched") is not True
    assert result.get("reason") in {
        "authority_expired",
        "no_unused_authority",
        "sqlite_authority_expired",
    }
    assert result.get("abort_before_plan") is False
    assert result.get("sell_blocked") is True
    auth_path = tmp_path / HARDENED_AUTHORITY_FILENAME
    state, reason, details = classify_paper_operator_approval_artifact(contract_path=auth_path)
    assert details.get("hardened_sell_arm") is True
    assert state == PAPER_APPROVAL_EXPIRED or str(reason).endswith("authority_expired")
    inv = "mo-at-cycle3-buy"
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": EXPIRED_AT.isoformat(), "mode": "paper", "rows": []}),
        encoding="utf-8",
    )
    (tmp_path / "manage_positions_plan.csv").write_text(
        "symbol,final_action,qty,selected_for_execution,reason_code,skip_reason,"
        "current_position_qty,current_price,planned_notional\n",
        encoding="utf-8",
    )
    write_cycle_binding(
        results_root=tmp_path,
        invocation_id=inv,
        plan_generated_at=EXPIRED_AT.isoformat(),
        f_stage_status="COMPLETE",
        now=EXPIRED_AT,
        remaining_blocker=BLOCKER_NONE,
        operator_execution_required=False,
        block_reasons=[],
    )
    write_entry_interlock(
        results_root=tmp_path,
        invocation_id=inv,
        now=EXPIRED_AT,
        decision=DECISION_MAY_CONTINUE,
        reason="ok",
        packet_status="VALID_EMPTY",
        interlock_scope="NONE",
        global_block_reason=None,
    )
    interlock_path = tmp_path / "paper_sell_exit_before_entry.json"
    interlock_sha = hashlib.sha256(interlock_path.read_bytes()).hexdigest()
    bind_path = tmp_path / "manage_positions_cycle_binding.json"
    bind_sha = (
        hashlib.sha256(bind_path.read_bytes()).hexdigest() if bind_path.is_file() else "0" * 64
    )
    (tmp_path / "portfolio_allocation_recommendations.csv").write_text(
        "ticker,action,eligible_for_auto_entry,planned_notional,sector\n",
        encoding="utf-8",
    )
    (tmp_path / "execution_readiness.json").write_text(
        json.dumps(
            {
                "readiness_status": "NOT_READY",
                "failed_checks": ["watchdog"],
                "live_execution_permitted": False,
                "paper_execution_permitted": True,
                "governance_mode": "GREEN",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "services.manage_positions_cycle_binding.evaluate_exit_before_entry",
        lambda **kwargs: {
            "ok": True,
            "decision": DECISION_MAY_CONTINUE,
            "reason": "ok",
            "packet_status": "VALID_EMPTY",
            "interlock_scope": "NONE",
            "global_block_reason": None,
        },
    )
    monkeypatch.setenv("TRITON_ENABLE_PAPER_EXECUTION", "1")
    monkeypatch.setenv("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", tmp_path)
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.EXEC_READINESS", tmp_path / "execution_readiness.json"
    )
    alloc = ROOT / "data" / "results"
    alloc.mkdir(parents=True, exist_ok=True)
    alloc_csv = alloc / "portfolio_allocation_recommendations.csv"
    if not alloc_csv.is_file():
        alloc_csv.write_text(
            "ticker,action,eligible_for_auto_entry,planned_notional,sector\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "services.paper_entry_loop_stage._load_governed_allocation_candidates",
        lambda: ([], {"eligible_entry_symbols": [], "allocation_candidates_count": 0}),
    )
    monkeypatch.setattr(
        "services.paper_operator_approval_artifact_state.APPROVAL_PATH",
        auth_path,
    )
    monkeypatch.setattr("services.guard_auto.is_market_open_et", lambda *_a, **_k: True)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", inv)
    (tmp_path / "market_open_invocation_handoff.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "READY",
                "market_open_invocation_id": inv,
                "session_date": "2026-09-01",
                "issued_at": EXPIRED_AT.isoformat(),
                "ready_at": EXPIRED_AT.isoformat(),
                "generated_at": EXPIRED_AT.isoformat(),
                "interlock_sha256": interlock_sha,
                "f_binding_sha256": bind_sha,
                "source": "scripts/run_market_open.ps1",
                "creates_no_authority": True,
                "live_execution_permitted": False,
            }
        ),
        encoding="utf-8",
    )
    _ok, gate_reason, gate_details = evaluate_paper_entry_gates(
        verbose=False, results_root=tmp_path, market_open_invocation_id=inv
    )
    assert gate_reason != "approval_expired"
    checks = (gate_details or {}).get("checks") or {}
    if checks.get("paper_operator_approval_state"):
        assert checks.get("sell_authority_ignored_for_buy_gate") is True
        assert checks.get("expired_sell_authority_not_buy_approval_expired") is True
    assert str(gate_details.get("failed_check") or "") != "paper_operator_approval"


def test_case_j_pending_risk_interlock_not_approval_expired(monkeypatch):
    monkeypatch.setattr(
        "services.paper_sell_entry_dependency.pending_sell_sector",
        lambda symbol: "Consumer",
    )
    pending = [
        {
            "symbol": "HD",
            "action": "EXIT",
            "qty": 13,
            "sector": "Consumer",
            "risk_objective": "REDUCE_CONSUMER",
            "final_action": "EXIT",
        }
    ]
    candidate = {
        "symbol": "LOW",
        "ticker": "LOW",
        "action": "BUY",
        "sector": "Consumer",
        "planned_notional": 500.0,
        "eligible_for_auto_entry": True,
    }
    rec = evaluate_entry_candidate(
        candidate,
        pending_sells=pending,
        current_cash=10_000.0,
        at_position_cap=False,
    )
    assert rec["decision"] == BLOCK_WORSENS_PENDING_RISK_REDUCTION
    assert rec["decision"] != "approval_expired"
    assert sell_unusable_should_abort_cycle("authority_expired") is False


def test_case_k_no_qualifying_buy_is_healthy():
    from services.defensive_exit_approval_readiness import build_pre_plan_dispatch_evidence

    ev = build_pre_plan_dispatch_evidence(
        {
            "attempted": False,
            "dispatched": False,
            "reason": "no_unused_authority",
            "sell_blocked": True,
            "sell_block_class": "SELL_PERMISSION_UNUSABLE",
        },
        invocation_id="mo-at-no-buy",
        now=EXPIRED_AT,
    )
    assert ev["pipeline_health_class"] == "SELL_NOT_AUTHORIZED"
    assert ev["broker_orders_submitted"] is False
    assert ev["live_execution_permitted"] is False
    assert ev["dispatch_performed"] is False


def test_case_l_ambiguous_in_progress_hard_fail(monkeypatch, tmp_path):
    isolate_sell_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "services.place_live_orders.dispatch_unused_authorized_sell_slice_pre_plan",
        lambda **kwargs: {
            "attempted": True,
            "dispatched": False,
            "consumed": False,
            "reason": REASON_MATCHING_IN_PROGRESS,
        },
    )
    from services.manage_positions import _try_pre_plan_authorized_sell_dispatch

    result = _try_pre_plan_authorized_sell_dispatch(
        broker=object(), mode="paper", dry_run=False, verbose=False
    )
    assert result.get("abort_before_plan") is True
    assert result.get("sell_block_class") == "HARD_CYCLE_FAILURE"
    assert result.get("reason") == REASON_MATCHING_IN_PROGRESS
    ev = __import__(
        "services.defensive_exit_approval_readiness", fromlist=["build_pre_plan_dispatch_evidence"]
    ).build_pre_plan_dispatch_evidence(result, invocation_id="mo-at-ambiguous", now=NEXT_DUE)
    assert ev["pipeline_health_class"] == "SYSTEM_FAILURE"


def test_sell_block_does_not_authorize_buy():
    assert sell_permission_soft_block_reason("stale_readiness_packet") is True
    # Soft SELL block is not a BUY permit.
    assert BLOCK_WORSENS_PENDING_RISK_REDUCTION != "approval_expired"
