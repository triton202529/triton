"""Phase 198M-C — mixed F-plan SELL extraction and BOM-free handoff.

PAPER ONLY. Isolated tmp_path. Fake broker only. No production writes.
Does not modify paper_entry_loop_stage.py or run_continuous_loop.ps1.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from dashboard.paper_sell_arm_controls import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_GREEN,
    LABEL_ARMED,
    LABEL_NOT_REQUIRED,
    LABEL_READY,
    map_sell_arm_visual_status,
)
from services.defensive_exit_approval_readiness import (
    PACKET_NAME,
    RECOGNIZED_ENTRY_ONLY_ACTIONS,
    generate_from_manage_artifacts,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    DECISION_CANDIDATE_AWARE,
    DECISION_MAY_CONTINUE,
    F2_GENERATION_FAILURE,
    F2_OK,
    F2_PENDING_SELL_ARM,
    INVOCATION_ENV,
    evaluate_exit_before_entry,
    selected_row_hash,
    write_cycle_binding,
)
from services.paper_entry_loop_stage import (
    REASON_INVOCATION_HANDOFF_BINDING,
    REASON_INVOCATION_HANDOFF_NOT_READY,
    REASON_INVOCATION_HANDOFF_SESSION,
    REASON_INVOCATION_HANDOFF_STALE,
    REASON_INVOCATION_MALFORMED,
    REASON_INVOCATION_MISSING,
    REASON_INVOCATION_SOURCE_MISMATCH,
    _current_session_date_et,
    resolve_entry_invocation_id,
    validate_market_open_invocation_handoff,
)
from services.paper_sell_arm_authority import BrokerSnapshotView, readiness_packet_hash
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import (
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
    begin_paper_sell_dispatch,
)
from services.paper_sell_arm_store import db_path_for_store_root, inspect_sqlite_arm_snapshot
from services.paper_sell_entry_dependency import (
    ALLOW_EXISTING_GATES,
    BLOCK_SAME_SYMBOL_PENDING_SELL,
    context_from_interlock,
    filter_entry_candidates,
)
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    NetworkFakeAlpacaBroker,
    block_real_network,
)

ROOT = Path(__file__).resolve().parents[1]
MARKET_OPEN = ROOT / "scripts" / "run_market_open.ps1"
PROD_PACKET = ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"
PROD_PACKET_SHA = "3a2c0dc9f5529edfe06dadc7482948b0741b2e5146dbb85dcec893ae8eec402c"
PROD_ARM = ROOT / "data" / "live" / "paper_arm_confirm.json"
REPO_SQLITE = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
NOW = datetime(2026, 8, 17, 14, 0, tzinfo=timezone.utc)
INV = "mo-20260817_140000-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    stub_dir_fsync_ok(monkeypatch)
    block_real_network(monkeypatch)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    for key in list(os.environ):
        if "ALPACA" in key.upper() or key.startswith("APCA_"):
            monkeypatch.delenv(key, raising=False)
    for key in (
        "TRITON_PAPER_SELL_ARM_STORE_ROOT",
        "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_DB",
        "RESULTS_ROOT",
        INVOCATION_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    before_pkt = PROD_PACKET.read_bytes() if PROD_PACKET.is_file() else None
    before_arm = PROD_ARM.read_bytes() if PROD_ARM.is_file() else None
    yield
    _ACTIVE_PERMIT.set(None)
    after_pkt = PROD_PACKET.read_bytes() if PROD_PACKET.is_file() else None
    after_arm = PROD_ARM.read_bytes() if PROD_ARM.is_file() else None
    assert after_pkt == before_pkt
    assert hashlib.sha256(after_pkt).hexdigest() == PROD_PACKET_SHA
    assert after_arm == before_arm
    assert not REPO_SQLITE.is_file()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(results: Path, rows, *, ts=None):
    ts = ts or NOW.isoformat()
    results.mkdir(parents=True, exist_ok=True)
    (results / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": ts, "orders_executed": 0, "blocked": True, "mode": "paper"}),
        encoding="utf-8",
    )
    fieldnames = [
        "symbol",
        "final_action",
        "qty",
        "selected_for_execution",
        "reason_code",
        "skip_reason",
        "current_position_qty",
        "current_price",
        "planned_notional",
        "order_side",
        "status",
    ]
    with (results / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            item = {k: "" for k in fieldnames}
            item.update(row)
            if "selected_for_execution" in item:
                sel = item["selected_for_execution"]
                item["selected_for_execution"] = (
                    "true" if sel in (True, "true", "True") else str(sel)
                )
            w.writerow(item)
    return ts


def _row(
    symbol, action, qty, *, selected=True, side="", reason="EXIT_HIGH_DRAG_PROTECTION", price=332.5
):
    return {
        "symbol": symbol,
        "final_action": action,
        "qty": qty,
        "selected_for_execution": "true" if selected else "false",
        "reason_code": reason,
        "skip_reason": "",
        "current_position_qty": qty,
        "current_price": price,
        "planned_notional": float(qty) * float(price),
        "order_side": side,
        "status": "planned" if selected else "skipped",
    }


def _bind(
    results: Path, monkeypatch, *, invocation_id=INV, remaining=BLOCKER_MISSING_SELL_AUTHORITY
):
    monkeypatch.setenv(INVOCATION_ENV, invocation_id)
    plan = json.loads((results / "manage_positions_plan.json").read_text(encoding="utf-8"))
    return write_cycle_binding(
        results_root=results,
        invocation_id=invocation_id,
        plan_generated_at=str(plan["timestamp"]),
        f_stage_status="COMPLETE",
        now=NOW,
        remaining_blocker=remaining,
        operator_execution_required=remaining == BLOCKER_MISSING_SELL_AUTHORITY,
        block_reasons=(
            ["DISPATCH_BLOCKED_APPROVAL_REQUIRED"]
            if remaining == BLOCKER_MISSING_SELL_AUTHORITY
            else []
        ),
    )


def _handoff_fn_source() -> str:
    src = MARKET_OPEN.read_text(encoding="utf-8")
    start = src.index("function Write-Utf8NoBomAtomicFile")
    end = src.index("# Supersede any prior READY")
    block = src[start:end].strip()
    assert "UTF8Encoding($false)" in block
    assert "Set-Content" not in block
    return block


def _ps_write_handoff(
    dest: Path,
    *,
    status: str,
    inv: str,
    session_date: str,
    issued_at: str,
    interlock_sha: str = "",
    f_binding_sha: str = "",
) -> subprocess.CompletedProcess:
    dest.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
$ErrorActionPreference = 'Stop'
$script:HandoffPath = {json.dumps(str(dest))}
$script:MarketOpenInvocationId = {json.dumps(inv)}
$script:HandoffSessionDate = {json.dumps(session_date)}
$script:HandoffIssuedAt = {json.dumps(issued_at)}
{_handoff_fn_source()}
Write-InvocationHandoff -Status {json.dumps(status)} -InterlockSha256 {json.dumps(interlock_sha)} -FBindingSha256 {json.dumps(f_binding_sha)}
"""
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _assert_no_bom(path: Path) -> bytes:
    raw = path.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", raw[:8].hex()
    json.loads(raw.decode("utf-8"))
    json.loads(path.read_text(encoding="utf-8"))
    return raw


def test_script_uses_bom_free_writer_not_setcontent_utf8():
    src = MARKET_OPEN.read_text(encoding="utf-8")
    assert "New-Object System.Text.UTF8Encoding($false)" in src
    assert "function Write-Utf8NoBomAtomicFile" in src
    assert "Set-Content -Path $tmp -Encoding utf8" not in src
    tree_ok = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$errs=$null; $null=[System.Management.Automation.Language.Parser]::ParseFile("
            f"{json.dumps(str(MARKET_OPEN))}, [ref]$null, [ref]$errs); "
            "if($errs.Count){ $errs | ForEach-Object { $_.ToString() }; exit 1 } else { 'PS_PARSE_OK' }",
        ],
        capture_output=True,
        text=True,
    )
    assert "PS_PARSE_OK" in (tree_ok.stdout or "")
    assert "BUY" in RECOGNIZED_ENTRY_ONLY_ACTIONS
    assert "ADD" in RECOGNIZED_ENTRY_ONLY_ACTIONS
    assert "OPEN_NEW" in RECOGNIZED_ENTRY_ONLY_ACTIONS


def test_mixed_plan_excludes_arkk_add_writes_generated_at(tmp_path, monkeypatch):
    rows = [
        _row("TSLA", "EXIT", 8, side="sell"),
        _row("ARKK", "ADD", 8, side="buy", reason="ADD_APPROVED", price=81.7),
    ]
    _write_plan(tmp_path, rows)
    binding = _bind(tmp_path, monkeypatch)
    whole = selected_row_hash(
        list(
            csv.DictReader(
                (tmp_path / "manage_positions_plan.csv").read_text(encoding="utf-8").splitlines()
            )
        )
    )
    assert binding["selected_row_hash"] == whole
    prior = tmp_path / PACKET_NAME
    prior.write_text(json.dumps({"legacy": True}), encoding="utf-8")
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is True
    assert result["exit_code"] == F2_PENDING_SELL_ARM
    assert result["wrote"] is True
    pkt = result["packet"]
    assert pkt["generated_at"]
    assert pkt["market_open_invocation_id"] == INV
    assert pkt["selected_row_hash"] == whole
    assert pkt["plan_hash"] == whole
    assert pkt["source_plan_sha256"] == binding["source_plan_sha256"]
    symbols = [a["symbol"] for a in pkt["approval_ready_actions"]]
    assert symbols == ["TSLA"]
    assert all(a["action"] != "ADD" for a in pkt["approval_ready_actions"])
    assert pkt["excluded_entry_only_count"] == 1
    assert "ARKK" in pkt["excluded_entry_only_symbols"]
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    vis = map_sell_arm_visual_status(pres)
    assert vis["label"] == LABEL_READY
    assert vis["color"] == COLOR_AMBER
    assert pres["arm_enabled"] is True
    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["challenge_count"] == 0
    assert snap["authority_count"] == 0
    assert result["broker"] == 0


def test_buy_add_only_valid_empty(tmp_path, monkeypatch):
    _write_plan(tmp_path, [_row("ARKK", "ADD", 8, side="buy", reason="ADD_APPROVED", price=81.7)])
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is True
    assert result["exit_code"] == F2_OK
    assert result["packet"]["generated_at"]
    assert result["packet"]["approval_ready_actions"] == []
    assert result["packet"]["excluded_entry_only_count"] == 1
    assert result["interlock_decision"] == DECISION_MAY_CONTINUE
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "x.sqlite3",
    )
    vis = map_sell_arm_visual_status(pres)
    assert vis["label"] == LABEL_NOT_REQUIRED
    assert vis["color"] == COLOR_BLUE


def test_sell_only_pending_arm_unchanged(tmp_path, monkeypatch):
    _write_plan(tmp_path, [_row("TSLA", "EXIT", 8, side="sell")])
    binding = _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["exit_code"] == F2_PENDING_SELL_ARM
    assert result["packet"]["approval_ready_actions"][0]["symbol"] == "TSLA"
    assert result["packet"]["excluded_entry_only_count"] == 0
    assert result["packet"]["selected_row_hash"] == binding["selected_row_hash"]


def test_malformed_unknown_action_fail_closed_preserves_packet(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        [
            _row("TSLA", "EXIT", 8, side="sell"),
            _row("MSFT", "WEIRD", 1, side="sell"),
        ],
    )
    _bind(tmp_path, monkeypatch)
    prior = tmp_path / PACKET_NAME
    prior.write_text(json.dumps({"legacy": True, "keep": 1}), encoding="utf-8")
    before = prior.read_text(encoding="utf-8")
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is False
    assert result["exit_code"] == F2_GENERATION_FAILURE
    assert result["wrote"] is False
    assert result["previous_preserved"] is True
    assert prior.read_text(encoding="utf-8") == before
    assert result["broker"] == 0


def test_contradictory_side_fail_closed(tmp_path, monkeypatch):
    _write_plan(tmp_path, [_row("TSLA", "EXIT", 8, side="buy")])
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is False
    assert result["exit_code"] == F2_GENERATION_FAILURE
    assert "contradictory_side_action" in str(result.get("reason"))


def test_zero_qty_sell_fail_closed(tmp_path, monkeypatch):
    _write_plan(tmp_path, [_row("TSLA", "EXIT", 0, side="sell")])
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is False
    assert result["exit_code"] == F2_GENERATION_FAILURE


def test_open_new_excluded_from_sell_packet(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        [
            _row("TSLA", "EXIT", 8, side="sell"),
            _row("MSFT", "OPEN_NEW", 1, side="buy", reason="ENTRY_APPROVED", price=400.0),
        ],
    )
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is True
    assert result["exit_code"] == F2_PENDING_SELL_ARM
    symbols = [a["symbol"] for a in result["packet"]["approval_ready_actions"]]
    assert symbols == ["TSLA"]
    assert result["packet"]["excluded_entry_only_count"] == 1
    assert "MSFT" in result["packet"]["excluded_entry_only_symbols"]


def test_cannot_hide_malformed_behind_buy_exclusion(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        [
            _row("ARKK", "ADD", 8, side="buy", reason="ADD_APPROVED", price=81.7),
            _row("TSLA", "EXIT", -1, side="sell"),
        ],
    )
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert result["ok"] is False
    assert result["exit_code"] == F2_GENERATION_FAILURE


def test_ps_pending_and_ready_bom_free_and_entry_accepts(tmp_path, monkeypatch):
    _write_plan(tmp_path, [_row("TSLA", "EXIT", 8, side="sell")])
    _bind(tmp_path, monkeypatch)
    f2 = generate_from_manage_artifacts(results_root=tmp_path, now=NOW, expected_invocation_id=INV)
    assert f2["ok"] is True
    dest = tmp_path / "market_open_invocation_handoff.json"
    session = _current_session_date_et(now=datetime.now(timezone.utc))
    issued = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime(
        "%Y-%m-%dT%H:%M:%S.0000000Z"
    )
    pending = _ps_write_handoff(
        dest, status="PENDING", inv=INV, session_date=session, issued_at=issued
    )
    assert pending.returncode == 0, pending.stderr
    raw_p = _assert_no_bom(dest)
    assert raw_p[:3] != b"\xef\xbb\xbf"
    doc_p = json.loads(dest.read_text(encoding="utf-8"))
    assert doc_p["status"] == "PENDING"
    assert doc_p["schema_version"] == 2
    inv, err, _ = validate_market_open_invocation_handoff(
        results_root=tmp_path, now=datetime.now(timezone.utc)
    )
    assert inv is None
    assert err == REASON_INVOCATION_HANDOFF_NOT_READY

    interlock = tmp_path / "paper_sell_exit_before_entry.json"
    binding = tmp_path / "manage_positions_cycle_binding.json"
    ready = _ps_write_handoff(
        dest,
        status="READY",
        inv=INV,
        session_date=session,
        issued_at=issued,
        interlock_sha=_sha(interlock),
        f_binding_sha=_sha(binding),
    )
    assert ready.returncode == 0, ready.stderr
    _assert_no_bom(dest)
    doc_r = json.loads(dest.read_text(encoding="utf-8"))
    assert doc_r["status"] == "READY"
    assert doc_r["interlock_sha256"] == _sha(interlock)
    resolved, err2 = resolve_entry_invocation_id(
        results_root=tmp_path, now=datetime.now(timezone.utc)
    )
    assert err2 is None
    assert resolved == INV


def test_pending_supersedes_prior_ready(tmp_path):
    dest = tmp_path / "market_open_invocation_handoff.json"
    session = _current_session_date_et(now=datetime.now(timezone.utc))
    issued = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime(
        "%Y-%m-%dT%H:%M:%S.0000000Z"
    )
    (tmp_path / "paper_sell_exit_before_entry.json").write_text("{}", encoding="utf-8")
    first = _ps_write_handoff(
        dest,
        status="READY",
        inv=INV,
        session_date=session,
        issued_at=issued,
        interlock_sha=_sha(tmp_path / "paper_sell_exit_before_entry.json"),
        f_binding_sha="b" * 64,
    )
    assert first.returncode == 0, first.stderr
    second = _ps_write_handoff(
        dest,
        status="PENDING",
        inv="mo-20260817_140100-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        session_date=session,
        issued_at=issued,
    )
    assert second.returncode == 0, second.stderr
    _assert_no_bom(dest)
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["status"] == "PENDING"
    assert doc["market_open_invocation_id"].startswith("mo-20260817_140100")
    inv, err, _ = validate_market_open_invocation_handoff(
        results_root=tmp_path, now=datetime.now(timezone.utc)
    )
    assert inv is None
    assert err == REASON_INVOCATION_HANDOFF_NOT_READY


def test_handoff_fail_closed_matrix(tmp_path, monkeypatch):
    clock = datetime.now(timezone.utc)
    session = _current_session_date_et(now=clock)
    missing_inv, missing_err = resolve_entry_invocation_id(results_root=tmp_path, now=clock)
    assert missing_inv is None
    assert missing_err == REASON_INVOCATION_MISSING

    dest = tmp_path / "market_open_invocation_handoff.json"
    dest.write_text("{not-json", encoding="utf-8")
    inv, err, _ = validate_market_open_invocation_handoff(results_root=tmp_path, now=clock)
    assert err == REASON_INVOCATION_MALFORMED

    _write_plan(tmp_path, [_row("TSLA", "EXIT", 8, side="sell")])
    _bind(tmp_path, monkeypatch)
    generate_from_manage_artifacts(results_root=tmp_path, now=NOW, expected_invocation_id=INV)
    interlock = tmp_path / "paper_sell_exit_before_entry.json"
    issued = (clock - timedelta(seconds=5)).isoformat()
    payload = {
        "schema_version": 2,
        "status": "READY",
        "market_open_invocation_id": INV,
        "session_date": session,
        "issued_at": issued,
        "ready_at": issued,
        "generated_at": issued,
        "interlock_sha256": _sha(interlock),
        "f_binding_sha256": "c" * 64,
        "source": "scripts/run_market_open.ps1",
        "creates_no_authority": True,
        "live_execution_permitted": False,
        "broker_orders_submitted": False,
    }
    dest.write_text(json.dumps(payload), encoding="utf-8")
    ok_inv, ok_err, _ = validate_market_open_invocation_handoff(results_root=tmp_path, now=clock)
    assert ok_err is None
    assert ok_inv == INV

    stale = dict(payload)
    stale["ready_at"] = (clock - timedelta(hours=2)).isoformat()
    stale["issued_at"] = stale["ready_at"]
    dest.write_text(json.dumps(stale), encoding="utf-8")
    _, err, _ = validate_market_open_invocation_handoff(results_root=tmp_path, now=clock)
    assert err == REASON_INVOCATION_HANDOFF_STALE

    future = dict(payload)
    future["ready_at"] = (clock + timedelta(hours=1)).isoformat()
    future["issued_at"] = future["ready_at"]
    dest.write_text(json.dumps(future), encoding="utf-8")
    _, err, _ = validate_market_open_invocation_handoff(results_root=tmp_path, now=clock)
    assert err == REASON_INVOCATION_HANDOFF_STALE

    wrong_sess = dict(payload)
    wrong_sess["session_date"] = "2020-01-01"
    dest.write_text(json.dumps(wrong_sess), encoding="utf-8")
    _, err, _ = validate_market_open_invocation_handoff(results_root=tmp_path, now=clock)
    assert err == REASON_INVOCATION_HANDOFF_SESSION

    mismatch = dict(payload)
    mismatch["interlock_sha256"] = "a" * 64
    dest.write_text(json.dumps(mismatch), encoding="utf-8")
    _, err, _ = validate_market_open_invocation_handoff(results_root=tmp_path, now=clock)
    assert err == REASON_INVOCATION_HANDOFF_BINDING

    dest.write_text(json.dumps(payload), encoding="utf-8")
    wrong_inv, wrong_err = resolve_entry_invocation_id(
        explicit="mo-20260817_999999-ffffffffffffffffffffffffffffffff",
        results_root=tmp_path,
        now=clock,
    )
    assert wrong_inv is None
    assert wrong_err == REASON_INVOCATION_SOURCE_MISMATCH


def test_failed_replace_does_not_expose_partial_ready(tmp_path):
    dest = tmp_path / "market_open_invocation_handoff.json"
    session = _current_session_date_et(now=datetime.now(timezone.utc))
    issued = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime(
        "%Y-%m-%dT%H:%M:%S.0000000Z"
    )
    first = _ps_write_handoff(
        dest, status="PENDING", inv=INV, session_date=session, issued_at=issued
    )
    assert first.returncode == 0, first.stderr
    pending_bytes = dest.read_bytes()
    tmp_blocker = Path(str(dest) + ".tmp")
    tmp_blocker.mkdir()
    failed = _ps_write_handoff(
        dest,
        status="READY",
        inv=INV,
        session_date=session,
        issued_at=issued,
        interlock_sha="d" * 64,
        f_binding_sha="e" * 64,
    )
    assert failed.returncode != 0
    assert dest.read_bytes() == pending_bytes
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["status"] == "PENDING"


def test_two_cycle_mixed_plan_fake_broker(tmp_path, monkeypatch):
    import services.paper_sell_arm_authority as auth_mod
    import services.paper_sell_arm_runtime_enforcement as enf

    monkeypatch.setattr(auth_mod, "LIVE_GOVERNANCE", tmp_path)
    monkeypatch.setattr(enf, "LIVE_GOVERNANCE", tmp_path)
    rows = [
        _row("TSLA", "EXIT", 8, side="sell"),
        _row("ARKK", "ADD", 8, side="buy", reason="ADD_APPROVED", price=81.7),
    ]
    _write_plan(tmp_path, rows)
    binding = _bind(tmp_path, monkeypatch)
    dest = tmp_path / "market_open_invocation_handoff.json"
    session = _current_session_date_et(now=datetime.now(timezone.utc))
    issued = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime(
        "%Y-%m-%dT%H:%M:%S.0000000Z"
    )
    pending = _ps_write_handoff(
        dest, status="PENDING", inv=INV, session_date=session, issued_at=issued
    )
    assert pending.returncode == 0, pending.stderr
    _assert_no_bom(dest)

    cycle1 = generate_from_manage_artifacts(
        results_root=tmp_path, now=NOW, expected_invocation_id=INV
    )
    assert cycle1["exit_code"] == F2_PENDING_SELL_ARM
    assert [a["symbol"] for a in cycle1["packet"]["approval_ready_actions"]] == ["TSLA"]
    assert "ARKK" not in [a["symbol"] for a in cycle1["packet"]["approval_ready_actions"]]
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock["decision"] == DECISION_CANDIDATE_AWARE
    assert interlock["proceeds_available_for_entry"] == 0
    assert interlock["selected_row_hash"] == binding["selected_row_hash"]
    ctx = context_from_interlock(interlock)
    allowed, decisions = filter_entry_candidates(
        [
            {"ticker": "MSFT", "allocation_action": "OPEN_NEW", "est_notional": 500.0},
            {"ticker": "TSLA", "allocation_action": "ADD", "est_notional": 200.0},
        ],
        pending_sells=ctx["pending_sells"],
        current_cash=10000.0,
    )
    by_sym = {d["symbol"]: d["decision"] for d in decisions}
    assert by_sym["MSFT"] == ALLOW_EXISTING_GATES
    assert by_sym["TSLA"] == BLOCK_SAME_SYMBOL_PENDING_SELL
    assert [c["ticker"] for c in allowed] == ["MSFT"]

    ready = _ps_write_handoff(
        dest,
        status="READY",
        inv=INV,
        session_date=session,
        issued_at=issued,
        interlock_sha=_sha(tmp_path / "paper_sell_exit_before_entry.json"),
        f_binding_sha=_sha(tmp_path / "manage_positions_cycle_binding.json"),
    )
    assert ready.returncode == 0, ready.stderr
    _assert_no_bom(dest)
    resolved, err = resolve_entry_invocation_id(
        results_root=tmp_path, now=datetime.now(timezone.utc)
    )
    assert err is None
    assert resolved == INV

    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    vis = map_sell_arm_visual_status(pres)
    assert vis["label"] == LABEL_READY
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert armed["ok"] is True
    bad = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase="NOPE",
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert bad["ok"] is False
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert confirmed["ok"] is True
    snap = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=1)
    )
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["dispatch_count"] == 0
    vis2 = map_sell_arm_visual_status(
        build_presentation(
            store_root=tmp_path,
            results_root=tmp_path,
            now=NOW + timedelta(minutes=1),
            paper_mode=True,
            live_blocked=True,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
        )
    )
    assert vis2["label"] == LABEL_ARMED
    assert vis2["color"] == COLOR_GREEN

    packet = cycle1["packet"]
    auth = snap["latest_authority"]
    payload = dict(auth.get("payload") or {})
    challenge_id = payload.get("confirmation_challenge_id") or auth.get("confirmation_challenge_id")
    challenge_hash = payload.get("confirmation_challenge_hash") or auth.get(
        "confirmation_challenge_hash"
    )
    fingerprint = payload.get("account_fingerprint") or auth.get("account_fingerprint")
    assert challenge_id
    (tmp_path / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": NOW.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (NOW + timedelta(hours=4)).isoformat(),
                "allow_execute": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available\nTSLA,8,8\n", encoding="utf-8"
    )
    (tmp_path / "open_orders_snapshot.csv").write_text("symbol,side,qty\n", encoding="utf-8")
    (tmp_path / "independent_dispatch_bindings.json").write_text(
        json.dumps(
            {
                "plan_hash": packet["plan_hash"],
                "cycle_id": packet["cycle_id"],
                "run_id": packet["run_id"],
                "batch_id": packet["batch_id"],
                "plan_id": packet["plan_id"],
                "trading_date": NOW.date().isoformat(),
                "account_fingerprint": fingerprint,
                "readiness_packet_hash": readiness_packet_hash(packet),
                "confirmation_challenge_id": challenge_id,
                "confirmation_challenge_hash": challenge_hash,
                "paper_execution_authorized": True,
                "market_session": "MARKET_OPEN",
                "market_session_observed_at": NOW.isoformat(),
                "market_session_timezone": "America/New_York",
                "lifecycle_actions": {"TSLA": "EXIT"},
            }
        ),
        encoding="utf-8",
    )

    @dataclass
    class Planned:
        symbol: str = "TSLA"
        side: str = "sell"
        qty: float = 8.0
        limit_price: float = 332.5
        stance: str = "EXIT"
        client_order_id: str = "coid-198m-c"
        force_exit_override: bool = False
        ref_price: Optional[float] = 332.5
        management_action: Optional[str] = "EXIT"

    view = BrokerSnapshotView(
        positions={"TSLA": {"qty": 8.0, "qty_available": 8.0}},
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date=NOW.date().isoformat(),
        account_fingerprint=fingerprint,
        current_plan_hash=packet["plan_hash"],
        paper_buy_arm_valid=True,
        paper_execution_authorized=True,
        cycle_id=packet["cycle_id"],
        run_id=packet["run_id"],
        batch_id=packet["batch_id"],
        plan_id=packet["plan_id"],
        readiness_packet_hash=readiness_packet_hash(packet),
        confirmation_challenge_id=challenge_id,
        confirmation_challenge_hash=challenge_hash,
        evidence_independent=True,
        market_session="MARKET_OPEN",
        lifecycle_actions={"TSLA": "EXIT"},
    )
    fake = NetworkFakeAlpacaBroker(mode="paper")
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=[Planned()],
        broker_view=view,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        live_override=LIVE_BLOCKED,
        explicit_runtime_write=True,
    )
    assert guard.permitted is True
    fake.submit_order(
        symbol="TSLA",
        qty=8,
        side="sell",
        order_type="limit",
        time_in_force="day",
        limit_price=332.5,
        client_order_id="coid-198m-c",
    )
    assert len(fake.posts) == 1
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(
            symbol="TSLA",
            qty=8,
            side="sell",
            order_type="limit",
            time_in_force="day",
            limit_price=332.5,
            client_order_id="coid-198m-c-replay",
        )
    assert len(fake.posts) == 1
    snap2 = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=2)
    )
    assert snap2["dispatch_count"] >= 1 or snap2["latest_authority"]["state"] in {
        "CONSUMED",
        "DISPATCH_IN_PROGRESS",
    }

    _write_plan(tmp_path, rows, ts=(NOW + timedelta(minutes=5)).isoformat())
    write_cycle_binding(
        results_root=tmp_path,
        invocation_id="mo-20260817_140500-cccccccccccccccccccccccccccccccc",
        plan_generated_at=(NOW + timedelta(minutes=5)).isoformat(),
        f_stage_status="COMPLETE",
        now=NOW + timedelta(minutes=5),
        remaining_blocker=BLOCKER_MISSING_SELL_AUTHORITY,
        operator_execution_required=True,
        block_reasons=["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
        submitted_symbols=["TSLA"],
        submitted_outcomes={"TSLA": "SUBMITTED"},
        orders_executed=1,
    )
    monkeypatch.setenv(INVOCATION_ENV, "mo-20260817_140500-cccccccccccccccccccccccccccccccc")
    cycle2 = generate_from_manage_artifacts(
        results_root=tmp_path,
        now=NOW + timedelta(minutes=6),
        expected_invocation_id="mo-20260817_140500-cccccccccccccccccccccccccccccccc",
    )
    assert cycle2["ok"] is True
    sell_syms = [a["symbol"] for a in cycle2["packet"]["approval_ready_actions"]]
    assert "ARKK" not in sell_syms
    assert "TSLA" not in sell_syms
    assert cycle2["exit_code"] in {F2_OK, F2_PENDING_SELL_ARM}
    assert fake.fake_broker_call_count == 1
    assert not any(str(p.get("side") or "").lower() == "buy" for p in fake.submit_payloads)
