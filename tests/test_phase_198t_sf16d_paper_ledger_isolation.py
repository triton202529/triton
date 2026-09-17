"""Phase 198T-SF16D — pre-plan SELL paper-ledger root isolation.

Isolated worktree only. Fake broker. TEMP store. No production writer,
network, TRITON start, or authority arming against production.
"""

from __future__ import annotations

import builtins
import csv
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

_PROD = Path(r"C:\Users\akimw\triton")
_REPO = Path(__file__).resolve().parents[1]
if _REPO.resolve() == _PROD.resolve():
    raise RuntimeError("SF16D tests must not run in the production repository.")
if "isolated" not in str(_REPO).lower():
    raise RuntimeError("SF16D tests must run from an isolated worktree.")

from services.manage_positions_cycle_binding import selected_row_hash, sha256_file_bytes
from services.paper_order_ledger import LEDGER_RELATIVE, load_events
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
)
from services.paper_sell_arm_cap_slice import slice_reviewed_sells
from services.paper_sell_dispatch_session_evidence import write_dispatch_session_evidence
from services.place_live_orders import (
    CANONICAL_PRODUCTION_ROOT,
    IsolatedPaperLedgerRootError,
    ROOT,
    dispatch_unused_authorized_sell_slice_pre_plan,
    resolve_pre_plan_paper_ledger_root,
)
from tests._phase_198i_sell_fakes import (
    NetworkFakeAlpacaBroker,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    SESSION_OPEN_NOW,
    isolate_sell_store,
    readiness_packet,
    stub_unrelated_pre_gates,
)

OPEN_CAL = {
    "session_state": "OPEN",
    "reason": "regular_session_open",
    "session_date": "2026-08-11",
    "session_open_et": "2026-08-11T09:30:00-04:00",
    "session_close_et": "2026-08-11T16:00:00-04:00",
    "source": "services.market_calendar.regular_session_state",
    "as_of_utc": "2026-08-11T16:00:00Z",
    "as_of_et": "2026-08-11T12:00:00-04:00",
    "calendar": "XNYS",
}

_FORBIDDEN_HIT: List[str] = []


def _forbidden(path: Path) -> bool:
    text = str(path.resolve()).lower()
    return text.startswith(str((_PROD / "data" / "results").resolve()).lower()) or text.startswith(
        str((_PROD / "data" / "governance").resolve()).lower()
    )


@pytest.fixture(autouse=True)
def _block_production_data(monkeypatch):
    _FORBIDDEN_HIT.clear()
    real_open = builtins.open

    def guarded(file, mode="r", *args, **kwargs):
        try:
            path = Path(file)
        except TypeError:
            return real_open(file, mode, *args, **kwargs)
        writing = any(flag in str(mode) for flag in ("w", "a", "x", "+"))
        if _forbidden(path) and writing:
            _FORBIDDEN_HIT.append(f"{mode}:{path}")
            raise RuntimeError(f"SF16D blocked production path {mode} {path}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    yield
    if _FORBIDDEN_HIT:
        raise RuntimeError(f"production path touch: {_FORBIDDEN_HIT}")


def _arm_dia(tmp_path: Path, monkeypatch, *, auth_symbol: str = "DIA") -> Dict[str, Any]:
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", "mo-sf16d-isolated")
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    now = SESSION_OPEN_NOW
    (tmp_path / "manage_positions_plan.csv").write_text(
        "symbol,final_action,qty,selected_for_execution,reason_code\nDIA,EXIT,1,true,FORCE_EXIT\n",
        encoding="utf-8",
    )
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(),
                "mode": "paper",
                "rows": [
                    {
                        "symbol": "DIA",
                        "final_action": "EXIT",
                        "qty": 1.0,
                        "selected_for_execution": "true",
                        "reason_code": "FORCE_EXIT",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan_sha = sha256_file_bytes(tmp_path / "manage_positions_plan.csv")
    rows = list(
        csv.DictReader(
            (tmp_path / "manage_positions_plan.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    pkt = readiness_packet(symbol=auth_symbol, qty=1.0, price=517.779, now=now)
    pkt["generated_at"] = now.isoformat()
    pkt["valid_until"] = (now + timedelta(minutes=25)).isoformat()
    pkt["plan_hash"] = plan_sha
    pkt["source_plan_sha256"] = plan_sha
    pkt["selected_row_hash"] = selected_row_hash(rows)
    pkt["trading_date"] = now.date().isoformat()
    pkt["market_open_invocation_id"] = "mo-sf16d-isolated"
    pkt["invocation_id"] = "mo-sf16d-isolated"
    slice_doc = slice_reviewed_sells(pkt, cap=APPROVAL_CLI_CAP)
    (tmp_path / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (now + timedelta(hours=4)).isoformat(),
                "allow_execute": True,
            }
        ),
        encoding="utf-8",
    )
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=pkt,
        account_raw="acct-sf16d",
        trading_date=now.date().isoformat(),
        now=now,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        explicit_runtime_write=True,
        authorized_slice=slice_doc,
    )
    confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=pkt,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=pkt["plan_hash"],
        now=now,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        explicit_runtime_write=True,
        authorized_slice=slice_doc,
    )
    (tmp_path / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(pkt), encoding="utf-8"
    )
    (tmp_path / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available,current_price\nDIA,1,1,517.779\n",
        encoding="utf-8",
    )
    (tmp_path / "open_orders_snapshot.csv").write_text("symbol,side,qty\n", encoding="utf-8")
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=now,
        invocation_id="mo-sf16d-isolated",
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    return {"now": now, "pkt": pkt}


def _fake_broker() -> NetworkFakeAlpacaBroker:
    return NetworkFakeAlpacaBroker(
        mode="paper",
        positions=[
            {"symbol": "DIA", "qty": "1", "qty_available": "1", "avg_entry_price": "464.90"}
        ],
        quote_bid=517.0,
        quote_ask=517.779,
    )


def test_default_root_is_module_repo_root():
    assert resolve_pre_plan_paper_ledger_root() == ROOT
    assert (
        resolve_pre_plan_paper_ledger_root()
        == Path(__import__("services.place_live_orders", fromlist=["x"]).__file__)
        .resolve()
        .parents[1]
    )
    assert resolve_pre_plan_paper_ledger_root(store_root=ROOT / "data" / "governance") == ROOT
    assert (
        resolve_pre_plan_paper_ledger_root(
            store_root=CANONICAL_PRODUCTION_ROOT / "data" / "governance"
        )
        == ROOT
    )


def test_explicit_temp_store_is_not_production_root(tmp_path):
    resolved = resolve_pre_plan_paper_ledger_root(store_root=tmp_path)
    assert resolved == tmp_path.resolve()
    assert resolved != _PROD.resolve()
    assert resolved != ROOT.resolve()


def test_missing_or_production_data_root_fails_closed():
    with pytest.raises(IsolatedPaperLedgerRootError) as missing:
        resolve_pre_plan_paper_ledger_root(store_root=_REPO / "missing-sf16d-root")
    assert missing.value.reason == "isolated_ledger_root_unavailable"
    with pytest.raises(IsolatedPaperLedgerRootError) as inside:
        resolve_pre_plan_paper_ledger_root(store_root=_PROD / "data" / "results")
    assert inside.value.reason == "isolated_ledger_root_inside_production_data"


def test_missing_root_dispatch_does_not_submit(tmp_path, monkeypatch):
    _arm_dia(tmp_path, monkeypatch)
    fake = _fake_broker()
    missing = tmp_path / "not-created"
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=missing,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=SESSION_OPEN_NOW,
    )
    assert result.get("attempted") is False
    assert result.get("dispatched") is False
    assert result.get("consumed") is False
    assert result.get("reason") == "isolated_ledger_root_unavailable"
    assert fake.fake_broker_call_count == 0
    assert not list(getattr(fake, "submit_payloads", []) or [])


def test_production_results_root_as_store_does_not_submit(tmp_path, monkeypatch):
    _arm_dia(tmp_path, monkeypatch)
    fake = _fake_broker()
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=_PROD / "data" / "results",
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=SESSION_OPEN_NOW,
    )
    assert result.get("dispatched") is False
    assert result.get("reason") == "isolated_ledger_root_inside_production_data"
    assert fake.fake_broker_call_count == 0


def test_scoped_unused_writes_temp_ledger_not_production(tmp_path, monkeypatch):
    _arm_dia(tmp_path, monkeypatch, auth_symbol="DIA")
    fake = _fake_broker()
    before = (_PROD / "data" / "results" / "paper_order_events.jsonl").read_bytes()
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=SESSION_OPEN_NOW,
        verbose=True,
    )
    after = (_PROD / "data" / "results" / "paper_order_events.jsonl").read_bytes()
    assert result.get("dispatched") is True
    assert result.get("paper_order_ledger_root") == str(tmp_path.resolve())
    assert fake.fake_broker_call_count == 1
    payloads = list(getattr(fake, "submit_payloads", []) or [])
    assert payloads and payloads[0]["symbol"] == "DIA"
    ledger = tmp_path / LEDGER_RELATIVE
    assert ledger.is_file()
    events = load_events(ledger)
    assert events
    assert all(e.get("event_type") == "ACTION_SUBMITTED" for e in events)
    assert all(e.get("symbol") == "DIA" for e in events)
    assert all(e.get("broker_order_id") == "fake-ord-1" for e in events)
    assert after == before
    assert not (_PROD / "data" / "results" / "paper_order_events.jsonl").samefile(ledger)


def test_wrong_symbol_zero_submits(tmp_path, monkeypatch):
    _arm_dia(tmp_path, monkeypatch, auth_symbol="HD")
    fake = _fake_broker()
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=SESSION_OPEN_NOW,
    )
    assert result.get("dispatched") is False
    assert fake.fake_broker_call_count == 0
    assert not (tmp_path / LEDGER_RELATIVE).is_file()


def test_consumed_zero_submits(tmp_path, monkeypatch):
    _arm_dia(tmp_path, monkeypatch, auth_symbol="DIA")
    fake = _fake_broker()
    first = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=SESSION_OPEN_NOW,
    )
    assert first.get("dispatched") is True
    replay_broker = _fake_broker()
    replay = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=replay_broker,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=SESSION_OPEN_NOW,
    )
    assert replay.get("dispatched") is False
    assert replay_broker.fake_broker_call_count == 0
    assert replay.get("reason") in {
        "no_unused_authority",
        "authority_not_unused",
        "status_not_armed",
    }


def test_expired_zero_submits(tmp_path, monkeypatch):
    _arm_dia(tmp_path, monkeypatch, auth_symbol="DIA")
    fake = _fake_broker()
    later = SESSION_OPEN_NOW + timedelta(hours=5)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=later,
    )
    assert result.get("dispatched") is False
    assert fake.fake_broker_call_count == 0
    assert not (tmp_path / LEDGER_RELATIVE).is_file()
