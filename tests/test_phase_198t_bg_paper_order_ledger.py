"""Phase 198T-BG — unified paper-order ledger boundary and September 4 replay."""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.paper_order_ledger import (
    EVENT_ACTION_FAILED,
    EVENT_ACTION_SKIPPED,
    EVENT_ACTION_SUBMITTED,
    EVENT_ORDER_TERMINAL_RECONCILED,
    LEDGER_RELATIVE,
    LOCK_RELATIVE,
    PaperOrderLedgerError,
    SCHEMA_VERSION,
    append_events,
    build_event,
    consumed_authority_message,
    dispatch_visibility,
    filter_unified_rows,
    load_events,
    merge_activity_payload,
    merge_legacy_audit_rows,
    proceeds_summary,
    reconcile_from_recent_orders,
    record_sell_dispatch_events,
    reduce_events,
    reduce_ledger,
    safe_record_sell_dispatch,
    unified_order_rows,
)

ROOT = Path(__file__).resolve().parents[1]
BF = Path(r"C:\Users\akimw\AppData\Local\Temp\phase_198T_BF_20260904_195631")

pytestmark = pytest.mark.bn2a_isolation


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    (tmp_path / "data" / "results").mkdir(parents=True)
    return tmp_path


def _auth(aid: str, trading_date: str = "2026-09-04") -> dict:
    return {
        "authority_id": aid,
        "payload": {"trading_date": trading_date, "authority_id": aid},
    }


def _clip(symbol: str, qty: float, outcome: str = "UNCHANGED", **extra) -> dict:
    row = {
        "symbol": symbol,
        "action": "EXIT",
        "authorized_quantity": qty if qty else extra.get("authorized_quantity", 1),
        "authorized_max_notional": extra.get("authorized_max_notional", 100.0),
        "current_reference_price": extra.get("price", 10.0),
        "runtime_quantity": qty,
        "runtime_estimated_notional": extra.get("notional", qty * extra.get("price", 10.0)),
        "outcome": outcome,
        "reason": extra.get("reason", "authorized_quantity_fits_current_price"),
    }
    row.update(extra)
    return row


def _path_meta(path: Path):
    if not path.exists():
        return None
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def test_import_has_no_production_side_effects(tmp_root: Path):
    """Import must not create a ledger. Assert against an empty TEMP root.

    Production already contains paper_order_events.jsonl from live paper
    operation. That environmental fact is not an import side effect, so the
    'ledger is absent' assertion uses the injectable TEMP fixture instead of
    the production path. Production mtime/size must stay unchanged.
    """
    import services.paper_order_ledger as mod

    isolated = tmp_root
    assert not (isolated / LEDGER_RELATIVE).is_file()
    assert not (isolated / LOCK_RELATIVE).is_file()

    production = Path(r"C:\Users\akimw\triton")
    prod_ledger = production / LEDGER_RELATIVE
    prod_lock = production / LOCK_RELATIVE
    before_ledger = _path_meta(prod_ledger)
    before_lock = _path_meta(prod_lock)

    assert mod.ledger_path(isolated) == isolated / LEDGER_RELATIVE
    assert mod.lock_path(isolated) == isolated / LOCK_RELATIVE
    assert not (isolated / LEDGER_RELATIVE).is_file()
    assert not (isolated / LOCK_RELATIVE).is_file()

    before = (isolated / LEDGER_RELATIVE).is_file()
    assert (isolated / LEDGER_RELATIVE).is_file() is before
    assert _path_meta(prod_ledger) == before_ledger
    assert _path_meta(prod_lock) == before_lock
    assert "alpaca" not in Path(mod.__file__).read_text(encoding="utf-8").lower()


def test_duplicate_snapshot_reconciliation(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="sell-dispatch-a",
        clip_audit=[_clip("AAA", 2)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 2,
                "client_order_id": "c1",
                "broker_order_id": "b1",
                "broker_status": "submitted",
            }
        ],
        root=tmp_root,
    )
    recent = tmp_root / "data" / "results" / "recent_orders.csv"
    recent.write_text(
        "id,client_order_id,symbol,side,qty,filled_qty,filled_avg_price,status,submitted_at,filled_at,snapshot_ts\n"
        "b1,c1,AAA,sell,2,2,10.5,filled,2026-09-04T14:00:00Z,2026-09-04T14:00:01Z,2026-09-04T23:00:00Z\n",
        encoding="utf-8",
    )
    first = reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    second = reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    assert first["appended"] == 1
    assert second["appended"] == 0
    rows = reduce_ledger(root=tmp_root)
    filled = [r for r in rows if r.get("terminal_status") == "FILLED"]
    assert len(filled) == 1
    assert filled[0]["proceeds"] == 21.0


def test_duplicate_dispatch_invocation(tmp_root: Path):
    kwargs = dict(
        authority=_auth("sell-arm-a"),
        dispatch_id="sell-dispatch-a",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    a = record_sell_dispatch_events(**kwargs)
    b = record_sell_dispatch_events(**kwargs)
    assert a["appended"] == 1
    assert b["appended"] == 0
    assert b["skipped_duplicate"] == 1


def test_partial_then_full_fill(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 10)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 10,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    ev = build_event(
        event_type="ORDER_STATUS_RECONCILED",
        source_pipeline="recent_orders_snapshot",
        authority_id="sell-arm-a",
        dispatch_id="d1",
        action_ordinal=1,
        symbol="AAA",
        side="SELL",
        client_order_id="c1",
        broker_order_id="b1",
        broker_submission_status="PARTIALLY_FILLED",
        filled_qty=4,
        average_fill_price=10.0,
        terminal_status_proven=False,
    )
    append_events([ev], root=tmp_root)
    ev2 = build_event(
        event_type=EVENT_ORDER_TERMINAL_RECONCILED,
        source_pipeline="recent_orders_snapshot",
        authority_id="sell-arm-a",
        dispatch_id="d1",
        action_ordinal=1,
        symbol="AAA",
        side="SELL",
        client_order_id="c1",
        broker_order_id="b1",
        broker_terminal_status="FILLED",
        filled_qty=10,
        average_fill_price=10.2,
        filled_timestamp="2026-09-04T15:00:00Z",
        terminal_status_proven=True,
    )
    append_events([ev2], root=tmp_root)
    row = reduce_ledger(root=tmp_root)[0]
    assert row["terminal_status"] == "FILLED"
    assert row["proceeds"] == 102.0
    assert row["authority_id"] == "sell-arm-a"


def test_submitted_without_terminal(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    row = reduce_ledger(root=tmp_root)[0]
    assert row["action_disposition"] == "submitted"
    assert row["terminal_status"] is None
    assert row["proceeds"] is None
    assert row["terminal_status_proven"] is False


def test_missing_broker_order_id_still_visible(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c-only",
                "broker_order_id": None,
            }
        ],
        root=tmp_root,
    )
    row = reduce_ledger(root=tmp_root)[0]
    assert row["logical_key"] == "client:c-only"
    assert row["broker_order_id"] is None


def test_client_id_then_broker_correlation(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": None,
            }
        ],
        root=tmp_root,
    )
    recent = tmp_root / "data" / "results" / "recent_orders.csv"
    recent.write_text(
        "id,client_order_id,symbol,side,qty,filled_qty,filled_avg_price,status,submitted_at,filled_at\n"
        "broker-1,c1,AAA,sell,1,1,9.5,filled,2026-09-04T14:00:00Z,2026-09-04T14:00:01Z\n",
        encoding="utf-8",
    )
    reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    rows = reduce_ledger(root=tmp_root)
    assert len(rows) == 1
    assert rows[0]["broker_order_id"] == "broker-1"
    assert rows[0]["terminal_status"] == "FILLED"
    assert rows[0]["proceeds"] == 9.5


def test_mismatched_broker_client_does_not_weak_match(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    recent = tmp_root / "data" / "results" / "recent_orders.csv"
    recent.write_text(
        "id,client_order_id,symbol,side,qty,filled_qty,filled_avg_price,status,submitted_at,filled_at\n"
        "other,other-c,AAA,sell,1,1,9.5,filled,2026-09-04T14:00:00Z,2026-09-04T14:00:01Z\n",
        encoding="utf-8",
    )
    result = reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    assert result["appended"] == 0
    assert reduce_ledger(root=tmp_root)[0]["terminal_status"] is None


@pytest.mark.parametrize(
    "status,event_status",
    [
        ("rejected", "REJECTED"),
        ("canceled", "CANCELLED"),
        ("expired", "EXPIRED"),
        ("replaced", "REPLACED"),
    ],
)
def test_terminal_non_fill_statuses(tmp_root: Path, status: str, event_status: str):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    recent = tmp_root / "data" / "results" / "recent_orders.csv"
    recent.write_text(
        "id,client_order_id,symbol,side,qty,filled_qty,filled_avg_price,status,submitted_at,filled_at\n"
        f"b1,c1,AAA,sell,1,0,,{status},2026-09-04T14:00:00Z,\n",
        encoding="utf-8",
    )
    reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    row = reduce_ledger(root=tmp_root)[0]
    assert row["terminal_status"] == event_status
    assert row["proceeds"] is None


def test_malformed_ledger_line_fail_closed(tmp_root: Path):
    path = tmp_root / LEDGER_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(PaperOrderLedgerError) as exc:
        load_events(path)
    assert exc.value.reason == "ledger_malformed_line"


def test_interrupted_append_fail_closed(tmp_root: Path):
    path = tmp_root / LEDGER_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"event_id":"x"', encoding="utf-8")
    with pytest.raises(PaperOrderLedgerError) as exc:
        load_events(path)
    assert exc.value.reason == "ledger_incomplete_final_line"


def test_concurrent_append(tmp_root: Path):
    ev1 = build_event(
        event_type=EVENT_ACTION_SUBMITTED,
        source_pipeline="preplan_authorized_sell",
        dispatch_id="d1",
        action_ordinal=1,
        symbol="AAA",
        side="SELL",
        broker_order_id="b-conc-1",
        action_disposition="submitted",
    )
    ev2 = build_event(
        event_type=EVENT_ACTION_SUBMITTED,
        source_pipeline="preplan_authorized_sell",
        dispatch_id="d1",
        action_ordinal=2,
        symbol="BBB",
        side="SELL",
        broker_order_id="b-conc-2",
        action_disposition="submitted",
    )
    errors = []

    def _go(ev):
        try:
            append_events([ev], root=tmp_root)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_go, args=(ev1,))
    t2 = threading.Thread(target=_go, args=(ev2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    assert len(load_events(tmp_root / LEDGER_RELATIVE)) == 2


def test_ledger_failure_after_broker_does_not_replay(tmp_root: Path):
    path = tmp_root / LEDGER_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad}\n", encoding="utf-8")
    result = safe_record_sell_dispatch(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    assert result["ok"] is False
    assert result["replay_forbidden"] is True
    assert result["authority_reusable"] is False


def test_filled_does_not_regress_to_new():
    events = [
        build_event(
            event_type=EVENT_ACTION_SUBMITTED,
            source_pipeline="preplan_authorized_sell",
            dispatch_id="d1",
            action_ordinal=1,
            symbol="AAA",
            side="SELL",
            broker_order_id="b1",
            broker_submission_status="SUBMITTED",
            action_disposition="submitted",
        ),
        build_event(
            event_type=EVENT_ORDER_TERMINAL_RECONCILED,
            source_pipeline="recent_orders_snapshot",
            dispatch_id="d1",
            action_ordinal=1,
            symbol="AAA",
            side="SELL",
            broker_order_id="b1",
            broker_terminal_status="FILLED",
            filled_qty=1,
            average_fill_price=10,
            terminal_status_proven=True,
        ),
        build_event(
            event_type="ORDER_STATUS_RECONCILED",
            source_pipeline="recent_orders_snapshot",
            dispatch_id="d1",
            action_ordinal=1,
            symbol="AAA",
            side="SELL",
            broker_order_id="b1",
            broker_submission_status="NEW",
            terminal_status_proven=False,
        ),
    ]
    row = reduce_events(events)[0]
    assert row["terminal_status"] == "FILLED"
    assert row["invalid_regression"] == "FILLED->SUBMITTED"


def test_consumed_is_not_filled():
    ev = build_event(
        event_type=EVENT_ACTION_SUBMITTED,
        source_pipeline="preplan_authorized_sell",
        dispatch_id="d1",
        action_ordinal=1,
        symbol="AAA",
        side="SELL",
        broker_order_id="b1",
        extra={"authority_state": "CONSUMED"},
        action_disposition="submitted",
    )
    row = reduce_events([ev])[0]
    assert row["terminal_status"] is None
    assert row["proceeds"] is None


def test_skipped_and_failed_visibility_and_no_proceeds(tmp_root: Path):
    result = record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[
            _clip("AAA", 1),
            _clip(
                "BBB",
                0,
                outcome="SKIPPED_ZERO",
                reason="clipped_to_zero_vs_authorized_max_notional",
            ),
            _clip("CCC", 1),
        ],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        failures=[{"symbol": "CCC", "reason": "broker_submit_exception"}],
        root=tmp_root,
    )
    assert result["submitted"] == 1
    assert result["skipped"] == 1
    assert result["failed"] == 1
    rows = reduce_ledger(root=tmp_root)
    assert {r["action_disposition"] for r in rows} == {
        "submitted",
        "explicitly_skipped",
        "failed",
    }
    assert all(
        r.get("proceeds") in (None, 0) for r in rows if r["action_disposition"] != "submitted"
    )


def test_no_duplicate_proceeds_on_report_repeat(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 2, price=10.0, notional=20.0)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 2,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    recent = tmp_root / "data" / "results" / "recent_orders.csv"
    recent.write_text(
        "id,client_order_id,symbol,side,qty,filled_qty,filled_avg_price,status,submitted_at,filled_at\n"
        "b1,c1,AAA,sell,2,2,10,filled,2026-09-04T14:00:00Z,2026-09-04T14:00:01Z\n",
        encoding="utf-8",
    )
    reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    a = proceeds_summary(root=tmp_root)
    b = proceeds_summary(root=tmp_root)
    assert a["filled_proceeds"] == b["filled_proceeds"] == 20.0
    assert a["realized_pl"] is None


def test_mixed_buy_and_sell_no_duplicate(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "shared-id",
            }
        ],
        root=tmp_root,
    )
    audit = tmp_root / "data" / "results" / "paper_execution_audit.csv"
    audit.write_text(
        "timestamp,session,symbol,lifecycle_state,signal,action,qty,price,order_type,authorization_status,governance_authorized,execution_authorized,readiness,broker_mode,duplicate_check_result,submitted_order_id,status,skip_reason\n"
        "2026-08-14T18:53:33Z,s,HD,ADD,ADD,buy,4,340.41,limit,authorized,true,true,READY,paper,,shared-id,submitted,\n"
        "2026-08-14T18:53:33Z,s,HD,ADD,ADD,buy,4,340.41,limit,authorized,true,true,READY,paper,,buy-only,submitted,\n",
        encoding="utf-8",
    )
    rows = unified_order_rows(root=tmp_root)
    ids = [r.get("broker_order_id") for r in rows]
    assert ids.count("shared-id") == 1
    assert "buy-only" in ids


def test_same_symbol_two_batches_and_same_day(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-1"),
        dispatch_id="d1",
        clip_audit=[_clip("BITO", 3)],
        submissions=[
            {
                "symbol": "BITO",
                "qty": 3,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    record_sell_dispatch_events(
        authority=_auth("sell-arm-2"),
        dispatch_id="d2",
        clip_audit=[_clip("BITO", 1)],
        submissions=[
            {
                "symbol": "BITO",
                "qty": 1,
                "client_order_id": "c2",
                "broker_order_id": "b2",
            }
        ],
        root=tmp_root,
    )
    rows = [r for r in reduce_ledger(root=tmp_root) if r.get("symbol") == "BITO"]
    assert len(rows) == 2
    assert {r["dispatch_id"] for r in rows} == {"d1", "d2"}


def test_activity_and_orders_filters_idempotent(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-a"),
        dispatch_id="d1",
        clip_audit=[_clip("AAA", 1), _clip("BBB", 0, outcome="SKIPPED_ZERO")],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            }
        ],
        root=tmp_root,
    )
    payload = merge_activity_payload(
        {"events": [], "event_count": 0, "sources_used": []}, root=tmp_root
    )
    again = merge_activity_payload(payload, root=tmp_root)
    ids = [e["event_id"] for e in again["events"]]
    assert len(ids) == len(set(ids))
    shown = filter_unified_rows(
        unified_order_rows(root=tmp_root), side="SELL", trading_date="2026-09-04"
    )
    assert shown


def test_stale_banner_suppressed_when_consumed_dispatch():
    vis = {
        "submitted": 4,
        "skipped": 1,
        "failed": 0,
        "filled": 0,
        "has_dispatch": True,
    }
    msg = consumed_authority_message(vis)
    assert "No orders were submitted" not in msg
    assert "4 submitted" in msg
    assert "1 skipped" in msg


def test_new_packet_not_mixed_with_prior_dispatch():
    vis = dispatch_visibility(
        [
            {
                "authority_id": "old",
                "dispatch_id": "d-old",
                "kind": "order",
                "action_disposition": "submitted",
                "broker_order_id": "x",
            }
        ],
        authority_id="new",
    )
    assert vis["has_dispatch"] is False
    assert vis["submitted"] == 0


def test_sept4_isolated_replay_three_batches(tmp_root: Path):
    assert BF.is_dir()
    auths = json.loads((BF / "sqlite" / "sept4_authorities.json").read_text(encoding="utf-8"))
    disps = json.loads((BF / "sqlite" / "sept4_dispatches.json").read_text(encoding="utf-8"))
    recent = json.loads(
        (BF / "local" / "recent_orders_sept4_sells.json").read_text(encoding="utf-8")
    )
    by_id = {r["id"]: r for r in recent}
    expected = {
        "sell-arm-audit-5317d118550d": (4, 4, 0, 0, 4564.91),
        "sell-arm-audit-5c5af8562c11": (3, 3, 0, 0, 4986.070005),
        "sell-arm-audit-ac41e32906fc": (5, 4, 1, 0, 4368.63),
    }
    for auth in sorted(auths, key=lambda a: a["created_at"]):
        disp = next(d for d in disps if d["authority_id"] == auth["authority_id"])
        clips = (disp.get("intent") or {}).get("clip_audit") or []
        broker_ids = [x for x in str(disp.get("broker_ref") or "").split(",") if x]
        submitted_syms = [c["symbol"] for c in clips if c.get("runtime_quantity")]
        submissions = []
        for sym, bid in zip(submitted_syms, broker_ids):
            rec = by_id[bid]
            submissions.append(
                {
                    "symbol": sym,
                    "qty": rec.get("qty"),
                    "client_order_id": rec.get("client_order_id"),
                    "broker_order_id": bid,
                    "broker_status": "submitted",
                }
            )
        result = record_sell_dispatch_events(
            authority=auth,
            dispatch_id=disp["dispatch_id"],
            clip_audit=clips,
            submissions=submissions,
            root=tmp_root,
        )
        exp = expected[auth["authority_id"]]
        assert (
            result["authorized"],
            result["submitted"],
            result["skipped"],
            result["failed"],
        ) == exp[:4]
    recent_path = tmp_root / "data" / "results" / "recent_orders.csv"
    with recent_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "client_order_id",
                "symbol",
                "side",
                "qty",
                "filled_qty",
                "filled_avg_price",
                "status",
                "submitted_at",
                "filled_at",
            ],
        )
        w.writeheader()
        for r in recent:
            w.writerow(
                {
                    "id": r.get("id"),
                    "client_order_id": r.get("client_order_id"),
                    "symbol": r.get("symbol"),
                    "side": r.get("side"),
                    "qty": r.get("qty"),
                    "filled_qty": r.get("filled_qty"),
                    "filled_avg_price": r.get("filled_avg_price"),
                    "status": r.get("status"),
                    "submitted_at": r.get("submitted_at"),
                    "filled_at": r.get("filled_at"),
                }
            )
    reconcile_from_recent_orders(root=tmp_root, recent_path=recent_path)
    reconcile_from_recent_orders(root=tmp_root, recent_path=recent_path)
    rows = reduce_ledger(root=tmp_root)
    orders = [r for r in rows if r.get("kind") == "order"]
    skipped = [r for r in rows if r.get("action_disposition") == "explicitly_skipped"]
    assert len(orders) == 11
    assert len(skipped) == 1
    assert skipped[0]["symbol"] == "ARKK"
    assert {r["symbol"] for r in orders} <= {
        "TSLA",
        "GLD",
        "ARKK",
        "BITO",
        "BRK.B",
        "BLK",
        "DIA",
    }
    assert all(r.get("side") == "SELL" for r in orders)
    assert all(r.get("terminal_status") == "FILLED" for r in orders)
    total = proceeds_summary(root=tmp_root, trading_date="2026-09-04")
    assert total["filled_count"] == 11
    assert abs(total["filled_proceeds"] - 13919.610005) < 1e-6
    by_auth = {}
    for r in orders:
        by_auth.setdefault(r["authority_id"], 0.0)
        by_auth[r["authority_id"]] += float(r["proceeds"])
    assert abs(by_auth["sell-arm-audit-5317d118550d"] - 4564.91) < 1e-6
    assert abs(by_auth["sell-arm-audit-5c5af8562c11"] - 4986.070005) < 1e-6
    assert abs(by_auth["sell-arm-audit-ac41e32906fc"] - 4368.63) < 1e-6
    assert (ROOT / LEDGER_RELATIVE).is_file() is False or ROOT.resolve() != Path(
        r"C:\Users\akimw\triton"
    )


def test_foreign_historical_intent_does_not_attach_to_new_dispatch(tmp_root: Path):
    record_sell_dispatch_events(
        authority=_auth("sell-arm-foreign"),
        dispatch_id="sell-dispatch-foreign",
        clip_audit=[_clip("AAA", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c-old",
                "broker_order_id": "b-old",
            }
        ],
        root=tmp_root,
    )
    record_sell_dispatch_events(
        authority=_auth("sell-arm-current"),
        dispatch_id="sell-dispatch-current",
        clip_audit=[_clip("BBB", 1)],
        submissions=[
            {
                "symbol": "BBB",
                "qty": 1,
                "client_order_id": "c-new",
                "broker_order_id": "b-new",
            }
        ],
        root=tmp_root,
    )
    vis = dispatch_visibility(reduce_ledger(root=tmp_root), authority_id="sell-arm-current")
    assert vis["submitted"] == 1
    assert "sell-dispatch-foreign" not in vis["dispatch_ids"]
    assert vis["dispatch_ids"] == ["sell-dispatch-current"]


def test_single_use_replay_does_not_duplicate_events(tmp_root: Path):
    kwargs = dict(
        authority=_auth("sell-arm-a"),
        dispatch_id="sell-dispatch-a",
        clip_audit=[_clip("AAA", 1), _clip("BBB", 1)],
        submissions=[
            {
                "symbol": "AAA",
                "qty": 1,
                "client_order_id": "c1",
                "broker_order_id": "b1",
            },
            {
                "symbol": "BBB",
                "qty": 1,
                "client_order_id": "c2",
                "broker_order_id": "b2",
            },
        ],
        root=tmp_root,
    )
    first = record_sell_dispatch_events(**kwargs)
    replay = record_sell_dispatch_events(**kwargs)
    assert first["appended"] == 2
    assert replay["appended"] == 0
    assert replay["skipped_duplicate"] == 2
    rows = [r for r in reduce_ledger(root=tmp_root) if r.get("kind") == "order"]
    assert len(rows) == 2


def test_schema_choice_and_version():
    assert LEDGER_RELATIVE == "data/results/paper_order_events.jsonl"
    assert SCHEMA_VERSION == "198T-BG-PAPER-ORDER-LEDGER-1"
