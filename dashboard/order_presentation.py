"""Presentation helpers for Orders counters and historical plans.

Read-only. Does not write the ledger, contact a broker, or change authority.
Displayed broker orders come only from the authoritative ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from services.paper_order_ledger import (
    SOURCE_FPLAN,
    canonical_symbol,
    dispatch_visibility,
    ledger_path,
    load_events,
    money,
    reduce_ledger,
)


class AuthoritativeLedgerError(RuntimeError):
    """Fail-closed: displayed orders are missing the authoritative ledger."""


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PLAN_SIDES = frozenset({"PLAN", "PLANNED", "FPLAN", "PLANNING", "HISTORICAL_PLAN"})
HISTORICAL_PLAN_SOURCES = frozenset(
    {SOURCE_FPLAN, "execution_plan", "manage_positions_plan", "historical_plan"}
)


def is_historical_plan_row(row: Mapping[str, Any]) -> bool:
    """True for legacy planning records that are not broker orders."""
    if str(row.get("kind") or "") == "plan":
        return True
    side = str(row.get("side") or "").upper()
    if side in PLAN_SIDES:
        return True
    source = str(row.get("source_pipeline") or "")
    if source in HISTORICAL_PLAN_SOURCES:
        return True
    disposition = str(row.get("action_disposition") or "").strip().lower()
    if disposition in {"planned", "plan", "authorized"} and not row.get("broker_order_id"):
        return True
    return False


def _legacy_row_is_plan(row: Mapping[str, Any]) -> bool:
    action = str(row.get("action") or row.get("signal") or "").strip().upper()
    status = str(row.get("status") or "").strip().upper()
    return action in PLAN_SIDES or status in PLAN_SIDES


def load_historical_plan_rows(*, root: Path) -> List[Dict[str, Any]]:
    """Read-only historical plans from the frozen audit. Not broker orders."""
    import csv

    audit = Path(root) / "data" / "results" / "paper_execution_audit.csv"
    if not audit.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with audit.open(encoding="utf-8", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            if not _legacy_row_is_plan(row):
                continue
            out.append(
                {
                    "logical_key": f"plan:{idx}:{row.get('symbol') or ''}",
                    "kind": "plan",
                    "authority_id": None,
                    "dispatch_id": None,
                    "symbol": canonical_symbol(row.get("symbol")),
                    "side": str(row.get("action") or row.get("signal") or "").upper() or None,
                    "trading_date": str(row.get("timestamp") or "")[:10],
                    "source_pipeline": SOURCE_FPLAN,
                    "selected_qty": money(row.get("qty")),
                    "submission_status": None,
                    "terminal_status": None,
                    "terminal_status_proven": False,
                    "broker_order_id": None,
                    "client_order_id": None,
                    "filled_qty": None,
                    "average_fill_price": None,
                    "proceeds": None,
                    "event_timestamp": row.get("timestamp"),
                    "fees_commissions": "UNPROVEN",
                    "source_artifact": str(audit),
                    "action_disposition": "planned",
                }
            )
    return out


def split_unified_order_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate broker/action rows from legacy planning records."""
    broker: List[Dict[str, Any]] = []
    plans: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if is_historical_plan_row(item):
            plans.append(item)
        else:
            broker.append(item)
    return broker, plans


def presentation_order_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Operator-facing counters. Planning records are never broker orders."""
    broker, plans = split_unified_order_rows(rows)
    vis = dispatch_visibility(broker)
    skipped = [r for r in broker if r.get("action_disposition") == "explicitly_skipped"]
    submitted = [
        r
        for r in broker
        if r.get("action_disposition") == "submitted"
        or (r.get("kind") == "order" and r.get("broker_order_id"))
    ]
    filled = [
        r
        for r in submitted
        if r.get("terminal_status") == "FILLED" and r.get("terminal_status_proven")
    ]
    open_rows = [r for r in submitted if not r.get("terminal_status_proven")]
    return {
        "broker_orders": len(submitted),
        "filled": len(filled),
        "open": len(open_rows),
        "skipped": len(skipped),
        "failed": len([r for r in broker if r.get("action_disposition") == "failed"]),
        "rejected": len(
            [
                r
                for r in submitted
                if str(r.get("terminal_status") or "") in {"REJECTED", "CANCELLED", "EXPIRED"}
            ]
        ),
        "failed_or_rejected": int(vis.get("failed") or 0) + int(vis.get("rejected") or 0),
        "historical_plans": len(plans),
        "visibility": vis,
        "broker_rows": broker,
        "plan_rows": plans,
    }


def display_side(value: Any) -> str:
    """Never present Plan as an order side."""
    text = str(value or "").strip()
    if not text:
        return "Unavailable"
    if text.upper() in PLAN_SIDES or text.upper() == "PLAN":
        return "Historical plan"
    from dashboard.display_labels import display_status

    return display_status(text)


def plan_table_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    from dashboard.display_labels import display_status, quantity_text
    from dashboard import operator_ui as oui

    return {
        "Symbol": row.get("symbol") or "Unavailable",
        "Record type": "Historical plan",
        "Quantity": quantity_text(row.get("selected_qty") or row.get("qty")),
        "Recorded status": display_status(
            row.get("action_disposition") or row.get("submission_status") or "planned"
        ),
        "Time": oui.format_timestamp(
            row.get("event_timestamp") or row.get("last_event_timestamp") or row.get("trading_date")
        ),
    }


def _row_match_keys(row: Mapping[str, Any]) -> Tuple[str, ...]:
    keys = []
    if row.get("broker_order_id"):
        keys.append(f"broker:{row['broker_order_id']}")
    if row.get("client_order_id"):
        keys.append(f"client:{row['client_order_id']}")
    if row.get("logical_key"):
        keys.append(f"logical:{row['logical_key']}")
    keys.append(
        "tuple:"
        f"{row.get('authority_id')}|{row.get('dispatch_id')}|{row.get('symbol')}|"
        f"{row.get('side')}|{row.get('action_disposition')}|{row.get('selected_qty')}"
    )
    return tuple(keys)


def assert_rows_derived_from_ledger_events(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> None:
    event_keys = {key for event in events for key in _row_match_keys(event)}
    if not rows:
        raise AuthoritativeLedgerError("no_displayed_orders")
    for row in rows:
        if not any(key in event_keys for key in _row_match_keys(row)):
            raise AuthoritativeLedgerError(
                f"displayed_order_not_in_ledger:{row.get('symbol')}:{row.get('broker_order_id')}"
            )


def load_authoritative_presentation(
    *,
    root: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load Orders presentation from the copied ledger only.

    Historical plans may come from the frozen audit. Broker orders never do.
    A missing or mismatched ledger fails closed instead of inventing rows.
    """
    root_p = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    ledger = ledger_path(root_p)
    if not ledger.is_file():
        raise AuthoritativeLedgerError("authoritative_ledger_missing")
    digest = file_sha256(ledger)
    if expected_sha256 and digest != expected_sha256:
        raise AuthoritativeLedgerError(f"authoritative_ledger_hash_mismatch:{digest}")
    events = load_events(ledger)
    broker = [dict(row) for row in reduce_ledger(root=root_p) if not is_historical_plan_row(row)]
    assert_rows_derived_from_ledger_events(broker, events)
    plans = load_historical_plan_rows(root=root_p)
    counts = presentation_order_counts(broker + plans)
    counts["ledger_sha256"] = digest
    counts["ledger_events"] = len(events)
    counts["ledger_path"] = str(ledger)
    return counts
