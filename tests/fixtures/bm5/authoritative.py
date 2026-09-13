"""Authoritative BM5 frozen copies. Isolated writes only. Never writes production."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

EXPECTED_LEDGER_SHA256 = "6ce03a05bf553eb8a9dce5938f964f2a24dd1f1e4ee1437e5f9f87d4cc6e9aee"
EXPECTED_LEDGER_EVENTS = 25
EXPECTED_BROKER_ORDERS = 12
EXPECTED_FILLED = 12
EXPECTED_OPEN = 0
EXPECTED_SKIPPED = 1
EXPECTED_FAILED = 0

GE_AUTHORITY_ID = "sell-arm-audit-0fd67b6383de"
GE_DISPATCH_ID = "sell-dispatch-5135a17d7fe74121"
GE_BROKER_ORDER_ID = "32e66fe5-abea-4807-96ea-9abb81b243ea"
GE_CLIENT_ORDER_ID = "triton-preplan-GE-EXIT-09b6490e74"
GE_SYMBOL = "GE"
GE_SIDE = "SELL"
GE_QTY = 8
GE_FILL_PRICE = 338.4
GE_BROKER_SUBMITTED = "2026-09-08T15:18:03.782Z"
GE_BROKER_FILLED = "2026-09-08T15:18:04.610Z"

SAME_CYCLE = "UNAVAILABLE_SAME_CYCLE_DISPATCH_RECONCILIATION"

PROD_ROOT = Path(r"C:\Users\akimw\triton")
ISO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
LEDGER_RELATIVE = "data/results/paper_order_events.jsonl"
CAPTURE_RELATIVE = "tmp/preview_capture.json"

SNAPSHOT_RELS: tuple[str, ...] = (
    LEDGER_RELATIVE,
    "data/results/paper_execution_audit.csv",
    "data/results/portfolio_history.csv",
    "data/results/positions_snapshot.csv",
    "data/results/performance_risk_overlay.csv",
    "data/results/signal_lifecycle.csv",
    "data/results/heartbeat.json",
    "data/results/data_freshness_refresh_audit.json",
    "data/results/watchdog_status.json",
    "data/results/execution_readiness.json",
    "data/results/defensive_exit_approval_readiness.json",
    "data/results/open_orders_snapshot.csv",
    "data/results/recent_orders.csv",
    "data/results/manage_positions_pre_plan_dispatch.json",
    "data/live/paper_sell_arm.sqlite3",
    "data/live/paper_arm_confirm.json",
    "data/governance/paper_sell_arm_authority.json",
    "data/governance/paper_sell_arm_state.json",
    "data/governance/paper_sell_arm_consumption.json",
    "data/governance/paper_sell_arm_audit.jsonl",
    "data/governance/paper_sell_arm_dispatch_permit.json",
    "data/governance/paper_sell_arm_dispatch_intent.json",
    "data/governance/paper_sell_confirmation_challenge.json",
)


class AuthoritativeLedgerError(RuntimeError):
    """Fail-closed: authoritative ledger missing, mismatched, or replaced."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_ledger_path() -> Path:
    return FROZEN_DIR / LEDGER_RELATIVE


def require_frozen_ledger() -> Path:
    path = frozen_ledger_path()
    if not path.is_file():
        raise AuthoritativeLedgerError("frozen_ledger_missing")
    digest = file_sha256(path)
    if digest != EXPECTED_LEDGER_SHA256:
        raise AuthoritativeLedgerError(f"frozen_ledger_hash_mismatch:{digest}")
    return path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def materialize_authoritative_fixture(root: Path) -> Dict[str, Any]:
    """Copy frozen authoritative bytes into root. Never generates synthetic orders."""
    root = Path(root)
    if root.resolve() == PROD_ROOT.resolve():
        raise AuthoritativeLedgerError("refusing_production_materialize")
    ledger_src = require_frozen_ledger()
    copied: List[str] = []
    for rel in SNAPSHOT_RELS:
        src = FROZEN_DIR / rel
        if not src.is_file():
            if rel == LEDGER_RELATIVE:
                raise AuthoritativeLedgerError("frozen_ledger_missing")
            continue
        copy_file(src, root / rel)
        copied.append(rel)
    dest_ledger = root / LEDGER_RELATIVE
    digest = file_sha256(dest_ledger)
    if digest != EXPECTED_LEDGER_SHA256:
        raise AuthoritativeLedgerError(f"materialized_ledger_hash_mismatch:{digest}")
    if dest_ledger.resolve() == ledger_src.resolve() and root.resolve() != FROZEN_DIR.resolve():
        raise AuthoritativeLedgerError("materialize_did_not_copy")
    return {
        "root": str(root),
        "ledger": str(dest_ledger),
        "ledger_sha256": digest,
        "copied": copied,
    }


def write_preview_capture(root: Path, *, captured_at: Optional[str] = None) -> Dict[str, Any]:
    captured = captured_at or datetime.now(timezone.utc).isoformat()
    payload = {
        "preview_data_captured_at": captured,
        "source": "production_read_only_after_bm5_hash_proof",
        "ledger_sha256": EXPECTED_LEDGER_SHA256,
        "ledger_events": EXPECTED_LEDGER_EVENTS,
        "production_root": str(PROD_ROOT),
        "isolated_root": str(root),
    }
    path = root / CAPTURE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def preview_capture_label(captured_at: str) -> str:
    text = str(captured_at or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(ZoneInfo("America/New_York"))
        stamp = local.strftime("%Y-%m-%d %H:%M:%S ET")
    except ValueError:
        stamp = captured_at
    return f"Preview data captured at {stamp}"


def displayed_order_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
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


def ledger_event_keys(event: Mapping[str, Any]) -> tuple[str, ...]:
    keys = []
    if event.get("broker_order_id"):
        keys.append(f"broker:{event['broker_order_id']}")
    if event.get("client_order_id"):
        keys.append(f"client:{event['client_order_id']}")
    keys.append(
        "tuple:"
        f"{event.get('authority_id')}|{event.get('dispatch_id')}|{event.get('symbol')}|"
        f"{event.get('side')}|{event.get('action_disposition')}|{event.get('selected_qty')}"
    )
    return tuple(keys)


def assert_rows_derived_from_events(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> None:
    event_keys = {key for event in events for key in ledger_event_keys(event)}
    if not rows:
        raise AuthoritativeLedgerError("no_displayed_orders")
    for row in rows:
        if not any(key in event_keys for key in displayed_order_keys(row)):
            raise AuthoritativeLedgerError(
                f"displayed_order_not_in_ledger:{row.get('symbol')}:{row.get('broker_order_id')}"
            )


def sha256_map(root: Path, rels: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for rel in rels:
        path = Path(root) / rel
        out[rel] = file_sha256(path) if path.is_file() else None
    return out
