"""Durable NYSE-session entry budget.

Per-order and per-cycle caps stay in execute_trades. This budget is an
additional refusal keyed by the NYSE session date, not an invocation tag.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from services.symbol_identity import identity_key

SCHEMA = "SESSION-ENTRY-BUDGET-1"
DEFAULT_MAX_NEW_NAMES = 10
DEFAULT_MAX_ADDS = 3
OPEN_NEW = "OPEN_NEW"
ADD = "ADD"


class SessionBudgetError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def budget_path(root: Path, session_date: str) -> Path:
    return Path(root) / "data" / "live" / "session_entry_budget" / f"{session_date}.json"


def _empty(session_date: str) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "session_date": session_date,
        "max_new_names": DEFAULT_MAX_NEW_NAMES,
        "max_adds": DEFAULT_MAX_ADDS,
        "reservations": [],
    }


def _load(path: Path, session_date: str) -> Dict[str, Any]:
    if not path.exists():
        return _empty(session_date)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionBudgetError("budget_unreadable") from exc
    if not isinstance(doc, dict) or str(doc.get("schema") or "") != SCHEMA:
        raise SessionBudgetError("budget_unreadable")
    if str(doc.get("session_date") or "") != session_date:
        raise SessionBudgetError("budget_session_mismatch")
    if not isinstance(doc.get("reservations"), list):
        raise SessionBudgetError("budget_unreadable")
    return doc


def _atomic_write(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise SessionBudgetError("budget_unwritable") from exc


def _symbol(value: Any) -> str:
    """Session slot key. Class-share aliases share one reservation."""
    return identity_key(value)


def _reservations(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [row for row in doc.get("reservations") or [] if isinstance(row, dict)]


def _opened_this_session(rows: Sequence[Mapping[str, Any]], symbol: str) -> bool:
    return any(
        _symbol(row.get("symbol")) == symbol and str(row.get("action") or "") == OPEN_NEW
        for row in rows
    )


def _add_count(rows: Sequence[Mapping[str, Any]], symbol: Optional[str] = None) -> int:
    return sum(
        1
        for row in rows
        if str(row.get("action") or "") == ADD
        and (symbol is None or _symbol(row.get("symbol")) == symbol)
    )


def _new_name_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(
        {_symbol(row.get("symbol")) for row in rows if str(row.get("action") or "") == OPEN_NEW}
    )


def decide_entry(
    doc: Mapping[str, Any],
    *,
    symbol: str,
    action: str,
    held_at_open: bool,
    terminal_sell_proven: bool = False,
    broker_flat: bool = False,
) -> Dict[str, Any]:
    """Pure decision. A refused decision does not mutate the document."""
    symbol = _symbol(symbol)
    action = str(action or "").upper()
    rows = _reservations(doc)
    max_new = int(doc.get("max_new_names") or DEFAULT_MAX_NEW_NAMES)
    max_adds = int(doc.get("max_adds") or DEFAULT_MAX_ADDS)
    if not symbol or action not in {OPEN_NEW, ADD}:
        return {"ok": False, "reason": "budget_action_invalid", "symbol": symbol}
    prior = [row for row in rows if _symbol(row.get("symbol")) == symbol]
    reentry = (
        action == OPEN_NEW and terminal_sell_proven is True and broker_flat is True and bool(prior)
    )
    if action == ADD and _opened_this_session(rows, symbol):
        return {"ok": False, "reason": "opened_today_cannot_add", "symbol": symbol}
    if prior and not reentry:
        return {"ok": False, "reason": "session_symbol_already_used", "symbol": symbol}
    if action == ADD and not held_at_open:
        return {
            "ok": False,
            "reason": "add_requires_position_held_at_open",
            "symbol": symbol,
        }
    if action == ADD and _add_count(rows, symbol) >= 1:
        return {"ok": False, "reason": "session_add_already_used", "symbol": symbol}
    if action == ADD and _add_count(rows) >= max_adds:
        return {"ok": False, "reason": "session_add_cap", "symbol": symbol}
    if action == OPEN_NEW and _new_name_count(rows) >= max_new:
        return {"ok": False, "reason": "session_new_name_cap", "symbol": symbol}
    return {"ok": True, "reason": "reserved", "symbol": symbol, "reentry": reentry}


def preview_entries(
    root: Path,
    session_date: str,
    entries: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Read-only preview. A missing file is an empty budget. A bad file refuses."""
    path = budget_path(root, session_date)
    try:
        doc = _load(path, session_date)
    except SessionBudgetError as exc:
        return {"ok": False, "reason": exc.reason, "refusals": []}
    refusals = []
    shadow = json.loads(json.dumps(doc))
    for entry in entries:
        decision = decide_entry(
            shadow,
            symbol=str(entry.get("symbol") or ""),
            action=str(entry.get("action") or ""),
            held_at_open=bool(entry.get("held_at_open")),
            terminal_sell_proven=bool(entry.get("terminal_sell_proven")),
            broker_flat=bool(entry.get("broker_flat")),
        )
        if not decision["ok"]:
            refusals.append(decision)
            continue
        shadow["reservations"].append(
            {
                "symbol": decision["symbol"],
                "action": str(entry.get("action") or "").upper(),
            }
        )
    if refusals:
        return {"ok": False, "reason": refusals[0]["reason"], "refusals": refusals}
    return {"ok": True, "reason": "preview_ok", "refusals": []}


def reserve_entries(
    root: Path,
    session_date: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    invocation_tag: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Atomically reserve every entry or reserve none. Read/write failure refuses."""
    if not session_date:
        return {"ok": False, "reason": "budget_session_missing", "refusals": []}
    path = budget_path(root, session_date)
    try:
        doc = _load(path, session_date)
    except SessionBudgetError as exc:
        return {"ok": False, "reason": exc.reason, "refusals": []}
    stamp = (
        (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    accepted: List[Dict[str, Any]] = []
    refusals: List[Dict[str, Any]] = []
    for entry in entries:
        decision = decide_entry(
            doc,
            symbol=str(entry.get("symbol") or ""),
            action=str(entry.get("action") or ""),
            held_at_open=bool(entry.get("held_at_open")),
            terminal_sell_proven=bool(entry.get("terminal_sell_proven")),
            broker_flat=bool(entry.get("broker_flat")),
        )
        if not decision["ok"]:
            refusals.append(decision)
            continue
        accepted.append(
            {
                "symbol": decision["symbol"],
                "action": str(entry.get("action") or "").upper(),
                "invocation_tag": str(invocation_tag or ""),
                "reserved_at": stamp,
                "terminal_sell_reentry": bool(decision.get("reentry")),
            }
        )
        doc["reservations"].append(accepted[-1])
    if refusals:
        return {"ok": False, "reason": refusals[0]["reason"], "refusals": refusals}
    if not accepted:
        return {"ok": True, "reason": "no_entries", "refusals": []}
    try:
        _atomic_write(path, doc)
    except SessionBudgetError as exc:
        return {"ok": False, "reason": exc.reason, "refusals": []}
    return {"ok": True, "reason": "reserved", "reserved": accepted, "refusals": []}
