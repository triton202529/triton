"""Display-only exit-source classification for broker-activity sell rows.

Reads audited local evidence. Performs no writes, broker calls, or accounting.
A client-order prefix is not an exit source.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

EXIT_SOURCE_TRITON = "TRITON exit"
EXIT_SOURCE_MANUAL = "Manual exit — operator profit-taking"
EXIT_SOURCE_UNLINKED = "Unlinked broker activity"

MANUAL_PROFIT_TAKING_REASON = "Manual operator exit / profit-taking"
PIPELINE_PREPLAN_SELL = "preplan_authorized_sell"
PIPELINE_MANUAL_EXIT = "manual_portfolio_exit"
EVENT_ACTION_SUBMITTED = "ACTION_SUBMITTED"
EVENT_ORDER_TERMINAL = "ORDER_TERMINAL_RECONCILED"

_SELL_SIDES = {"sell", "sell_to_close"}


def is_exit_side(side: Any) -> bool:
    text = str(side or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text in _SELL_SIDES


def _text(value: Any) -> str:
    return str(value or "").strip()


def _symbol(value: Any) -> str:
    return _text(value).upper()


def _qty(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _qty_eq(left: Any, right: Any) -> bool:
    a = _qty(left)
    b = _qty(right)
    return a is not None and b is not None and a == b


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _json_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _readonly_rows(path: Path, sql: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    from services.manual_portfolio_exit import immutable_readonly_sqlite_uri

    try:
        conn = sqlite3.connect(immutable_readonly_sqlite_uri(path), uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    conn.row_factory = sqlite3.Row
    try:
        return [{key: row[key] for key in row.keys()} for row in conn.execute(sql)]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _broker_ids(value: Any) -> set:
    return {part.strip() for part in _text(value).split(",") if part.strip()}


def _action_name(item: Mapping[str, Any]) -> str:
    return _text(item.get("action")).upper()


def _is_exit_action(item: Mapping[str, Any]) -> bool:
    if _action_name(item) == "EXIT":
        return True
    side = str(item.get("side") or "").strip().lower().replace("-", "_").replace(" ", "_")
    return side in _SELL_SIDES


def _bounds_from_authority(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    canonical = [
        item for item in _json_list(row.get("canonical_batch_json")) if isinstance(item, dict)
    ]
    payload = _json_obj(row.get("payload_json"))
    actions = payload.get("authorized_actions")
    if not isinstance(actions, list):
        actions = payload.get("authorized_scope")
    payload_bounds = [item for item in actions or [] if isinstance(item, dict)]
    if canonical and payload_bounds and not _same_bounds(canonical, payload_bounds):
        return []
    return canonical or payload_bounds


def _qty_key(value: Any) -> str:
    qty = _qty(value)
    if qty is None:
        return ""
    return format(qty.normalize(), "f")


def _same_bounds(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    def key(item: Mapping[str, Any]) -> tuple:
        return (
            _symbol(item.get("symbol")),
            _action_name(item),
            _qty_key(item.get("max_qty")),
        )

    return sorted(key(item) for item in left) == sorted(key(item) for item in right)


def _exit_bound_covers(bounds: Sequence[Mapping[str, Any]], symbol: str, filled: Decimal) -> bool:
    hits = [item for item in bounds if _symbol(item.get("symbol")) == symbol]
    if len(hits) != 1:
        return False
    bound = hits[0]
    if _action_name(bound) != "EXIT":
        return False
    max_qty = _qty(bound.get("max_qty"))
    if max_qty is None or filled <= 0 or filled > max_qty:
        return False
    return True


def _requested_matches(
    requested: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    filled: Decimal,
    client_order_id: str,
    require_client: bool,
) -> bool:
    hits = [
        item
        for item in requested
        if isinstance(item, dict)
        and _symbol(item.get("symbol")) == symbol
        and _is_exit_action(item)
    ]
    if len(hits) != 1:
        return False
    action = hits[0]
    action_qty = _qty(action.get("qty"))
    if action_qty is None:
        action_qty = _qty(action.get("selected_qty"))
    if action_qty != filled:
        return False
    client = _text(action.get("client_order_id"))
    if require_client or client:
        if client != client_order_id:
            return False
    return True


def _gate_symbols(decision: Mapping[str, Any]) -> set:
    raw = decision.get("symbol(s)")
    if raw in (None, ""):
        raw = decision.get("symbol") or decision.get("symbols") or ""
    return {part.strip().upper() for part in str(raw).replace(";", ",").split(",") if part.strip()}


class ExitSourceIndex:
    """In-memory, read-only view of the local exit-source evidence."""

    def __init__(self) -> None:
        self.triton_authorities: Dict[str, Dict[str, Any]] = {}
        self.triton_dispatches: Dict[str, Dict[str, Any]] = {}
        self.manual_authorities: Dict[str, Dict[str, Any]] = {}
        self.manual_dispatches: Dict[str, Dict[str, Any]] = {}
        self.manual_requests: List[Dict[str, Any]] = []
        self.manual_receipts: List[Dict[str, Any]] = []
        self.gate_decisions: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    @classmethod
    def load(cls, root: Path) -> "ExitSourceIndex":
        index = cls()
        root = Path(root)
        index._load_arm_sqlite(root / "data" / "live" / "paper_sell_arm.sqlite3", manual=False)
        index._load_arm_sqlite(
            root
            / "data"
            / "governance"
            / "manual_portfolio_exit"
            / "manual_portfolio_exit.sqlite3",
            manual=True,
        )
        index._load_gate(root / "data" / "governance" / "execution_gate_decisions.json")
        index._load_ledger(root / "data" / "results" / "paper_order_events.jsonl")
        manual_root = root / "data" / "governance" / "manual_portfolio_exit"
        index.manual_requests = _read_json_dir(manual_root / "requests")
        index.manual_receipts = _read_json_dir(manual_root / "dispatch_terminal")
        return index

    def classify(
        self,
        *,
        broker_order_id: str,
        client_order_id: str,
        symbol: str,
        filled_qty: Any,
    ) -> str:
        filled = _qty(filled_qty)
        broker = _text(broker_order_id)
        client = _text(client_order_id)
        sym = _symbol(symbol)
        if filled is None or filled <= 0 or not broker or not client or not sym:
            return EXIT_SOURCE_UNLINKED
        triton = self._triton_exit(broker, client, sym, filled)
        manual = self._manual_exit(broker, client, sym, filled)
        if triton and not manual:
            return EXIT_SOURCE_TRITON
        if manual and not triton:
            return EXIT_SOURCE_MANUAL
        return EXIT_SOURCE_UNLINKED

    def _load_arm_sqlite(self, path: Path, *, manual: bool) -> None:
        authorities = _readonly_rows(
            path,
            """
            SELECT authority_id, state, consuming_dispatch_id, canonical_batch_json, payload_json
            FROM sell_authorities
            """,
        )
        dispatches = _readonly_rows(
            path,
            """
            SELECT dispatch_id, authority_id, state, broker_ref, intent_json
            FROM sell_dispatches
            """,
        )
        auth_out: Dict[str, Dict[str, Any]] = {}
        for row in authorities:
            authority_id = _text(row.get("authority_id"))
            if not authority_id:
                continue
            auth_out[authority_id] = {
                "authority_id": authority_id,
                "state": _text(row.get("state")).upper(),
                "consuming_dispatch_id": _text(row.get("consuming_dispatch_id")),
                "bounds": _bounds_from_authority(row),
            }
        disp_out: Dict[str, Dict[str, Any]] = {}
        for row in dispatches:
            dispatch_id = _text(row.get("dispatch_id"))
            if not dispatch_id:
                continue
            intent = _json_obj(row.get("intent_json"))
            requested = intent.get("requested")
            if not isinstance(requested, list):
                requested = intent.get("actions") if isinstance(intent.get("actions"), list) else []
            disp_out[dispatch_id] = {
                "dispatch_id": dispatch_id,
                "authority_id": _text(row.get("authority_id")),
                "state": _text(row.get("state")).upper(),
                "broker_ids": _broker_ids(row.get("broker_ref"))
                | _broker_ids(intent.get("broker_order_id")),
                "intent_broker_order_id": _text(intent.get("broker_order_id")),
                "intent_client_order_id": _text(intent.get("client_order_id")),
                "requested": [item for item in requested if isinstance(item, dict)],
            }
        if manual:
            self.manual_authorities = auth_out
            self.manual_dispatches = disp_out
        else:
            self.triton_authorities = auth_out
            self.triton_dispatches = disp_out

    def _load_gate(self, path: Path) -> None:
        payload = _load_json(path)
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return
        self.gate_decisions = [row for row in records if isinstance(row, dict)]

    def _load_ledger(self, path: Path) -> None:
        if not path.is_file():
            return
        kept = {EVENT_ACTION_SUBMITTED, EVENT_ORDER_TERMINAL}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return
        events: List[Dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event_type") in kept:
                events.append(row)
        self.events = events

    def _triton_exit(self, broker: str, client: str, symbol: str, filled: Decimal) -> bool:
        submitted = self._single_submitted(broker, PIPELINE_PREPLAN_SELL, client, symbol, filled)
        terminal = self._proven_fill(broker, client, symbol, filled)
        if submitted is None or terminal is None:
            return False
        authority_id = _text(submitted.get("authority_id"))
        dispatch_id = _text(submitted.get("dispatch_id"))
        if not authority_id or not dispatch_id:
            return False
        if _text(terminal.get("authority_id")) != authority_id:
            return False
        if _text(terminal.get("dispatch_id")) != dispatch_id:
            return False
        authority = self.triton_authorities.get(authority_id)
        dispatch = self.triton_dispatches.get(dispatch_id)
        if authority is None or dispatch is None:
            return False
        if authority["state"] != "CONSUMED" or authority["consuming_dispatch_id"] != dispatch_id:
            return False
        if dispatch["state"] != "SUBMITTED" or dispatch["authority_id"] != authority_id:
            return False
        if broker not in dispatch["broker_ids"]:
            return False
        if not _exit_bound_covers(authority["bounds"], symbol, filled):
            return False
        if not _requested_matches(
            dispatch["requested"],
            symbol=symbol,
            filled=filled,
            client_order_id=client,
            require_client=True,
        ):
            return False
        return self._gate_allows(broker, client, symbol, filled)

    def _manual_exit(self, broker: str, client: str, symbol: str, filled: Decimal) -> bool:
        requests = [
            row
            for row in self.manual_requests
            if _text(row.get("reason")) == MANUAL_PROFIT_TAKING_REASON
            and _text(row.get("broker_order_id")) == broker
            and _text(row.get("client_order_id")) == client
            and _symbol(row.get("symbol")) == symbol
            and _qty_eq(row.get("selected_qty") or row.get("qty"), filled)
        ]
        if len(requests) != 1:
            return False
        request = requests[0]
        authority_id = _text(request.get("authority_id"))
        dispatch_id = _text(request.get("dispatch_id"))
        if not authority_id or not dispatch_id:
            return False
        authority = self.manual_authorities.get(authority_id)
        dispatch = self.manual_dispatches.get(dispatch_id)
        if authority is None or dispatch is None:
            return False
        if authority["state"] != "CONSUMED" or authority["consuming_dispatch_id"] != dispatch_id:
            return False
        if dispatch["state"] != "SUBMITTED" or dispatch["authority_id"] != authority_id:
            return False
        if broker not in dispatch["broker_ids"]:
            return False
        if dispatch["intent_broker_order_id"] and dispatch["intent_broker_order_id"] != broker:
            return False
        if dispatch["intent_client_order_id"] and dispatch["intent_client_order_id"] != client:
            return False
        if not _requested_matches(
            dispatch["requested"],
            symbol=symbol,
            filled=filled,
            client_order_id=client,
            require_client=False,
        ):
            return False
        receipts = [
            row
            for row in self.manual_receipts
            if _text(row.get("reconciliation_state")) == "terminal_reconciled"
            and _text(row.get("broker_order_id")) == broker
            and _text(row.get("client_order_id")) == client
            and _symbol(row.get("symbol")) == symbol
            and _qty_eq(row.get("selected_qty") or row.get("qty"), filled)
            and _text(row.get("authority_id")) == authority_id
            and _text(row.get("dispatch_id")) == dispatch_id
        ]
        if len(receipts) != 1:
            return False
        submitted = self._single_submitted(broker, PIPELINE_MANUAL_EXIT, client, symbol, filled)
        terminal = self._proven_fill(broker, client, symbol, filled)
        if submitted is None or terminal is None:
            return False
        if _text(submitted.get("authority_id")) != authority_id:
            return False
        if _text(submitted.get("dispatch_id")) != dispatch_id:
            return False
        if _text(terminal.get("authority_id")) != authority_id:
            return False
        if _text(terminal.get("dispatch_id")) != dispatch_id:
            return False
        return True

    def _single_submitted(
        self,
        broker: str,
        pipeline: str,
        client: str,
        symbol: str,
        filled: Decimal,
    ) -> Optional[Dict[str, Any]]:
        rows = [
            event
            for event in self.events
            if event.get("event_type") == EVENT_ACTION_SUBMITTED
            and _text(event.get("broker_order_id")) == broker
        ]
        if len(rows) != 1:
            return None
        event = rows[0]
        if _text(event.get("source_pipeline")) != pipeline:
            return None
        if _text(event.get("client_order_id")) != client:
            return None
        if _symbol(event.get("symbol")) != symbol:
            return None
        if not _qty_eq(event.get("selected_qty"), filled):
            return None
        side = _text(event.get("side"))
        if side and not is_exit_side(side):
            return None
        return event

    def _proven_fill(
        self,
        broker: str,
        client: str,
        symbol: str,
        filled: Decimal,
    ) -> Optional[Dict[str, Any]]:
        rows = [
            event
            for event in self.events
            if event.get("event_type") == EVENT_ORDER_TERMINAL
            and _text(event.get("broker_order_id")) == broker
        ]
        proven = [
            event
            for event in rows
            if event.get("terminal_status_proven") is True
            and _text(event.get("broker_terminal_status")).upper() == "FILLED"
        ]
        if len(proven) != 1:
            return None
        event = proven[0]
        if _text(event.get("client_order_id")) != client:
            return None
        if _symbol(event.get("symbol")) != symbol:
            return None
        if not _qty_eq(event.get("filled_qty"), filled):
            return None
        for other in rows:
            if other is event:
                continue
            if _text(other.get("client_order_id")) not in ("", client):
                return None
            if _symbol(other.get("symbol")) not in ("", symbol):
                return None
            other_qty = _qty(other.get("filled_qty"))
            if other_qty is not None and other_qty != filled:
                return None
            status = _text(other.get("broker_terminal_status")).upper()
            if status and status != "FILLED":
                return None
        return event

    def _gate_allows(self, broker: str, client: str, symbol: str, filled: Decimal) -> bool:
        rows = [
            decision
            for decision in self.gate_decisions
            if _text(decision.get("execution_identifier")) == client
        ]
        if not rows:
            return False
        matched_symbol = False
        for decision in rows:
            verdict = _text(decision.get("execution_gate_decision")).upper()
            if verdict != "ALLOW" or decision.get("allowed") is False:
                return False
            symbols = _gate_symbols(decision)
            if not symbols or symbol not in symbols:
                return False
            matched_symbol = True
            quantity = decision.get("quantity")
            if _text(quantity) and not _qty_eq(quantity, filled):
                return False
            for key in ("broker_order_id", "order_id"):
                recorded = _text(decision.get(key))
                if recorded and recorded != broker:
                    return False
        return matched_symbol


def _read_json_dir(path: Path) -> List[Dict[str, Any]]:
    if not path.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for file_path in sorted(path.glob("*.json")):
        payload = _load_json(file_path)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
