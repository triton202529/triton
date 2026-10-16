"""October 15 BITO manual-exit ledger recovery.

The dry-run proposes three events and writes nothing. Persistent write flags
stay false. One future invocation can proceed only when its invocation token
matches a one-time local manifest; that manifest is removed only after the
write succeeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import sqlite3
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from services.paper_order_ledger import (
    EVENT_ACTION_SUBMITTED,
    EVENT_ORDER_TERMINAL_RECONCILED,
    PaperOrderLedgerError,
    append_events,
    ledger_path,
    load_events,
    manual_exit_submission_event,
    record_manual_exit_submission,
    status_reconciliation_event,
)
from services.performance_accounting import (
    EVENTS_RELATIVE,
    _fill_event_from_terminal,
    load_events as load_accounting_events,
    record_proven_terminal_fills,
)

WRITE_MODE_ENABLED = False
EXECUTOR_ENABLED = False
APPROVED_HEAD = "be0d54ebe9c68512389af6e533c153e88147d0d0"
TARGET_BROKER_ORDER_ID = "507112c1-8984-4450-8c67-f9d1bca5a8ab"
TARGET_CLIENT_ORDER_ID = "manual-BITO-0f628c831b6e31d43583ceb6"
TARGET_REQUEST_ID = "0f628c831b6e31d43583ceb610e13db1ed32c1bd727ad32e5856d5e6e0a2215f"
TARGET_AUTHORITY_ID = "manual-auth-0f628c831b6e31d43583ceb6"
TARGET_DISPATCH_ID = "sell-dispatch-f40ccbd63cf54e52"
TARGET_SYMBOL = "BITO"
TARGET_QTY = Decimal("450")
TARGET_FILL_PRICE = Decimal("11.01")
TARGET_CREATED_AT = "2026-10-15T16:14:33.48798992Z"
TARGET_SUBMITTED_AT = "2026-10-15T16:14:33.495870609Z"
TARGET_FILLED_AT = "2026-10-15T16:14:33.914668051Z"
TARGET_FILL_TIMESTAMP = "2026-10-15T16:14:33.914668+00:00"
TARGET_UPDATED_AT = "2026-10-15T16:14:33.916512203Z"
TARGET_STATUS = "filled"
GOOGL_BROKER_ORDER_ID = "924938e3-813d-4007-9384-b36fb639cd66"
EXPECTED_ACTION_SUBMITTED_EVENT_ID = (
    "3a979aa39b42013e02bb3c431c10d788c57277daa9dcd18ac220cc3b94a489d4"
)
EXPECTED_TERMINAL_EVENT_ID = "fe597bdac633f99aff0142b2862d33ebf8b419f5b20c9293d571bab023a83528"
EXPECTED_FILL_SELL_EVENT_ID = "074134fa1bdbfc863192763a87803f2c2f7b3d204428906214f3504088a1d6ab"
AUTHORIZATION_RELATIVE = Path(
    "data/governance/manual_portfolio_exit/october15_bito_recovery.authorization"
)
_MANIFEST_FIELDS = (
    "executor_baseline_commit",
    "execution_commit",
    "broker_order_id",
    "client_order_id",
    "request_id",
    "authority_id",
    "dispatch_id",
    "symbol",
    "quantity",
    "fill_price",
    "fill_timestamp",
    "action_submitted_event_id",
    "order_terminal_reconciled_event_id",
    "fill_sell_event_id",
    "paper_order_events_sha256",
    "performance_accounting_events_sha256",
    "nonce",
)
_STORE_RELATIVE = Path("data/governance/manual_portfolio_exit")
_RUNTIME_LOCKS = (
    Path("data/live/continuous_loop.lock"),
    Path("data/live/watchdog_loop.lock"),
    Path("data/live/daily_refresh.lock"),
    Path("data/daily_refresh.lock"),
)
_WRITER_SCRIPT_MARKERS = (
    "run_daily_refresh.ps1",
    "run_market_open.ps1",
    "run_data_freshness_refresh.ps1",
    "snapshot_live_orders",
)
_BANNED_EVENT_KEYS = frozenset(
    {"fee", "fee_amount", "commission", "realized_pl", "net_realized_pl"}
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fail(reason: str, **details: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "wrote": False, "reason": reason, "dry_run": True}
    if details:
        payload["details"] = details
    return payload


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None
    if not number.is_finite():
        return None
    return number


def _load_target_row(path: Path, broker_order_id: str) -> Dict[str, Any]:
    if not path.is_file():
        return _fail("snapshot_missing", path=str(path))
    matches = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("id") or "").strip() == broker_order_id:
                matches.append(row)
    if not matches:
        return _fail("snapshot_row_missing", broker_order_id=broker_order_id)
    if len(matches) != 1:
        return _fail("snapshot_row_duplicated", count=len(matches))
    return {"ok": True, "row": matches[0]}


def _timestamp_mismatches(row: Mapping[str, Any]) -> Dict[str, str]:
    expected = {
        "created_at": TARGET_CREATED_AT,
        "submitted_at": TARGET_SUBMITTED_AT,
        "filled_at": TARGET_FILLED_AT,
        "updated_at": TARGET_UPDATED_AT,
    }
    mismatches = {}
    for field, wanted in expected.items():
        found = str(row.get(field) or "")
        if found != wanted:
            mismatches[field] = found
    return mismatches


def _identity_mismatches(row: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches: Dict[str, Any] = {}
    if str(row.get("client_order_id") or "") != TARGET_CLIENT_ORDER_ID:
        mismatches["client_order_id"] = row.get("client_order_id")
    if str(row.get("symbol") or "").upper() != TARGET_SYMBOL:
        mismatches["symbol"] = row.get("symbol")
    if str(row.get("status") or "").strip().lower() != TARGET_STATUS:
        mismatches["status"] = row.get("status")
    for field, wanted in (("qty", TARGET_QTY), ("filled_qty", TARGET_QTY)):
        found = _decimal(row.get(field))
        if found != wanted:
            mismatches[field] = row.get(field)
    price = _decimal(row.get("filled_avg_price"))
    if price != TARGET_FILL_PRICE:
        mismatches["filled_avg_price"] = row.get("filled_avg_price")
    mismatches.update(_timestamp_mismatches(row))
    return mismatches


def _existing_target_events(root: Path) -> Dict[str, Any]:
    found = []
    orders = ledger_path(root)
    if orders.is_file():
        for event in load_events(orders):
            if str(event.get("broker_order_id") or "") != TARGET_BROKER_ORDER_ID:
                continue
            if event.get("event_type") in {EVENT_ACTION_SUBMITTED, EVENT_ORDER_TERMINAL_RECONCILED}:
                found.append(str(event.get("event_type")))
    accounting = root / EVENTS_RELATIVE
    if accounting.is_file():
        for line in accounting.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            key = str(event.get("logical_key") or "")
            if key.startswith(f"fill:{TARGET_BROKER_ORDER_ID}:"):
                found.append(str(event.get("event_type") or "FILL_SELL"))
    return {"ok": True, "events": found}


def _refuse(reason: str, **details: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "wrote": False,
        "dry_run": False,
        "reason": reason,
        "executor_enabled": EXECUTOR_ENABLED,
    }
    if details:
        payload["details"] = details
    return payload


def _incomplete(reason: str, **details: Any) -> Dict[str, Any]:
    payload = _refuse(reason, **details)
    payload["wrote"] = True
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ledger_hashes(root: Path) -> Dict[str, str]:
    order_path = ledger_path(root)
    accounting_path = root / EVENTS_RELATIVE
    order_bytes = order_path.read_bytes() if order_path.is_file() else b""
    accounting_bytes = accounting_path.read_bytes() if accounting_path.is_file() else b""
    return {
        "paper_order_events_sha256": _sha256_bytes(order_bytes),
        "performance_accounting_events_sha256": _sha256_bytes(accounting_bytes),
    }


def _repository_head() -> str:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _full_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_descendant(ancestor: str, descendant: str) -> Optional[bool]:
    """True when descendant is ancestor or a later commit on its history."""
    import subprocess

    if not _full_sha(ancestor) or not _full_sha(descendant):
        return False
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    return None


def _executor_source_committed() -> bool:
    """True only when this module's worktree bytes match the committed blob."""
    import subprocess

    relative = "services/manual_exit_ledger_recovery.py"
    try:
        difference = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{relative}"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return difference.returncode == 0 and tracked.returncode == 0


def execution_commit_for_manifest() -> Dict[str, Any]:
    """Identify the commit a future manifest may pin. Does not write a manifest."""
    if not _executor_source_committed():
        return _refuse("executor_not_committed")
    head = _repository_head()
    if not _full_sha(head):
        return _refuse("head_unreadable")
    descended = _is_descendant(APPROVED_HEAD, head)
    if descended is None:
        return _refuse("commit_ancestry_unreadable")
    if not descended:
        return _refuse("execution_commit_not_descendant")
    return {
        "ok": True,
        "wrote": False,
        "executor_baseline_commit": APPROVED_HEAD,
        "execution_commit": head,
    }


def _load_manifest(root: Path) -> Dict[str, Any]:
    path = root / AUTHORIZATION_RELATIVE
    if not path.is_file():
        return _refuse("authorization_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _refuse("authorization_unreadable", error=type(exc).__name__)
    if not isinstance(payload, dict) or set(payload) != set(_MANIFEST_FIELDS):
        return _refuse("manifest_rejected", fields=["shape"])
    if any(
        not isinstance(payload.get(field), str) or payload.get(field) == ""
        for field in _MANIFEST_FIELDS
    ):
        return _refuse("manifest_rejected", fields=["shape"])
    return {"ok": True, "manifest": payload}


def _manifest_field_mismatches(manifest: Mapping[str, Any]) -> list[str]:
    expected = {
        "executor_baseline_commit": APPROVED_HEAD,
        "broker_order_id": TARGET_BROKER_ORDER_ID,
        "client_order_id": TARGET_CLIENT_ORDER_ID,
        "request_id": TARGET_REQUEST_ID,
        "authority_id": TARGET_AUTHORITY_ID,
        "dispatch_id": TARGET_DISPATCH_ID,
        "symbol": TARGET_SYMBOL,
        "quantity": format(TARGET_QTY, "f"),
        "fill_price": format(TARGET_FILL_PRICE, "f"),
        "fill_timestamp": TARGET_FILL_TIMESTAMP,
        "action_submitted_event_id": EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        "order_terminal_reconciled_event_id": EXPECTED_TERMINAL_EVENT_ID,
        "fill_sell_event_id": EXPECTED_FILL_SELL_EVENT_ID,
    }
    return [field for field, wanted in expected.items() if manifest.get(field) != wanted]


def _authorize_manifest(root: Path, invocation_token: str) -> Dict[str, Any]:
    """Validate the one-time manifest. Does not create, rewrite, or remove it."""
    loaded = _load_manifest(root)
    if not loaded.get("ok"):
        return loaded
    manifest = loaded["manifest"]
    nonce = str(manifest.get("nonce") or "")
    supplied = str(invocation_token or "")
    if not hmac.compare_digest(supplied, nonce):
        return _refuse("authorization_mismatch")
    mismatches = _manifest_field_mismatches(manifest)
    if "broker_order_id" in mismatches:
        return _refuse("not_recovery_target", broker_order_id=manifest.get("broker_order_id"))
    if mismatches:
        return _refuse("manifest_rejected", fields=sorted(mismatches))
    execution_commit = str(manifest.get("execution_commit") or "")
    if not _full_sha(execution_commit):
        return _refuse("manifest_rejected", fields=["execution_commit"])
    try:
        actual_head = _repository_head()
    except (OSError, RuntimeError) as exc:
        return _refuse("head_unreadable", error=type(exc).__name__)
    if actual_head != execution_commit:
        return _refuse("head_mismatch")
    descended = _is_descendant(APPROVED_HEAD, execution_commit)
    if descended is None:
        return _refuse("commit_ancestry_unreadable")
    if not descended:
        return _refuse("execution_commit_not_descendant")
    actual_hashes = _ledger_hashes(root)
    hash_fields = [
        field for field, actual in actual_hashes.items() if manifest.get(field) != actual
    ]
    if hash_fields:
        return _refuse("ledger_hash_mismatch", fields=sorted(hash_fields))
    return {"ok": True}


def _consume_manifest(root: Path) -> Dict[str, Any]:
    path = root / AUTHORIZATION_RELATIVE
    try:
        path.unlink()
    except OSError as exc:
        return _incomplete("manifest_not_consumed", error=type(exc).__name__)
    if path.exists():
        return _incomplete("manifest_not_consumed")
    return {"ok": True}


def _lock_held(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text or set(text) <= {"0"}:
        return False
    return True


def _scan_repo_runtime_processes() -> Dict[str, Any]:
    """Read-only process scan for the production root. Fail closed on any error."""
    import subprocess

    if sys.platform != "win32":
        return {"ok": False, "reason": "runtime_process_scan_failed", "error": "non_windows"}
    try:
        from services.runtime_process_detector import scan_loop_processes_for_control

        loops = scan_loop_processes_for_control()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": "runtime_process_scan_failed", "error": type(exc).__name__}
    pids = []
    for found_pids, method, _warnings in loops.values():
        if method == "failed":
            return {"ok": False, "reason": "runtime_process_scan_failed"}
        pids.extend(int(pid) for pid in found_pids)
    pattern = "|".join(_WRITER_SCRIPT_MARKERS)
    script = (
        f"$rx = '{pattern}'; "
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.ProcessId -ne $PID -and $_.CommandLine -and ($_.CommandLine -match $rx) "
        "-and ($_.CommandLine -notmatch 'Get-CimInstance') "
        "} | Select-Object -ExpandProperty ProcessId"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": "runtime_process_scan_failed", "error": type(exc).__name__}
    if completed.returncode != 0:
        return {"ok": False, "reason": "runtime_process_scan_failed", "error": "powershell_exit"}
    for line in (completed.stdout or "").splitlines():
        if line.strip().isdigit():
            pids.append(int(line.strip()))
    return {"ok": True, "pids": pids}


def _active_runtime_processes(root: Path) -> Dict[str, Any]:
    if root.resolve() != _repo_root().resolve():
        return {"ok": True, "pids": []}
    return _scan_repo_runtime_processes()


def _runtime_obstacles(root: Path) -> Dict[str, Any]:
    held = [path.as_posix() for path in _RUNTIME_LOCKS if _lock_held(root / path)]
    if held:
        return _refuse("runtime_lock", locks=held)
    try:
        processes = _active_runtime_processes(root)
    except (OSError, RuntimeError) as exc:
        return _refuse("runtime_process_scan_failed", error=type(exc).__name__)
    if not processes.get("ok"):
        return _refuse(
            str(processes.get("reason") or "runtime_process_scan_failed"),
            error=processes.get("error"),
        )
    pids = [int(pid) for pid in processes.get("pids") or []]
    if pids:
        return _refuse("runtime_process_active", count=len(pids))
    return {"ok": True}


def _supplied_identity_refusal(
    *,
    broker_order_id: str,
    client_order_id: str,
    request_id: str,
    authority_id: str,
    dispatch_id: str,
    symbol: str,
    quantity: Any,
    fill_price: Any,
    fill_timestamp: str,
) -> Optional[Dict[str, Any]]:
    supplied = str(broker_order_id or "").strip()
    if supplied == GOOGL_BROKER_ORDER_ID or supplied != TARGET_BROKER_ORDER_ID:
        return _refuse("not_recovery_target", broker_order_id=supplied)
    mismatches: Dict[str, Any] = {}
    pairs = {
        "client_order_id": (str(client_order_id or "").strip(), TARGET_CLIENT_ORDER_ID),
        "request_id": (str(request_id or "").strip(), TARGET_REQUEST_ID),
        "authority_id": (str(authority_id or "").strip(), TARGET_AUTHORITY_ID),
        "dispatch_id": (str(dispatch_id or "").strip(), TARGET_DISPATCH_ID),
        "symbol": (str(symbol or "").strip().upper(), TARGET_SYMBOL),
        "fill_timestamp": (str(fill_timestamp or "").strip(), TARGET_FILL_TIMESTAMP),
    }
    for field, (found, wanted) in pairs.items():
        if found != wanted:
            mismatches[field] = found
    if _decimal(quantity) != TARGET_QTY:
        mismatches["quantity"] = str(quantity)
    if _decimal(fill_price) != TARGET_FILL_PRICE:
        mismatches["fill_price"] = str(fill_price)
    if mismatches:
        return _refuse("target_mismatch", fields=sorted(mismatches))
    return None


def _store_root(root: Path) -> Path:
    return root / _STORE_RELATIVE


def _receipt_path(root: Path) -> Path:
    return _store_root(root) / "requests" / f"{TARGET_REQUEST_ID}.json"


def _receipt_mismatches(payload: Mapping[str, Any]) -> list[str]:
    mismatches = []
    if str(payload.get("request_id") or "") != TARGET_REQUEST_ID:
        mismatches.append("request_id")
    if str(payload.get("broker_order_id") or "") != TARGET_BROKER_ORDER_ID:
        mismatches.append("broker_order_id")
    if str(payload.get("symbol") or "").upper() != TARGET_SYMBOL:
        mismatches.append("symbol")
    if _decimal(payload.get("selected_qty")) != TARGET_QTY:
        mismatches.append("selected_qty")
    if str(payload.get("status") or "") != "submitted":
        mismatches.append("status")
    if str(payload.get("broker_status") or "").lower() != "pending_new":
        mismatches.append("broker_status")
    for field, wanted in (
        ("client_order_id", TARGET_CLIENT_ORDER_ID),
        ("authority_id", TARGET_AUTHORITY_ID),
        ("dispatch_id", TARGET_DISPATCH_ID),
    ):
        if (
            field in payload
            and payload.get(field) not in (None, "")
            and payload.get(field) != wanted
        ):
            mismatches.append(field)
    return mismatches


def _load_receipt(root: Path) -> Dict[str, Any]:
    path = _receipt_path(root)
    if not path.is_file():
        return _refuse("receipt_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _refuse("receipt_unreadable", error=type(exc).__name__)
    if not isinstance(payload, dict):
        return _refuse("receipt_unreadable")
    mismatches = _receipt_mismatches(payload)
    if mismatches:
        return _refuse("target_mismatch", fields=sorted(mismatches))
    return {"ok": True, "receipt": payload}


def _dispatch_db(root: Path) -> Path:
    return _store_root(root) / "manual_portfolio_exit.sqlite3"


def _inspect_dispatch(root: Path) -> Dict[str, Any]:
    path = _dispatch_db(root)
    if not path.is_file():
        return _refuse("dispatch_missing")
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return _refuse("dispatch_unreadable", error=type(exc).__name__)
    connection.row_factory = sqlite3.Row
    try:
        dispatch = connection.execute(
            """
            SELECT dispatch_id, authority_id, state, broker_ref, symbols_submitted_json
            FROM sell_dispatches WHERE dispatch_id=?
            """,
            (TARGET_DISPATCH_ID,),
        ).fetchone()
        authority = connection.execute(
            "SELECT authority_id, state, final_result FROM sell_authorities WHERE authority_id=?",
            (TARGET_AUTHORITY_ID,),
        ).fetchone()
    except sqlite3.Error as exc:
        return _refuse("dispatch_unreadable", error=type(exc).__name__)
    finally:
        connection.close()
    if dispatch is None:
        return _refuse("dispatch_missing")
    if authority is None:
        return _refuse("authority_missing")
    mismatches = []
    if str(dispatch["authority_id"]) != TARGET_AUTHORITY_ID:
        mismatches.append("authority_id")
    if str(dispatch["state"]) != "SUBMITTED":
        mismatches.append("dispatch_state")
    if str(dispatch["broker_ref"] or "") != "":
        mismatches.append("broker_ref")
    try:
        symbols = json.loads(dispatch["symbols_submitted_json"] or "[]")
    except json.JSONDecodeError:
        symbols = None
    if symbols != [TARGET_SYMBOL]:
        mismatches.append("symbols_submitted")
    if str(authority["state"]) != "CONSUMED":
        mismatches.append("authority_state")
    if str(authority["final_result"] or "") != "DISPATCH_INTENT":
        mismatches.append("authority_final_result")
    if mismatches:
        return _refuse("target_mismatch", fields=sorted(mismatches))
    return {"ok": True}


def _has_banned_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if _BANNED_EVENT_KEYS.intersection(value):
            return True
        return any(_has_banned_key(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_banned_key(item) for item in value)
    return False


def _built_target_events(row: Mapping[str, Any]) -> Dict[str, Any]:
    from services.paper_order_ledger import _row_from_recent

    submission = manual_exit_submission_event(
        symbol=TARGET_SYMBOL,
        qty=TARGET_QTY,
        client_order_id=TARGET_CLIENT_ORDER_ID,
        broker_order_id=TARGET_BROKER_ORDER_ID,
        authority_id=TARGET_AUTHORITY_ID,
        dispatch_id=TARGET_DISPATCH_ID,
        event_timestamp=TARGET_SUBMITTED_AT,
    )
    normalized = _row_from_recent(row)
    terminal = status_reconciliation_event(
        rec=normalized,
        match={
            "authority_id": TARGET_AUTHORITY_ID,
            "dispatch_id": TARGET_DISPATCH_ID,
            "action_ordinal": None,
            "trading_date": "2026-10-15",
            "symbol": TARGET_SYMBOL,
            "side": "SELL",
            "selected_qty": 450.0,
            "client_order_id": TARGET_CLIENT_ORDER_ID,
            "broker_order_id": TARGET_BROKER_ORDER_ID,
        },
        method="broker_order_id",
        snapshot_ts=str(row.get("snapshot_ts") or "") or None,
        source_artifact="recent_orders.csv",
    )
    fill = _fill_event_from_terminal(terminal)
    if fill is None:
        return {"ok": False, "reason": "fill_event_not_proposed"}
    return {
        "ok": True,
        "action_submitted": submission,
        "order_terminal_reconciled": terminal,
        "fill_sell": fill,
    }


def _proposal(row: Mapping[str, Any]) -> Dict[str, Any]:
    built = _built_target_events(row)
    if not built.get("ok"):
        return _fail(str(built.get("reason") or "fill_event_not_proposed"))
    submission = built["action_submitted"]
    terminal = built["order_terminal_reconciled"]
    fill = built["fill_sell"]
    return {
        "ok": True,
        "wrote": False,
        "dry_run": True,
        "broker_order_id": TARGET_BROKER_ORDER_ID,
        "action_submitted": submission,
        "order_terminal_reconciled": terminal,
        "fill_sell": fill,
        "proposed_receipt": {
            "request_id": TARGET_REQUEST_ID,
            "status": "submitted",
            "broker_status": TARGET_STATUS,
            "broker_order_id": TARGET_BROKER_ORDER_ID,
            "client_order_id": TARGET_CLIENT_ORDER_ID,
            "reconciliation_state": "terminal_reconciled",
        },
        "proposed_dispatch": {
            "dispatch_id": TARGET_DISPATCH_ID,
            "authority_id": TARGET_AUTHORITY_ID,
            "state": "SUBMITTED",
            "broker_ref": TARGET_BROKER_ORDER_ID,
        },
        "fee": None,
        "commission": None,
        "net_realized_pl": None,
        "realized_pl": None,
        "parquet_modified": False,
    }


def dry_run_october15_bito_recovery(
    broker_order_id: str,
    *,
    recent_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Propose the three ledger identities for this exact order. Writes nothing."""
    root_p = Path(root) if root is not None else _repo_root()
    csv_path = (
        Path(recent_path)
        if recent_path is not None
        else root_p / "data" / "results" / "recent_orders.csv"
    )
    supplied = str(broker_order_id or "").strip()
    if supplied == GOOGL_BROKER_ORDER_ID:
        return _fail("not_recovery_target", broker_order_id=supplied, symbol="GOOGL")
    if supplied != TARGET_BROKER_ORDER_ID:
        return _fail("not_recovery_target", broker_order_id=supplied)
    loaded = _load_target_row(csv_path, supplied)
    if not loaded.get("ok"):
        return loaded
    row = loaded["row"]
    mismatches = _identity_mismatches(row)
    if mismatches:
        return _fail("target_mismatch", fields=sorted(mismatches))
    try:
        present = _existing_target_events(root_p)
    except (OSError, json.JSONDecodeError, PaperOrderLedgerError) as exc:
        return _fail("ledger_unreadable", error=type(exc).__name__)
    if present.get("events"):
        return _fail("target_event_exists", events=present["events"])
    return _proposal(row)


def _event_identity_refusal(built: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    expected = (
        ("action_submitted", EXPECTED_ACTION_SUBMITTED_EVENT_ID, EVENT_ACTION_SUBMITTED),
        ("order_terminal_reconciled", EXPECTED_TERMINAL_EVENT_ID, EVENT_ORDER_TERMINAL_RECONCILED),
        ("fill_sell", EXPECTED_FILL_SELL_EVENT_ID, "FILL_SELL"),
    )
    for key, event_id, event_type in expected:
        event = built.get(key)
        if not isinstance(event, Mapping):
            return _refuse("event_identity_mismatch", event=key)
        if (
            str(event.get("event_id") or "") != event_id
            or str(event.get("event_type") or "") != event_type
        ):
            return _refuse("event_identity_mismatch", event=key)
        if _has_banned_key(event):
            return _refuse("unexpected_pnl_field", event=key)
    fill = built["fill_sell"]
    if not str(fill.get("logical_key") or "").startswith(f"fill:{TARGET_BROKER_ORDER_ID}:"):
        return _refuse("event_identity_mismatch", event="fill_sell")
    return None


def _mark_receipt_filled(root: Path) -> Dict[str, Any]:
    from services.manual_portfolio_exit import _save_receipt

    loaded = _load_receipt(root)
    if not loaded.get("ok"):
        return _incomplete(str(loaded.get("reason") or "receipt_missing"), stage="receipt")
    payload = dict(loaded["receipt"])
    payload["broker_status"] = TARGET_STATUS
    payload["reconciliation_state"] = "terminal_reconciled"
    payload["status"] = "submitted"
    _save_receipt(_store_root(root), payload)
    saved = json.loads(_receipt_path(root).read_text(encoding="utf-8"))
    if saved.get("broker_status") != TARGET_STATUS or saved.get("request_id") != TARGET_REQUEST_ID:
        return _incomplete("receipt_not_updated")
    if saved.get("status") != "submitted":
        return _incomplete("receipt_not_updated")
    return {"ok": True}


def _set_dispatch_broker_ref(root: Path) -> Dict[str, Any]:
    from services.paper_sell_arm_store import PaperSellArmStoreError, write_connection

    path = _dispatch_db(root)
    if not path.is_file():
        return _incomplete("dispatch_missing", stage="dispatch")
    try:
        with write_connection(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                before = connection.total_changes
                connection.execute(
                    """
                    UPDATE sell_dispatches
                    SET broker_ref=?
                    WHERE dispatch_id=?
                      AND authority_id=?
                      AND state='SUBMITTED'
                      AND (broker_ref IS NULL OR broker_ref='')
                    """,
                    (TARGET_BROKER_ORDER_ID, TARGET_DISPATCH_ID, TARGET_AUTHORITY_ID),
                )
                if connection.total_changes - before != 1:
                    connection.execute("ROLLBACK")
                    return _incomplete("dispatch_not_updated")
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                connection.execute("ROLLBACK")
                return _incomplete("dispatch_not_updated", error=type(exc).__name__)
    except (OSError, sqlite3.Error, PaperSellArmStoreError) as exc:
        return _incomplete("dispatch_not_updated", error=type(exc).__name__)
    return {"ok": True}


def execute_october15_bito_ledger_recovery(
    *,
    broker_order_id: str,
    client_order_id: str,
    request_id: str,
    authority_id: str,
    dispatch_id: str,
    symbol: str,
    quantity: Any,
    fill_price: Any,
    fill_timestamp: str,
    authorization_token: str,
    root: Optional[Path] = None,
    recent_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append the three October 15 BITO events for one matching manifest.

    Persistent flags stay false. This function does not call the broker, create
    a manifest, create an authority, or write ledger.parquet. The manifest is
    removed only after every permitted write succeeds.
    """
    if root is None:
        return _refuse("root_required")
    root_p = Path(root)
    if not root_p.is_dir():
        return _refuse("root_missing")
    authorized = _authorize_manifest(root_p, authorization_token)
    if not authorized.get("ok"):
        return authorized
    identity = _supplied_identity_refusal(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        request_id=request_id,
        authority_id=authority_id,
        dispatch_id=dispatch_id,
        symbol=symbol,
        quantity=quantity,
        fill_price=fill_price,
        fill_timestamp=fill_timestamp,
    )
    if identity is not None:
        return identity
    runtime = _runtime_obstacles(root_p)
    if not runtime.get("ok"):
        return runtime
    csv_path = (
        Path(recent_path)
        if recent_path is not None
        else root_p / "data" / "results" / "recent_orders.csv"
    )
    loaded = _load_target_row(csv_path, TARGET_BROKER_ORDER_ID)
    if not loaded.get("ok"):
        details = loaded.get("details") if isinstance(loaded.get("details"), dict) else {}
        return _refuse(str(loaded.get("reason") or "snapshot_missing"), **details)
    mismatches = _identity_mismatches(loaded["row"])
    if mismatches:
        return _refuse("target_mismatch", fields=sorted(mismatches))
    try:
        present = _existing_target_events(root_p)
    except (OSError, json.JSONDecodeError, PaperOrderLedgerError) as exc:
        return _refuse("ledger_unreadable", error=type(exc).__name__)
    if present.get("events"):
        return _refuse("target_event_exists", events=present["events"])
    receipt = _load_receipt(root_p)
    if not receipt.get("ok"):
        return receipt
    dispatch = _inspect_dispatch(root_p)
    if not dispatch.get("ok"):
        return dispatch
    built = _built_target_events(loaded["row"])
    if not built.get("ok"):
        return _refuse(str(built.get("reason") or "fill_event_not_proposed"))
    identity_events = _event_identity_refusal(built)
    if identity_events is not None:
        return identity_events
    order_path = ledger_path(root_p)
    accounting_path = root_p / EVENTS_RELATIVE
    before_order_bytes = order_path.read_bytes() if order_path.is_file() else b""
    before_fill_bytes = accounting_path.read_bytes() if accounting_path.is_file() else b""
    before_order_ids = (
        [event.get("event_id") for event in load_events(order_path)] if before_order_bytes else []
    )
    before_fill_ids = (
        [event.get("event_id") for event in load_accounting_events(accounting_path)]
        if before_fill_bytes
        else []
    )
    recorded = record_manual_exit_submission(
        symbol=TARGET_SYMBOL,
        qty=TARGET_QTY,
        client_order_id=TARGET_CLIENT_ORDER_ID,
        broker_order_id=TARGET_BROKER_ORDER_ID,
        authority_id=TARGET_AUTHORITY_ID,
        dispatch_id=TARGET_DISPATCH_ID,
        event_timestamp=TARGET_SUBMITTED_AT,
        root=root_p,
    )
    submitted_event = recorded.get("event") if isinstance(recorded.get("event"), Mapping) else {}
    if (
        recorded.get("appended") != 1
        or submitted_event.get("event_id") != EXPECTED_ACTION_SUBMITTED_EVENT_ID
    ):
        if recorded.get("appended"):
            return _incomplete("submission_not_recorded", reason_detail=recorded.get("reason"))
        return _refuse("submission_not_recorded", reason_detail=recorded.get("reason"))
    terminal = built["order_terminal_reconciled"]
    appended = append_events([terminal], root=root_p)
    written = appended.get("written_events") or []
    if (
        appended.get("appended") != 1
        or not written
        or written[0].get("event_id") != EXPECTED_TERMINAL_EVENT_ID
    ):
        return _incomplete("terminal_not_recorded")
    filled = record_proven_terminal_fills([terminal], root=root_p)
    if filled.get("appended") != 1:
        return _incomplete("fill_not_recorded")
    order_bytes = order_path.read_bytes()
    fill_bytes = accounting_path.read_bytes()
    if not order_bytes.startswith(before_order_bytes) or not fill_bytes.startswith(
        before_fill_bytes
    ):
        return _incomplete("ledger_history_rewritten")
    orders = load_events(order_path)
    accounting = load_accounting_events(accounting_path)
    if [event.get("event_id") for event in orders] != before_order_ids + [
        EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        EXPECTED_TERMINAL_EVENT_ID,
    ]:
        return _incomplete("ledger_identity_mismatch")
    if [event.get("event_id") for event in accounting] != before_fill_ids + [
        EXPECTED_FILL_SELL_EVENT_ID
    ]:
        return _incomplete("fill_not_recorded")
    if accounting[-1].get("event_type") != "FILL_SELL" or _has_banned_key(accounting[-1]):
        return _incomplete("unexpected_pnl_field")
    marked = _mark_receipt_filled(root_p)
    if not marked.get("ok"):
        return marked
    bound = _set_dispatch_broker_ref(root_p)
    if not bound.get("ok"):
        return bound
    consumed = _consume_manifest(root_p)
    if not consumed.get("ok"):
        return consumed
    return {
        "ok": True,
        "wrote": True,
        "dry_run": False,
        "reason": "recovered",
        "executor_enabled": False,
        "manifest_consumed": True,
        "broker_order_id": TARGET_BROKER_ORDER_ID,
        "event_ids": {
            EVENT_ACTION_SUBMITTED: EXPECTED_ACTION_SUBMITTED_EVENT_ID,
            EVENT_ORDER_TERMINAL_RECONCILED: EXPECTED_TERMINAL_EVENT_ID,
            "FILL_SELL": EXPECTED_FILL_SELL_EVENT_ID,
        },
        "receipt_broker_status": TARGET_STATUS,
        "dispatch_broker_ref": TARGET_BROKER_ORDER_ID,
        "broker_called": False,
        "fee": None,
        "commission": None,
        "net_realized_pl": None,
        "realized_pl": None,
        "parquet_modified": False,
    }


def execute_october15_bito_recovery(
    broker_order_id: str = "",
    *,
    operator_authorization: Optional[str] = None,
    target_confirmation: Optional[str] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Refuse every write. A future mode must be enabled separately and confirmed."""
    del broker_order_id, _ignored
    authorized = bool(operator_authorization and target_confirmation)
    return {
        "ok": False,
        "wrote": False,
        "dry_run": True,
        "reason": "write_mode_not_enabled",
        "write_mode_enabled": WRITE_MODE_ENABLED,
        "authorization_present": authorized,
        "required": ["separate_operator_authorization", "target_order_confirmation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    blocked = [
        arg
        for arg in args
        if arg in {"--write", "--execute", "--apply"} or arg.startswith("--write=")
    ]
    if blocked or "--write" in " ".join(args):
        print(json.dumps(execute_october15_bito_recovery(), sort_keys=True))
        return 2
    parser = argparse.ArgumentParser(
        description="Dry-run the October 15 BITO manual-exit recovery."
    )
    parser.add_argument("--broker-order-id", required=True)
    parser.add_argument("--recent-orders", type=Path)
    parser.add_argument("--root", type=Path)
    parsed = parser.parse_args(args)
    report = dry_run_october15_bito_recovery(
        parsed.broker_order_id,
        recent_path=parsed.recent_orders,
        root=parsed.root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
