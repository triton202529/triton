"""Durable paper-only execution decision audit trail (Phase 148C)."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
AUDIT_CSV = RESULTS / "paper_execution_audit.csv"

AUDIT_COLUMNS = [
    "timestamp",
    "session",
    "symbol",
    "lifecycle_state",
    "signal",
    "action",
    "qty",
    "price",
    "order_type",
    "authorization_status",
    "governance_authorized",
    "execution_authorized",
    "readiness",
    "broker_mode",
    "duplicate_check_result",
    "submitted_order_id",
    "status",
    "skip_reason",
]

_DUPLICATE_REASON_CODES = {
    "DUPLICATE_OPEN_ORDER",
    "IN_FLIGHT_ORDER",
    "IN_FLIGHT_ALREADY_SATISFIED",
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bool_csv(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_order_id(detail: str) -> str:
    m = re.search(r"order_id=([^\s;]+)", detail or "")
    return m.group(1).strip() if m else ""


def load_authorization_context() -> Dict[str, Any]:
    gov = _read_json(RESULTS / "governance_authorization.json")
    ready = _read_json(RESULTS / "execution_readiness.json")
    env_ok = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1"
    overall = gov.get("overall_authorization")
    if env_ok and gov.get("execution_authorized") is not True:
        auth_status = "paper_env_only"
    elif overall is True:
        auth_status = "authorized"
    elif overall is False:
        auth_status = "denied"
    else:
        auth_status = "unknown"
    return {
        "authorization_status": auth_status,
        "governance_authorized": gov.get("governance_authorized"),
        "execution_authorized": gov.get("execution_authorized"),
        "readiness": _safe_str(ready.get("readiness_status")),
        "paper_env_enabled": env_ok,
    }


def _duplicate_check_result(reason_code: str, status: str) -> str:
    rc = _safe_str(reason_code).upper()
    st = _safe_str(status).lower()
    if rc == "DUPLICATE_OPEN_ORDER":
        return "blocked_duplicate"
    if rc in _DUPLICATE_REASON_CODES or (rc == "IN_FLIGHT_ORDER" and st == "kept"):
        return "satisfied_in_flight"
    return ""


def _audit_action_from_parts(
    *,
    side: str = "",
    plan_action: str = "",
    diag_status: str = "",
    reason_code: str = "",
) -> str:
    side_l = _safe_str(side).lower()
    if side_l in ("buy", "sell"):
        return side_l
    pa = _safe_str(plan_action).lower()
    if pa in ("plan", "buy", "sell"):
        return "buy" if pa == "plan" else pa
    st = _safe_str(diag_status).lower()
    rc = _safe_str(reason_code).upper()
    if st in ("submitted", "planned", "kept") and rc not in _DUPLICATE_REASON_CODES:
        return side_l or "plan"
    if st in ("dropped", "blocked", "skipped"):
        return "skip"
    return pa or side_l or st or ""


def build_audit_row(
    *,
    mode: str,
    session: str,
    symbol: str = "",
    lifecycle_state: str = "",
    signal: str = "",
    action: str = "",
    qty: Any = "",
    price: Any = "",
    order_type: str = "",
    status: str = "",
    skip_reason: str = "",
    duplicate_check_result: str = "",
    submitted_order_id: str = "",
    timestamp: Optional[str] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    auth = ctx or load_authorization_context()
    return {
        "timestamp": timestamp or _utc_iso(),
        "session": _safe_str(session),
        "symbol": _safe_str(symbol).upper(),
        "lifecycle_state": _safe_str(lifecycle_state),
        "signal": _safe_str(signal),
        "action": _safe_str(action),
        "qty": _safe_str(qty),
        "price": _safe_str(price),
        "order_type": _safe_str(order_type),
        "authorization_status": _safe_str(auth.get("authorization_status")),
        "governance_authorized": _bool_csv(auth.get("governance_authorized")),
        "execution_authorized": _bool_csv(auth.get("execution_authorized")),
        "readiness": _safe_str(auth.get("readiness")),
        "broker_mode": "paper" if _safe_str(mode).lower() == "paper" else "",
        "duplicate_check_result": _safe_str(duplicate_check_result),
        "submitted_order_id": _safe_str(submitted_order_id),
        "status": _safe_str(status),
        "skip_reason": _safe_str(skip_reason),
    }


def append_duplicate_block_audit(
    *,
    mode: str,
    session: str,
    symbol: str,
    side: str,
    reason: str,
    qty: Any = "",
    price: Any = "",
    order_type: str = "",
    ctx: Optional[Dict[str, Any]] = None,
) -> None:
    auth = ctx or load_authorization_context()
    append_audit_row(
        mode,
        build_audit_row(
            mode=mode,
            session=session,
            symbol=_safe_str(symbol),
            action=_safe_str(side).lower(),
            qty=qty,
            price=price,
            order_type=order_type,
            status="blocked_duplicate",
            skip_reason=reason,
            duplicate_check_result="blocked_duplicate",
            ctx=auth,
        ),
    )


def append_audit_row(mode: str, row: Dict[str, str]) -> None:
    """Append one audit row. Live mode rows are never written."""
    if _safe_str(mode).lower() != "paper":
        return
    if _safe_str(row.get("broker_mode")).lower() == "live":
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_header = not AUDIT_CSV.is_file() or AUDIT_CSV.stat().st_size == 0
    with AUDIT_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in AUDIT_COLUMNS})


def row_from_drop_diag(
    diag: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    auth = ctx or load_authorization_context()
    reason_code = _safe_str(diag.get("reason_code"))
    reason_detail = _safe_str(diag.get("reason_detail"))
    diag_status = _safe_str(diag.get("status")).lower()
    skip_reason = reason_code
    if reason_detail and diag_status in ("dropped", "blocked", "skipped"):
        skip_reason = f"{reason_code}:{reason_detail}" if reason_code else reason_detail
    elif reason_detail and reason_code == "SUBMITTED":
        skip_reason = ""

    status = diag_status
    if reason_code == "SUBMITTED":
        status = "submitted"
    elif reason_code in _DUPLICATE_REASON_CODES and diag_status == "blocked":
        status = "blocked_duplicate"
    elif reason_code in _DUPLICATE_REASON_CODES and diag_status == "kept":
        status = "duplicate_satisfied"
    elif diag_status == "kept" and reason_code == "KEPT":
        status = "planned"
        skip_reason = ""
    elif diag_status == "dropped":
        status = "skipped"

    submitted_order_id = _extract_order_id(reason_detail)
    stance_u = _safe_str(diag.get("stance")).upper()
    side_action = ""
    if stance_u in ("BUY", "ADD"):
        side_action = "buy"
    elif stance_u in ("TRIM", "EXIT", "ROTATE_EXIT"):
        side_action = "sell"
    return build_audit_row(
        mode=_safe_str(diag.get("run_mode") or "paper"),
        session=_safe_str(diag.get("session")),
        symbol=_safe_str(diag.get("symbol") or diag.get("ticker")),
        lifecycle_state=_safe_str(diag.get("stance")),
        signal=_safe_str(diag.get("opportunity_type") or diag.get("stance")),
        action=side_action
        or _audit_action_from_parts(
            diag_status=diag_status,
            reason_code=reason_code,
        ),
        qty=diag.get("planned_qty", ""),
        price="",
        order_type="",
        status=status,
        skip_reason=skip_reason,
        duplicate_check_result=_duplicate_check_result(reason_code, diag_status),
        submitted_order_id=submitted_order_id,
        timestamp=_safe_str(diag.get("timestamp")) or None,
        ctx=auth,
    )


def row_from_plan_line(
    plan_line: Any,
    *,
    session: str,
    mode: str,
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    auth = ctx or load_authorization_context()
    pl_status = _safe_str(getattr(plan_line, "status", "")).lower()
    skip_reason = _safe_str(getattr(plan_line, "skip_reason", ""))
    planned = getattr(plan_line, "planned", None)
    stance = _safe_str(getattr(plan_line, "stance", ""))
    signal = _safe_str(getattr(plan_line, "opportunity_type", "") or stance)
    action = _safe_str(getattr(plan_line, "action", ""))
    symbol = _safe_str(getattr(plan_line, "symbol", ""))
    qty = ""
    price = ""
    order_type = ""
    status = pl_status
    if planned is not None:
        qty = getattr(planned, "qty", "")
        price = getattr(planned, "limit_price", "")
        order_type = getattr(planned, "order_type", "")
        side = _safe_str(getattr(planned, "side", ""))
        if side:
            action = side
        if pl_status == "planned":
            status = "planned"
    elif pl_status == "skipped":
        status = "skipped"
        action = action or "skip"
    return build_audit_row(
        mode=mode,
        session=session,
        symbol=symbol,
        lifecycle_state=stance,
        signal=signal,
        action=action,
        qty=qty,
        price=price,
        order_type=order_type,
        status=status,
        skip_reason=skip_reason,
        ctx=auth,
    )


def row_from_planned_order(
    order: Any,
    *,
    session: str,
    mode: str,
    status: str = "planned",
    skip_reason: str = "",
    duplicate_check_result: str = "",
    submitted_order_id: str = "",
    lifecycle_state: str = "",
    signal: str = "",
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    auth = ctx or load_authorization_context()
    return build_audit_row(
        mode=mode,
        session=session,
        symbol=_safe_str(getattr(order, "symbol", "")),
        lifecycle_state=lifecycle_state or _safe_str(getattr(order, "stance", "")),
        signal=signal or _safe_str(getattr(order, "stance", "")),
        action=_safe_str(getattr(order, "side", "")),
        qty=getattr(order, "qty", ""),
        price=getattr(order, "limit_price", ""),
        order_type=_safe_str(getattr(order, "order_type", "")),
        status=status,
        skip_reason=skip_reason,
        duplicate_check_result=duplicate_check_result,
        submitted_order_id=submitted_order_id,
        ctx=auth,
    )


def write_execute_trades_plan_audit(
    plan_lines: Sequence[Any],
    *,
    session: str,
    mode: str,
) -> int:
    if _safe_str(mode).lower() != "paper":
        return 0
    ctx = load_authorization_context()
    n = 0
    for pl in plan_lines:
        append_audit_row(mode, row_from_plan_line(pl, session=session, mode=mode, ctx=ctx))
        n += 1
    if n:
        print(f"[PAPER_EXEC_AUDIT] wrote {n} row(s) from execute_trades plan session={session}")
    return n


def flush_placement_audit(
    *,
    mode: str,
    session: str,
    diag_rows: Iterable[Dict[str, Any]],
    planned_orders: Optional[Sequence[Any]] = None,
    dry_run: bool = False,
) -> int:
    if _safe_str(mode).lower() != "paper":
        return 0
    ctx = load_authorization_context()
    n = 0
    submitted_keys: set = set()
    for diag in diag_rows:
        if not isinstance(diag, dict):
            continue
        if dry_run and _safe_str(diag.get("reason_code")) == "KEPT":
            continue
        append_audit_row(mode, row_from_drop_diag(diag, ctx=ctx))
        n += 1
        sym = _safe_str(diag.get("symbol")).upper()
        if _safe_str(diag.get("reason_code")) == "SUBMITTED":
            submitted_keys.add(sym)

    if dry_run and planned_orders:
        for order in planned_orders:
            sym = _safe_str(getattr(order, "symbol", "")).upper()
            if sym in submitted_keys:
                continue
            append_audit_row(
                mode,
                row_from_planned_order(
                    order,
                    session=session,
                    mode=mode,
                    status="planned",
                    ctx=ctx,
                ),
            )
            n += 1

    if n:
        print(
            f"[PAPER_EXEC_AUDIT] wrote {n} row(s) from placement session={session} dry_run={dry_run}"
        )
    return n


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Paper execution audit utilities")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args(argv)
    if args.show_context:
        print(json.dumps(load_authorization_context(), indent=2))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
