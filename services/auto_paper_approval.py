"""
auto_paper_approval.py
Controlled automatic PAPER-ONLY governance approval (Phase 196B).

Never approves live requests. Requires TRITON_ENABLE_AUTO_PAPER_APPROVAL=1.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

EXEC_READINESS = RESULTS / "execution_readiness.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
AUDIT_JSON = RESULTS / "auto_paper_approval_audit.json"
AUDIT_CSV = RESULTS / "auto_paper_approval_audit.csv"

AUTO_APPROVAL_AUTHORITY = "TRITON_AUTO_PAPER_APPROVAL"
AUTO_APPROVAL_RATIONALE = (
    "Automatic paper approval after market-open readiness, paper ARM, "
    "broker snapshot, and governance gate validation"
)

AUDIT_FIELDS = (
    "timestamp",
    "request_id",
    "symbols",
    "approval_authority",
    "rationale",
    "mode",
    "live_execution_permitted",
    "readiness_status",
    "paper_arm_valid",
    "duplicate_order_check",
    "duplicate_position_check",
    "market_status",
    "execution_path",
    "status",
    "refusal_reason",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _append_audit_csv(row: Dict[str, Any]) -> None:
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    if AUDIT_CSV.is_file():
        with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: str(row.get(k, "")) for k in AUDIT_FIELDS})
    tmp = AUDIT_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, AUDIT_CSV)


def _file_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0


def _norm_symbols(symbols: Sequence[str]) -> List[str]:
    return sorted({str(s).strip().upper() for s in symbols if str(s).strip()})


def _log(verbose: bool, status: str, *, reason: str = "", request_id: str = "") -> None:
    msg = f"[PAPER_AUTO_APPROVAL] status={status}"
    if request_id:
        msg += f" request_id={request_id}"
    if reason:
        msg += f" reason={reason}"
    print(msg, flush=True)


def _write_audit(
    *,
    status: str,
    refusal_reason: str = "",
    request_id: str = "",
    symbols: Optional[Sequence[str]] = None,
    checks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checks = checks or {}
    row = {
        "timestamp": _utc_now_iso(),
        "request_id": request_id,
        "symbols": ",".join(_norm_symbols(symbols or [])),
        "approval_authority": AUTO_APPROVAL_AUTHORITY if status == "PASS" else "",
        "rationale": AUTO_APPROVAL_RATIONALE if status == "PASS" else "",
        "mode": "paper",
        "live_execution_permitted": str(checks.get("live_execution_permitted", False)).lower(),
        "readiness_status": str(checks.get("readiness_status") or ""),
        "paper_arm_valid": str(checks.get("paper_arm_valid", False)).lower(),
        "duplicate_order_check": str(checks.get("duplicate_order_check", "OK")),
        "duplicate_position_check": str(checks.get("duplicate_position_check", "OK")),
        "market_status": str(checks.get("market_status") or ""),
        "execution_path": "capital_deployment",
        "status": status,
        "refusal_reason": refusal_reason,
    }
    _append_audit_csv(row)
    doc = _load_json(AUDIT_JSON)
    events = list(doc.get("events") or [])
    events.append(row)
    doc["events"] = events[-200:]
    doc["last_event"] = row
    doc["generated_at"] = _utc_now_iso()
    _atomic_write_json(AUDIT_JSON, doc)
    return row


def _duplicate_checks(symbols: Sequence[str]) -> Tuple[bool, str, Dict[str, str]]:
    from services.paper_entry_loop_stage import _load_held_symbols, _load_open_buy_keys

    held = _load_held_symbols()
    open_buy = _load_open_buy_keys()
    sym_set = set(_norm_symbols(symbols))
    held_overlap = sym_set & held
    order_overlap = {s for s, side in open_buy if side in ("BUY", "B") and s in sym_set}
    if held_overlap:
        return (
            False,
            f"ALREADY_HELD:{','.join(sorted(held_overlap))}",
            {
                "duplicate_order_check": "OK",
                "duplicate_position_check": "FAIL",
            },
        )
    if order_overlap:
        return (
            False,
            f"OPEN_ORDER:{','.join(sorted(order_overlap))}",
            {
                "duplicate_order_check": "FAIL",
                "duplicate_position_check": "OK",
            },
        )
    return (
        True,
        "OK",
        {
            "duplicate_order_check": "OK",
            "duplicate_position_check": "OK",
        },
    )


def _validate_auto_approval_gates(
    symbols: Sequence[str],
) -> Tuple[bool, str, Dict[str, Any]]:
    checks: Dict[str, Any] = {"symbols": _norm_symbols(symbols)}

    if os.environ.get("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "").strip() != "1":
        return False, "AUTO_PAPER_APPROVAL_DISABLED", checks
    if os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() != "1":
        return False, "PAPER_EXECUTION_DISABLED", checks
    if os.environ.get("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "").strip() != "1":
        return False, "AUTO_PAPER_ENTRIES_DISABLED", checks

    from services.guard_auto import is_market_open_et

    market_open = is_market_open_et(datetime.now(timezone.utc))
    checks["market_open"] = market_open
    checks["market_status"] = "OPEN" if market_open else "CLOSED"
    if not market_open:
        return False, "MARKET_CLOSED", checks

    readiness = _load_json(EXEC_READINESS)
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    live_permitted = bool(readiness.get("live_execution_permitted", True))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))
    checks["readiness_status"] = readiness_status
    checks["live_execution_permitted"] = live_permitted
    checks["paper_execution_permitted"] = paper_permitted
    if readiness_status != "READY":
        return False, f"READINESS_{readiness_status}", checks
    if live_permitted:
        return False, "LIVE_EXECUTION_PERMITTED", checks
    if not paper_permitted:
        return False, "PAPER_NOT_PERMITTED", checks

    snap_age = _file_age_minutes(OPEN_ORDERS)
    checks["broker_snapshot_age_minutes"] = snap_age
    if snap_age is None:
        return False, "BROKER_SNAPSHOT_MISSING", checks
    if snap_age > 120:
        return False, "BROKER_SNAPSHOT_STALE", checks

    try:
        from services.arm_mode import is_paper_arm_confirm_valid, load_arm_config, validate_mode

        arm_cfg = load_arm_config()
        arm_valid = is_paper_arm_confirm_valid(arm_cfg)
        checks["paper_arm_valid"] = arm_valid
        checks["arm_mode"] = validate_mode(arm_cfg.get("mode"))
        if validate_mode(arm_cfg.get("mode")) == "ASSISTED" and not arm_valid:
            return False, "ARM_EXPIRED_OR_MISSING", checks
    except Exception as e:
        checks["paper_arm_valid"] = False
        return False, f"ARM_CHECK_ERROR:{e}", checks

    if not symbols:
        return False, "NO_SYMBOLS", checks

    dup_ok, dup_reason, dup_meta = _duplicate_checks(symbols)
    checks.update(dup_meta)
    if not dup_ok:
        return False, dup_reason, checks

    return True, "OK", checks


def _find_pending_for_symbols(symbols: Sequence[str]) -> Optional[Dict[str, Any]]:
    from services.execution_approval_workflow import (
        _load_requests,
        validate_paper_capital_deployment_approvable,
    )

    sym_set = set(_norm_symbols(symbols))
    for req in reversed(_load_requests()):
        if req.get("approval_status") != "PENDING":
            continue
        ok, _ = validate_paper_capital_deployment_approvable(req)
        if not ok:
            continue
        req_syms = {
            s.strip().upper() for s in str(req.get("symbol(s)") or "").split(",") if s.strip()
        }
        if sym_set <= req_syms or sym_set & req_syms:
            return req
    return None


def _ensure_pending_request(
    symbols: Sequence[str],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    from services.execution_approval_workflow import (
        create_approval_request,
        evaluate_governed_execution,
        find_approved_request,
        find_latest_pending_paper_capital_deployment,
    )
    from services.execution_governance_council import determine_authority_level

    ctx = dict(context or {})
    ctx.setdefault("paperMode", True)
    ctx.setdefault("liveMode", False)
    ctx.setdefault("lifecycleApproved", True)
    ctx.setdefault("riskConstraintsSatisfied", True)
    ctx.setdefault("governanceApproved", False)

    if ctx.get("liveMode") is True:
        return None, "LIVE_CONTEXT_REFUSED"

    existing = find_approved_request(
        "",
        "capital_deployment",
        symbols=symbols,
        context=ctx,
    )
    if existing:
        return existing, "ALREADY_APPROVED"

    pending = _find_pending_for_symbols(symbols) or find_latest_pending_paper_capital_deployment()
    if pending:
        return pending, "EXISTING_PENDING"

    evaluation = evaluate_governed_execution(
        "capital_deployment",
        ctx,
        record_evidence=False,
        record_requests=False,
        source="auto_paper_approval",
        symbols=symbols,
        action_requested="capital_deployment",
    )
    if evaluation.get("execution_path") == "live_execution" or ctx.get("liveMode") is True:
        return None, "LIVE_PATH_REFUSED"
    if str(evaluation.get("decision") or "").upper() not in ("ALLOW", "REVIEW_REQUIRED"):
        return None, f"GATE_{evaluation.get('decision')}"

    authority_level, triggers = determine_authority_level(
        path="capital_deployment",
        context=ctx,
        evaluation=evaluation,
        action_requested="capital_deployment",
    )
    req = create_approval_request(
        path="capital_deployment",
        context=ctx,
        evaluation=evaluation,
        authority_level=authority_level,
        triggering_conditions=triggers,
        symbols=symbols,
        action_requested="capital_deployment",
    )
    return req, "CREATED"


def attempt_auto_paper_approval(
    symbols: Sequence[str],
    *,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Attempt automatic paper-only governance approval.

    Returns (ok, reason, details). Never approves live requests.
    """
    ok, reason, checks = _validate_auto_approval_gates(symbols)
    if not ok:
        _log(verbose, "REFUSED", reason=reason)
        audit = _write_audit(
            status="REFUSED", refusal_reason=reason, symbols=symbols, checks=checks
        )
        return False, reason, {"checks": checks, "audit": audit}

    from services.execution_approval_workflow import (
        approve_request,
        find_approved_request,
        validate_paper_capital_deployment_approvable,
    )

    ctx = dict(context or {"paperMode": True, "liveMode": False})
    approved = find_approved_request(
        "",
        "capital_deployment",
        symbols=symbols,
        context=ctx,
    )
    if approved:
        _log(verbose, "PASS", request_id=str(approved.get("approval_id") or ""))
        audit = _write_audit(
            status="PASS",
            request_id=str(approved.get("approval_id") or ""),
            symbols=symbols,
            checks=checks,
        )
        return True, "ALREADY_APPROVED", {"checks": checks, "approved": approved, "audit": audit}

    pending, pending_reason = _ensure_pending_request(symbols, context=ctx)
    if not pending:
        _log(verbose, "REFUSED", reason=pending_reason)
        audit = _write_audit(
            status="REFUSED",
            refusal_reason=pending_reason,
            symbols=symbols,
            checks=checks,
        )
        return False, pending_reason, {"checks": checks, "audit": audit}

    if pending.get("approval_status") == "APPROVED":
        rid = str(pending.get("approval_id") or "")
        _log(verbose, "PASS", request_id=rid)
        audit = _write_audit(status="PASS", request_id=rid, symbols=symbols, checks=checks)
        return True, pending_reason, {"checks": checks, "approved": pending, "audit": audit}

    ok_val, val_reason = validate_paper_capital_deployment_approvable(pending)
    if not ok_val:
        _log(verbose, "REFUSED", reason=val_reason)
        audit = _write_audit(
            status="REFUSED",
            refusal_reason=val_reason,
            request_id=str(pending.get("approval_id") or ""),
            symbols=symbols,
            checks=checks,
        )
        return False, val_reason, {"checks": checks, "audit": audit}

    updated, approve_reason = approve_request(
        str(pending.get("approval_id")),
        approving_authority=AUTO_APPROVAL_AUTHORITY,
        approval_rationale=AUTO_APPROVAL_RATIONALE,
        validate_paper_capital=True,
    )
    if not updated:
        _log(verbose, "REFUSED", reason=approve_reason)
        audit = _write_audit(
            status="REFUSED",
            refusal_reason=approve_reason,
            request_id=str(pending.get("approval_id") or ""),
            symbols=symbols,
            checks=checks,
        )
        return False, approve_reason, {"checks": checks, "audit": audit}

    rid = str(updated.get("approval_id") or "")
    _log(verbose, "PASS", request_id=rid)
    audit = _write_audit(status="PASS", request_id=rid, symbols=symbols, checks=checks)
    return True, pending_reason, {"checks": checks, "approved": updated, "audit": audit}
