"""
auto_paper_approval.py
Controlled automatic PAPER-ONLY governance approval (Phase 196B / 196D).

Never approves live requests. Requires TRITON_ENABLE_AUTO_PAPER_APPROVAL=1.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"

EXEC_READINESS = RESULTS / "execution_readiness.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
AUDIT_JSON = RESULTS / "auto_paper_approval_audit.json"
AUDIT_CSV = RESULTS / "auto_paper_approval_audit.csv"
APPROVAL_REGISTRY = GOVERNANCE / "execution_approval_requests.json"

AUTO_APPROVAL_AUTHORITY = "TRITON_AUTO_PAPER_APPROVAL"
AUTO_APPROVAL_RATIONALE = (
    "Automatic paper approval after market-open readiness, paper ARM, "
    "broker snapshot, and governance gate validation"
)

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-^]{0,15}$")

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
    "downstream_gate_name",
    "downstream_gate_status",
    "downstream_gate_reason",
    "downstream_block_reasons",
    "approval_request_created",
    "approval_request_match_status",
    "approval_registry_path",
    "governance_approval_present_after_auto",
    "symbols_requested",
    "symbols_approved",
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


def _build_paper_approval_context(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Paper capital_deployment gate context — satisfies downstream execution gate checks."""
    ctx = dict(context or {})
    ctx["paperMode"] = True
    ctx["liveMode"] = False
    ctx.setdefault("lifecycleApproved", True)
    ctx.setdefault("allocationApproved", True)
    ctx.setdefault("riskConstraintsSatisfied", True)
    ctx.setdefault("executionAuthorized", True)
    ctx.setdefault("governanceApproved", False)
    return ctx


def _load_risk_flags() -> Dict[str, str]:
    flags: Dict[str, str] = {}
    if not RISK_OVERLAY.is_file():
        return flags
    try:
        import pandas as pd

        df = pd.read_csv(RISK_OVERLAY, low_memory=False)
        sym_col = "ticker" if "ticker" in df.columns else "symbol"
        flag_col = "risk_flag" if "risk_flag" in df.columns else "drag_flag"
        if sym_col not in df.columns:
            return flags
        for _, row in df.iterrows():
            sym = str(row.get(sym_col) or "").strip().upper()
            if sym:
                flags[sym] = str(row.get(flag_col) or "OK")
    except Exception:
        pass
    return flags


def _validate_symbols(symbols: Sequence[str]) -> Tuple[bool, str, List[str]]:
    norm = _norm_symbols(symbols)
    if not norm:
        return False, "NO_SYMBOLS", []
    block_reasons: List[str] = []
    risk_flags = _load_risk_flags()
    for sym in norm:
        if sym in ("NAN", "NONE", "NULL"):
            return False, f"INVALID_SYMBOL:{sym}", block_reasons
        if not _SYMBOL_RE.match(sym):
            return False, f"INVALID_SYMBOL:{sym}", block_reasons
        flag = risk_flags.get(sym, "OK")
        risk_set = {p for p in (str(flag).strip().upper().split("|")) if p and p != "OK"}
        if "BLOCK_NEW_BUY" in risk_set or "FORCE_EXIT" in risk_set:
            block_reasons.append(f"RISK_OVERLAY_BLOCK:{sym}")
    if block_reasons:
        return False, "|".join(block_reasons), block_reasons
    return True, "OK", block_reasons


def _evaluate_downstream_gate(
    symbols: Sequence[str],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from services.triton_execution_gate import evaluate_execution_gate

    ctx = _build_paper_approval_context(context)
    evaluation = evaluate_execution_gate(
        "capital_deployment",
        ctx,
        record_evidence=False,
        source="auto_paper_approval",
        symbols=symbols,
        action_requested="capital_deployment",
    )
    decision = str(evaluation.get("decision") or "").upper()
    reason = str(evaluation.get("reason") or "")
    return {
        "downstream_gate_name": "execution_gate:capital_deployment",
        "downstream_gate_status": decision,
        "downstream_gate_reason": reason,
        "downstream_block_reasons": reason if decision == "BLOCK" else "",
        "gate_evaluation": evaluation,
        "gate_context": ctx,
    }


def _format_gate_refusal(gate_info: Dict[str, Any]) -> str:
    status = gate_info.get("downstream_gate_status") or "BLOCK"
    reason = gate_info.get("downstream_gate_reason") or "unknown"
    if status == "BLOCK":
        return f"GATE_BLOCK:capital_deployment:{reason}"
    return f"GATE_{status}:{reason}"


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
    gate_info: Optional[Dict[str, Any]] = None,
    match_status: str = "",
    request_created: bool = False,
    governance_visible: Optional[bool] = None,
    symbols_approved: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    checks = checks or {}
    gate_info = gate_info or {}
    sym_list = _norm_symbols(symbols or [])
    row = {
        "timestamp": _utc_now_iso(),
        "request_id": request_id,
        "symbols": ",".join(sym_list),
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
        "downstream_gate_name": str(gate_info.get("downstream_gate_name") or ""),
        "downstream_gate_status": str(gate_info.get("downstream_gate_status") or ""),
        "downstream_gate_reason": str(gate_info.get("downstream_gate_reason") or ""),
        "downstream_block_reasons": str(gate_info.get("downstream_block_reasons") or ""),
        "approval_request_created": str(request_created).lower(),
        "approval_request_id": request_id,
        "approval_request_match_status": match_status,
        "approval_registry_path": str(APPROVAL_REGISTRY),
        "governance_approval_present_after_auto": (
            str(governance_visible).lower() if governance_visible is not None else ""
        ),
        "symbols_requested": ",".join(sym_list),
        "symbols_approved": ",".join(_norm_symbols(symbols_approved or [])),
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

    sym_ok, sym_reason, sym_blocks = _validate_symbols(symbols)
    checks["symbol_block_reasons"] = sym_blocks
    if not sym_ok:
        return False, sym_reason, checks

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
        if sym_set <= req_syms:
            return req
    return None


def _ensure_pending_request(
    symbols: Sequence[str],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    from services.execution_approval_workflow import (
        create_approval_request,
        find_approved_request,
    )
    from services.execution_governance_council import determine_authority_level

    ctx = _build_paper_approval_context(context)
    gate_info = _evaluate_downstream_gate(symbols, context=ctx)
    gate_eval = gate_info.get("gate_evaluation") or {}
    decision = str(gate_eval.get("decision") or "").upper()

    if ctx.get("liveMode") is True:
        return None, "LIVE_CONTEXT_REFUSED", gate_info

    existing = find_approved_request(
        "",
        "capital_deployment",
        symbols=symbols,
        context=ctx,
    )
    if existing:
        gate_info["approval_request_match_status"] = "ALREADY_APPROVED"
        return existing, "ALREADY_APPROVED", gate_info

    pending = _find_pending_for_symbols(symbols)
    if pending:
        gate_info["approval_request_match_status"] = "EXISTING_PENDING"
        return pending, "EXISTING_PENDING", gate_info

    if decision not in ("ALLOW", "REVIEW_REQUIRED"):
        gate_info["approval_request_match_status"] = "NOT_CREATED_GATE_BLOCKED"
        return None, _format_gate_refusal(gate_info), gate_info

    if decision == "REVIEW_REQUIRED":
        gate_info["approval_request_match_status"] = "NOT_CREATED_GATE_REVIEW_REQUIRED"
        return None, _format_gate_refusal(gate_info), gate_info

    authority_level, triggers = determine_authority_level(
        path="capital_deployment",
        context=ctx,
        evaluation=gate_eval,
        action_requested="capital_deployment",
    )
    req = create_approval_request(
        path="capital_deployment",
        context=ctx,
        evaluation=gate_eval,
        authority_level=authority_level,
        triggering_conditions=triggers,
        symbols=symbols,
        action_requested="capital_deployment",
    )
    gate_info["approval_request_match_status"] = "CREATED"
    gate_info["approval_request_created"] = True
    return req, "CREATED", gate_info


def _governance_visible(
    symbols: Sequence[str],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    from services.execution_approval_workflow import find_approved_request

    ctx = _build_paper_approval_context(context)
    approved = find_approved_request(
        "",
        "capital_deployment",
        symbols=symbols,
        context=ctx,
    )
    return approved is not None, approved


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
        validate_paper_capital_deployment_approvable,
    )

    ctx = _build_paper_approval_context(context)
    visible, approved = _governance_visible(symbols, context=ctx)
    if visible and approved:
        rid = str(approved.get("approval_id") or "")
        _log(verbose, "PASS", request_id=rid)
        audit = _write_audit(
            status="PASS",
            request_id=rid,
            symbols=symbols,
            checks=checks,
            match_status="ALREADY_APPROVED",
            governance_visible=True,
            symbols_approved=_norm_symbols(symbols),
        )
        return (
            True,
            "ALREADY_APPROVED",
            {
                "checks": checks,
                "approved": approved,
                "audit": audit,
                "governance_approval_present_after_auto": True,
            },
        )

    pending, pending_reason, gate_info = _ensure_pending_request(symbols, context=ctx)
    match_status = str(gate_info.get("approval_request_match_status") or pending_reason)
    request_created = bool(gate_info.get("approval_request_created"))

    if not pending:
        refusal = pending_reason
        _log(verbose, "REFUSED", reason=refusal)
        audit = _write_audit(
            status="REFUSED",
            refusal_reason=refusal,
            symbols=symbols,
            checks=checks,
            gate_info=gate_info,
            match_status=match_status,
            request_created=request_created,
            governance_visible=False,
        )
        return (
            False,
            refusal,
            {
                "checks": checks,
                "audit": audit,
                "gate": gate_info,
                "approval_request_match_status": match_status,
            },
        )

    rid = str(pending.get("approval_id") or "")

    if pending.get("approval_status") == "APPROVED":
        visible, _ = _governance_visible(symbols, context=ctx)
        _log(verbose, "PASS", request_id=rid)
        audit = _write_audit(
            status="PASS",
            request_id=rid,
            symbols=symbols,
            checks=checks,
            gate_info=gate_info,
            match_status=match_status,
            request_created=request_created,
            governance_visible=visible,
            symbols_approved=_norm_symbols(symbols),
        )
        return (
            True,
            pending_reason,
            {
                "checks": checks,
                "approved": pending,
                "audit": audit,
                "governance_approval_present_after_auto": visible,
            },
        )

    ok_val, val_reason = validate_paper_capital_deployment_approvable(pending)
    if not ok_val:
        _log(verbose, "REFUSED", reason=val_reason)
        audit = _write_audit(
            status="REFUSED",
            refusal_reason=val_reason,
            request_id=rid,
            symbols=symbols,
            checks=checks,
            gate_info=gate_info,
            match_status=match_status,
            request_created=request_created,
            governance_visible=False,
        )
        return False, val_reason, {"checks": checks, "audit": audit, "gate": gate_info}

    updated, approve_reason = approve_request(
        rid,
        approving_authority=AUTO_APPROVAL_AUTHORITY,
        approval_rationale=AUTO_APPROVAL_RATIONALE,
        validate_paper_capital=True,
    )
    if not updated:
        _log(verbose, "REFUSED", reason=approve_reason)
        audit = _write_audit(
            status="REFUSED",
            refusal_reason=approve_reason,
            request_id=rid,
            symbols=symbols,
            checks=checks,
            gate_info=gate_info,
            match_status=match_status,
            request_created=request_created,
            governance_visible=False,
        )
        return False, approve_reason, {"checks": checks, "audit": audit, "gate": gate_info}

    rid = str(updated.get("approval_id") or "")
    visible, approved_after = _governance_visible(symbols, context=ctx)
    _log(verbose, "PASS", request_id=rid)
    audit = _write_audit(
        status="PASS",
        request_id=rid,
        symbols=symbols,
        checks=checks,
        gate_info=gate_info,
        match_status=match_status,
        request_created=request_created,
        governance_visible=visible,
        symbols_approved=_norm_symbols(symbols),
    )
    return (
        True,
        pending_reason,
        {
            "checks": checks,
            "approved": updated,
            "audit": audit,
            "gate": gate_info,
            "governance_approval_present_after_auto": visible,
            "approval_request_match_status": match_status,
            "approved_visible": approved_after,
        },
    )


def parse_approved_symbols(approved_req: Optional[Dict[str, Any]]) -> List[str]:
    if not approved_req:
        return []
    raw = approved_req.get("symbol(s)") or approved_req.get("symbols") or ""
    return _norm_symbols(str(raw).split(","))


def approval_symbols_exact_match(
    approved_req: Optional[Dict[str, Any]],
    symbols: Sequence[str],
) -> bool:
    if not approved_req:
        return False
    return set(parse_approved_symbols(approved_req)) == set(_norm_symbols(symbols))


def ensure_governance_approval_for_symbols(
    symbols: Sequence[str],
    *,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    allow_auto_retry: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Ensure governance approval exists for the exact final symbol set.
    Optionally runs auto paper approval when enabled.
    """
    from services.execution_approval_workflow import find_approved_request

    sym_list = _norm_symbols(symbols)
    details: Dict[str, Any] = {
        "symbols_requested": sym_list,
        "approved_symbols": [],
        "selection_approval_sync": False,
    }
    if not sym_list:
        return False, "NO_SYMBOLS", details

    ctx = _build_paper_approval_context(context)
    approved = find_approved_request(
        "",
        "capital_deployment",
        symbols=sym_list,
        context=ctx,
    )
    if approval_symbols_exact_match(approved, sym_list):
        details["approved_symbols"] = parse_approved_symbols(approved)
        details["selection_approval_sync"] = True
        details["governance_approval_present"] = True
        details["approval_id"] = str((approved or {}).get("approval_id") or "")
        return True, "OK", details

    auto_enabled = (
        allow_auto_retry and os.environ.get("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "").strip() == "1"
    )
    details["auto_paper_approval_enabled"] = auto_enabled
    if auto_enabled:
        auto_ok, auto_reason, auto_details = attempt_auto_paper_approval(
            sym_list,
            context=ctx,
            verbose=verbose,
        )
        details["auto_paper_approval"] = auto_details
        approved = find_approved_request(
            "",
            "capital_deployment",
            symbols=sym_list,
            context=ctx,
        )
        if auto_ok and approval_symbols_exact_match(approved, sym_list):
            details["approved_symbols"] = parse_approved_symbols(approved)
            details["selection_approval_sync"] = True
            details["governance_approval_present"] = True
            details["approval_id"] = str((approved or {}).get("approval_id") or "")
            return True, auto_reason, details
        if not auto_ok:
            return False, auto_reason, details

    details["approved_symbols"] = parse_approved_symbols(approved)
    if approved and not approval_symbols_exact_match(approved, sym_list):
        return False, "FINAL_SELECTION_NOT_APPROVED", details
    return False, "GOVERNANCE_APPROVAL_REQUIRED", details
