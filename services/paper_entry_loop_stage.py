"""
paper_entry_loop_stage.py
Controlled paper-entry execution stage for run_continuous_loop.ps1 (Phase 195Y).

Invokes execute_trades --mode paper --execute only when all safety gates pass.
Never submits live orders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
CONFIG = ROOT / "config"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

EXEC_READINESS = RESULTS / "execution_readiness.json"
TRADE_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
ALLOC_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
AUDIT_JSON = RESULTS / "paper_entry_loop_stage_audit.json"
LAST_MARKET_DECISION_JSON = RESULTS / "paper_entry_loop_stage_last_market_decision.json"
DISPATCH_CONTRACT_PATH = LIVE / "paper_entry_dispatch_contract.json"
DISPATCH_CONTRACT_TTL_MINUTES = 15
EXEC_TRADES_CFG = CONFIG / "execute_trades.json"
GOVERNED_ALLOC_SOURCE = "portfolio_allocation_recommendations.csv"

_EXECUTE_INVOKED = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _python_exe() -> str:
    return str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable


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


def _norm(val: Any) -> str:
    return str(val or "").strip().upper()


def _file_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0


def _find_execute_trades_running() -> Tuple[bool, List[str]]:
    if sys.platform != "win32":
        return False, []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" "
        "| Where-Object { $_.CommandLine -like '*execute_trades*' -and $_.CommandLine -like '*--execute*' } "
        "| Select-Object -ExpandProperty CommandLine"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
        return len(lines) > 0, lines
    except (subprocess.SubprocessError, OSError):
        return False, []


def _load_open_buy_keys() -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    if not OPEN_ORDERS.is_file():
        return keys
    try:
        df = pd.read_csv(OPEN_ORDERS, low_memory=False)
        for _, row in df.iterrows():
            sym = _norm(row.get("symbol") or row.get("ticker"))
            side = _norm(row.get("side") or row.get("action") or "BUY")
            if sym and side in ("BUY", "B"):
                keys.add((sym, "BUY"))
    except Exception:
        pass
    return keys


def _load_held_symbols() -> Set[str]:
    held: Set[str] = set()
    if not POSITIONS.is_file():
        return held
    try:
        df = pd.read_csv(POSITIONS, low_memory=False)
        sym_col = "symbol" if "symbol" in df.columns else "ticker"
        if sym_col not in df.columns:
            return held
        for _, row in df.iterrows():
            sym = _norm(row.get(sym_col))
            qty = row.get("qty") or row.get("quantity") or 0
            try:
                if sym and float(qty) != 0:
                    held.add(sym)
            except (TypeError, ValueError):
                if sym:
                    held.add(sym)
    except Exception:
        pass
    return held


def _effective_max_positions() -> int:
    cfg = _load_json(EXEC_TRADES_CFG)
    try:
        return int(cfg.get("max_positions", 25) or 25)
    except (TypeError, ValueError):
        return 25


def _load_risk_map() -> Dict[str, str]:
    risk_map: Dict[str, str] = {}
    if not RISK_OVERLAY.is_file():
        return risk_map
    try:
        ro = pd.read_csv(RISK_OVERLAY, low_memory=False)
        sym_col = "ticker" if "ticker" in ro.columns else "symbol"
        flag_col = "risk_flag" if "risk_flag" in ro.columns else "drag_flag"
        if sym_col in ro.columns:
            for _, row in ro.iterrows():
                sym = _norm(row.get(sym_col))
                if sym:
                    risk_map[sym] = str(row.get(flag_col) or "OK")
    except Exception:
        pass
    return risk_map


def _load_trade_opportunity_enrichment() -> Dict[str, Dict[str, Any]]:
    """Optional edge/confidence enrichment; not an execution source."""
    out: Dict[str, Dict[str, Any]] = {}
    if not TRADE_OPPORTUNITIES.is_file():
        return out
    try:
        df = pd.read_csv(TRADE_OPPORTUNITIES, low_memory=False)
        sym_col = "ticker" if "ticker" in df.columns else "symbol"
        if sym_col not in df.columns:
            return out
        for _, row in df.iterrows():
            sym = _norm(row.get(sym_col))
            if sym:
                out[sym] = row.to_dict()
    except Exception:
        pass
    return out


def _allocation_sym_col(df: pd.DataFrame) -> Optional[str]:
    if "ticker" in df.columns:
        return "ticker"
    if "symbol" in df.columns:
        return "symbol"
    return None


def _evaluate_allocation_entry_row(
    *,
    sym: str,
    action: str,
    row: Dict[str, Any],
    risk_map: Dict[str, str],
    open_buy: Set[Tuple[str, str]],
    held: Set[str],
    at_cap: bool,
) -> Tuple[bool, List[str], str]:
    blocked_reasons: List[str] = []
    row_flag = str(row.get("risk_flag") or "OK")
    risk_flag = risk_map.get(sym, row_flag)
    risk_set = {p for p in (_norm(x) for x in risk_flag.split("|")) if p and p != "OK"}

    if not sym:
        return False, ["MISSING_TICKER"], risk_flag
    if action == "BLOCK":
        blocked_reasons.append("ALLOCATION_BLOCK")
    if "BLOCK_NEW_BUY" in risk_set:
        blocked_reasons.append("RISK_OVERLAY_BLOCK")
    if "FORCE_EXIT" in risk_set:
        blocked_reasons.append("FORCE_EXIT")
    if action == "OPEN_NEW" and sym in held:
        blocked_reasons.append("ALREADY_HELD")
    if action == "ADD" and sym not in held:
        blocked_reasons.append("ADD_REQUIRES_EXISTING_POSITION")
    if action not in ("OPEN_NEW", "ADD"):
        blocked_reasons.append(f"ALLOCATION_ACTION_{action or 'UNKNOWN'}")
    if (sym, "BUY") in open_buy:
        blocked_reasons.append("OPEN_BUY_ORDER")
    if at_cap and action == "OPEN_NEW":
        blocked_reasons.append("MAX_POSITIONS")

    try:
        from services.duplicate_order_guard import check_duplicate_order

        session = datetime.now(timezone.utc).strftime("%Y%m%d")
        dup, dup_reason = check_duplicate_order(
            symbol=sym,
            side="buy",
            session=session,
            open_side_keys=open_buy,
        )
        if dup:
            blocked_reasons.append(f"DUPLICATE:{dup_reason}")
    except Exception:
        pass

    return len(blocked_reasons) == 0, blocked_reasons, risk_flag


def _load_governed_allocation_candidates() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Governed paper-entry candidates from portfolio_allocation_recommendations.csv.
    trade_opportunities.csv is enrichment-only (not an execution source).
    """
    handoff: Dict[str, Any] = {
        "source_file_used": str(ALLOC_RECOMMENDATIONS.relative_to(ROOT)).replace("\\", "/"),
        "source_file_mtime": None,
        "allocation_candidates_count": 0,
        "allocation_open_new_count": 0,
        "allocation_add_count": 0,
        "eligible_entry_symbols": [],
        "rejected_entry_symbols": {},
        "governed_source": GOVERNED_ALLOC_SOURCE,
    }
    if not ALLOC_RECOMMENDATIONS.is_file():
        handoff["source_error"] = "allocation_recommendations_missing"
        return [], handoff

    try:
        mtime = datetime.fromtimestamp(
            ALLOC_RECOMMENDATIONS.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        handoff["source_file_mtime"] = mtime
        df = pd.read_csv(ALLOC_RECOMMENDATIONS, low_memory=False)
    except Exception as exc:
        handoff["source_error"] = str(exc)
        return [], handoff

    sym_col = _allocation_sym_col(df)
    if not sym_col or df.empty or "recommended_action" not in df.columns:
        handoff["source_error"] = "allocation_schema_invalid"
        return [], handoff

    risk_map = _load_risk_map()
    opp_enrich = _load_trade_opportunity_enrichment()
    open_buy = _load_open_buy_keys()
    held = _load_held_symbols()
    max_pos = _effective_max_positions()
    at_cap = len(held) >= max_pos

    candidates: List[Dict[str, Any]] = []
    rejected: Dict[str, List[str]] = {}
    eligible_symbols: List[str] = []

    for _, row in df.iterrows():
        sym = _norm(row.get(sym_col))
        action = _norm(row.get("recommended_action"))
        if action not in ("OPEN_NEW", "ADD"):
            continue

        handoff["allocation_candidates_count"] += 1
        if action == "OPEN_NEW":
            handoff["allocation_open_new_count"] += 1
        else:
            handoff["allocation_add_count"] += 1

        row_dict = row.to_dict()
        eligible, blocked_reasons, risk_flag = _evaluate_allocation_entry_row(
            sym=sym,
            action=action,
            row=row_dict,
            risk_map=risk_map,
            open_buy=open_buy,
            held=held,
            at_cap=at_cap,
        )
        if eligible:
            eligible_symbols.append(sym)
        else:
            rejected[sym] = blocked_reasons

        enrich = opp_enrich.get(sym, {})
        candidates.append(
            {
                "ticker": sym,
                "source": GOVERNED_ALLOC_SOURCE,
                "opportunity_type": _norm(enrich.get("opportunity_type") or action),
                "edge_score": row_dict.get("edge_score") or enrich.get("edge_score"),
                "confidence": row_dict.get("confidence") or enrich.get("confidence"),
                "allocation_score": row_dict.get("allocation_score"),
                "eligible_for_auto_entry": eligible,
                "block_reasons": "|".join(blocked_reasons) if blocked_reasons else "",
                "risk_flag": risk_flag,
                "allocation_action": action,
                "is_currently_held": sym in held,
            }
        )

    handoff["eligible_entry_symbols"] = sorted(set(eligible_symbols))
    handoff["rejected_entry_symbols"] = rejected
    return candidates, handoff


def _load_entry_candidates() -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Backward-compatible wrapper; governed allocation is the execution source."""
    candidates, _handoff = _load_governed_allocation_candidates()
    df = pd.DataFrame()
    if ALLOC_RECOMMENDATIONS.is_file():
        try:
            df = pd.read_csv(ALLOC_RECOMMENDATIONS, low_memory=False)
        except Exception:
            pass
    return candidates, df


def _planner_rejection_detail(preview_meta: Dict[str, Any]) -> str:
    if preview_meta.get("final_selection_block_reason"):
        return str(preview_meta["final_selection_block_reason"])
    skip = preview_meta.get("skip_reasons") or {}
    if isinstance(skip, dict) and skip:
        top = next(iter(skip.items()))
        return f"planner_skip:{top[0]}={top[1]}"
    if preview_meta.get("lifecycle_blocked"):
        return f"lifecycle_blocked:{preview_meta.get('lifecycle_reason', '')}"
    if preview_meta.get("empty_opportunities"):
        return "not_in_planner_universe"
    warnings = preview_meta.get("preview_warnings") or preview_meta.get("warnings") or []
    if warnings:
        return str(warnings[0])
    return ""


def _check_allocation_planner_contract(
    eligible_symbols: List[str],
    final_symbols: List[str],
    preview_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    eligible_set = {_norm(s) for s in eligible_symbols if s}
    final_set = {_norm(s) for s in final_symbols if s}
    detail: Dict[str, Any] = {
        "eligible_allocation_symbols": sorted(eligible_set),
        "planner_preview_symbols": sorted(final_set),
        "contract_mismatch_detected": False,
        "planner_rejection_detail": _planner_rejection_detail(preview_meta),
    }

    if not eligible_set:
        detail["contract_status"] = "NO_ELIGIBLE_ALLOCATION"
        return True, "OK", detail

    if not final_set:
        detail["contract_mismatch_detected"] = True
        detail["contract_status"] = "PLANNER_EMPTY"
        return (
            False,
            "CONTRACT_MISMATCH:allocation_eligible_but_planner_empty",
            detail,
        )

    outside = sorted(final_set - eligible_set)
    if outside:
        detail["contract_mismatch_detected"] = True
        detail["planner_symbols_outside_allocation"] = outside
        detail["contract_status"] = "PLANNER_OUTSIDE_ALLOCATION"
        return (
            False,
            f"CONTRACT_MISMATCH:planner={sorted(final_set)} outside allocation_eligible={sorted(eligible_set)}",
            detail,
        )

    detail["contract_status"] = "ALIGNED"
    return True, "OK", detail


def _merge_handoff_into_details(details: Dict[str, Any], handoff: Dict[str, Any]) -> None:
    checks = details.setdefault("checks", {})
    for key in (
        "source_file_used",
        "source_file_mtime",
        "allocation_candidates_count",
        "allocation_open_new_count",
        "allocation_add_count",
        "eligible_entry_symbols",
        "rejected_entry_symbols",
        "governed_source",
    ):
        if key in handoff:
            checks[key] = handoff[key]
    details["allocation_handoff"] = handoff


def _flatten_audit_fields(
    audit: Dict[str, Any],
    *,
    details: Dict[str, Any],
    handoff: Dict[str, Any],
) -> None:
    checks = details.get("checks") or {}
    from services.paper_arm_status import classify_paper_arm_confirm_status

    try:
        from services.arm_mode import is_paper_arm_confirm_valid, load_arm_config

        paper_arm_valid = is_paper_arm_confirm_valid(load_arm_config())
    except Exception:
        paper_arm_valid = classify_paper_arm_confirm_status() == "VALID"

    held = _load_held_symbols()
    open_buy = _load_open_buy_keys()

    audit.update(
        {
            "allocation_candidates_count": handoff.get("allocation_candidates_count", 0),
            "allocation_open_new_count": handoff.get("allocation_open_new_count", 0),
            "allocation_add_count": handoff.get("allocation_add_count", 0),
            "eligible_entry_symbols": handoff.get("eligible_entry_symbols") or [],
            "rejected_entry_symbols": handoff.get("rejected_entry_symbols") or {},
            "source_file_used": handoff.get("source_file_used"),
            "source_file_mtime": handoff.get("source_file_mtime"),
            "paper_arm_status": "VALID" if paper_arm_valid else classify_paper_arm_confirm_status(),
            "execution_readiness_status": checks.get("execution_readiness_status"),
            "market_open": checks.get("market_open"),
            "open_orders_count": len(open_buy),
            "positions_count": len(held),
            "final_selected_symbols": checks.get("final_selected_symbols") or [],
            "final_selection_status": checks.get("final_selection_status"),
            "final_selection_block_reason": checks.get("final_selection_block_reason"),
            "planner_preview_symbols": checks.get("planner_preview_symbols") or [],
            "planner_input_mode": checks.get("planner_input_mode"),
            "planner_considered_symbols": checks.get("planner_considered_symbols") or [],
            "planner_selected_symbols": checks.get("planner_selected_symbols") or [],
            "planner_rejected_symbols": checks.get("planner_rejected_symbols") or {},
            "contract_mismatch_detected": bool(checks.get("contract_mismatch_detected")),
        }
    )


def _write_stage_audit(audit: Dict[str, Any], *, market_open: bool) -> None:
    _atomic_write_json(AUDIT_JSON, audit)
    if market_open:
        _atomic_write_json(LAST_MARKET_DECISION_JSON, audit)


def _dispatch_contract_checksum(doc: Dict[str, Any]) -> str:
    payload = {k: v for k, v in doc.items() if k != "checksum"}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_paper_entry_dispatch_contract(
    details: Dict[str, Any],
    handoff: Dict[str, Any],
) -> Dict[str, Any]:
    """Write governed dispatch contract before execute_trades subprocess."""
    checks = details.get("checks") or {}
    from services.paper_arm_status import classify_paper_arm_confirm_status

    final_selected = list(
        checks.get("planner_selected_symbols") or checks.get("final_selected_symbols") or []
    )
    allowed = list(
        checks.get("eligible_entry_symbols") or handoff.get("eligible_entry_symbols") or []
    )
    approved = list(
        details.get("symbols_approved") or checks.get("approved_symbols") or final_selected
    )
    rejected = dict(
        checks.get("planner_rejected_symbols") or handoff.get("rejected_entry_symbols") or {}
    )

    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=DISPATCH_CONTRACT_TTL_MINUTES)
    dispatch_id = uuid.uuid4().hex[:16]

    try:
        from services.arm_mode import is_paper_arm_confirm_valid, load_arm_config

        paper_arm_valid = is_paper_arm_confirm_valid(load_arm_config())
        paper_arm_status = "VALID" if paper_arm_valid else classify_paper_arm_confirm_status()
    except Exception:
        paper_arm_status = classify_paper_arm_confirm_status()

    readiness = _load_json(EXEC_READINESS)

    contract: Dict[str, Any] = {
        "dispatch_id": dispatch_id,
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "paper_entry_loop_stage",
        "mode": "paper",
        "governed_source": GOVERNED_ALLOC_SOURCE,
        "approved_symbols": sorted({_norm(s) for s in approved if s}),
        "final_selected_symbols": sorted({_norm(s) for s in final_selected if s}),
        "allowed_symbols": sorted({_norm(s) for s in allowed if s}),
        "rejected_symbols": rejected,
        "planner_input_mode": checks.get("planner_input_mode") or "allocation",
        "approval_request_id": str(details.get("approval_request_id") or ""),
        "paper_arm_status": paper_arm_status,
        "execution_readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
        "market_open": bool(checks.get("market_open")),
        "live_execution_permitted": False,
        "contract_expires_at_utc": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    contract["approved_symbols"] = sorted(contract["approved_symbols"])
    contract["final_selected_symbols"] = sorted(contract["final_selected_symbols"])
    contract["allowed_symbols"] = sorted(contract["allowed_symbols"])
    contract["checksum"] = _dispatch_contract_checksum(contract)
    _atomic_write_json(DISPATCH_CONTRACT_PATH, contract)
    return contract


def _risk_blocked_in_selection(symbols: List[str]) -> List[str]:
    from services.auto_paper_approval import _load_risk_flags

    flags = _load_risk_flags()
    blocked: List[str] = []
    for sym in (_norm(s) for s in symbols):
        if not sym:
            continue
        flag = flags.get(sym, "OK")
        risk_set = {p for p in (_norm(x) for x in str(flag).split("|")) if p and p != "OK"}
        if "BLOCK_NEW_BUY" in risk_set or "FORCE_EXIT" in risk_set:
            blocked.append(sym)
    return blocked


def _preview_final_planned_symbols(
    *,
    verbose: bool = False,
    allowed_symbols: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    from services.execute_trades import preview_planned_buy_symbols

    if allowed_symbols:
        return preview_planned_buy_symbols(
            "paper",
            verbose=verbose,
            for_execute=True,
            source="allocation",
            allowed_symbols=allowed_symbols,
        )
    return preview_planned_buy_symbols("paper", verbose=verbose, for_execute=True)


def _verify_final_selection_before_execute(
    expected_symbols: List[str],
    *,
    verbose: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Re-preview planner output and ensure governance approval matches exactly."""
    from services.auto_paper_approval import (
        _build_paper_approval_context,
        ensure_governance_approval_for_symbols,
        parse_approved_symbols,
    )
    from services.execution_approval_workflow import find_approved_request

    sync_details: Dict[str, Any] = {
        "expected_symbols": sorted({_norm(s) for s in expected_symbols if s}),
    }
    final_symbols, preview_meta = _preview_final_planned_symbols(
        verbose=verbose,
        allowed_symbols=list(sync_details["expected_symbols"]),
    )
    sync_details["planner_preview"] = preview_meta
    sync_details["final_selected_symbols"] = final_symbols

    blocked = _risk_blocked_in_selection(final_symbols)
    if blocked:
        sync_details["risk_blocked_in_final_selection"] = blocked
        return False, f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{','.join(blocked)}", sync_details

    expected_set = set(sync_details["expected_symbols"])
    final_set = set(final_symbols)
    if expected_set != final_set:
        sync_details["selection_drift"] = {
            "expected": sorted(expected_set),
            "final": sorted(final_set),
        }

    gov_ctx = _build_paper_approval_context({"paperMode": True, "liveMode": False})
    approved = find_approved_request(
        "",
        "capital_deployment",
        symbols=final_symbols,
        context=gov_ctx,
    )
    sync_details["approved_symbols"] = parse_approved_symbols(approved)

    from services.auto_paper_approval import approval_symbols_exact_match

    if approval_symbols_exact_match(approved, final_symbols):
        sync_details["selection_approval_sync"] = True
        return True, "OK", sync_details

    ok, reason, gov_details = ensure_governance_approval_for_symbols(
        final_symbols,
        context=gov_ctx,
        verbose=verbose,
        allow_auto_retry=True,
    )
    sync_details["governance_sync"] = gov_details
    if ok:
        sync_details["selection_approval_sync"] = True
        sync_details["approved_symbols"] = gov_details.get("approved_symbols") or final_symbols
        return True, "OK", sync_details

    if final_set != expected_set:
        return False, "FINAL_SELECTION_NOT_APPROVED", sync_details
    return False, reason or "FINAL_SELECTION_NOT_APPROVED", sync_details


def evaluate_paper_entry_gates(*, verbose: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    details: Dict[str, Any] = {"checks": {}, "candidates": []}

    candidates, handoff = _load_governed_allocation_candidates()
    details["candidates"] = candidates
    _merge_handoff_into_details(details, handoff)

    def block(reason: str, check: str, *, safe: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        details["failed_check"] = check
        details["safe_block"] = safe
        label = "SAFE_BLOCK" if safe else "FAIL"
        if verbose:
            print(f"[PAPER_ENTRY_GATE] status={label} reason={reason}", flush=True)
        return False, reason, details

    paper_env = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() == "1"
    auto_env = os.environ.get("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "").strip() == "1"
    details["checks"]["TRITON_ENABLE_PAPER_EXECUTION"] = paper_env
    details["checks"]["TRITON_ENABLE_AUTO_PAPER_ENTRIES"] = auto_env
    if not paper_env:
        return block("TRITON_ENABLE_PAPER_EXECUTION not set to 1", "paper_execution_env")
    if not auto_env:
        return block("TRITON_ENABLE_AUTO_PAPER_ENTRIES not set to 1", "auto_paper_entries_env")

    if not ALLOC_RECOMMENDATIONS.is_file():
        return block(
            "portfolio_allocation_recommendations.csv missing",
            "allocation_source",
        )

    from services.guard_auto import is_market_open_et

    now_utc = datetime.now(timezone.utc)
    market_open = is_market_open_et(now_utc)
    details["checks"]["market_open"] = market_open
    if not market_open:
        if verbose:
            print("[PAPER_AUTO_APPROVAL] status=REFUSED reason=MARKET_CLOSED", flush=True)
        return block("market is closed", "market_open", safe=True)

    readiness = _load_json(EXEC_READINESS)
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    live_permitted = bool(readiness.get("live_execution_permitted", True))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))
    failed_checks = list(readiness.get("failed_checks") or [])
    details["checks"]["execution_readiness_status"] = readiness_status
    details["checks"]["live_execution_permitted"] = live_permitted
    details["checks"]["paper_execution_permitted"] = paper_permitted
    details["checks"]["readiness_failed_checks"] = failed_checks
    if readiness_status != "READY":
        stale_hint = ""
        if "lifecycle" in failed_checks or "signals" in failed_checks:
            stale_hint = " (stale lifecycle/signals — run phase_196a --repair)"
        return block(f"execution_readiness={readiness_status}{stale_hint}", "execution_readiness")
    if live_permitted:
        return block("live_execution_permitted must be false", "live_block")
    if not paper_permitted:
        return block("paper_execution_permitted must be true", "paper_authorization")

    running, cmd_lines = _find_execute_trades_running()
    details["checks"]["execute_trades_execute_running"] = running
    if running:
        return block("execute_trades --execute already running", "concurrent_execute")

    snap_age = _file_age_minutes(OPEN_ORDERS)
    details["checks"]["open_orders_snapshot_age_minutes"] = snap_age
    if snap_age is None:
        return block("broker open_orders snapshot missing", "broker_snapshot")
    if snap_age > 120:
        return block(f"broker snapshot stale ({snap_age:.0f}m)", "broker_snapshot")

    try:
        from services.arm_mode import (
            is_paper_arm_confirm_valid,
            load_arm_config,
            resolve_arm_mutation_permissions,
            validate_mode,
        )

        arm_cfg = load_arm_config()
        arm_perms, arm_blocks = resolve_arm_mutation_permissions(arm_cfg)
        arm_mode = validate_mode(arm_cfg.get("mode"))
        arm_confirm_valid = is_paper_arm_confirm_valid(arm_cfg)
        details["checks"]["arm_mode"] = arm_mode
        details["checks"]["arm_confirm_valid"] = arm_confirm_valid
        details["checks"]["arm_permissions"] = arm_perms
        details["checks"]["arm_block_reasons"] = arm_blocks
        if arm_mode == "ASSISTED" and not arm_confirm_valid:
            return block("ARM_EXPIRED_OR_MISSING paper_arm_confirm", "arm_permission")
        if not arm_perms.get("execute_trades"):
            reason = "ARM permission denied"
            if arm_blocks:
                reason += f": {','.join(arm_blocks)}"
            if "ASSISTED_CONFIRMATION" in reason or not arm_confirm_valid:
                reason = "ARM_EXPIRED_OR_MISSING"
            return block(reason, "arm_permission")
    except Exception as e:
        return block(f"ARM check error: {e}", "arm_permission")

    candidates, handoff_reload = _load_governed_allocation_candidates()
    details["candidates"] = candidates
    _merge_handoff_into_details(details, handoff_reload)
    eligible = [c for c in candidates if c.get("eligible_for_auto_entry")]
    details["checks"]["entry_rows_total"] = len(candidates)
    details["checks"]["entry_rows_eligible"] = len(eligible)
    if not candidates:
        return block(
            "no OPEN_NEW/ADD rows in portfolio_allocation_recommendations.csv",
            "entry_candidates",
        )
    if not eligible:
        return block(
            "no eligible allocation candidates after risk/duplicate/cap gates",
            "entry_candidates",
        )

    try:
        eligible_syms = list(handoff_reload.get("eligible_entry_symbols") or [])
        final_symbols, preview_meta = _preview_final_planned_symbols(
            verbose=verbose,
            allowed_symbols=eligible_syms,
        )
        details["planner_preview"] = preview_meta
        details["checks"]["eligible_entry_symbols"] = eligible_syms
        details["checks"]["final_selected_symbols"] = final_symbols
        details["checks"]["planner_preview_symbols"] = final_symbols
        details["checks"]["planner_input_mode"] = preview_meta.get("planner_input_mode")
        details["checks"]["planner_considered_symbols"] = preview_meta.get(
            "planner_considered_symbols"
        )
        details["checks"]["planner_selected_symbols"] = (
            preview_meta.get("planner_selected_symbols") or final_symbols
        )
        details["checks"]["planner_rejected_symbols"] = (
            preview_meta.get("planner_rejected_symbols") or {}
        )
        details["checks"]["preview_candidate"] = preview_meta.get("preview_candidate")
        details["checks"]["preview_candidate_risk_flag"] = preview_meta.get(
            "preview_candidate_risk_flag"
        )
        details["checks"]["preview_candidate_executable"] = preview_meta.get(
            "preview_candidate_executable"
        )
        details["checks"]["governed_executable_preview_candidate"] = preview_meta.get(
            "governed_executable_preview_candidate"
        )
        details["checks"]["allocation_action"] = preview_meta.get("allocation_action")
        details["checks"]["final_selection_status"] = preview_meta.get("final_selection_status")
        details["checks"]["final_selection_block_reason"] = preview_meta.get(
            "final_selection_block_reason"
        ) or _planner_rejection_detail(preview_meta)
        if preview_meta.get("preview_warnings"):
            details["checks"]["preview_warnings"] = preview_meta.get("preview_warnings")

        contract_ok, contract_reason, contract_detail = _check_allocation_planner_contract(
            eligible_syms,
            final_symbols,
            preview_meta,
        )
        details["allocation_planner_contract"] = contract_detail
        details["checks"]["contract_mismatch_detected"] = contract_detail.get(
            "contract_mismatch_detected"
        )
        if not contract_ok:
            return block(contract_reason, "contract_mismatch")

        blocked_risk = _risk_blocked_in_selection(final_symbols)
        if blocked_risk:
            details["checks"]["risk_blocked_in_final_selection"] = blocked_risk
            return block(
                f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{','.join(blocked_risk)}",
                "final_selection",
            )

        if not final_symbols:
            if preview_meta.get("lifecycle_blocked"):
                return block(
                    f"planner lifecycle blocked: {preview_meta.get('lifecycle_reason', '')}",
                    "planner_preview",
                    safe=True,
                )
            return block(
                "NO_FINAL_PLANNED_ENTRIES after planner filtering",
                "final_selection",
                safe=True,
            )

        from services.auto_paper_approval import ensure_governance_approval_for_symbols

        gov_ok, gov_reason, gov_details = ensure_governance_approval_for_symbols(
            final_symbols,
            context={"paperMode": True, "liveMode": False},
            verbose=verbose,
            allow_auto_retry=True,
        )
        details["governance_approval"] = gov_details
        details["checks"]["governance_approval_present"] = bool(
            gov_details.get("governance_approval_present")
        )
        details["checks"]["governance_approval_present_after_auto"] = bool(
            gov_details.get("governance_approval_present")
        )
        details["checks"]["symbols_requested"] = final_symbols
        details["checks"]["approved_symbols"] = gov_details.get("approved_symbols") or []
        details["checks"]["selection_approval_sync"] = bool(
            gov_details.get("selection_approval_sync")
        )
        details["approval_request_id"] = gov_details.get("approval_id") or ""
        if gov_details.get("auto_paper_approval"):
            auto_details = gov_details["auto_paper_approval"]
            details["auto_paper_approval"] = auto_details
            gate = auto_details.get("gate") or {}
            details["downstream_gate_name"] = gate.get("downstream_gate_name")
            details["downstream_gate_status"] = gate.get("downstream_gate_status")
            details["downstream_gate_reason"] = gate.get("downstream_gate_reason")
            details["downstream_block_reasons"] = gate.get("downstream_block_reasons")
            details["approval_request_created"] = gate.get("approval_request_created")
            if not details["approval_request_id"]:
                details["approval_request_id"] = (
                    (auto_details.get("audit") or {}).get("request_id")
                    or (auto_details.get("approved") or {}).get("approval_id")
                    or ""
                )
            details["approval_request_match_status"] = auto_details.get(
                "approval_request_match_status"
            ) or gate.get("approval_request_match_status")
        details["approval_registry_path"] = str(
            ROOT / "data" / "governance" / "execution_approval_requests.json"
        )
        details["symbols_approved"] = gov_details.get("approved_symbols") or []

        if not gov_ok:
            if gov_reason == "FINAL_SELECTION_NOT_APPROVED":
                return block("FINAL_SELECTION_NOT_APPROVED", "final_selection")
            if gov_details.get("auto_paper_approval") and not gov_details.get(
                "governance_approval_present"
            ):
                return block(
                    "AUTO_APPROVAL_NOT_VISIBLE_TO_EXECUTION_GATE",
                    "governance_approval",
                )
            return block(
                gov_reason if gov_reason else "GOVERNANCE_APPROVAL_REQUIRED",
                "governance_approval",
            )
    except Exception as e:
        return block(f"governance approval check error: {e}", "governance_approval")

    global _EXECUTE_INVOKED
    if _EXECUTE_INVOKED:
        return block("execute_trades already invoked this cycle", "single_invoke_guard")

    if verbose:
        print(
            f"[PAPER_ENTRY_GATE] status=PASS reason=OK "
            f"final_selected={details['checks'].get('final_selected_symbols')}",
            flush=True,
        )
    details["failed_check"] = None
    details["safe_block"] = False
    return True, "OK", details


def run_paper_entry_stage(*, verbose: bool = False, dry_run: bool = False) -> int:
    global _EXECUTE_INVOKED

    gate_ok, reason, details = evaluate_paper_entry_gates(verbose=verbose)
    handoff = details.get("allocation_handoff") or {}
    market_open = bool((details.get("checks") or {}).get("market_open"))

    audit: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "gate_pass": gate_ok,
        "gate_reason": reason,
        "dry_run": dry_run,
        "executed": False,
        "execute_exit_code": None,
        "safe_block": bool(details.get("safe_block")),
        "details": details,
    }
    _flatten_audit_fields(audit, details=details, handoff=handoff)

    if not gate_ok:
        safe = bool(details.get("safe_block"))
        if reason == "market is closed":
            print("[PAPER_ENTRY_EXECUTION_BLOCKED] reason=MARKET_CLOSED", flush=True)
        elif safe:
            print(f"[PAPER_ENTRY_EXECUTION_BLOCKED] reason={reason} (SAFE_BLOCK)", flush=True)
        else:
            print(f"[PAPER_ENTRY_EXECUTION_BLOCKED] reason={reason}", flush=True)
        _write_stage_audit(audit, market_open=market_open)
        return 0

    if dry_run:
        print(
            "[PAPER_ENTRY_EXECUTION] dry_run=true; gates passed, skipping execute_trades",
            flush=True,
        )
        audit["message"] = "dry_run_skip"
        _write_stage_audit(audit, market_open=market_open)
        return 0

    if _EXECUTE_INVOKED:
        print("[PAPER_ENTRY_EXECUTION_BLOCKED] reason=already_invoked_this_process", flush=True)
        return 0

    expected = list(details.get("checks", {}).get("eligible_entry_symbols") or [])
    sync_ok, sync_reason, sync_details = _verify_final_selection_before_execute(
        expected,
        verbose=verbose,
    )
    details["pre_execute_selection_sync"] = sync_details
    audit["details"] = details
    _flatten_audit_fields(audit, details=details, handoff=handoff)
    if not sync_ok:
        print(f"[PAPER_ENTRY_EXECUTION_BLOCKED] reason={sync_reason}", flush=True)
        audit["gate_pass"] = False
        audit["gate_reason"] = sync_reason
        audit["executed"] = False
        _write_stage_audit(audit, market_open=market_open)
        return 0

    dispatch_contract = _write_paper_entry_dispatch_contract(details, handoff)
    dispatch_symbols = list(dispatch_contract.get("final_selected_symbols") or [])
    audit["dispatch_contract_path"] = str(DISPATCH_CONTRACT_PATH.relative_to(ROOT)).replace(
        "\\", "/"
    )
    audit["dispatch_id"] = dispatch_contract.get("dispatch_id")
    audit["dispatch_symbols"] = dispatch_symbols
    audit["execute_trades_dispatch_mode"] = True

    _EXECUTE_INVOKED = True
    py = _python_exe()
    cmd = [
        py,
        "-m",
        "services.execute_trades",
        "--mode",
        "paper",
        "--execute",
        "--verbose",
        "--dispatch-contract",
        str(DISPATCH_CONTRACT_PATH),
    ]
    print("[PAPER_ENTRY_EXECUTION] starting execute_trades paper --execute (dispatch)", flush=True)
    if verbose:
        print(f"[PAPER_ENTRY_EXECUTION] CMD: {' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
        rc = int(proc.returncode)
    except Exception as e:
        rc = 1
        print(f"[PAPER_ENTRY_EXECUTION] error={e}", flush=True)
    dur = time.perf_counter() - t0

    dispatch_audit = _load_json(RESULTS / "execute_trades_dispatch_audit.json")
    exec_selected = list(dispatch_audit.get("execute_trades_selected_symbols") or [])
    dispatch_mismatch = bool(dispatch_audit.get("dispatch_contract_mismatch"))
    block_reason = str(dispatch_audit.get("block_reason") or "")
    if not block_reason and rc == 2:
        block_reason = "execute_trades_blocked"

    print(f"[PAPER_ENTRY_EXECUTION] completed exit_code={rc} duration_sec={dur:.1f}", flush=True)
    audit["executed"] = True
    audit["execute_exit_code"] = rc
    audit["duration_sec"] = round(dur, 3)
    audit["execute_trades_selected_symbols"] = exec_selected
    audit["dispatch_contract_mismatch"] = dispatch_mismatch
    audit["broker_orders_submitted"] = bool(dispatch_audit.get("broker_orders_submitted"))
    audit["execution_block_reason"] = block_reason
    _write_stage_audit(audit, market_open=market_open)
    return 0 if rc in (0, 2) else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Controlled paper entry stage for continuous loop")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="Evaluate gates only; never call execute_trades"
    )
    ap.add_argument("--evaluate-only", action="store_true", help="Alias for --dry-run")
    args = ap.parse_args(argv)
    dry = bool(args.dry_run or args.evaluate_only)
    return run_paper_entry_stage(verbose=bool(args.verbose), dry_run=dry)


if __name__ == "__main__":
    sys.exit(main())
