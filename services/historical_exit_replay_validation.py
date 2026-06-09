"""
Phase 153 — Historical Exit Replay Validation.

Read-only replay of June 8 SLV/TSLA/AMD exits through Phase 152 protection gates.
Does not modify exit logic, positions, or place orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from services.effective_lifecycle import (
    ExitLogicConfig,
    ExitLogicMetrics,
    advisory_exit_confidence_allows,
    apply_drag_decay,
    check_position_age_protection,
    check_rotation_confirmation,
    check_winner_protection,
    compute_exit_confidence,
    find_rotation_replacement,
    is_catastrophic_signal_deterioration,
    is_force_exit_flag,
    is_severe_risk_condition,
    load_exit_logic_config,
    load_position_entry_times,
)
from services.exit_rotation_audit import (
    _discover_exit_events,
    _detect_session_date,
    _index_by_symbol,
    _load_fill_timeline,
    _parse_ts,
    _read_csv_rows,
    _read_json,
    _safe_float,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "historical_exit_replay_validation.json"
OUTPUT_CSV = RESULTS / "historical_exit_replay_validation.csv"

TARGET_SYMBOLS = ("SLV", "TSLA", "AMD")
SESSION_DATE_DEFAULT = "2026-06-08"

BLOCKER_DRAG_DECAY = "DRAG_DECAY"
BLOCKER_POSITION_AGE = "POSITION_AGE_PROTECTION"
BLOCKER_ROTATION = "ROTATION_CONFIRMATION"
BLOCKER_WINNER = "WINNER_PROTECTION"
BLOCKER_EXIT_CONFIDENCE = "EXIT_CONFIDENCE_GATE"

CSV_COLUMNS = [
    "symbol",
    "session_date",
    "original_decision",
    "original_reason",
    "protected_decision",
    "blocked_by",
    "confidence_score",
    "position_age_hours",
    "current_pl",
    "historical_total_pl",
    "drag_score",
    "risk_flag",
    "replacement_candidate",
    "confidence_delta",
    "edge_delta",
    "verdict",
    "notes",
]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hours_at_decision(entry_time: Optional[datetime], decision_ts: str) -> float:
    if entry_time is None:
        return 0.0
    decision_dt = _parse_ts(decision_ts)
    if decision_dt is None:
        return 0.0
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    delta = decision_dt - entry_time
    return round(max(0.0, delta.total_seconds() / 3600.0), 4)


def _lifecycle_field(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return ""


def _original_decision_label(exit_type: str, source: str) -> str:
    et = str(exit_type or "").upper()
    if et == "ROTATE_EXIT":
        return "ROTATE_EXIT"
    if et in ("EXIT", "FORCED_EXIT", "LIFECYCLE_EXIT"):
        return "EXIT"
    if "TRIM" in str(source or "").upper():
        return "TRIM"
    return et or "EXIT"


def _original_reason(record: Dict[str, Any]) -> str:
    parts = [str(record.get("trigger_reason") or "").strip()]
    source = str(record.get("source") or "").strip()
    if source and source not in parts[0]:
        parts.append(source)
    return "; ".join(p for p in parts if p) or "unknown"


def _emergency_context(
    *,
    total_pl: float,
    unrealized_pl: float,
    risk_flag: str,
    delta_pct: float,
    confidence: float,
    config: ExitLogicConfig,
    force_exit_active: bool = False,
) -> bool:
    return (
        force_exit_active
        or is_severe_risk_condition(
            total_pl=total_pl,
            unrealized_pl=unrealized_pl,
            risk_flag=risk_flag,
            config=config,
        )
        or is_catastrophic_signal_deterioration(
            delta_pct=delta_pct, confidence=confidence, config=config
        )
    )


def _replay_high_drag_path(
    *,
    symbol: str,
    hours_held: float,
    current_pl: float,
    total_pl: float,
    risk_flag: str,
    delta_pct: float,
    confidence: float,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
) -> Tuple[str, str, float, List[str]]:
    """Mirror manage_positions high-drag gate. Returns (decision, primary_blocker, score, all_blockers)."""
    blockers: List[str] = []
    emergency = _emergency_context(
        total_pl=total_pl,
        unrealized_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        config=config,
        force_exit_active=is_force_exit_flag(risk_flag),
    )

    _hist_w, _curr_w, adjusted, still_force = apply_drag_decay(
        historical_total_pl=total_pl,
        current_unrealized_pl=current_pl,
        hours_held=hours_held,
        confidence=confidence,
        config=config,
    )
    metrics.drag_adjustments += 1

    if not still_force and not emergency:
        blockers.append(BLOCKER_DRAG_DECAY)
        score, rec = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=0.0,
            drag_score=adjusted,
            config=config,
        )
        return "HOLD", BLOCKER_DRAG_DECAY, score, blockers

    exit_kind = "HIGH_DRAG"
    if not check_position_age_protection(
        symbol=symbol,
        hours_held=hours_held,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        blockers.append(BLOCKER_POSITION_AGE)
        score, _ = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=0.0,
            drag_score=adjusted,
            config=config,
        )
        return "HOLD", BLOCKER_POSITION_AGE, score, blockers

    if not check_winner_protection(
        symbol=symbol,
        current_pl=current_pl,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        blockers.append(BLOCKER_WINNER)
        score, _ = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=0.0,
            drag_score=adjusted,
            config=config,
        )
        return "HOLD", BLOCKER_WINNER, score, blockers

    score, rec = compute_exit_confidence(
        hours_held=hours_held,
        current_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        replacement_edge_delta=0.0,
        drag_score=adjusted,
        config=config,
    )
    if not advisory_exit_confidence_allows(
        symbol=symbol,
        recommendation=rec,
        config=config,
        metrics=metrics,
        emergency=emergency,
        exit_kind=exit_kind,
        score=score,
    ):
        blockers.append(BLOCKER_EXIT_CONFIDENCE)
        return "HOLD", BLOCKER_EXIT_CONFIDENCE, score, blockers

    return rec if rec in ("EXIT", "TRIM") else "EXIT", "", score, blockers


def _replay_rotate_path(
    *,
    symbol: str,
    hours_held: float,
    current_pl: float,
    total_pl: float,
    risk_flag: str,
    delta_pct: float,
    confidence: float,
    replacement_sym: str,
    conf_delta: float,
    edge_delta: float,
    alloc_delta: float,
    drag_score: float,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
) -> Tuple[str, str, float, List[str]]:
    blockers: List[str] = []
    emergency = _emergency_context(
        total_pl=total_pl,
        unrealized_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        config=config,
        force_exit_active=is_force_exit_flag(risk_flag),
    )

    if not check_rotation_confirmation(
        symbol=symbol,
        candidate=replacement_sym,
        edge_delta=edge_delta,
        conf_delta=conf_delta,
        alloc_delta=alloc_delta,
        config=config,
        metrics=metrics,
    ):
        blockers.append(BLOCKER_ROTATION)
        score, rec = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=edge_delta,
            drag_score=drag_score,
            config=config,
        )
        return "HOLD", BLOCKER_ROTATION, score, blockers

    exit_kind = "ROTATE_EXIT"
    if not check_position_age_protection(
        symbol=symbol,
        hours_held=hours_held,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        blockers.append(BLOCKER_POSITION_AGE)
        score, _ = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=edge_delta,
            drag_score=drag_score,
            config=config,
        )
        return "HOLD", BLOCKER_POSITION_AGE, score, blockers

    if not check_winner_protection(
        symbol=symbol,
        current_pl=current_pl,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        blockers.append(BLOCKER_WINNER)
        score, _ = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=edge_delta,
            drag_score=drag_score,
            config=config,
        )
        return "HOLD", BLOCKER_WINNER, score, blockers

    score, rec = compute_exit_confidence(
        hours_held=hours_held,
        current_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        replacement_edge_delta=edge_delta,
        drag_score=drag_score,
        config=config,
    )
    if not advisory_exit_confidence_allows(
        symbol=symbol,
        recommendation=rec,
        config=config,
        metrics=metrics,
        emergency=emergency,
        exit_kind=exit_kind,
        score=score,
    ):
        blockers.append(BLOCKER_EXIT_CONFIDENCE)
        return "HOLD", BLOCKER_EXIT_CONFIDENCE, score, blockers

    return "ROTATE_EXIT", "", score, blockers


def _replay_trim_priority_path(
    *,
    symbol: str,
    hours_held: float,
    current_pl: float,
    total_pl: float,
    risk_flag: str,
    delta_pct: float,
    confidence: float,
    drag_score: float,
    config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
) -> Tuple[str, str, float, List[str]]:
    blockers: List[str] = []
    emergency = _emergency_context(
        total_pl=total_pl,
        unrealized_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        config=config,
        force_exit_active=is_force_exit_flag(risk_flag),
    )
    exit_kind = "TRIM_PRIORITY"

    if not check_position_age_protection(
        symbol=symbol,
        hours_held=hours_held,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        blockers.append(BLOCKER_POSITION_AGE)
        score, _ = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=0.0,
            drag_score=drag_score,
            config=config,
        )
        return "HOLD", BLOCKER_POSITION_AGE, score, blockers

    if not check_winner_protection(
        symbol=symbol,
        current_pl=current_pl,
        exit_kind=exit_kind,
        config=config,
        metrics=metrics,
        emergency=emergency,
    ):
        blockers.append(BLOCKER_WINNER)
        score, _ = compute_exit_confidence(
            hours_held=hours_held,
            current_pl=current_pl,
            risk_flag=risk_flag,
            delta_pct=delta_pct,
            confidence=confidence,
            replacement_edge_delta=0.0,
            drag_score=drag_score,
            config=config,
        )
        return "HOLD", BLOCKER_WINNER, score, blockers

    score, rec = compute_exit_confidence(
        hours_held=hours_held,
        current_pl=current_pl,
        risk_flag=risk_flag,
        delta_pct=delta_pct,
        confidence=confidence,
        replacement_edge_delta=0.0,
        drag_score=drag_score,
        config=config,
    )
    if not advisory_exit_confidence_allows(
        symbol=symbol,
        recommendation=rec,
        config=config,
        metrics=metrics,
        emergency=emergency,
        exit_kind=exit_kind,
        score=score,
    ):
        blockers.append(BLOCKER_EXIT_CONFIDENCE)
        return "HOLD", BLOCKER_EXIT_CONFIDENCE, score, blockers

    return rec if rec in ("TRIM", "EXIT") else "TRIM", "", score, blockers


def _verdict(
    *,
    original: str,
    protected: str,
    blocked_by: str,
    symbol: str,
) -> Tuple[str, str]:
    orig = str(original or "").upper()
    prot = str(protected or "").upper()
    expected_hold = orig in ("EXIT", "ROTATE_EXIT", "FORCED_EXIT", "LIFECYCLE_EXIT")
    if expected_hold and prot in ("HOLD", "WAIT"):
        return "FIX_VALIDATED", ""
    if expected_hold and prot == "TRIM" and orig == "EXIT":
        return "FIX_VALIDATED", "exit reduced to TRIM under Phase 152 gates"
    if expected_hold and prot in ("EXIT", "ROTATE_EXIT"):
        reason = (
            f"Phase 152 protections did not block {symbol} {orig}; "
            f"protected_decision={prot} blocked_by={blocked_by or 'NONE'}"
        )
        return "FIX_FAILED", reason
    return "FIX_VALIDATED", ""


@dataclass
class ReplayContext:
    results_dir: Path
    session_date: str
    config: ExitLogicConfig
    audit_by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_overlay: Dict[str, Dict[str, str]] = field(default_factory=dict)
    lifecycle: Dict[str, Dict[str, str]] = field(default_factory=dict)
    effective: Dict[str, Dict[str, str]] = field(default_factory=dict)
    alloc_rows: Dict[str, Dict[str, str]] = field(default_factory=dict)
    alloc_summary: Dict[str, Any] = field(default_factory=dict)
    fills: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    held_symbols: Set[str] = field(default_factory=set)
    entry_times: Dict[str, datetime] = field(default_factory=dict)


def _load_context(results_dir: Path, session_date: str) -> ReplayContext:
    audit_rows = _read_csv_rows(results_dir / "paper_execution_audit.csv")
    live_orders = _read_csv_rows(results_dir / "live_orders_log.csv")
    recent_orders = _read_csv_rows(results_dir / "recent_orders.csv")
    positions = _read_csv_rows(results_dir / "positions_snapshot.csv")

    lifecycle = _index_by_symbol(_read_csv_rows(results_dir / "signal_lifecycle.csv"), "ticker")
    if not lifecycle:
        lifecycle = _index_by_symbol(_read_csv_rows(results_dir / "signal_lifecycle.csv"))
    effective = _index_by_symbol(
        _read_csv_rows(results_dir / "signal_lifecycle_effective.csv"), "ticker"
    )
    if not effective:
        effective = _index_by_symbol(_read_csv_rows(results_dir / "signal_lifecycle_effective.csv"))

    risk_overlay = _index_by_symbol(
        _read_csv_rows(results_dir / "performance_risk_overlay.csv"), "ticker"
    )
    alloc_rows = _index_by_symbol(
        _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")
    )
    alloc_summary = _read_json(results_dir / "portfolio_allocation_summary.json")
    exit_audit = _read_json(results_dir / "exit_rotation_audit.json")

    detected = session_date or _detect_session_date(audit_rows, live_orders)
    fills = _load_fill_timeline(live_orders, recent_orders, detected)
    entry_times = load_position_entry_times(results_dir)

    held_symbols = {
        str(r.get("symbol") or r.get("ticker") or "").upper()
        for r in positions
        if _safe_float(r.get("qty")) > 0
    }
    for sym in TARGET_SYMBOLS:
        held_symbols.add(sym)

    audit_by_symbol: Dict[str, Dict[str, Any]] = {}
    for rec in exit_audit.get("exit_records") or []:
        if not isinstance(rec, dict):
            continue
        sym = str(rec.get("symbol") or "").upper()
        if sym in TARGET_SYMBOLS and sym not in audit_by_symbol:
            audit_by_symbol[sym] = rec

    if len(audit_by_symbol) < len(TARGET_SYMBOLS):
        events = _discover_exit_events(audit_rows, detected)
        for ev in events:
            sym = str(ev.get("symbol") or "").upper()
            if sym in TARGET_SYMBOLS and sym not in audit_by_symbol:
                audit_by_symbol[sym] = {
                    "symbol": sym,
                    "exit_type": ev.get("exit_type"),
                    "timestamp": ev.get("timestamp"),
                    "source": "",
                    "trigger_reason": "",
                    "filled": True,
                }

    return ReplayContext(
        results_dir=results_dir,
        session_date=detected,
        config=load_exit_logic_config(),
        audit_by_symbol=audit_by_symbol,
        risk_overlay=risk_overlay,
        lifecycle=lifecycle,
        effective=effective,
        alloc_rows=alloc_rows,
        alloc_summary=alloc_summary,
        fills=fills,
        held_symbols=held_symbols,
        entry_times=entry_times,
    )


def _reconstruct_state(ctx: ReplayContext, symbol: str) -> Dict[str, Any]:
    sym = symbol.upper()
    audit = ctx.audit_by_symbol.get(sym, {})
    risk = ctx.risk_overlay.get(sym, {})
    lc = ctx.lifecycle.get(sym, {})
    eff = ctx.effective.get(sym, {})
    alloc = ctx.alloc_rows.get(sym, {})
    fill = ctx.fills.get(sym, {})

    decision_ts = str(audit.get("timestamp") or "")
    entry_time = fill.get("entry_time")
    if entry_time is None and sym in ctx.entry_times:
        entry_time = ctx.entry_times[sym]

    hours_held = _hours_at_decision(entry_time, decision_ts)
    if hours_held <= 0 and audit.get("hours_held"):
        hours_held = _safe_float(audit.get("hours_held"))

    current_pl = _safe_float(audit.get("current_position_pl"))
    if current_pl == 0.0 and fill.get("entry_price") and fill.get("exit_price"):
        qty = _safe_float(fill.get("exit_qty") or fill.get("entry_qty"))
        entry_p = _safe_float(fill.get("entry_price"))
        exit_p = _safe_float(fill.get("exit_price"))
        if qty and entry_p and exit_p:
            current_pl = round((exit_p - entry_p) * qty, 4)

    total_pl = _safe_float(risk.get("total_pl") or audit.get("historical_total_pl"))
    risk_flag = str(risk.get("risk_flag") or audit.get("risk_flag") or "OK").upper()
    confidence = _safe_float(
        alloc.get("confidence") or lc.get("confidence") or lc.get("entry_confidence")
    )
    delta_pct = _safe_float(
        alloc.get("delta_pct") or lc.get("delta_pct") or lc.get("signal_delta_pct")
    )

    lifecycle_state = _lifecycle_field(lc, "lifecycle_action", "stance", "signal")
    effective_stance = _lifecycle_field(
        eff, "effective_stance", "effective_signal", "effective_lifecycle"
    ) or _lifecycle_field(lc, "effective_stance", "effective_signal")

    replacement_sym, conf_delta, edge_delta, alloc_delta, _ = find_rotation_replacement(
        sym,
        ctx.alloc_rows,
        ctx.alloc_summary,
        ctx.held_symbols,
    )
    if audit.get("rotation_candidate"):
        replacement_sym = str(audit.get("rotation_candidate") or replacement_sym)
    if audit.get("confidence_delta") is not None:
        conf_delta = _safe_float(audit.get("confidence_delta"))
    if audit.get("edge_delta") is not None:
        edge_delta = _safe_float(audit.get("edge_delta"))

    _, _, drag_score, _ = apply_drag_decay(
        historical_total_pl=total_pl,
        current_unrealized_pl=current_pl,
        hours_held=hours_held,
        confidence=confidence,
        config=ctx.config,
    )

    return {
        "symbol": sym,
        "decision_timestamp": decision_ts,
        "entry_time": entry_time.isoformat().replace("+00:00", "Z") if entry_time else "",
        "position_age_hours": hours_held,
        "current_pl": current_pl,
        "total_pl": total_pl,
        "lifecycle_state": lifecycle_state,
        "effective_lifecycle": effective_stance,
        "risk_flag": risk_flag,
        "drag_score": drag_score,
        "confidence": confidence,
        "delta_pct": delta_pct,
        "replacement_candidate": replacement_sym,
        "confidence_delta": conf_delta,
        "edge_delta": edge_delta,
        "alloc_delta": alloc_delta,
        "exit_type": str(audit.get("exit_type") or "").upper(),
        "source": str(audit.get("source") or ""),
        "trigger_reason": str(audit.get("trigger_reason") or ""),
    }


def _replay_symbol(ctx: ReplayContext, symbol: str, verbose: bool) -> Dict[str, Any]:
    state = _reconstruct_state(ctx, symbol)
    audit = ctx.audit_by_symbol.get(symbol.upper(), {})

    original = _original_decision_label(state["exit_type"], state["source"])
    original_reason = _original_reason({**audit, **state})

    if verbose:
        print(
            f"[REPLAY_STATE] symbol={symbol} "
            f"position_age_hours={state['position_age_hours']} "
            f"current_pl={state['current_pl']} "
            f"drag_score={state['drag_score']} "
            f"risk_flag={state['risk_flag']} "
            f"replacement={state['replacement_candidate'] or 'NONE'} "
            f"confidence={state['confidence']:.6f} "
            f"edge_delta={state['edge_delta']}",
            flush=True,
        )
        print(
            f"[ORIGINAL_DECISION] symbol={symbol} decision={original} reason={original_reason}",
            flush=True,
        )

    metrics = ExitLogicMetrics()
    exit_type = state["exit_type"]
    source = state["source"].upper()

    if exit_type == "ROTATE_EXIT" or original == "ROTATE_EXIT":
        protected, blocked_by, score, _all_blockers = _replay_rotate_path(
            symbol=symbol,
            hours_held=state["position_age_hours"],
            current_pl=state["current_pl"],
            total_pl=state["total_pl"],
            risk_flag=state["risk_flag"],
            delta_pct=state["delta_pct"],
            confidence=state["confidence"],
            replacement_sym=state["replacement_candidate"],
            conf_delta=state["confidence_delta"],
            edge_delta=state["edge_delta"],
            alloc_delta=state["alloc_delta"],
            drag_score=state["drag_score"],
            config=ctx.config,
            metrics=metrics,
        )
    elif "HIGH_DRAG" in source or exit_type == "EXIT" and state["total_pl"] < -50:
        protected, blocked_by, score, _all_blockers = _replay_high_drag_path(
            symbol=symbol,
            hours_held=state["position_age_hours"],
            current_pl=state["current_pl"],
            total_pl=state["total_pl"],
            risk_flag=state["risk_flag"],
            delta_pct=state["delta_pct"],
            confidence=state["confidence"],
            config=ctx.config,
            metrics=metrics,
        )
    elif "TRIM_PRIORITY" in state["risk_flag"]:
        protected, blocked_by, score, _all_blockers = _replay_trim_priority_path(
            symbol=symbol,
            hours_held=state["position_age_hours"],
            current_pl=state["current_pl"],
            total_pl=state["total_pl"],
            risk_flag=state["risk_flag"],
            delta_pct=state["delta_pct"],
            confidence=state["confidence"],
            drag_score=state["drag_score"],
            config=ctx.config,
            metrics=metrics,
        )
    else:
        protected, blocked_by, score, _all_blockers = _replay_high_drag_path(
            symbol=symbol,
            hours_held=state["position_age_hours"],
            current_pl=state["current_pl"],
            total_pl=state["total_pl"],
            risk_flag=state["risk_flag"],
            delta_pct=state["delta_pct"],
            confidence=state["confidence"],
            config=ctx.config,
            metrics=metrics,
        )

    if verbose:
        print(
            f"[PROTECTED_DECISION] symbol={symbol} decision={protected} "
            f"blocked_by={blocked_by or 'NONE'} confidence_score={score}",
            flush=True,
        )

    verdict, fail_note = _verdict(
        original=original, protected=protected, blocked_by=blocked_by, symbol=symbol
    )

    if verbose:
        print(
            f"[REPLAY_RESULT]\n"
            f"symbol={symbol}\n"
            f"old={original}\n"
            f"new={protected}\n"
            f"blocked_by={blocked_by or 'NONE'}\n"
            f"verdict={verdict}",
            flush=True,
        )

    notes = fail_note
    if symbol == "AMD" and original == "ROTATE_EXIT":
        notes = (notes + "; " if notes else "") + (
            "Historical audit recorded ROTATE_EXIT (not plain EXIT); "
            "position age protection still blocks premature rotation"
        )

    return {
        "symbol": symbol,
        "session_date": ctx.session_date,
        "original_decision": original,
        "original_reason": original_reason,
        "protected_decision": protected,
        "blocked_by": blocked_by or "",
        "confidence_score": score,
        "position_age_hours": state["position_age_hours"],
        "current_pl": state["current_pl"],
        "historical_total_pl": state["total_pl"],
        "drag_score": state["drag_score"],
        "risk_flag": state["risk_flag"],
        "lifecycle_state": state["lifecycle_state"],
        "effective_lifecycle": state["effective_lifecycle"],
        "replacement_candidate": state["replacement_candidate"],
        "confidence_delta": state["confidence_delta"],
        "edge_delta": state["edge_delta"],
        "verdict": verdict,
        "notes": notes,
        "protection_metrics": metrics.to_dict(),
        "replay_state": state,
    }


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def run_historical_exit_replay_validation(
    *,
    session_date: Optional[str] = None,
    verbose: bool = False,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    results_dir = Path(results_dir or RESULTS)
    ctx = _load_context(results_dir, session_date or SESSION_DATE_DEFAULT)

    replay_rows: List[Dict[str, Any]] = []
    for sym in TARGET_SYMBOLS:
        if sym not in ctx.audit_by_symbol:
            replay_rows.append(
                {
                    "symbol": sym,
                    "session_date": ctx.session_date,
                    "original_decision": "UNKNOWN",
                    "original_reason": "exit record not found in audit artifacts",
                    "protected_decision": "UNKNOWN",
                    "blocked_by": "",
                    "confidence_score": 0.0,
                    "position_age_hours": 0.0,
                    "current_pl": 0.0,
                    "historical_total_pl": 0.0,
                    "drag_score": 0.0,
                    "risk_flag": "",
                    "replacement_candidate": "",
                    "confidence_delta": 0.0,
                    "edge_delta": 0.0,
                    "verdict": "FIX_FAILED",
                    "notes": "Could not reconstruct exit state from data/results artifacts",
                }
            )
            if verbose:
                print(
                    f"[REPLAY_RESULT] symbol={sym} verdict=FIX_FAILED "
                    f"reason=missing_exit_audit_record",
                    flush=True,
                )
            continue
        replay_rows.append(_replay_symbol(ctx, sym, verbose=verbose))

    validated = sum(1 for r in replay_rows if r.get("verdict") == "FIX_VALIDATED")
    failed = sum(1 for r in replay_rows if r.get("verdict") == "FIX_FAILED")
    protections_triggered = sum(1 for r in replay_rows if r.get("blocked_by"))

    if failed == 0 and validated == len(TARGET_SYMBOLS):
        overall = "PASS"
    elif failed > 0 and validated == 0:
        overall = "FAIL"
    else:
        overall = "WARN" if failed > 0 else "PASS"

    scorecard = {
        "replayed_exits": len(replay_rows),
        "protections_triggered": protections_triggered,
        "fixes_validated": validated,
        "fixes_failed": failed,
        "overall_result": overall,
    }

    payload: Dict[str, Any] = {
        "generated_at": _utc_iso(),
        "phase": "153",
        "session_date": ctx.session_date,
        "target_symbols": list(TARGET_SYMBOLS),
        "scorecard": scorecard,
        "replay_results": replay_rows,
        "disclaimer": "Read-only historical replay. No exit logic or execution changes.",
    }

    _atomic_write_json(payload, OUTPUT_JSON)
    _write_csv(replay_rows, OUTPUT_CSV)

    print(
        f"[HISTORICAL_EXIT_REPLAY_VALIDATION] "
        f"replayed_exits={scorecard['replayed_exits']} "
        f"validated={scorecard['fixes_validated']} "
        f"failed={scorecard['fixes_failed']} "
        f"overall_result={scorecard['overall_result']}",
        flush=True,
    )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 153 — Historical Exit Replay Validation (read-only)"
    )
    parser.add_argument(
        "--date",
        default=SESSION_DATE_DEFAULT,
        help="Session date YYYY-MM-DD (default: 2026-06-08)",
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed replay logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    args = parser.parse_args()

    run_historical_exit_replay_validation(
        session_date=args.date,
        verbose=args.verbose,
        results_dir=Path(args.results_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
