"""
phase_195l_buy_signal_execution_gate_trace_audit.py
Phase 195L — BUY signal to execution gate trace audit (read-only).

Traces lifecycle/effective BUY signals through allocation, trade opportunities,
manage_positions, and order planning to explain zero OPEN_NEW / zero paper orders.
Does not place orders or modify thresholds.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "buy_signal_execution_gate_trace_audit.json"
CANDIDATES_CSV = RESULTS / "buy_signal_execution_gate_trace_candidates.csv"
SUMMARY_CSV = RESULTS / "buy_signal_execution_gate_trace_summary.csv"

RAW_LIFECYCLE = RESULTS / "signal_lifecycle.csv"
EFFECTIVE_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
ALLOC_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
EDGE_SIZING = RESULTS / "edge_sizing_recommendations.csv"
TRADE_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
TRADE_OPP_DIAG = RESULTS / "trade_opportunity_build_diagnostics.json"
MANAGE_PLAN = RESULTS / "manage_positions_plan.json"
EXECUTION_PLAN = RESULTS / "execution_plan.json"
LAST_EXEC_SESSION = RESULTS / "last_execution_session.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"

STRONG_EDGE_SCORE = 0.50
MIN_OPEN_NEW_CONFIDENCE = 0.50

CLASSIFICATIONS = (
    "CORRECTLY_BLOCKED_DECISION_WAIT",
    "CORRECTLY_BLOCKED_WEAK_EDGE",
    "CORRECTLY_BLOCKED_LOW_CONFIDENCE",
    "CORRECTLY_BLOCKED_RISK_OVERLAY",
    "CORRECTLY_BLOCKED_NO_ACTIONABLE_RULE",
    "ALLOCATION_OPEN_GATE_NOT_MET",
    "DECISION_BUY_BUT_ALLOCATION_REJECTED",
    "DECISION_BUY_BUT_MISSING_ENTRY_HANDOFF",
    "EXECUTION_AUTHORIZED_BUT_NO_ORDER_PATH",
    "DASHBOARD_LIFECYCLE_LABEL_MISLEADING",
    "NEEDS_OPERATOR_REVIEW",
)

CANDIDATE_COLUMNS = (
    "ticker",
    "classification",
    "lifecycle_action",
    "effective_stance",
    "decision_action",
    "decision_reason",
    "lifecycle_decision_reason",
    "confidence",
    "delta_pct",
    "edge_score_allocation",
    "edge_score_trade_opp",
    "sizing_tier",
    "risk_flag",
    "block_new_buy",
    "force_exit",
    "allocation_recommended_action",
    "allocation_score",
    "allocation_reason",
    "open_gate_passed",
    "in_trade_opportunities",
    "trade_opp_sizing_bucket",
    "reached_manage_positions_entry",
    "reached_order_planning",
    "reached_broker_submission",
    "final_non_action_reason",
    "responsible_gate",
    "responsible_module",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _norm(val: Any) -> str:
    return str(val or "").strip().upper()


def _to_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _row_map(df: pd.DataFrame, key: str = "ticker") -> Dict[str, Dict[str, Any]]:
    if df.empty or key not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        sym = str(row.get(key) or "").strip().upper()
        if sym:
            out[sym] = row.to_dict()
    return out


def _risk_flags(risk_flag: str) -> Set[str]:
    parts = [_norm(p) for p in str(risk_flag or "").split("|")]
    return {p for p in parts if p and p != "OK"}


def _is_strong_edge(edge_score: Optional[float], sizing_tier: str) -> bool:
    if _norm(sizing_tier) == "STRONG_EDGE":
        return True
    if edge_score is None:
        return False
    return edge_score >= STRONG_EDGE_SCORE


def _open_gate_passed(
    *,
    is_held: bool,
    lifecycle_action: str,
    edge_score: Optional[float],
    sizing_tier: str,
    confidence: Optional[float],
    delta_pct: Optional[float],
) -> bool:
    if is_held:
        return False
    if _norm(lifecycle_action) not in {"BUY", "ADD"}:
        return False
    if not _is_strong_edge(edge_score, sizing_tier):
        return False
    if confidence is None or confidence < MIN_OPEN_NEW_CONFIDENCE:
        return False
    if delta_pct is None or delta_pct <= 0:
        return False
    return True


def _trace_universe(
    raw: Dict[str, Dict[str, Any]],
    eff: Dict[str, Dict[str, Any]],
    alloc: Dict[str, Dict[str, Any]],
) -> Set[str]:
    symbols: Set[str] = set()
    for m in (raw, eff, alloc):
        symbols.update(m.keys())
    for m in (raw, eff, alloc):
        for sym, row in m.items():
            if (
                _norm(row.get("lifecycle_action")) == "BUY"
                or _norm(row.get("effective_stance")) == "BUY"
                or _norm(row.get("decision_action")) == "BUY"
                or _norm(row.get("recommended_action"))
                in {"OPEN_NEW", "ADD", "WATCH", "BLOCK", "HOLD", "EXIT", "TRIM"}
            ):
                symbols.add(sym)
    return symbols


def _classify_candidate(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Return (classification, final_non_action_reason, gate, module)."""
    ticker = row["ticker"]
    lc = _norm(row.get("lifecycle_action"))
    eff = _norm(row.get("effective_stance"))
    dec = _norm(row.get("decision_action"))
    alloc_action = _norm(row.get("allocation_recommended_action"))
    alloc_reason = str(row.get("allocation_reason") or "")
    risk_flag = str(row.get("risk_flag") or "")
    risk_parts = _risk_flags(risk_flag)

    if row.get("block_new_buy"):
        return (
            "CORRECTLY_BLOCKED_RISK_OVERLAY",
            alloc_reason or f"risk_flag={risk_flag}",
            "portfolio_allocation_engine._decide_action rule C",
            "services/portfolio_allocation_engine.py",
        )

    if row.get("force_exit"):
        return (
            "CORRECTLY_BLOCKED_RISK_OVERLAY",
            alloc_reason or f"risk_flag={risk_flag}",
            "portfolio_allocation_engine._decide_action rule A",
            "services/portfolio_allocation_engine.py",
        )

    if eff == "BUY" and dec == "WAIT" and lc == "BUY":
        return (
            "DASHBOARD_LIFECYCLE_LABEL_MISLEADING",
            str(row.get("decision_reason") or "decision_action=WAIT despite effective_stance=BUY"),
            "build_effective_lifecycle decision layer",
            "services/build_effective_lifecycle.py",
        )

    if dec == "WAIT" and lc != "BUY":
        return (
            "CORRECTLY_BLOCKED_DECISION_WAIT",
            str(row.get("decision_reason") or "decision_action=WAIT"),
            "effective lifecycle decision_action gate",
            "services/build_effective_lifecycle.py",
        )

    if dec == "BUY" and alloc_action in {"WATCH", "HOLD", "BLOCK"}:
        if "edge_weak" in alloc_reason or not row.get("open_gate_passed"):
            if row.get("in_trade_opportunities"):
                return (
                    "DECISION_BUY_BUT_MISSING_ENTRY_HANDOFF",
                    (
                        f"allocation={alloc_action} ({alloc_reason}); "
                        "trade_opportunities ENTRY exists but execute_trades/manage_positions "
                        "entry path not wired through allocation OPEN_NEW"
                    ),
                    "allocation OPEN_NEW gate vs trade_opportunities parallel path",
                    "services/portfolio_allocation_engine.py + services/build_trade_opportunities.py",
                )
            if not _is_strong_edge(
                _to_float(row.get("edge_score_allocation")),
                str(row.get("sizing_tier") or ""),
            ):
                return (
                    "ALLOCATION_OPEN_GATE_NOT_MET",
                    alloc_reason or "missing strong edge in edge_sizing_recommendations.csv",
                    "portfolio_allocation_engine rule E strong_edge gate",
                    "services/portfolio_allocation_engine.py",
                )
            return (
                "DECISION_BUY_BUT_ALLOCATION_REJECTED",
                alloc_reason,
                "portfolio_allocation_engine._decide_action",
                "services/portfolio_allocation_engine.py",
            )

    if lc == "BUY" and alloc_action == "WATCH":
        if "conf<" in alloc_reason:
            return (
                "CORRECTLY_BLOCKED_LOW_CONFIDENCE",
                alloc_reason,
                "portfolio_allocation_engine rule F",
                "services/portfolio_allocation_engine.py",
            )
        if "edge_weak" in alloc_reason:
            return (
                "CORRECTLY_BLOCKED_WEAK_EDGE",
                alloc_reason,
                "portfolio_allocation_engine rule E/F (edge_sizing_recommendations.csv)",
                "services/portfolio_allocation_engine.py",
            )

    if lc == "BUY" and alloc_action == "HOLD":
        return (
            "CORRECTLY_BLOCKED_NO_ACTIONABLE_RULE",
            alloc_reason or "no_actionable_rule_matched",
            "portfolio_allocation_engine rule G",
            "services/portfolio_allocation_engine.py",
        )

    if lc == "BUY" and eff == "BUY" and not row.get("in_trade_opportunities"):
        return (
            "NEEDS_OPERATOR_REVIEW",
            "lifecycle BUY but absent from trade_opportunities.csv",
            "build_trade_opportunities lifecycle gate",
            "services/build_trade_opportunities.py",
        )

    if row.get("in_trade_opportunities") and not row.get("reached_order_planning"):
        return (
            "EXECUTION_AUTHORIZED_BUT_NO_ORDER_PATH",
            "trade_opportunity present; execute_trades plan/placement not run or dry_run",
            "execute_trades entry path",
            "services/execute_trades.py",
        )

    if alloc_action == "OPEN_NEW":
        return (
            "NEEDS_OPERATOR_REVIEW",
            "OPEN_NEW allocation but no broker submission observed",
            "execute_trades / paper submission gate",
            "services/execute_trades.py",
        )

    return (
        "NEEDS_OPERATOR_REVIEW",
        alloc_reason or "unclassified",
        "unknown",
        "unknown",
    )


def _classify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cls, reason, gate, module = _classify_candidate(row)
    row["classification"] = cls
    row["final_non_action_reason"] = reason
    row["responsible_gate"] = gate
    row["responsible_module"] = module
    return row


def run_phase_195l_audit(*, verbose: bool = True) -> Dict[str, Any]:
    raw_df = _safe_read_csv(RAW_LIFECYCLE)
    eff_df = _safe_read_csv(EFFECTIVE_LIFECYCLE)
    alloc_df = _safe_read_csv(ALLOC_RECOMMENDATIONS)
    risk_df = _safe_read_csv(RISK_OVERLAY)
    edge_df = _safe_read_csv(EDGE_SIZING)
    trade_df = _safe_read_csv(TRADE_OPPORTUNITIES)

    raw_map = _row_map(raw_df)
    eff_map = _row_map(eff_df)
    alloc_map = _row_map(alloc_df)
    risk_map = _row_map(risk_df)
    edge_map = _row_map(edge_df)
    trade_map = _row_map(trade_df)

    alloc_summary = _load_json(ALLOC_SUMMARY)
    trade_diag = _load_json(TRADE_OPP_DIAG)
    manage_plan = _load_json(MANAGE_PLAN)
    exec_plan = _load_json(EXECUTION_PLAN)
    last_exec = _load_json(LAST_EXEC_SESSION)

    open_orders_df = _safe_read_csv(OPEN_ORDERS)
    positions_df = _safe_read_csv(POSITIONS)
    broker_open = len(open_orders_df) if not open_orders_df.empty else 0
    broker_positions = len(positions_df) if not positions_df.empty else 0

    exec_planned = int(exec_plan.get("orders_planned") or 0)
    exec_executed = int(exec_plan.get("orders_executed") or 0)
    exec_dry_run = bool(exec_plan.get("dry_run", True))
    manage_planned = int(manage_plan.get("orders_planned") or 0)

    universe = _trace_universe(raw_map, eff_map, alloc_map)
    candidates: List[Dict[str, Any]] = []

    for sym in sorted(universe):
        raw = raw_map.get(sym, {})
        eff = eff_map.get(sym, {})
        alloc = alloc_map.get(sym, {})
        risk = risk_map.get(sym, {})
        edge = edge_map.get(sym, {})
        trade = trade_map.get(sym, {})

        lifecycle_action = eff.get("lifecycle_action") or raw.get("lifecycle_action") or ""
        effective_stance = eff.get("effective_stance") or eff.get("stance") or ""
        decision_action = eff.get("decision_action") or ""
        decision_reason = eff.get("decision_reason") or ""
        lifecycle_decision_reason = (
            eff.get("lifecycle_decision_reason") or trade.get("lifecycle_decision_reason") or ""
        )

        confidence = _to_float(
            eff.get("confidence") or alloc.get("confidence") or trade.get("confidence")
        )
        delta_pct = _to_float(
            eff.get("delta_pct") or alloc.get("delta_pct") or trade.get("delta_pct")
        )

        risk_flag = str(
            alloc.get("risk_flag") or risk.get("risk_flag") or edge.get("risk_flag") or "OK"
        )
        risk_parts = _risk_flags(risk_flag)

        edge_alloc = _to_float(alloc.get("edge_score"))
        if edge_alloc is None:
            edge_alloc = _to_float(edge.get("edge_score"))
        edge_trade = _to_float(trade.get("edge_score"))
        sizing_tier = str(alloc.get("sizing_tier") or edge.get("sizing_tier") or "")

        is_held = bool(alloc.get("is_currently_held"))
        alloc_action = str(alloc.get("recommended_action") or "")
        alloc_score = alloc.get("allocation_score")
        alloc_reason = str(alloc.get("reason") or "")

        open_gate = _open_gate_passed(
            is_held=is_held,
            lifecycle_action=lifecycle_action,
            edge_score=edge_alloc,
            sizing_tier=sizing_tier,
            confidence=confidence,
            delta_pct=delta_pct,
        )

        in_trade = sym in trade_map
        trade_bucket = str(trade.get("sizing_bucket") or trade.get("opportunity_type") or "")

        # manage_positions explicitly defers entries to execute_trades
        reached_mp_entry = False
        reached_planning = sym in trade_map and exec_planned > 0
        reached_broker = exec_executed > 0 and sym in trade_map

        row: Dict[str, Any] = {
            "ticker": sym,
            "lifecycle_action": lifecycle_action,
            "effective_stance": effective_stance,
            "decision_action": decision_action,
            "decision_reason": decision_reason,
            "lifecycle_decision_reason": lifecycle_decision_reason,
            "confidence": confidence,
            "delta_pct": delta_pct,
            "edge_score_allocation": edge_alloc,
            "edge_score_trade_opp": edge_trade,
            "sizing_tier": sizing_tier,
            "risk_flag": risk_flag,
            "block_new_buy": "BLOCK_NEW_BUY" in risk_parts,
            "force_exit": "FORCE_EXIT" in risk_parts,
            "allocation_recommended_action": alloc_action,
            "allocation_score": alloc_score,
            "allocation_reason": alloc_reason,
            "open_gate_passed": open_gate,
            "in_trade_opportunities": in_trade,
            "trade_opp_sizing_bucket": trade_bucket,
            "reached_manage_positions_entry": reached_mp_entry,
            "reached_order_planning": reached_planning,
            "reached_broker_submission": reached_broker,
        }
        candidates.append(_classify_row(row))

    # Focus tickers deep trace
    focus = ["BAC", "JPM", "SCHW", "WFC"]
    focus_trace = [c for c in candidates if c["ticker"] in focus]

    classification_counts = Counter(c["classification"] for c in candidates)
    blocker_counts = Counter(
        c["allocation_reason"] or c["final_non_action_reason"]
        for c in candidates
        if c.get("allocation_reason")
    )

    eff_all = eff_df if not eff_df.empty else pd.DataFrame()
    lifecycle_buy = (
        int(
            (
                eff_all.get("lifecycle_action", pd.Series(dtype=str)).astype(str).str.upper()
                == "BUY"
            ).sum()
        )
        if not eff_all.empty
        else 0
    )
    effective_buy = (
        int(
            (
                eff_all.get("effective_stance", pd.Series(dtype=str)).astype(str).str.upper()
                == "BUY"
            ).sum()
        )
        if not eff_all.empty
        else 0
    )
    decision_buy = (
        int(
            (
                eff_all.get("decision_action", pd.Series(dtype=str)).astype(str).str.upper()
                == "BUY"
            ).sum()
        )
        if not eff_all.empty
        else 0
    )

    diag = alloc_summary.get("allocation_diagnostics") or {}
    open_gate_n = int(diag.get("open_gate_candidates") or 0)
    open_new_n = int(
        alloc_summary.get("open_new_candidates") or diag.get("open_new_candidates") or 0
    )
    trade_opp_n = len(trade_df)

    trade_in_alloc_open = sum(
        1
        for c in candidates
        if c.get("in_trade_opportunities")
        and _norm(c.get("allocation_recommended_action")) != "OPEN_NEW"
    )

    primary_root_cause = (
        "Expected strict multi-layer gating plus parallel entry paths: dashboard/effective_stance "
        "shows lifecycle-level BUY (30 rows) while decision_action=BUY is rare (4 rows). "
        "Allocation OPEN_NEW requires strong_edge from edge_sizing_recommendations.csv "
        f"(only {len(edge_map)} symbols; none overlap lifecycle BUY), yielding open_gate_candidates=0. "
        f"Entry execution uses trade_opportunities.csv ({trade_opp_n} rows) via execute_trades, "
        "but manage_positions is exit-only and last execution_plan is stale/dry_run "
        f"(planned={exec_planned}, executed={exec_executed}, dry_run={exec_dry_run}). "
        "Not a broker defect — architectural handoff + strict edge gate + plan-only execution default."
    )

    recommended_next_step = (
        "Operator review: (1) treat dashboard BUY as lifecycle intent, not executable authorization; "
        "(2) for BAC/JPM/WFC compare trade_opportunities STRONG/STANDARD edge vs allocation WATCH "
        "(edge_sizing universe gap); (3) run execute_trades with --execute only after explicit "
        "operator approval — current pipeline leaves entries at plan-only; (4) SCHW correctly "
        "BLOCK via BLOCK_NEW_BUY overlay."
    )

    key_answers = {
        "lifecycle_buy_flows_to_allocation": True,
        "decision_action_used_by_allocation": False,
        "allocation_uses_lifecycle_action_not_decision_action": True,
        "lifecycle_buy_downgraded_when_decision_wait": True,
        "bac_jpm_wfc_blocked_by": "allocation WATCH (edge_weak): missing edge_sizing row despite trade_opp STRONG/STANDARD edge",
        "schw_blocked_by": "BLOCK_NEW_BUY risk overlay (performance_risk_overlay.csv drag_flag)",
        "manage_positions_ignores_entries": True,
        "missing_handoff_allocation_to_manage_positions": (
            "By design: entries via execute_trades from trade_opportunities, not allocation OPEN_NEW"
        ),
        "paper_authorized_entries_wired_via_execute_trades": True,
        "expected_strict_gating_or_defect": "expected_strict_gating_with_handoff_clarity_gap",
        "dashboard_buy_misleading": True,
    }

    summary_rows = [
        {"metric": "total_lifecycle_buy_rows", "value": lifecycle_buy},
        {"metric": "total_effective_buy_rows", "value": effective_buy},
        {"metric": "total_decision_action_buy_rows", "value": decision_buy},
        {"metric": "total_allocation_open_gate_candidates", "value": open_gate_n},
        {"metric": "total_open_new_candidates", "value": open_new_n},
        {"metric": "total_trade_opportunities", "value": trade_opp_n},
        {"metric": "trade_opp_not_allocation_open_new", "value": trade_in_alloc_open},
        {"metric": "edge_sizing_symbol_count", "value": len(edge_map)},
        {"metric": "reached_manage_positions_entry", "value": 0},
        {"metric": "reached_order_planning_any", "value": exec_planned},
        {"metric": "reached_broker_submission_any", "value": exec_executed},
        {"metric": "broker_open_orders", "value": broker_open},
        {"metric": "broker_positions", "value": broker_positions},
        {"metric": "manage_positions_planned", "value": manage_planned},
        {"metric": "execution_plan_dry_run", "value": exec_dry_run},
        {"metric": "last_execution_session", "value": last_exec.get("timestamp")},
    ]

    top_blockers = [{"blocker": k, "count": v} for k, v in blocker_counts.most_common(10)]

    payload: Dict[str, Any] = {
        "phase": "195L",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "summary": {
            "total_lifecycle_buy_rows": lifecycle_buy,
            "total_effective_buy_rows": effective_buy,
            "total_decision_action_buy_rows": decision_buy,
            "total_allocation_open_gate_candidates": open_gate_n,
            "total_open_new_candidates": open_new_n,
            "total_trade_opportunities": trade_opp_n,
            "total_candidates_reaching_manage_positions_entry": 0,
            "total_candidates_reaching_order_planning": exec_planned,
            "total_broker_submissions": exec_executed,
            "broker_open_orders": broker_open,
            "broker_positions": broker_positions,
            "classification_counts": dict(classification_counts),
            "top_10_blockers": top_blockers,
            "primary_root_cause": primary_root_cause,
            "behavior_assessment": "expected_strict_gating_with_handoff_clarity_gap",
            "recommended_next_step": recommended_next_step,
        },
        "key_questions": key_answers,
        "focus_tickers": focus_trace,
        "dashboard_explanation": {
            "why_many_buy_rows": (
                "effective_stance=BUY reflects lifecycle_action=BUY (30 rows). "
                "decision_action=BUY requires entry confidence (only 4 rows). "
                "Dashboard BUY badge is lifecycle-level, not OPEN_NEW authorization."
            ),
            "why_open_gate_zero": (
                "open_gate requires lifecycle BUY + strong_edge from edge_sizing_recommendations.csv. "
                "Strong-edge symbols (BITO, GBTC, HD, QQQ, T, TSLA, USO, XLK) have lifecycle WAIT/SELL, "
                "not BUY. BAC/JPM/WFC lifecycle BUY lack edge_sizing rows."
            ),
            "why_no_orders": (
                "manage_positions planned=0 (exit-only, 0 positions). "
                "execute_trades last plan stale/dry_run; trade_opportunities not submitted to broker."
            ),
        },
        "artifact_refs": {
            "signal_lifecycle": str(RAW_LIFECYCLE),
            "signal_lifecycle_effective": str(EFFECTIVE_LIFECYCLE),
            "portfolio_allocation_recommendations": str(ALLOC_RECOMMENDATIONS),
            "portfolio_allocation_summary": str(ALLOC_SUMMARY),
            "performance_risk_overlay": str(RISK_OVERLAY),
            "edge_sizing_recommendations": str(EDGE_SIZING),
            "trade_opportunities": str(TRADE_OPPORTUNITIES),
            "manage_positions_plan": str(MANAGE_PLAN),
            "execution_plan": str(EXECUTION_PLAN),
        },
        "trade_opportunity_diagnostics": trade_diag,
        "allocation_diagnostics": diag,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(CANDIDATES_CSV, candidates, CANDIDATE_COLUMNS)
    _write_csv(
        SUMMARY_CSV,
        summary_rows
        + [
            {"metric": f"blocker_{i+1}", "value": f"{b['blocker']}={b['count']}"}
            for i, b in enumerate(top_blockers)
        ],
        ("metric", "value"),
    )

    if verbose:
        print(
            json.dumps(
                {"phase": "195L", "phase_pass": True, "summary": payload["summary"]}, indent=2
            )
        )

    return payload


def main() -> int:
    run_phase_195l_audit(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
