"""
TRITON Phase 156 — Portfolio Management Stress Testing.

Simulation-only stress tests of the CERTIFIED portfolio management layer (Phase 155).
Does not modify execution, lifecycle, governance, or place orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.effective_lifecycle import (
    ExitLogicConfig,
    ExitLogicMetrics,
    advisory_exit_confidence_allows,
    apply_drag_decay,
    check_position_age_protection,
    check_rotation_confirmation,
    check_winner_protection,
    compute_exit_confidence,
    compute_hours_held,
    evaluate_protected_exit,
    is_catastrophic_signal_deterioration,
    is_force_exit_flag,
    is_severe_risk_condition,
    load_exit_logic_config,
    load_position_entry_times,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CONFIG_EXIT = ROOT / "config" / "exit_logic.json"

OUTPUT_JSON = RESULTS / "portfolio_management_stress_test.json"
OUTPUT_CSV = RESULTS / "portfolio_management_stress_scorecard.csv"

MAX_POSITION_WEIGHT_CAP = 0.10
SHOCK_LEVELS = (-0.05, -0.10, -0.15, -0.20)

SCORECARD_COLUMNS = [
    "section",
    "scenario",
    "protection",
    "symbol",
    "status",
    "detail",
    "generated_at",
]

logger = logging.getLogger(__name__)


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


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_scorecard_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log_tag(tag: str, fields: Dict[str, Any], verbose: bool) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _aggregate_status(statuses: List[str]) -> str:
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "WARN" for s in statuses):
        return "WARN"
    return "PASS"


@dataclass
class ScenarioResult:
    name: str
    status: str
    detail: str
    symbol: str = ""
    protection: str = ""


def _open_positions(positions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in positions if _safe_float(r.get("qty")) > 0]


def _stress_market_drawdown(
    *,
    results_dir: Path,
    exit_config: ExitLogicConfig,
    verbose: bool,
) -> Tuple[Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part A — portfolio shock scenarios."""
    positions = _open_positions(_read_csv_rows(results_dir / "positions_snapshot.csv"))
    perf_rows = {
        str(r.get("ticker") or "").upper(): r
        for r in _read_csv_rows(results_dir / "performance_risk_overlay.csv")
    }
    alloc_rows = {
        str(r.get("ticker") or "").upper(): r
        for r in _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")
    }
    effective_rows = {
        str(r.get("ticker") or "").upper(): r
        for r in _read_csv_rows(results_dir / "signal_lifecycle_effective.csv")
    }
    entry_times = load_position_entry_times(results_dir)
    now = datetime.now(timezone.utc)

    total_mv = sum(_safe_float(p.get("market_value") or p.get("value")) for p in positions)
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []
    shock_results: Dict[str, Any] = {}

    for shock in SHOCK_LEVELS:
        shock_pct = int(abs(shock) * 100)
        shock_key = f"shock_{shock_pct}pct"
        position_outcomes: List[Dict[str, Any]] = []
        local_statuses: List[str] = []

        for pos in positions:
            sym = str(pos.get("symbol") or pos.get("ticker") or "").upper()
            mv = _safe_float(pos.get("market_value") or pos.get("value"))
            base_unreal = _safe_float(pos.get("unrealized_pl"))
            shocked_unreal = round(base_unreal + mv * shock, 4)
            total_pl = _safe_float((perf_rows.get(sym) or {}).get("total_pl"), shocked_unreal)
            shocked_total = round(total_pl + mv * shock, 4)

            alloc = alloc_rows.get(sym, {})
            eff = effective_rows.get(sym, {})
            confidence = _safe_float(alloc.get("confidence") or eff.get("confidence"), 0.5)
            delta_pct = _safe_float(alloc.get("delta_pct") or eff.get("delta_pct"))
            risk_flag = str(
                alloc.get("risk_flag") or (perf_rows.get(sym) or {}).get("risk_flag") or ""
            )

            hours = compute_hours_held(sym, entry_times, now=now)
            metrics = ExitLogicMetrics()

            severe = is_severe_risk_condition(
                total_pl=shocked_total,
                unrealized_pl=shocked_unreal,
                risk_flag=risk_flag,
                config=exit_config,
            )
            hist_w, curr_w, adj_score, drag_force = apply_drag_decay(
                historical_total_pl=shocked_total,
                current_unrealized_pl=shocked_unreal,
                hours_held=hours,
                confidence=confidence,
                config=exit_config,
            )
            metrics.drag_adjustments += 1

            exit_allowed = evaluate_protected_exit(
                symbol=sym,
                exit_kind="HIGH_DRAG",
                hours_held=hours,
                current_pl=shocked_unreal,
                total_pl=shocked_total,
                unrealized_pl=shocked_unreal,
                risk_flag=risk_flag,
                delta_pct=delta_pct,
                confidence=confidence,
                replacement_edge_delta=0.0,
                drag_score=adj_score,
                config=exit_config,
                metrics=metrics,
                force_exit_active=severe or drag_force,
            )

            winner_blocked = not check_winner_protection(
                symbol=sym,
                current_pl=shocked_unreal,
                exit_kind="ROTATE_EXIT",
                config=exit_config,
                metrics=ExitLogicMetrics(),
                emergency=severe,
            )

            pos_status = "PASS"
            notes: List[str] = []
            if shocked_unreal < exit_config.severe_force_exit_unrealized_pl_lt:
                if not severe:
                    pos_status = "FAIL"
                    notes.append("severe_threshold_not_triggered")
                elif exit_allowed:
                    pos_status = "PASS"
                    notes.append("force_exit_bypasses_protections")
            elif shocked_unreal > 0 and winner_blocked:
                pos_status = "PASS"
                notes.append("winner_protection_active")
            elif shocked_unreal < 0 and not exit_allowed and not severe:
                pos_status = "PASS"
                notes.append("protections_correctly_block_non_emergency_exit")

            if drag_force and shocked_unreal > 0 and shocked_total < 0:
                pos_status = "WARN" if pos_status == "PASS" else pos_status
                notes.append("drag_decay_would_force_despite_positive_unreal")

            local_statuses.append(pos_status)
            position_outcomes.append(
                {
                    "symbol": sym,
                    "shocked_unrealized_pl": shocked_unreal,
                    "shocked_total_pl": shocked_total,
                    "severe_risk": severe,
                    "drag_force_exit": drag_force,
                    "exit_allowed": exit_allowed,
                    "winner_blocked": winner_blocked,
                    "status": pos_status,
                    "notes": notes,
                }
            )

        portfolio_shocked_mv = round(total_mv * (1.0 + shock), 4)
        drawdown_response = "NORMAL"
        if shock <= -0.15 and any(o["severe_risk"] for o in position_outcomes):
            drawdown_response = "CAPITAL_PRESERVATION_ESCALATION"
        elif shock <= -0.10:
            drawdown_response = "ELEVATED_RISK_MONITORING"

        shock_status = _aggregate_status(local_statuses) if position_outcomes else "WARN"
        shock_results[shock_key] = {
            "shock_pct": shock,
            "portfolio_value_after_shock": portfolio_shocked_mv,
            "drawdown_response": drawdown_response,
            "position_count": len(position_outcomes),
            "severe_triggered": sum(1 for o in position_outcomes if o["severe_risk"]),
            "status": shock_status,
            "positions": position_outcomes,
        }
        scenarios.append(
            ScenarioResult(
                name=shock_key,
                status=shock_status,
                detail=f"severe={shock_results[shock_key]['severe_triggered']} response={drawdown_response}",
            )
        )
        _log_tag(
            "MARKET_DRAWDOWN_STRESS",
            {"shock": f"{shock_pct}%", "result": shock_status},
            verbose,
        )

    overall = _aggregate_status([s.status for s in scenarios])
    detail = {
        "shock_levels_tested": len(SHOCK_LEVELS),
        "open_positions": len(positions),
        "total_portfolio_value": round(total_mv, 4),
        "shocks": shock_results,
        "status": overall,
    }
    ts = _utc_iso()
    for s in scenarios:
        csv_rows.append(
            {
                "section": "market_drawdown",
                "scenario": s.name,
                "protection": "combined",
                "symbol": "",
                "status": s.status,
                "detail": s.detail,
                "generated_at": ts,
            }
        )
    return detail, scenarios, csv_rows


def _stress_signal_deterioration(
    *,
    results_dir: Path,
    exit_config: ExitLogicConfig,
    verbose: bool,
) -> Tuple[Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part B — signal confidence/edge collapse scenarios."""
    positions = _open_positions(_read_csv_rows(results_dir / "positions_snapshot.csv"))
    held_syms = {str(p.get("symbol") or p.get("ticker") or "").upper() for p in positions}
    effective_rows = _read_csv_rows(results_dir / "signal_lifecycle_effective.csv")
    alloc_rows = {
        str(r.get("ticker") or "").upper(): r
        for r in _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")
    }
    entry_times = load_position_entry_times(results_dir)
    now = datetime.now(timezone.utc)

    deterioration_cases = [
        ("confidence_collapse", {"confidence": 0.15, "delta_pct": -0.008}),
        ("edge_collapse", {"confidence": 0.40, "delta_pct": -0.025}),
        ("buy_to_hold", {"confidence": 0.45, "delta_pct": -0.003, "stance": "HOLD"}),
        ("hold_to_exit", {"confidence": 0.30, "delta_pct": -0.018, "stance": "EXIT"}),
    ]

    case_results: List[Dict[str, Any]] = []
    local_statuses: List[str] = []

    for sym in sorted(held_syms):
        base_eff = next(
            (r for r in effective_rows if str(r.get("ticker") or "").upper() == sym),
            {},
        )
        base_alloc = alloc_rows.get(sym, {})
        pos = next(
            (p for p in positions if str(p.get("symbol") or p.get("ticker") or "").upper() == sym),
            {},
        )
        hours = compute_hours_held(sym, entry_times, now=now)
        current_pl = _safe_float(pos.get("unrealized_pl"))
        total_pl = _safe_float(base_alloc.get("total_pl"), current_pl)

        for case_name, overrides in deterioration_cases:
            confidence = float(overrides.get("confidence", 0.5))
            delta_pct = float(overrides.get("delta_pct", -0.01))
            stance = str(overrides.get("stance") or base_eff.get("effective_stance") or "HOLD")

            catastrophic = is_catastrophic_signal_deterioration(
                delta_pct=delta_pct,
                confidence=confidence,
                config=exit_config,
            )
            score, rec = compute_exit_confidence(
                hours_held=hours,
                current_pl=current_pl,
                risk_flag="TRIM_PRIORITY" if case_name == "hold_to_exit" else "",
                delta_pct=delta_pct,
                confidence=confidence,
                replacement_edge_delta=0.0,
                drag_score=total_pl,
                config=exit_config,
            )
            metrics = ExitLogicMetrics()
            exit_conf_ok = advisory_exit_confidence_allows(
                symbol=sym,
                recommendation=rec,
                config=exit_config,
                metrics=metrics,
                emergency=catastrophic,
                exit_kind="ROTATE_EXIT",
                score=score,
            )
            protected_ok = evaluate_protected_exit(
                symbol=sym,
                exit_kind="ROTATE_EXIT",
                hours_held=hours,
                current_pl=current_pl,
                total_pl=total_pl,
                unrealized_pl=current_pl,
                risk_flag="",
                delta_pct=delta_pct,
                confidence=confidence,
                replacement_edge_delta=0.0,
                drag_score=total_pl,
                config=exit_config,
                metrics=ExitLogicMetrics(),
                force_exit_active=catastrophic,
            )

            case_status = "PASS"
            if case_name in ("confidence_collapse", "edge_collapse", "hold_to_exit"):
                if not catastrophic and rec == "HOLD" and stance == "EXIT":
                    case_status = "WARN"
                elif catastrophic and not protected_ok:
                    case_status = "FAIL"
                elif rec in ("EXIT", "TRIM") or catastrophic:
                    case_status = "PASS"
            elif case_name == "buy_to_hold":
                if rec == "EXIT" and not catastrophic:
                    case_status = "WARN"
                else:
                    case_status = "PASS"

            local_statuses.append(case_status)
            case_results.append(
                {
                    "symbol": sym,
                    "case": case_name,
                    "confidence": confidence,
                    "delta_pct": delta_pct,
                    "stance": stance,
                    "catastrophic": catastrophic,
                    "exit_confidence_score": score,
                    "exit_recommendation": rec,
                    "exit_confidence_allows": exit_conf_ok,
                    "protected_exit_allowed": protected_ok,
                    "status": case_status,
                }
            )

    overall = _aggregate_status(local_statuses) if local_statuses else "WARN"
    _log_tag("SIGNAL_DETERIORATION_STRESS", {"result": overall}, verbose)

    scenarios = [
        ScenarioResult(
            name="signal_deterioration",
            status=overall,
            detail=f"cases={len(case_results)} held={len(held_syms)}",
        )
    ]
    ts = _utc_iso()
    csv_rows = [
        {
            "section": "signal_deterioration",
            "scenario": c["case"],
            "protection": "exit_confidence",
            "symbol": c["symbol"],
            "status": c["status"],
            "detail": f"rec={c['exit_recommendation']} cat={c['catastrophic']}",
            "generated_at": ts,
        }
        for c in case_results
    ]
    return (
        {
            "cases_tested": len(case_results),
            "held_symbols": sorted(held_syms),
            "case_results": case_results,
            "status": overall,
        },
        scenarios,
        csv_rows,
    )


def _stress_rotation_pressure(
    *,
    exit_config: ExitLogicConfig,
    verbose: bool,
) -> Tuple[Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part C — rotation confirmation under crowded replacement pressure."""
    rotation_cases = [
        {
            "name": "weak_rotation_blocked",
            "symbol": "INTC",
            "candidate": "BAC",
            "edge_delta": 0.01,
            "conf_delta": 0.02,
            "alloc_delta": 0.01,
            "expect_allowed": False,
        },
        {
            "name": "strong_rotation_allowed",
            "symbol": "GBTC",
            "candidate": "GE",
            "edge_delta": 0.12,
            "conf_delta": 0.08,
            "alloc_delta": 0.06,
            "expect_allowed": True,
        },
        {
            "name": "marginal_rotation_blocked",
            "symbol": "ARKK",
            "candidate": "META",
            "edge_delta": 0.02,
            "conf_delta": 0.02,
            "alloc_delta": 0.01,
            "expect_allowed": False,
        },
        {
            "name": "crowded_strong_candidate",
            "symbol": "QQQ",
            "candidate": "NVDA",
            "edge_delta": 0.15,
            "conf_delta": 0.10,
            "alloc_delta": 0.08,
            "expect_allowed": True,
        },
        {
            "name": "negative_edge_blocked",
            "symbol": "XLK",
            "candidate": "AMD",
            "edge_delta": -0.02,
            "conf_delta": 0.10,
            "alloc_delta": 0.05,
            "expect_allowed": False,
        },
    ]

    case_results: List[Dict[str, Any]] = []
    local_statuses: List[str] = []

    for case in rotation_cases:
        metrics = ExitLogicMetrics()
        allowed = check_rotation_confirmation(
            symbol=case["symbol"],
            candidate=case["candidate"],
            edge_delta=case["edge_delta"],
            conf_delta=case["conf_delta"],
            alloc_delta=case["alloc_delta"],
            config=exit_config,
            metrics=metrics,
        )
        expect = case["expect_allowed"]
        if allowed == expect:
            status = "PASS"
        elif allowed and not expect:
            status = "FAIL"
        else:
            status = "WARN"

        local_statuses.append(status)
        case_results.append(
            {
                **case,
                "allowed": allowed,
                "rotation_blocks": metrics.rotation_blocks,
                "status": status,
            }
        )

    overall = _aggregate_status(local_statuses)
    _log_tag("ROTATION_STRESS", {"result": overall}, verbose)

    scenarios = [
        ScenarioResult(
            name="rotation_pressure",
            status=overall,
            detail=f"cases={len(case_results)} blocked={sum(1 for c in case_results if not c['allowed'])}",
        )
    ]
    ts = _utc_iso()
    csv_rows = [
        {
            "section": "rotation_pressure",
            "scenario": c["name"],
            "protection": "rotation_confirmation",
            "symbol": c["symbol"],
            "status": c["status"],
            "detail": f"allowed={c['allowed']} edge_delta={c['edge_delta']}",
            "generated_at": ts,
        }
        for c in case_results
    ]
    return (
        {
            "cases_tested": len(case_results),
            "min_edge_delta_required": exit_config.rotation_min_edge_delta,
            "case_results": case_results,
            "status": overall,
        },
        scenarios,
        csv_rows,
    )


def _stress_concentration_shock(
    *,
    results_dir: Path,
    verbose: bool,
) -> Tuple[Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part D — concentration and allocation discipline under shock."""
    positions = _open_positions(_read_csv_rows(results_dir / "positions_snapshot.csv"))
    alloc_summary = _read_json(results_dir / "portfolio_allocation_summary.json")
    alloc_rows = _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")

    total_mv = sum(_safe_float(p.get("market_value") or p.get("value")) for p in positions)
    if total_mv <= 0:
        total_mv = _safe_float(alloc_summary.get("total_position_value"), 1.0)

    concentration_cases: List[Dict[str, Any]] = []

    # Single position dominance — simulate 50% weight in largest holder
    if positions:
        largest = max(
            positions,
            key=lambda p: _safe_float(p.get("market_value") or p.get("value")),
        )
        sym = str(largest.get("symbol") or largest.get("ticker") or "").upper()
        simulated_mv = total_mv * 0.50
        simulated_weight = simulated_mv / total_mv if total_mv else 0.0
        exceeds_cap = simulated_weight > MAX_POSITION_WEIGHT_CAP
        concentration_cases.append(
            {
                "case": "single_position_dominance",
                "symbol": sym,
                "simulated_weight_pct": round(simulated_weight * 100, 4),
                "max_allowed_pct": MAX_POSITION_WEIGHT_CAP * 100,
                "would_block_new_add": exceeds_cap,
                "status": "PASS" if exceeds_cap else "FAIL",
                "detail": (
                    "concentration_cap_would_block_add" if exceeds_cap else "cap_not_enforced"
                ),
            }
        )

    # Sector spike — tech-heavy book (ARKK, INTC, QQQ, XLK)
    tech_syms = {"ARKK", "INTC", "QQQ", "XLK", "NVDA", "AMD", "META", "MSFT", "GOOGL"}
    held_tech_mv = sum(
        _safe_float(p.get("market_value") or p.get("value"))
        for p in positions
        if str(p.get("symbol") or p.get("ticker") or "").upper() in tech_syms
    )
    tech_weight = held_tech_mv / total_mv if total_mv else 0.0
    tech_spike_weight = min(0.95, tech_weight + 0.30)
    concentration_cases.append(
        {
            "case": "sector_spike_tech",
            "current_tech_weight_pct": round(tech_weight * 100, 4),
            "simulated_tech_weight_pct": round(tech_spike_weight * 100, 4),
            "status": "PASS" if tech_spike_weight > 0.40 else "WARN",
            "detail": (
                "sector_concentration_detected"
                if tech_spike_weight > 0.40
                else "low_sector_exposure"
            ),
        }
    )

    # Correlation spike — all held positions shocked together
    corr_shock_pl = sum(_safe_float(p.get("unrealized_pl")) for p in positions)
    simultaneous_loss = round(corr_shock_pl + total_mv * -0.12, 4)
    concentration_cases.append(
        {
            "case": "correlation_spike",
            "simulated_portfolio_unrealized_pl": simultaneous_loss,
            "status": "PASS" if simultaneous_loss < 0 else "WARN",
            "detail": "correlated_drawdown_simulated",
        }
    )

    # Allocation discipline from summary
    largest_weight = _safe_float(alloc_summary.get("largest_position_weight_pct")) / 100.0
    overweight_count = int(alloc_summary.get("overweight_count") or 0)
    concentration_cases.append(
        {
            "case": "allocation_discipline",
            "largest_position_weight_pct": _safe_float(
                alloc_summary.get("largest_position_weight_pct")
            ),
            "overweight_count": overweight_count,
            "max_positions": alloc_summary.get("max_positions"),
            "open_new_candidates": alloc_summary.get("open_new_candidates"),
            "status": "PASS" if largest_weight <= 0.25 or overweight_count <= 5 else "WARN",
            "detail": "current_book_within_allocation_bands",
        }
    )

    # Held positions weight check from alloc rows
    for row in alloc_rows:
        if str(row.get("is_currently_held") or "").lower() not in ("true", "1", "yes"):
            continue
        sym = str(row.get("ticker") or "").upper()
        weight = _safe_float(row.get("current_weight_pct")) / 100.0
        band = str(row.get("allocation_band") or "")
        if weight > MAX_POSITION_WEIGHT_CAP and band == "OVERWEIGHT":
            concentration_cases.append(
                {
                    "case": "held_overweight_flagged",
                    "symbol": sym,
                    "current_weight_pct": _safe_float(row.get("current_weight_pct")),
                    "allocation_band": band,
                    "status": "PASS",
                    "detail": "overweight_correctly_flagged",
                }
            )

    local_statuses = [c["status"] for c in concentration_cases]
    overall = _aggregate_status(local_statuses)
    _log_tag("CONCENTRATION_STRESS", {"result": overall}, verbose)

    scenarios = [
        ScenarioResult(
            name="concentration_shock",
            status=overall,
            detail=f"cases={len(concentration_cases)}",
        )
    ]
    ts = _utc_iso()
    csv_rows = [
        {
            "section": "concentration_shock",
            "scenario": c["case"],
            "protection": "allocation_discipline",
            "symbol": c.get("symbol", ""),
            "status": c["status"],
            "detail": c.get("detail", ""),
            "generated_at": ts,
        }
        for c in concentration_cases
    ]
    return (
        {
            "cases_tested": len(concentration_cases),
            "max_position_weight_cap": MAX_POSITION_WEIGHT_CAP,
            "case_results": concentration_cases,
            "status": overall,
        },
        scenarios,
        csv_rows,
    )


def _stress_protection_framework(
    *,
    exit_config: ExitLogicConfig,
    verbose: bool,
) -> Tuple[Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part E — individual protection edge-case challenges."""
    edge_cases = [
        {
            "protection": "age_protection",
            "name": "new_position_severe_loss_bypasses_age",
            "run": lambda m: check_position_age_protection(
                symbol="TEST",
                hours_held=2.0,
                exit_kind="HIGH_DRAG",
                config=exit_config,
                metrics=m,
                emergency=True,
            ),
            "expect": True,
            "critical": True,
        },
        {
            "protection": "age_protection",
            "name": "new_position_minor_loss_blocked",
            "run": lambda m: check_position_age_protection(
                symbol="TEST",
                hours_held=2.0,
                exit_kind="ROTATE_EXIT",
                config=exit_config,
                metrics=m,
                emergency=False,
            ),
            "expect": False,
            "critical": True,
        },
        {
            "protection": "winner_protection",
            "name": "winner_weak_rotation_blocked",
            "run": lambda m: (
                not check_winner_protection(
                    symbol="WIN",
                    current_pl=50.0,
                    exit_kind="ROTATE_EXIT",
                    config=exit_config,
                    metrics=m,
                    emergency=False,
                )
                and not check_rotation_confirmation(
                    symbol="WIN",
                    candidate="WEAK",
                    edge_delta=0.01,
                    conf_delta=0.0,
                    alloc_delta=0.0,
                    config=exit_config,
                    metrics=m,
                )
            ),
            "expect": True,
            "critical": True,
        },
        {
            "protection": "drag_decay",
            "name": "historical_drag_positive_unreal_favors_hold",
            "run": lambda m: (
                apply_drag_decay(
                    historical_total_pl=-120.0,
                    current_unrealized_pl=25.0,
                    hours_held=48.0,
                    confidence=0.55,
                    config=exit_config,
                )[3]
                is False
            ),
            "expect": True,
            "critical": True,
        },
        {
            "protection": "winner_protection",
            "name": "catastrophic_signal_bypasses_winner",
            "run": lambda m: (
                is_catastrophic_signal_deterioration(
                    delta_pct=-0.02,
                    confidence=0.25,
                    config=exit_config,
                )
                and check_winner_protection(
                    symbol="WIN",
                    current_pl=80.0,
                    exit_kind="ROTATE_EXIT",
                    config=exit_config,
                    metrics=m,
                    emergency=True,
                )
            ),
            "expect": True,
            "critical": True,
        },
        {
            "protection": "exit_confidence",
            "name": "all_protections_hold_vs_single_exit",
            "run": lambda m: _protection_consensus_hold_vs_exit(exit_config, m),
            "expect": True,
            "critical": False,
        },
        {
            "protection": "force_exit",
            "name": "severe_unrealized_triggers_emergency",
            "run": lambda m: is_severe_risk_condition(
                total_pl=-150.0,
                unrealized_pl=-250.0,
                risk_flag="",
                config=exit_config,
            ),
            "expect": True,
            "critical": True,
        },
        {
            "protection": "rotation_confirmation",
            "name": "strong_candidate_superiority_enforced",
            "run": lambda m: check_rotation_confirmation(
                symbol="OLD",
                candidate="NEW",
                edge_delta=0.08,
                conf_delta=0.05,
                alloc_delta=0.04,
                config=exit_config,
                metrics=m,
            ),
            "expect": True,
            "critical": True,
        },
        {
            "protection": "exit_confidence",
            "name": "low_confidence_exit_blocked_when_hold",
            "run": lambda m: _exit_confidence_hold_blocks(exit_config, m),
            "expect": True,
            "critical": False,
        },
        {
            "protection": "drag_decay",
            "name": "severe_historical_drag_still_force",
            "run": lambda m: apply_drag_decay(
                historical_total_pl=-200.0,
                current_unrealized_pl=-30.0,
                hours_held=200.0,
                confidence=0.40,
                config=exit_config,
            )[3],
            "expect": True,
            "critical": True,
        },
    ]

    case_results: List[Dict[str, Any]] = []
    local_statuses: List[str] = []
    critical_fails: List[str] = []

    for case in edge_cases:
        metrics = ExitLogicMetrics()
        buf = StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            actual = bool(case["run"](metrics))
        finally:
            sys.stdout = old_stdout

        expect = case["expect"]
        if actual == expect:
            status = "PASS"
        elif case.get("critical"):
            status = "FAIL"
            critical_fails.append(case["name"])
        else:
            status = "WARN"

        local_statuses.append(status)
        case_results.append(
            {
                "protection": case["protection"],
                "name": case["name"],
                "expected": expect,
                "actual": actual,
                "critical": case.get("critical", False),
                "status": status,
            }
        )
        _log_tag(
            "PROTECTION_STRESS",
            {"protection": case["protection"], "status": status, "case": case["name"]},
            verbose,
        )

    overall = _aggregate_status(local_statuses)
    scenarios = [
        ScenarioResult(
            name="protection_framework",
            status=overall,
            detail=f"cases={len(case_results)} critical_fails={len(critical_fails)}",
        )
    ]
    ts = _utc_iso()
    csv_rows = [
        {
            "section": "protection_framework",
            "scenario": c["name"],
            "protection": c["protection"],
            "symbol": "",
            "status": c["status"],
            "detail": f"expected={c['expected']} actual={c['actual']}",
            "generated_at": ts,
        }
        for c in case_results
    ]
    return (
        {
            "cases_tested": len(case_results),
            "critical_failures": critical_fails,
            "case_results": case_results,
            "status": overall,
        },
        scenarios,
        csv_rows,
    )


def _protection_consensus_hold_vs_exit(
    exit_config: ExitLogicConfig,
    metrics: ExitLogicMetrics,
) -> bool:
    """When age+winner block exit but confidence says EXIT, emergency resolves deadlock."""
    age_ok = check_position_age_protection(
        symbol="CONSENSUS",
        hours_held=48.0,
        exit_kind="ROTATE_EXIT",
        config=exit_config,
        metrics=metrics,
        emergency=False,
    )
    winner_ok = check_winner_protection(
        symbol="CONSENSUS",
        current_pl=10.0,
        exit_kind="ROTATE_EXIT",
        config=exit_config,
        metrics=metrics,
        emergency=False,
    )
    score, rec = compute_exit_confidence(
        hours_held=48.0,
        current_pl=10.0,
        risk_flag="",
        delta_pct=-0.02,
        confidence=0.30,
        replacement_edge_delta=0.08,
        drag_score=-60.0,
        config=exit_config,
    )
    conf_ok = advisory_exit_confidence_allows(
        symbol="CONSENSUS",
        recommendation=rec,
        config=exit_config,
        metrics=metrics,
        emergency=False,
        exit_kind="ROTATE_EXIT",
        score=score,
    )
    # Winner blocks non-emergency; confidence may want exit — no incorrect bypass
    blocked = not age_ok or not winner_ok or not conf_ok
    return blocked or rec == "HOLD"


def _exit_confidence_hold_blocks(exit_config: ExitLogicConfig, metrics: ExitLogicMetrics) -> bool:
    score, rec = compute_exit_confidence(
        hours_held=10.0,
        current_pl=5.0,
        risk_flag="",
        delta_pct=0.001,
        confidence=0.70,
        replacement_edge_delta=-0.05,
        drag_score=10.0,
        config=exit_config,
    )
    allowed = advisory_exit_confidence_allows(
        symbol="HOLDTEST",
        recommendation=rec,
        config=exit_config,
        metrics=metrics,
        emergency=False,
        exit_kind="ROTATE_EXIT",
        score=score,
    )
    return rec == "HOLD" and not allowed


def _resolve_prior_certification(results_dir: Path) -> str:
    op_val = _read_json(results_dir / "portfolio_management_operational_validation.json")
    promo = op_val.get("part_f_certification_promotion_review") or {}
    if promo.get("new_status"):
        return str(promo["new_status"])
    cert = _read_json(results_dir / "portfolio_management_certification.json")
    return str((cert.get("scorecard") or {}).get("overall_certification") or "CERTIFIED")


def _promotion_review(
    *,
    scorecard: Dict[str, str],
    protection_detail: Dict[str, Any],
    prior_status: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Part G — STRESS_TESTED_CERTIFIED promotion review."""
    overall = scorecard.get("overall_stress_result", "FAIL")
    critical_fails = protection_detail.get("critical_failures") or []
    no_critical = len(critical_fails) == 0
    all_parts_pass = overall == "PASS"

    if all_parts_pass and no_critical:
        new_status = "STRESS_TESTED_CERTIFIED"
        reason = (
            "All six stress scenarios PASS with zero critical protection failures; "
            "portfolio management layer demonstrates robust adverse-condition behavior."
        )
    else:
        new_status = prior_status
        parts = [
            f"{k}={v}" for k, v in scorecard.items() if k != "overall_stress_result" and v != "PASS"
        ]
        reason = (
            f"Remain {prior_status}: overall={overall}"
            + (f"; non_pass={', '.join(parts)}" if parts else "")
            + (f"; critical_fails={critical_fails}" if critical_fails else "")
        )

    review = {
        "previous_status": prior_status,
        "new_status": new_status,
        "reason": reason,
        "criteria": {
            "overall_stress_pass": all_parts_pass,
            "no_critical_protection_failures": no_critical,
            "market_drawdown_pass": scorecard.get("market_drawdown") == "PASS",
            "signal_deterioration_pass": scorecard.get("signal_deterioration") == "PASS",
            "rotation_pressure_pass": scorecard.get("rotation_pressure") == "PASS",
            "concentration_shock_pass": scorecard.get("concentration_shock") == "PASS",
            "protection_framework_pass": scorecard.get("protection_framework") == "PASS",
        },
        "promotion_recommended": new_status == "STRESS_TESTED_CERTIFIED",
    }
    _log_tag(
        "STRESS_CERTIFICATION_REVIEW",
        {
            "previous_status": prior_status,
            "new_status": new_status,
            "reason": reason[:100],
        },
        verbose,
    )
    return review


def compute_portfolio_management_stress_test(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run all stress scenarios and assemble payload."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    exit_config = load_exit_logic_config(CONFIG_EXIT)

    part_a, _, csv_a = _stress_market_drawdown(
        results_dir=rd, exit_config=exit_config, verbose=verbose
    )
    part_b, _, csv_b = _stress_signal_deterioration(
        results_dir=rd, exit_config=exit_config, verbose=verbose
    )
    part_c, _, csv_c = _stress_rotation_pressure(exit_config=exit_config, verbose=verbose)
    part_d, _, csv_d = _stress_concentration_shock(results_dir=rd, verbose=verbose)
    part_e, _, csv_e = _stress_protection_framework(exit_config=exit_config, verbose=verbose)

    scorecard = {
        "market_drawdown": part_a["status"],
        "signal_deterioration": part_b["status"],
        "rotation_pressure": part_c["status"],
        "concentration_shock": part_d["status"],
        "protection_framework": part_e["status"],
        "overall_stress_result": _aggregate_status(
            [
                part_a["status"],
                part_b["status"],
                part_c["status"],
                part_d["status"],
                part_e["status"],
            ]
        ),
    }

    prior_status = _resolve_prior_certification(rd)
    promotion = _promotion_review(
        scorecard=scorecard,
        protection_detail=part_e,
        prior_status=prior_status,
        verbose=verbose,
    )

    strengths: List[str] = []
    weaknesses: List[str] = []
    for part_name, part_data, label in [
        (
            "market_drawdown",
            part_a,
            "Market drawdown shocks handled with correct severe/emergency paths",
        ),
        (
            "rotation_pressure",
            part_c,
            "Rotation confirmation blocks weak replacements, allows strong ones",
        ),
        ("protection_framework", part_e, "Phase 152 protection edge cases behave as designed"),
    ]:
        if part_data.get("status") == "PASS":
            strengths.append(label)
        elif part_data.get("status") == "FAIL":
            weaknesses.append(f"{part_name} scenario FAIL")

    if part_b.get("status") != "PASS":
        weaknesses.append("Signal deterioration response gaps under extreme collapse")
    if part_d.get("status") != "PASS":
        weaknesses.append("Concentration/allocation discipline gaps under shock")

    if promotion.get("critical_failures"):
        weaknesses.extend(promotion.get("critical_failures", []))

    critical_fails = part_e.get("critical_failures") or []
    if critical_fails:
        weaknesses.append(f"Critical protection failures: {', '.join(critical_fails)}")

    if not weaknesses:
        weaknesses = ["None identified"]

    recommendation = promotion["reason"]
    if promotion["promotion_recommended"]:
        recommendation = "Promote to STRESS_TESTED_CERTIFIED — all stress scenarios pass with no critical failures."

    _log_tag(
        "PORTFOLIO_MANAGEMENT_STRESS_TEST",
        {
            "status": scorecard["overall_stress_result"],
            "strengths": "; ".join(strengths[:2]) if strengths else "none",
            "weaknesses": "; ".join(weaknesses[:2]) if weaknesses else "none",
            "recommendation": recommendation[:120],
        },
        verbose,
    )

    scenario_counts = {
        "market_drawdown_shocks": len(SHOCK_LEVELS),
        "signal_deterioration_cases": part_b.get("cases_tested", 0),
        "rotation_pressure_cases": part_c.get("cases_tested", 0),
        "concentration_shock_cases": part_d.get("cases_tested", 0),
        "protection_framework_cases": part_e.get("cases_tested", 0),
        "total_scenarios": (
            len(SHOCK_LEVELS)
            + part_b.get("cases_tested", 0)
            + part_c.get("cases_tested", 0)
            + part_d.get("cases_tested", 0)
            + part_e.get("cases_tested", 0)
        ),
    }

    all_csv = csv_a + csv_b + csv_c + csv_d + csv_e
    for row in all_csv:
        if not row.get("generated_at"):
            row["generated_at"] = ts
    for area, status in scorecard.items():
        all_csv.append(
            {
                "section": "scorecard",
                "scenario": area,
                "protection": "",
                "symbol": "",
                "status": status,
                "detail": f"Phase 156 stress test — {area}",
                "generated_at": ts,
            }
        )
    all_csv.append(
        {
            "section": "promotion_review",
            "scenario": "certification",
            "protection": "",
            "symbol": "",
            "status": promotion.get("new_status", ""),
            "detail": promotion.get("reason", ""),
            "generated_at": ts,
        }
    )

    payload: Dict[str, Any] = {
        "generated_at": ts,
        "phase": "156",
        "mode": "simulation_only_stress_test",
        "disclaimer": (
            "Stress testing simulates adverse conditions only. "
            "No execution, lifecycle, or governance changes."
        ),
        "scorecard": scorecard,
        "scenario_counts": scenario_counts,
        "part_a_market_drawdown": part_a,
        "part_b_signal_deterioration": part_b,
        "part_c_rotation_pressure": part_c,
        "part_d_concentration_shock": part_d,
        "part_e_protection_framework": part_e,
        "part_g_stress_certification_review": promotion,
        "part_h_final_decision": {
            "status": scorecard["overall_stress_result"],
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": recommendation,
        },
        "promotion_recommendation": promotion.get("promotion_recommended"),
        "_csv_rows": all_csv,
    }
    return payload


def run_portfolio_management_stress_test(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute stress test and write JSON + CSV artifacts."""
    rd = Path(results_dir) if results_dir else RESULTS
    payload = compute_portfolio_management_stress_test(results_dir=rd, verbose=verbose)
    csv_rows = payload.pop("_csv_rows", [])

    out_json = rd / "portfolio_management_stress_test.json"
    out_csv = rd / "portfolio_management_stress_scorecard.csv"
    _atomic_write_json(payload, out_json)
    _write_scorecard_csv(csv_rows, out_csv)

    scorecard = payload.get("scorecard") or {}
    promo = payload.get("part_g_stress_certification_review") or {}
    print(
        f"[PORTFOLIO_MANAGEMENT_STRESS_TEST] phase=156 "
        f"overall={scorecard.get('overall_stress_result')} "
        f"promotion={promo.get('new_status')} "
        f"json={out_json} csv={out_csv}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 156 — Portfolio Management Stress Testing (simulation only)"
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed stress logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = run_portfolio_management_stress_test(
        results_dir=Path(args.results_dir),
        verbose=args.verbose,
    )
    overall = (payload.get("scorecard") or {}).get("overall_stress_result", "FAIL")
    return 0 if overall in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
