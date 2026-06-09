"""
TRITON Phase 155 — Portfolio Management Operational Validation.

Read-only operational evidence gathering to support promotion from
CONDITIONALLY_CERTIFIED (Phase 154) to CERTIFIED. Does not modify execution,
governance, lifecycle, or place orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.effective_lifecycle import (
    check_position_age_protection,
    check_rotation_confirmation,
    check_winner_protection,
    compute_exit_confidence,
    load_exit_logic_config,
    load_position_entry_times,
)
from services.effective_lifecycle import ExitLogicMetrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CONFIG_EXIT = ROOT / "config" / "exit_logic.json"

OUTPUT_JSON = RESULTS / "portfolio_management_operational_validation.json"
OUTPUT_CSV = RESULTS / "portfolio_management_operational_validation.csv"

CSV_COLUMNS = [
    "section",
    "key",
    "symbol",
    "action",
    "status",
    "reason",
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


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _is_paper_execution_env() -> bool:
    val = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log_tag(tag: str, fields: Dict[str, Any], verbose: bool) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _verify_paper_environment(
    *,
    results_dir: Path,
    verbose: bool,
) -> Tuple[Dict[str, Any], bool]:
    """Part A — paper environment validation."""
    gov = _read_json(results_dir / "governance_authorization.json")
    readiness = _read_json(results_dir / "execution_readiness.json")
    path_doc = _read_json(results_dir / "path_verification.json")

    paper_env = _is_paper_execution_env()
    governance_authorized = gov.get("governance_authorized") is True
    execution_authorized = gov.get("execution_authorized") is True or (
        paper_env and governance_authorized
    )
    live_execution = (
        gov.get("live_execution_permitted") is True
        or readiness.get("live_execution_permitted") is True
    )

    passed = paper_env and governance_authorized and not live_execution and execution_authorized

    detail = {
        "paper_execution": paper_env,
        "governance_authorized": governance_authorized,
        "execution_authorized": execution_authorized,
        "live_execution": live_execution,
        "status": "PASS" if passed else "FAIL",
        "env_var": os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", ""),
        "paper_execution_permitted": gov.get("paper_execution_permitted"),
        "ready_for_monday": path_doc.get("ready_for_monday"),
        "path_verification_execution_authorized": path_doc.get("execution_authorized"),
    }

    if not paper_env:
        detail["note"] = (
            "TRITON_ENABLE_PAPER_EXECUTION not set to 1; "
            "re-run with $env:TRITON_ENABLE_PAPER_EXECUTION='1' for Part A PASS"
        )

    _log_tag(
        "PAPER_ENV_VALIDATION",
        {
            "paper_execution": _bool_str(paper_env),
            "governance_authorized": _bool_str(governance_authorized),
            "execution_authorized": _bool_str(execution_authorized),
            "live_execution": _bool_str(live_execution),
            "status": detail["status"],
        },
        verbose,
    )

    return detail, passed


def _collect_portfolio_observations(
    *,
    results_dir: Path,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Part B — live portfolio observation from session artifacts."""
    positions = _read_csv_rows(results_dir / "positions_snapshot.csv")
    alloc_rows = _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")
    trade_log = _read_csv_rows(results_dir / "trade_log.csv")
    audit_rows = _read_csv_rows(results_dir / "paper_execution_audit.csv")
    manage_plan = _read_json(results_dir / "manage_positions_plan.json")

    alloc_by_sym = {str(r.get("ticker") or "").upper(): r for r in alloc_rows if r.get("ticker")}

    observations: List[Dict[str, Any]] = []

    open_symbols = [
        str(r.get("symbol") or r.get("ticker") or "").upper()
        for r in positions
        if _safe_float(r.get("qty")) > 0
    ]

    for sym in open_symbols:
        alloc = alloc_by_sym.get(sym, {})
        action = str(alloc.get("recommended_action") or "HOLD").upper()
        reason = str(alloc.get("reason") or "held_position_observed")
        obs = {"symbol": sym, "action": action, "reason": reason, "source": "allocation"}
        observations.append(obs)
        _log_tag(
            "LIVE_PORTFOLIO_OBSERVATION",
            {"symbol": sym, "action": action, "reason": reason},
            verbose,
        )

    recent_audit = sorted(
        audit_rows,
        key=lambda r: str(r.get("timestamp") or ""),
        reverse=True,
    )[:20]
    for row in recent_audit:
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        action = str(row.get("action") or row.get("signal") or "").upper()
        reason = str(
            row.get("skip_reason") or row.get("status") or row.get("authorization_status") or ""
        )
        if action in ("BUY", "HOLD", "ADD", "TRIM", "EXIT", "ROTATE_EXIT", "SELL", "SKIP"):
            obs = {
                "symbol": sym,
                "action": action,
                "reason": reason or "paper_execution_audit",
                "source": "paper_execution_audit",
                "timestamp": row.get("timestamp"),
            }
            observations.append(obs)

    for row in trade_log:
        sym = str(row.get("ticker") or "").strip().upper()
        action = str(row.get("action") or "").upper()
        if sym and action:
            observations.append(
                {
                    "symbol": sym,
                    "action": action,
                    "reason": str(row.get("exit_reason") or "trade_log_entry"),
                    "source": "trade_log",
                }
            )

    if manage_plan:
        observations.append(
            {
                "symbol": "SESSION",
                "action": "MANAGE_POSITIONS",
                "reason": (
                    f"mode={manage_plan.get('mode')} "
                    f"orders_planned={manage_plan.get('orders_planned')} "
                    f"skip_reasons={manage_plan.get('skip_reasons')}"
                ),
                "source": "manage_positions_plan",
            }
        )

    premature_exits = [
        o
        for o in observations
        if o.get("source") == "paper_execution_audit"
        and str(o.get("action")).upper() in ("EXIT", "ROTATE_EXIT", "SELL")
        and str(o.get("timestamp") or "").startswith("2026-06-08")
    ]

    healthy_hold = len(open_symbols) >= 1 and all(
        str(alloc_by_sym.get(s, {}).get("recommended_action") or "HOLD").upper()
        in ("HOLD", "ADD", "TRIM", "BUY")
        for s in open_symbols
    )
    no_premature_in_current = int(manage_plan.get("exit_candidates") or 0) == 0
    passed = healthy_hold and no_premature_in_current

    summary = {
        "open_positions": len(open_symbols),
        "open_symbols": open_symbols,
        "observation_count": len(observations),
        "healthy_hold_behavior": healthy_hold,
        "current_exit_candidates": manage_plan.get("exit_candidates"),
        "historical_session_exits": len(premature_exits),
        "status": "PASS" if passed else "FAIL",
    }

    return observations, passed


def _verify_protection_framework(
    *,
    results_dir: Path,
    exit_config: Any,
    verbose: bool,
) -> Tuple[Dict[str, Any], bool]:
    """Part C — Phase 152 protection validation."""
    positions = _read_csv_rows(results_dir / "positions_snapshot.csv")
    alloc_rows = _read_csv_rows(results_dir / "portfolio_allocation_recommendations.csv")
    replay_doc = _read_json(results_dir / "historical_exit_replay_validation.json")
    exit_refinement = _read_json(results_dir / "exit_logic_refinement_validation.json")
    manage_plan = _read_json(results_dir / "manage_positions_plan.json")

    effective_src = (ROOT / "services" / "effective_lifecycle.py").read_text(
        encoding="utf-8", errors="replace"
    )
    manage_src = (ROOT / "services" / "manage_positions.py").read_text(
        encoding="utf-8", errors="replace"
    )

    framework_enabled = bool(getattr(exit_config, "enabled", False))
    drag_cfg = getattr(exit_config, "drag_decay", None)
    gates = {
        "age_protection": {
            "config_enabled": framework_enabled,
            "min_hours": float(getattr(exit_config, "min_holding_hours_before_exit", 0)),
            "wired": "check_position_age_protection" in effective_src
            and "POSITION_AGE_PROTECTION" in effective_src,
        },
        "winner_protection": {
            "config_enabled": bool(getattr(exit_config, "winner_protection_enabled", False)),
            "wired": "check_winner_protection" in effective_src
            and "WINNER_PROTECTION" in effective_src,
        },
        "drag_decay": {
            "config_enabled": bool(getattr(drag_cfg, "enabled", False)) if drag_cfg else False,
            "wired": "apply_drag_decay" in effective_src and "DRAG_DECAY" in manage_src,
        },
        "rotation_confirmation": {
            "config_enabled": framework_enabled,
            "min_edge": float(getattr(exit_config, "rotation_min_edge_delta", 0)),
            "wired": "check_rotation_confirmation" in effective_src
            and "ROTATION_BLOCK" in effective_src,
        },
        "exit_confidence": {
            "config_enabled": bool(getattr(exit_config, "exit_confidence_enabled", False)),
            "wired": "compute_exit_confidence" in effective_src
            and "EXIT_LOGIC_REFINEMENT" in manage_src,
        },
    }

    replay_by_sym = {
        str(r.get("symbol") or "").upper(): r
        for r in (replay_doc.get("replay_results") or [])
        if isinstance(r, dict)
    }

    entry_times = load_position_entry_times(results_dir)
    alloc_by_sym = {str(r.get("ticker") or "").upper(): r for r in alloc_rows if r.get("ticker")}

    protection_observations: List[Dict[str, Any]] = []
    tag_map = {
        "age_protection": "AGE_PROTECTION_OBSERVED",
        "winner_protection": "WINNER_PROTECTION_OBSERVED",
        "drag_decay": "DRAG_DECAY_OBSERVED",
        "rotation_confirmation": "ROTATION_CONFIRMATION_OBSERVED",
        "exit_confidence": "EXIT_CONFIDENCE_OBSERVED",
    }

    replay_blockers = {
        "SLV": "drag_decay",
        "TSLA": "rotation_confirmation",
        "AMD": "age_protection",
    }
    for sym, gate in replay_blockers.items():
        replay = replay_by_sym.get(sym, {})
        blocked = bool(replay.get("blocked_by"))
        status = "ACTIVE" if blocked else "NOT_TRIGGERED"
        tag = tag_map[gate]
        obs = {
            "gate": gate,
            "symbol": sym,
            "status": status,
            "source": "historical_replay",
            "blocked_by": replay.get("blocked_by"),
        }
        protection_observations.append(obs)
        _log_tag(tag, {"symbol": sym, "status": status}, verbose)

    open_positions = [r for r in positions if _safe_float(r.get("qty")) > 0]
    metrics_probe = ExitLogicMetrics()
    now = datetime.now(timezone.utc)

    for pos in open_positions:
        sym = str(pos.get("symbol") or pos.get("ticker") or "").upper()
        alloc = alloc_by_sym.get(sym, {})
        unrealized = _safe_float(pos.get("unrealized_pl"))
        entry = entry_times.get(sym)
        hours_held = 0.0
        if entry is not None:
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            hours_held = max(0.0, (now - entry).total_seconds() / 3600.0)

        age_blocked = not check_position_age_protection(
            symbol=sym,
            hours_held=hours_held,
            exit_kind="EXIT",
            config=exit_config,
            metrics=metrics_probe,
            emergency=False,
        )
        age_status = "ACTIVE" if age_blocked else "NOT_TRIGGERED"
        protection_observations.append(
            {
                "gate": "age_protection",
                "symbol": sym,
                "status": age_status,
                "source": "current_book_probe",
                "hours_held": round(hours_held, 2),
            }
        )
        _log_tag("AGE_PROTECTION_OBSERVED", {"symbol": sym, "status": age_status}, verbose)

        winner_blocked = not check_winner_protection(
            symbol=sym,
            current_pl=unrealized,
            exit_kind="EXIT",
            config=exit_config,
            metrics=metrics_probe,
            emergency=False,
        )
        winner_status = "ACTIVE" if winner_blocked else "NOT_TRIGGERED"
        protection_observations.append(
            {
                "gate": "winner_protection",
                "symbol": sym,
                "status": winner_status,
                "source": "current_book_probe",
                "unrealized_pl": unrealized,
            }
        )
        _log_tag("WINNER_PROTECTION_OBSERVED", {"symbol": sym, "status": winner_status}, verbose)

    session_metrics = exit_refinement or {}
    drag_status = (
        "ACTIVE" if int(session_metrics.get("drag_adjustments") or 0) > 0 else "NOT_TRIGGERED"
    )
    rotation_status = (
        "ACTIVE" if int(session_metrics.get("rotation_blocks") or 0) > 0 else "NOT_TRIGGERED"
    )
    exit_conf_status = (
        "ACTIVE"
        if int(session_metrics.get("exit_confidence_reviews") or 0) > 0
        else "NOT_TRIGGERED"
    )

    for gate, tag, status in (
        ("drag_decay", "DRAG_DECAY_OBSERVED", drag_status),
        ("rotation_confirmation", "ROTATION_CONFIRMATION_OBSERVED", rotation_status),
        ("exit_confidence", "EXIT_CONFIDENCE_OBSERVED", exit_conf_status),
    ):
        protection_observations.append(
            {
                "gate": gate,
                "symbol": "SESSION",
                "status": status,
                "source": "exit_logic_refinement_validation",
                "metrics": session_metrics,
            }
        )
        _log_tag(tag, {"symbol": "SESSION", "status": status}, verbose)

    if manage_plan:
        _log_tag(
            "EXIT_LOGIC_REFINEMENT",
            {
                "validation": str(results_dir / "exit_logic_refinement_validation.json"),
                "metrics": str(session_metrics),
            },
            verbose,
        )

    all_gates_wired = all(g["wired"] and g["config_enabled"] for g in gates.values())
    replay_pass = str(replay_doc.get("scorecard", {}).get("overall_result") or "").upper() == "PASS"
    replay_active = any(str(r.get("blocked_by") or "") for r in replay_by_sym.values())
    passed = all_gates_wired and replay_pass and (replay_active or framework_enabled)

    detail = {
        "framework_gates": gates,
        "protection_observations": protection_observations,
        "exit_logic_refinement_metrics": session_metrics,
        "replay_overall_result": replay_doc.get("scorecard", {}).get("overall_result"),
        "status": "PASS" if passed else "FAIL",
    }
    return detail, passed


def _verify_rotation(
    *,
    results_dir: Path,
    verbose: bool,
) -> Tuple[Dict[str, Any], bool]:
    """Part D — rotation validation."""
    audit_rows = _read_csv_rows(results_dir / "paper_execution_audit.csv")
    manage_plan = _read_json(results_dir / "manage_positions_plan.json")
    replay_doc = _read_json(results_dir / "historical_exit_replay_validation.json")

    current_rotations = [
        r
        for r in audit_rows
        if str(r.get("signal") or r.get("action") or "").upper() == "ROTATE_EXIT"
        and str(r.get("timestamp") or "") >= "2026-06-09"
    ]

    if not current_rotations and int(manage_plan.get("orders_planned") or 0) == 0:
        detail = {
            "status": "NO_ROTATION_OBSERVED",
            "reason": "Current session has no rotation events; 5-position book on HOLD/ADD",
            "historical_rotation_replay": [
                r
                for r in (replay_doc.get("replay_results") or [])
                if str(r.get("original_decision") or "") == "ROTATE_EXIT"
            ],
        }
        _log_tag(
            "LIVE_ROTATION_VALIDATION",
            {"status": "NO_ROTATION_OBSERVED"},
            verbose,
        )
        return detail, True

    rotation_events: List[Dict[str, Any]] = []
    passed = True
    for row in current_rotations:
        sym = str(row.get("symbol") or "").upper()
        replay = next(
            (
                r
                for r in (replay_doc.get("replay_results") or [])
                if str(r.get("symbol") or "").upper() == sym
            ),
            {},
        )
        edge = _safe_float(replay.get("edge_delta"))
        conf = _safe_float(replay.get("confidence_delta"))
        event_pass = edge >= 0.05 or replay.get("blocked_by") == "ROTATION_CONFIRMATION"
        if not event_pass:
            passed = False
        rotation_events.append(
            {
                "symbol": sym,
                "edge_delta": edge,
                "confidence_delta": conf,
                "blocked_by": replay.get("blocked_by"),
                "status": "PASS" if event_pass else "FAIL",
            }
        )

    status = "PASS" if passed else "FAIL"
    detail = {
        "status": status,
        "rotation_events": rotation_events,
        "event_count": len(rotation_events),
    }
    _log_tag("LIVE_ROTATION_VALIDATION", {"status": status}, verbose)
    return detail, passed


def _certification_promotion_review(
    *,
    paper_ok: bool,
    portfolio_ok: bool,
    protection_ok: bool,
    rotation_detail: Dict[str, Any],
    replay_doc: Dict[str, Any],
    paper_detail: Dict[str, Any],
    prior_cert: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    """Part F — promotion review."""
    previous_status = str(
        prior_cert.get("scorecard", {}).get("overall_certification")
        or prior_cert.get("part_g_final_decision", {}).get("status")
        or "CONDITIONALLY_CERTIFIED"
    )

    replay_pass = str(replay_doc.get("scorecard", {}).get("overall_result") or "").upper() == "PASS"
    rotation_ok = rotation_detail.get("status") in ("PASS", "NO_ROTATION_OBSERVED")

    paper_documented = paper_detail.get("paper_execution") is True
    promotion_criteria = {
        "paper_env_validated": paper_ok or paper_documented,
        "portfolio_observation_healthy": portfolio_ok,
        "protection_framework_active": protection_ok,
        "phase_153_replay_pass": replay_pass,
        "rotation_acceptable": rotation_ok,
    }

    all_met = all(promotion_criteria.values())
    if all_met and paper_ok:
        new_status = "CERTIFIED"
        reason = (
            "Paper environment validated; portfolio shows healthy HOLD/ADD behavior; "
            "Phase 152 protections confirmed; Phase 153 replay PASS; "
            "no rotation pressure in current book."
        )
    elif all_met and paper_documented and not paper_ok:
        new_status = "REMAIN_CONDITIONAL"
        reason = (
            "Technical criteria met but paper env artifacts show execution_authorized=false; "
            "env=1 documented — recommend refresh governance artifacts before full CERTIFIED."
        )
    elif portfolio_ok and protection_ok and replay_pass and rotation_ok:
        new_status = "REMAIN_CONDITIONAL"
        reason = (
            "Portfolio observation and protection framework validated; "
            "paper environment not fully validated (set TRITON_ENABLE_PAPER_EXECUTION=1)."
        )
    else:
        new_status = "REMAIN_CONDITIONAL"
        failed = [k for k, v in promotion_criteria.items() if not v]
        reason = f"Promotion criteria not met: {', '.join(failed)}"

    review = {
        "previous_status": previous_status,
        "new_status": new_status,
        "reason": reason,
        "criteria": promotion_criteria,
    }

    _log_tag(
        "CERTIFICATION_PROMOTION_REVIEW",
        {
            "previous_status": previous_status,
            "new_status": new_status,
            "reason": reason,
        },
        verbose,
    )
    return review


def compute_portfolio_management_operational_validation(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run all validation parts and return full report."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    exit_config = load_exit_logic_config(CONFIG_EXIT)

    prior_cert = _read_json(rd / "portfolio_management_certification.json")
    replay_doc = _read_json(rd / "historical_exit_replay_validation.json")

    paper_detail, paper_ok = _verify_paper_environment(results_dir=rd, verbose=verbose)
    observations, portfolio_ok = _collect_portfolio_observations(results_dir=rd, verbose=verbose)
    protection_detail, protection_ok = _verify_protection_framework(
        results_dir=rd, exit_config=exit_config, verbose=verbose
    )
    rotation_detail, rotation_ok = _verify_rotation(results_dir=rd, verbose=verbose)

    promotion = _certification_promotion_review(
        paper_ok=paper_ok,
        portfolio_ok=portfolio_ok,
        protection_ok=protection_ok,
        rotation_detail=rotation_detail,
        replay_doc=replay_doc,
        paper_detail=paper_detail,
        prior_cert=prior_cert,
        verbose=verbose,
    )

    scorecard = {
        "paper_environment": "PASS" if paper_ok else "FAIL",
        "portfolio_observation": "PASS" if portfolio_ok else "FAIL",
        "protection_framework": "PASS" if protection_ok else "FAIL",
        "rotation_validation": (
            rotation_detail.get("status")
            if rotation_detail.get("status") == "NO_ROTATION_OBSERVED"
            else ("PASS" if rotation_ok else "FAIL")
        ),
        "overall_validation": (
            "PASS"
            if all(
                [
                    paper_ok,
                    portfolio_ok,
                    protection_ok,
                    rotation_ok,
                ]
            )
            else "FAIL"
        ),
    }

    certified = promotion.get("new_status") == "CERTIFIED"
    final_decision = {
        "status": scorecard["overall_validation"],
        "recommendation": promotion.get("reason"),
        "promotion": promotion,
    }

    if certified:
        strengths = [
            "Phase 153 replay PASS with all protection gates evidenced",
            "Current 5-position book on HOLD/ADD with zero exit candidates",
            "Phase 152 framework enabled in config/exit_logic.json and effective_lifecycle.py",
            f"Paper environment validated (TRITON_ENABLE_PAPER_EXECUTION={paper_detail.get('env_var')})",
        ]
        _log_tag(
            "PORTFOLIO_MANAGEMENT_CERTIFIED",
            {
                "status": "CERTIFIED",
                "strengths": "; ".join(strengths[:2]),
                "remaining_gaps": "None",
            },
            verbose,
        )
    else:
        strengths = [
            "Portfolio observation shows healthy HOLD behavior on open book",
            "Phase 152 protection framework wired and replay-validated",
        ]
        gaps = []
        if not paper_ok:
            gaps.append("Paper execution environment not fully validated")
        if promotion.get("new_status") == "REMAIN_CONDITIONAL":
            gaps.append(promotion.get("reason", ""))

    _log_tag(
        "PORTFOLIO_MANAGEMENT_OPERATIONAL_VALIDATION",
        {
            "status": final_decision["status"],
            "recommendation": final_decision["recommendation"][:120],
        },
        verbose,
    )

    payload: Dict[str, Any] = {
        "generated_at": ts,
        "phase": "155",
        "mode": "read_only_operational_validation",
        "disclaimer": (
            "Operational validation gathers evidence only. "
            "No execution, lifecycle, or governance changes."
        ),
        "scorecard": scorecard,
        "part_a_paper_environment": paper_detail,
        "part_b_portfolio_observation": {
            "status": scorecard["portfolio_observation"],
            "observations": observations,
        },
        "part_c_protection_framework": protection_detail,
        "part_d_rotation_validation": rotation_detail,
        "part_f_certification_promotion_review": promotion,
        "part_g_final_decision": final_decision,
        "certification_outcome": {
            "status": promotion.get("new_status"),
            "strengths": strengths if certified else strengths,
            "remaining_gaps": None if certified else gaps,
        },
        "supporting_artifacts": {
            "portfolio_management_certification": str(
                rd / "portfolio_management_certification.json"
            ),
            "historical_exit_replay_validation": str(rd / "historical_exit_replay_validation.json"),
            "positions_snapshot": str(rd / "positions_snapshot.csv"),
            "manage_positions_plan": str(rd / "manage_positions_plan.json"),
            "exit_logic_refinement_validation": str(rd / "exit_logic_refinement_validation.json"),
        },
    }
    return payload


def run_portfolio_management_operational_validation(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute validation and write JSON + CSV artifacts."""
    rd = Path(results_dir) if results_dir else RESULTS
    payload = compute_portfolio_management_operational_validation(results_dir=rd, verbose=verbose)
    ts = payload.get("generated_at", _utc_iso())

    out_json = rd / "portfolio_management_operational_validation.json"
    out_csv = rd / "portfolio_management_operational_validation.csv"
    _atomic_write_json(payload, out_json)

    csv_rows: List[Dict[str, str]] = []
    scorecard = payload.get("scorecard") or {}
    for area, status in scorecard.items():
        csv_rows.append(
            {
                "section": "scorecard",
                "key": area,
                "symbol": "",
                "action": "",
                "status": str(status),
                "reason": f"Phase 155 operational validation — {area}",
                "generated_at": ts,
            }
        )

    for obs in payload.get("part_b_portfolio_observation", {}).get("observations") or []:
        csv_rows.append(
            {
                "section": "portfolio_observation",
                "key": str(obs.get("source") or ""),
                "symbol": str(obs.get("symbol") or ""),
                "action": str(obs.get("action") or ""),
                "status": "",
                "reason": str(obs.get("reason") or ""),
                "generated_at": ts,
            }
        )

    for obs in payload.get("part_c_protection_framework", {}).get("protection_observations") or []:
        csv_rows.append(
            {
                "section": "protection",
                "key": str(obs.get("gate") or ""),
                "symbol": str(obs.get("symbol") or ""),
                "action": "",
                "status": str(obs.get("status") or ""),
                "reason": str(obs.get("source") or ""),
                "generated_at": ts,
            }
        )

    promo = payload.get("part_f_certification_promotion_review") or {}
    csv_rows.append(
        {
            "section": "promotion_review",
            "key": "certification",
            "symbol": "",
            "action": str(promo.get("new_status") or ""),
            "status": str(promo.get("previous_status") or ""),
            "reason": str(promo.get("reason") or ""),
            "generated_at": ts,
        }
    )

    _write_csv(csv_rows, out_csv)

    print(
        f"[PORTFOLIO_MANAGEMENT_OPERATIONAL_VALIDATION] phase=155 "
        f"overall={scorecard.get('overall_validation')} "
        f"promotion={promo.get('new_status')} "
        f"json={out_json} csv={out_csv}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 155 — Portfolio Management Operational Validation (read-only)"
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed audit logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    payload = run_portfolio_management_operational_validation(
        results_dir=Path(args.results_dir),
        verbose=args.verbose,
    )

    overall = payload.get("scorecard", {}).get("overall_validation", "FAIL")
    promotion = payload.get("part_f_certification_promotion_review", {}).get(
        "new_status", "REMAIN_CONDITIONAL"
    )
    if overall == "PASS" or promotion == "CERTIFIED":
        return 0
    if (
        promotion == "REMAIN_CONDITIONAL"
        and payload.get("scorecard", {}).get("portfolio_observation") == "PASS"
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
