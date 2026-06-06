"""
TRITON Governance Authorization & Execution Readiness — Phases 16–18, 147.

Authorization gates, readiness assessment, and paper-mode protective trials.
Paper execution authorized only via policy gate (Phase 147). Live trading disabled.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_FRESHNESS_MINUTES = 30
SIGNALS_FRESHNESS_HOURS = 48
LIFECYCLE_FRESHNESS_HOURS = 48

GOVERNANCE_BLOCK_LEVELS = frozenset(
    {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }
)
GOVERNANCE_PERMIT_ESCALATION = frozenset({"GREEN", "YELLOW"})


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}")


def _enabled_policy_count(policy_doc: Dict[str, Any]) -> int:
    explicit = policy_doc.get("enabled_policy_count")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    return sum(
        1 for p in (policy_doc.get("policies") or []) if isinstance(p, dict) and p.get("enabled")
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _file_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (_utc_now() - mtime).total_seconds() / 60.0


def _policy_by_action(policy_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(p.get("action")): p
        for p in (policy_doc.get("policies") or [])
        if isinstance(p, dict) and p.get("action")
    }


def _approved_actions(queue_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for req in queue_doc.get("requests") or []:
        if not isinstance(req, dict):
            continue
        if str(req.get("status")) != "APPROVED":
            continue
        action = str(req.get("candidate_action") or "")
        fp = str(req.get("candidate_fingerprint") or "")
        if action:
            out[action] = req
        if fp:
            out[fp] = req
    return out


def _evaluate_governance_gate(
    gov_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    policy: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    escalation = str(cpe_doc.get("escalation_state") or gov_doc.get("escalation_state") or "")
    awareness = str(gov_doc.get("governance_awareness_level") or "")
    requires_green = bool((policy or {}).get("requires_green_governance", True))

    if awareness in GOVERNANCE_BLOCK_LEVELS:
        return False, f"Governance awareness blocks action ({awareness})"
    if requires_green and escalation not in GOVERNANCE_PERMIT_ESCALATION:
        return False, f"Escalation {escalation} does not meet green-governance requirement"
    if escalation in ("RED", "CRITICAL"):
        return False, f"Escalation state {escalation} requires review before authorization"
    return True, "Governance posture permits evaluation"


def _evaluate_operator_gate(
    candidate: Dict[str, Any],
    approved: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    action = str(candidate.get("candidate_action") or "")
    fp = str(candidate.get("candidate_id") or "")
    if fp in approved or action in approved:
        return True, "Operator approval present in queue"
    return False, "No approved operator authorization for this action"


def _evaluate_policy_gate(
    candidate: Dict[str, Any],
    policy_doc: Dict[str, Any],
    policies: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    action = str(candidate.get("candidate_action") or "")
    pol = policies.get(action)
    if not pol:
        return False, f"No protective policy defined for {action}"
    if not pol.get("enabled"):
        return False, f"Protective policy disabled for {action}"
    if _enabled_policy_count(policy_doc) <= 0:
        return False, "No protective policies enabled"
    if policy_doc.get("global_execution_enabled") is not True:
        return False, "Global execution remains disabled"
    return True, "Policy permits action (paper authorization layer — not executing)"


def _evaluate_execution_gate(policy_doc: Dict[str, Any]) -> Tuple[bool, str]:
    mode = str(policy_doc.get("mode") or "").lower()
    enabled_count = _enabled_policy_count(policy_doc)

    if policy_doc.get("live_execution_enabled") is True or mode == "live":
        _log("LIVE_EXECUTION_BLOCKED", "Live execution flag or mode detected")
        return False, "Live execution blocked — live mode not permitted"

    if policy_doc.get("automated_trading_permitted") is True and mode != "paper":
        _log("EXECUTION_GATE_BLOCK", "automated_trading_permitted=true outside paper mode")
        return False, "Automated live trading not permitted"

    paper_ok = policy_doc.get("paper_execution_enabled") is True
    global_ok = policy_doc.get("global_execution_enabled") is True
    live_ok = policy_doc.get("live_execution_enabled") is False
    auto_ok = policy_doc.get("automated_trading_permitted") is False or mode == "paper"

    if paper_ok and global_ok and live_ok and auto_ok and enabled_count > 0:
        _log("EXECUTION_GATE", "Paper execution authorized by policy gate")
        return True, "Paper execution authorized by policy gate"

    reasons: List[str] = []
    if not paper_ok:
        reasons.append("paper_execution_enabled is false")
    if not global_ok:
        reasons.append("global_execution_enabled is false")
    if not live_ok:
        reasons.append("live_execution_enabled must remain false")
    if not auto_ok:
        reasons.append("automated_trading_permitted blocks non-paper automation")
    if enabled_count <= 0:
        reasons.append("enabled_policy_count is 0")

    reason = "; ".join(reasons) if reasons else "Execution authorization disabled"
    _log("EXECUTION_GATE_BLOCK", reason)
    return False, reason


def compute_governance_authorization(
    *,
    candidates_doc: Dict[str, Any],
    queue_doc: Dict[str, Any],
    policy_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 16: four-layer authorization gate (read-only evaluation)."""
    ts = _iso_utc()
    policies = _policy_by_action(policy_doc)
    approved = _approved_actions(queue_doc)

    sys_gov, sys_gov_reason = _evaluate_governance_gate(gov_doc, cpe_doc, None)
    sys_op = len(approved) > 0
    sys_pol = _enabled_policy_count(policy_doc) > 0
    sys_exec, sys_exec_reason = _evaluate_execution_gate(policy_doc)

    if sys_exec:
        _log("PAPER_EXECUTION_AUTH", "System execution gate open for paper mode only")

    candidate_auths: List[Dict[str, Any]] = []
    for cand in candidates_doc.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        pol = policies.get(str(cand.get("candidate_action") or ""))
        gov_ok, gov_reason = _evaluate_governance_gate(gov_doc, cpe_doc, pol)
        op_ok, op_reason = _evaluate_operator_gate(cand, approved)
        pol_ok, pol_reason = _evaluate_policy_gate(cand, policy_doc, policies)
        exec_ok, exec_reason = _evaluate_execution_gate(policy_doc)
        overall = bool(gov_ok and op_ok and pol_ok and exec_ok)

        candidate_auths.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "candidate_action": cand.get("candidate_action"),
                "governance_authorized": gov_ok,
                "operator_authorized": op_ok,
                "policy_authorized": pol_ok,
                "execution_authorized": exec_ok,
                "overall_authorization": overall,
                "gate_reasons": {
                    "governance": gov_reason,
                    "operator": op_reason,
                    "policy": pol_reason,
                    "execution": exec_reason,
                },
            }
        )

    overall_system = bool(sys_gov and sys_op and sys_pol and sys_exec)

    return {
        "generated_at": ts,
        "governance_authorized": sys_gov,
        "operator_authorized": sys_op,
        "policy_authorized": sys_pol,
        "execution_authorized": sys_exec,
        "overall_authorization": overall_system,
        "gate_reasons": {
            "governance": sys_gov_reason,
            "operator": (
                "At least one operator approval present"
                if sys_op
                else "No approved operator authorizations"
            ),
            "policy": (
                "At least one protective policy enabled"
                if sys_pol
                else "All protective policies disabled"
            ),
            "execution": sys_exec_reason,
        },
        "paper_execution_permitted": sys_exec,
        "live_execution_permitted": False,
        "candidate_authorizations": candidate_auths,
        "authorization_questions": {
            "is_governance_permitting": sys_gov,
            "is_operator_approval_present": sys_op,
            "is_policy_enabled": sys_pol,
            "is_execution_allowed": sys_exec,
        },
        "disclaimer": (
            "Authorization evaluation only. Paper execution may be permitted when configured; "
            "live execution remains blocked. This layer does not place orders."
        ),
    }


def _check_data_freshness(
    *,
    watchdog_ts: str,
    cpi_doc: Dict[str, Any],
    results_dir: Path,
) -> Tuple[bool, str]:
    now = _utc_now()
    wd = _parse_iso(watchdog_ts)
    if wd and (now - wd).total_seconds() / 60.0 > DATA_FRESHNESS_MINUTES:
        return False, f"Watchdog data older than {DATA_FRESHNESS_MINUTES} minutes"
    cpi_ts = _parse_iso(cpi_doc.get("generated_at"))
    if cpi_ts and (now - cpi_ts).total_seconds() / 60.0 > DATA_FRESHNESS_MINUTES * 2:
        return False, "Capital preservation intelligence is stale"
    status_age = _file_age_minutes(results_dir / "watchdog_status.json")
    if status_age is not None and status_age > DATA_FRESHNESS_MINUTES:
        return False, "watchdog_status.json is stale"
    return True, "Core risk artifacts are fresh"


def _check_watchdog_health(watchdog_status: str, active_alerts: List[Any]) -> Tuple[bool, str]:
    stale_types = {str(a.get("alert_type")) for a in active_alerts if isinstance(a, dict)}
    if "STALE_HEARTBEAT" in stale_types:
        return False, "Stale heartbeat alert active"
    if str(watchdog_status).upper() not in ("OK", "WARN", "WARNING"):
        return False, f"Watchdog status is {watchdog_status}"
    return True, "Watchdog monitoring healthy"


def _check_broker_health(broker_connected: bool, broker_error: Optional[str]) -> Tuple[bool, str]:
    if not broker_connected:
        return False, broker_error or "Broker disconnected"
    return True, "Broker connectivity OK (read-only)"


def _check_lifecycle_integrity(results_dir: Path) -> Tuple[bool, str]:
    lifecycle_cfg = ROOT / "config" / "lifecycle_logic.json"
    lifecycle_csv = results_dir / "signal_lifecycle.csv"
    signals_csv = results_dir / "signals.csv"

    if lifecycle_cfg.is_file():
        cfg = _read_json(lifecycle_cfg)
        if cfg.get("enabled") is False:
            return False, "Lifecycle logic disabled in config"

    if not lifecycle_csv.is_file():
        return False, "signal_lifecycle.csv missing"

    age = _file_age_minutes(lifecycle_csv)
    if age is not None and age > LIFECYCLE_FRESHNESS_HOURS * 60:
        return False, "signal_lifecycle.csv is stale"

    if signals_csv.is_file():
        sig_mtime = lifecycle_csv.stat().st_mtime
        mtimes = [signals_csv.stat().st_mtime]
        mtimes.extend(p.stat().st_mtime for p in results_dir.glob("signals*.csv") if p.is_file())
        upstream_mtime = max(mtimes)
        if sig_mtime < upstream_mtime:
            return False, "Lifecycle state older than upstream signals"

    recon = results_dir / "lifecycle_reconciliation.csv"
    if recon.is_file() and recon.stat().st_size > 0:
        return True, "Lifecycle artifacts present and reconciled"
    return True, "Lifecycle CSV present (reconciliation optional)"


def _check_signal_freshness(results_dir: Path) -> Tuple[bool, str]:
    snapshot_path = results_dir / "signals_snapshot.json"
    if snapshot_path.is_file():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list) and rows:
                latest = None
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    ts = _parse_iso(row.get("generated_at_utc") or row.get("generated_at"))
                    if ts and (latest is None or ts > latest):
                        latest = ts
                if latest:
                    age_h = (_utc_now() - latest).total_seconds() / 3600.0
                    if age_h <= SIGNALS_FRESHNESS_HOURS:
                        return True, f"Signals snapshot fresh ({age_h:.1f}h old)"
                    return False, f"Signals snapshot stale ({age_h:.1f}h old)"
        except (json.JSONDecodeError, OSError):
            pass

    signals_csv = results_dir / "signals.csv"
    age = _file_age_minutes(signals_csv)
    if age is not None and age <= SIGNALS_FRESHNESS_HOURS * 60:
        return True, "signals.csv within freshness window"
    if signals_csv.is_file():
        return False, "signals.csv exceeds freshness window"
    return False, "No fresh signal artifacts found"


def _derive_readiness_status(checks: Dict[str, bool]) -> str:
    operational = ("data_freshness", "watchdog", "broker", "lifecycle", "signals")
    authorization = ("governance", "approval", "policy")

    if not all(checks.get(k) for k in operational):
        return "NOT_READY"
    if all(checks.get(k) for k in authorization):
        return "READY"
    return "PARTIALLY_READY"


def _append_readiness_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "readiness_status",
        "checks_passing",
        "checks_total",
        "governance",
        "approval",
        "policy",
        "data_freshness",
        "watchdog",
        "broker",
        "lifecycle",
        "signals",
    ]
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def compute_execution_readiness(
    *,
    results_dir: Path,
    auth_doc: Dict[str, Any],
    queue_doc: Dict[str, Any],
    policy_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    watchdog_status: str,
    watchdog_ts: str,
    broker_connected: bool,
    broker_error: Optional[str],
    active_alerts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Phase 17: eligibility assessment for future protective actions."""
    ts = _iso_utc()
    results_dir = Path(results_dir)

    gov_ok = bool(auth_doc.get("governance_authorized"))
    approval_ok = (queue_doc.get("counts") or {}).get("APPROVED", 0) > 0
    policy_enabled_ok = bool(auth_doc.get("policy_authorized"))
    paper_exec_ok = bool(auth_doc.get("execution_authorized"))
    policy_ok = policy_enabled_ok and paper_exec_ok

    data_ok, data_reason = _check_data_freshness(
        watchdog_ts=watchdog_ts, cpi_doc=cpi_doc, results_dir=results_dir
    )
    wd_ok, wd_reason = _check_watchdog_health(watchdog_status, active_alerts)
    broker_ok, broker_reason = _check_broker_health(broker_connected, broker_error)
    lifecycle_ok, lifecycle_reason = _check_lifecycle_integrity(results_dir)
    signals_ok, signals_reason = _check_signal_freshness(results_dir)

    checks = {
        "governance": gov_ok,
        "approval": approval_ok,
        "policy": policy_ok,
        "data_freshness": data_ok,
        "watchdog": wd_ok,
        "broker": broker_ok,
        "lifecycle": lifecycle_ok,
        "signals": signals_ok,
    }
    check_details = {
        "governance": auth_doc.get("gate_reasons", {}).get("governance", ""),
        "approval": "Approved requests present" if approval_ok else "No approved requests",
        "policy": (
            auth_doc.get("gate_reasons", {}).get("policy", "")
            + (
                f"; {auth_doc.get('gate_reasons', {}).get('execution', '')}"
                if not paper_exec_ok
                else ""
            )
        ).strip("; "),
        "data_freshness": data_reason,
        "watchdog": wd_reason,
        "broker": broker_reason,
        "lifecycle": lifecycle_reason,
        "signals": signals_reason,
    }

    passing = [k for k, v in checks.items() if v]
    failing = [k for k, v in checks.items() if not v]
    status = _derive_readiness_status(checks)

    history_path = results_dir / "execution_readiness_history.csv"
    _append_readiness_history(
        history_path,
        {
            "timestamp": ts,
            "readiness_status": status,
            "checks_passing": len(passing),
            "checks_total": len(checks),
            **{k: checks[k] for k in checks},
        },
    )

    return {
        "generated_at": ts,
        "readiness_status": status,
        "checks": checks,
        "check_details": check_details,
        "passing_checks": passing,
        "failed_checks": failing,
        "checks_passing_count": len(passing),
        "checks_total": len(checks),
        "mode": "PAPER_ONLY" if paper_exec_ok else "SIMULATION_ONLY",
        "paper_execution_permitted": paper_exec_ok,
        "live_execution_permitted": False,
        "disclaimer": (
            "Readiness assessment only. READY requires paper execution authorization "
            "and all operational checks. Live execution remains blocked."
        ),
    }


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _sim_by_type(sim_doc: Dict[str, Any], sim_type: str) -> Optional[Dict[str, Any]]:
    for sim in sim_doc.get("simulations") or []:
        if isinstance(sim, dict) and sim.get("simulation_type") == sim_type:
            return sim
    return None


def compute_protective_action_trials(
    *,
    positions: List[Dict[str, Any]],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    sim_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 18: paper-mode protective action trials (simulation only)."""
    ts = _iso_utc()
    sim_doc = sim_doc or {}
    baseline_cps = int(cpi_doc.get("capital_preservation_score") or 0)
    baseline_conc = (
        _safe_float((sim_doc.get("baseline") or {}).get("largest_concentration_pct"), 0.0) or 0.0
    )

    conc_sim = _sim_by_type(sim_doc, "concentration_cap") or {}
    dd_sim = _sim_by_type(sim_doc, "drawdown_protection") or {}
    exp_sim = _sim_by_type(sim_doc, "exposure_reduction") or {}
    risk_off_sim = _sim_by_type(sim_doc, "risk_off_posture") or {}

    conc_risk = _safe_float(conc_sim.get("risk_score_delta"), 18.0) or 18.0
    dd_risk = _safe_float(dd_sim.get("risk_score_delta"), 8.0) or 8.0
    exp_risk = _safe_float(exp_sim.get("risk_score_delta"), 12.0) or 12.0
    risk_off_risk = _safe_float(risk_off_sim.get("risk_score_delta"), 15.0) or 15.0

    trials = [
        {
            "trial_id": "paper_concentration_reduction",
            "trial_name": "Paper Concentration Reduction",
            "trial_type": "concentration",
            "estimated_cps_improvement": round(min(20.0, conc_risk * 0.7), 1),
            "estimated_risk_reduction": round(conc_risk, 1),
            "estimated_concentration_reduction": round(
                abs(_safe_float(conc_sim.get("concentration_delta"), baseline_conc * 0.5) or 0.0),
                2,
            ),
            "estimated_drawdown_improvement": round(
                abs(_safe_float(conc_sim.get("max_drawdown_delta"), 5.0) or 5.0), 2
            ),
            "expected_benefits": [
                "Lower single-name concentration",
                "Improved capital preservation score",
            ],
            "expected_risks": [
                "Potential return drag from diversification",
                "Paper simulation only — no orders placed",
            ],
            "status": "SIMULATION_ONLY",
            "execution_permitted": False,
            "mode": "PAPER",
        },
        {
            "trial_id": "paper_drawdown_reduction",
            "trial_name": "Paper Drawdown Reduction",
            "trial_type": "drawdown",
            "estimated_cps_improvement": round(min(15.0, dd_risk * 0.8), 1),
            "estimated_risk_reduction": round(dd_risk, 1),
            "estimated_concentration_reduction": 0.0,
            "estimated_drawdown_improvement": round(
                abs(_safe_float(dd_sim.get("max_drawdown_delta"), 3.0) or 3.0), 2
            ),
            "expected_benefits": [
                "Review of positions exceeding drawdown threshold",
                "Reduced tail drawdown exposure",
            ],
            "expected_risks": [
                "May crystallize losses in review scenarios",
                "Paper simulation only — no positions modified",
            ],
            "status": "SIMULATION_ONLY",
            "execution_permitted": False,
            "mode": "PAPER",
        },
        {
            "trial_id": "paper_exposure_reduction",
            "trial_name": "Paper Exposure Reduction",
            "trial_type": "exposure",
            "estimated_cps_improvement": round(min(18.0, exp_risk * 0.65), 1),
            "estimated_risk_reduction": round(exp_risk, 1),
            "estimated_concentration_reduction": round(
                abs(_safe_float(exp_sim.get("concentration_delta"), 5.0) or 5.0), 2
            ),
            "estimated_drawdown_improvement": round(
                abs(_safe_float(exp_sim.get("max_drawdown_delta"), 4.0) or 4.0), 2
            ),
            "expected_benefits": [
                "Lower gross market exposure",
                "Reduced volatility under stress",
            ],
            "expected_risks": [
                "Reduced upside participation",
                "Paper simulation only — no rebalancing executed",
            ],
            "status": "SIMULATION_ONLY",
            "execution_permitted": False,
            "mode": "PAPER",
        },
        {
            "trial_id": "paper_risk_off_transition",
            "trial_name": "Paper Risk-Off Transition",
            "trial_type": "risk_off",
            "estimated_cps_improvement": round(min(22.0, max(8.0, (70 - baseline_cps) * 0.35)), 1),
            "estimated_risk_reduction": round(risk_off_risk, 1),
            "estimated_concentration_reduction": round(baseline_conc * 0.15, 2),
            "estimated_drawdown_improvement": round(
                abs(_safe_float(risk_off_sim.get("max_drawdown_delta"), 6.0) or 6.0), 2
            ),
            "expected_benefits": [
                "Defensive posture under RED escalation",
                "Improved resilience if risk-off triggers",
            ],
            "expected_risks": [
                "Opportunity cost in recovering markets",
                f"Current escalation: {cpe_doc.get('escalation_state', 'unknown')}",
            ],
            "status": "SIMULATION_ONLY",
            "execution_permitted": False,
            "mode": "PAPER",
        },
    ]

    return {
        "generated_at": ts,
        "baseline_cps": baseline_cps,
        "trial_count": len(trials),
        "trials": trials,
        "disclaimer": "Paper-mode trials only. No live orders, broker actions, or portfolio changes.",
    }


def persist_governance_execution_readiness(
    *,
    results_dir: Path,
    positions: List[Dict[str, Any]],
    active_alerts: List[Dict[str, Any]],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    candidates_doc: Dict[str, Any],
    queue_doc: Dict[str, Any],
    policy_doc: Dict[str, Any],
    sim_doc: Optional[Dict[str, Any]],
    watchdog_status: str,
    watchdog_ts: str,
    broker_connected: bool,
    broker_error: Optional[str],
) -> Dict[str, Any]:
    """Run phases 16–18 and write JSON artifacts."""
    results_dir = Path(results_dir)

    auth_doc = compute_governance_authorization(
        candidates_doc=candidates_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        gov_doc=gov_doc,
        cpe_doc=cpe_doc,
    )
    _atomic_write_json(auth_doc, results_dir / "governance_authorization.json")

    readiness_doc = compute_execution_readiness(
        results_dir=results_dir,
        auth_doc=auth_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        gov_doc=gov_doc,
        cpi_doc=cpi_doc,
        watchdog_status=watchdog_status,
        watchdog_ts=watchdog_ts,
        broker_connected=broker_connected,
        broker_error=broker_error,
        active_alerts=active_alerts,
    )
    _atomic_write_json(readiness_doc, results_dir / "execution_readiness.json")

    trials_doc = compute_protective_action_trials(
        positions=positions,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        sim_doc=sim_doc,
    )
    _atomic_write_json(trials_doc, results_dir / "protective_action_trials.json")

    return {
        "governance_authorization": auth_doc,
        "execution_readiness": readiness_doc,
        "protective_action_trials": trials_doc,
    }


def main() -> int:
    """CLI: recompute governance authorization + readiness from existing artifacts."""
    results_dir = ROOT / "data" / "results"
    policy_doc = _read_json(results_dir / "protective_action_policy.json")
    candidates_doc = _read_json(results_dir / "defensive_action_candidates.json")
    queue_doc = _read_json(results_dir / "human_approval_queue.json")
    gov_doc = _read_json(results_dir / "governance_risk_summary.json")
    cpe_doc = _read_json(results_dir / "capital_preservation_escalation.json")
    cpi_doc = _read_json(results_dir / "capital_preservation_intelligence.json")
    alerts_doc = _read_json(results_dir / "watchdog_alerts.json")
    status_doc = _read_json(results_dir / "watchdog_status.json")

    auth_doc = compute_governance_authorization(
        candidates_doc=candidates_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        gov_doc=gov_doc,
        cpe_doc=cpe_doc,
    )
    _atomic_write_json(auth_doc, results_dir / "governance_authorization.json")

    readiness_doc = compute_execution_readiness(
        results_dir=results_dir,
        auth_doc=auth_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        gov_doc=gov_doc,
        cpi_doc=cpi_doc,
        watchdog_status=str(status_doc.get("watchdog_status") or "OK"),
        watchdog_ts=str(status_doc.get("timestamp") or _iso_utc()),
        broker_connected=bool(status_doc.get("broker_connected")),
        broker_error=status_doc.get("broker_error"),
        active_alerts=list(alerts_doc.get("active_alerts") or []),
    )
    _atomic_write_json(readiness_doc, results_dir / "execution_readiness.json")
    print(json.dumps({"authorization": auth_doc, "readiness": readiness_doc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
