"""
TRITON Phase 173 — Governance Authorization Trace (diagnostic only).

Explains why governance_authorized may be false while execution_authorized is
true, tracing every dependency back to source rules and artifacts. Produces a
block-reason registry, dependency tree, and remediation recommendations.

SAFETY: traceability only — does not modify governance decisions, lifecycle,
execution, broker, portfolio logic, or permissions.

Outputs (data/results/):
    governance_authorization_trace.json
    governance_block_reasons.csv
    governance_dependency_tree.json

Run:
    python -m services.governance_authorization_trace
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_authorization_trace.json"
OUT_BLOCKS = RESULTS / "governance_block_reasons.csv"
OUT_TREE = RESULTS / "governance_dependency_tree.json"

GOVERNANCE_SOURCE = "services/governance_execution_readiness.py"
GOVERNANCE_RULE = "_evaluate_governance_gate"
PERMIT_ESCALATION = {"GREEN", "YELLOW"}
BLOCK_AWARENESS = {
    "MANAGEMENT_REVIEW_REQUIRED",
    "BOARD_REVIEW_REQUIRED",
    "CRITICAL_INTERVENTION",
}


def _log(msg: str) -> None:
    print(f"[governance_authorization_trace] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(*paths: Path) -> Tuple[Path, Dict[str, Any]]:
    for p in paths:
        if p.is_file():
            return p, _read_json(p)
    return paths[0], {}


def _load_inputs() -> Dict[str, Any]:
    """Load authorization artifacts with canonical fallbacks."""
    auth_path, auth = _first_existing(
        RESULTS / "governance_authorization.json",
        RESULTS / "execution_authorization.json",
    )
    ready_path, readiness = _first_existing(
        RESULTS / "execution_readiness.json",
        RESULTS / "readiness_assessment.json",
    )
    wd_path, watchdog = _first_existing(
        RESULTS / "watchdog_status.json",
        RESULTS / "risk_watchdog.json",
    )
    cp_path, cp_eval = _first_existing(
        RESULTS / "capital_preservation_certification.json",
        RESULTS / "capital_preservation_evaluation.json",
        RESULTS / "capital_preservation_governor.json",
    )
    cpe_path, cpe = _first_existing(RESULTS / "capital_preservation_escalation.json")
    gov_path, gov_risk = _first_existing(RESULTS / "governance_risk_summary.json")
    policy_path, policy = _first_existing(
        RESULTS / "protective_action_policy.json",
        RESULTS / "policy_registry.json",
        RESULTS / "runtime_policy.json",
    )
    alloc_path, alloc = _first_existing(RESULTS / "portfolio_allocation_summary.json")
    gate_path, gate = _first_existing(RESULTS / "paper_submission_gate_audit.json")

    return {
        "paths": {
            "governance_authorization": str(auth_path),
            "execution_readiness": str(ready_path),
            "watchdog": str(wd_path),
            "capital_preservation": str(cp_path),
            "capital_preservation_escalation": str(cpe_path),
            "governance_risk_summary": str(gov_path),
            "protective_action_policy": str(policy_path),
            "portfolio_allocation_summary": str(alloc_path),
            "paper_submission_gate_audit": str(gate_path),
        },
        "governance_authorization": auth,
        "execution_readiness": readiness,
        "watchdog": watchdog,
        "capital_preservation": cp_eval,
        "capital_preservation_escalation": cpe,
        "governance_risk_summary": gov_risk,
        "protective_action_policy": policy,
        "portfolio_allocation_summary": alloc,
        "paper_submission_gate_audit": gate,
        "positions_snapshot": _read_csv(RESULTS / "positions_snapshot.csv"),
        "performance_intelligence": _read_csv(RESULTS / "performance_intelligence.csv"),
        "allocation_recommendations": _read_csv(
            RESULTS / "portfolio_allocation_recommendations.csv"
        ),
        "accountability_registry": _read_json(RESULTS / "accountability_registry.json"),
        "governance_policy_summary": _read_json(RESULTS / "governance_policy_summary.json"),
    }


def _trace_governance_gate(ctx: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["governance_authorization"]
    cpe = ctx["capital_preservation_escalation"]
    gov = ctx["governance_risk_summary"]
    cp = ctx["capital_preservation"]

    escalation = str(
        cpe.get("escalation_state")
        or cp.get("escalation_state")
        or gov.get("escalation_state")
        or "UNKNOWN"
    )
    awareness = str(
        gov.get("governance_awareness_level") or cp.get("governance_awareness_level") or ""
    )
    authorized = bool(auth.get("governance_authorized"))
    reason = str((auth.get("gate_reasons") or {}).get("governance") or "")

    checks: List[Dict[str, Any]] = []
    blocking: List[str] = []

    if awareness in BLOCK_AWARENESS:
        checks.append(
            {
                "check": "governance_awareness_level",
                "passed": False,
                "current_value": awareness,
                "required_value": f"not in {sorted(BLOCK_AWARENESS)}",
                "source_file": ctx["paths"]["governance_risk_summary"],
                "source_module": GOVERNANCE_SOURCE,
                "source_rule": GOVERNANCE_RULE,
            }
        )
        blocking.append(f"Governance awareness blocks action ({awareness})")

    permit_ok = escalation in PERMIT_ESCALATION
    checks.append(
        {
            "check": "escalation_state_permit",
            "passed": permit_ok,
            "current_value": escalation,
            "required_value": f"one of {sorted(PERMIT_ESCALATION)}",
            "source_file": ctx["paths"]["capital_preservation_escalation"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": GOVERNANCE_RULE,
            "rule_detail": "requires_green_governance defaults true; ORANGE/RED/CRITICAL block",
        }
    )
    if not permit_ok:
        blocking.append(f"Escalation {escalation} does not meet green-governance requirement")

    if escalation in ("RED", "CRITICAL"):
        checks.append(
            {
                "check": "escalation_red_critical",
                "passed": False,
                "current_value": escalation,
                "required_value": "not RED or CRITICAL",
                "source_file": ctx["paths"]["capital_preservation_escalation"],
                "source_module": GOVERNANCE_SOURCE,
                "source_rule": GOVERNANCE_RULE,
            }
        )

    return {
        "governance_authorized": authorized,
        "reason": reason
        or ("; ".join(blocking) if blocking else "Governance posture permits evaluation"),
        "escalation_state": escalation,
        "governance_awareness_level": awareness,
        "checks": checks,
        "blocking_conditions": blocking,
    }


def _trace_execution_gate(ctx: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["governance_authorization"]
    policy = ctx["protective_action_policy"]
    authorized = bool(auth.get("execution_authorized"))
    reason = str((auth.get("gate_reasons") or {}).get("execution") or "")

    checks = [
        {
            "check": "paper_execution_enabled",
            "passed": policy.get("paper_execution_enabled") is True,
            "current_value": policy.get("paper_execution_enabled"),
            "required_value": True,
            "source_file": ctx["paths"]["protective_action_policy"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_evaluate_execution_gate",
        },
        {
            "check": "global_execution_enabled",
            "passed": policy.get("global_execution_enabled") is True,
            "current_value": policy.get("global_execution_enabled"),
            "required_value": True,
            "source_file": ctx["paths"]["protective_action_policy"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_evaluate_execution_gate",
        },
        {
            "check": "live_execution_enabled",
            "passed": policy.get("live_execution_enabled") is False,
            "current_value": policy.get("live_execution_enabled"),
            "required_value": False,
            "source_file": ctx["paths"]["protective_action_policy"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_evaluate_execution_gate",
        },
        {
            "check": "enabled_policy_count",
            "passed": int(policy.get("enabled_policy_count") or 0) > 0,
            "current_value": policy.get("enabled_policy_count"),
            "required_value": "> 0",
            "source_file": ctx["paths"]["protective_action_policy"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_evaluate_execution_gate",
        },
    ]

    return {
        "execution_authorized": authorized,
        "reason": reason,
        "checks": checks,
        "paper_execution_permitted": bool(auth.get("paper_execution_permitted")),
        "live_execution_permitted": bool(auth.get("live_execution_permitted")),
    }


def _trace_readiness(ctx: Dict[str, Any]) -> Dict[str, Any]:
    readiness = ctx["execution_readiness"]
    status = str(readiness.get("readiness_status") or "UNKNOWN")
    checks = readiness.get("checks") or {}
    details = readiness.get("check_details") or {}
    failed = list(readiness.get("failed_checks") or [])

    trace_rows: List[Dict[str, Any]] = []
    owners = {
        "governance": "Capital Preservation / Governance",
        "approval": "Human Approval Queue",
        "policy": "Protective Action Policy",
        "data_freshness": "Risk Watchdog",
        "watchdog": "Risk Watchdog",
        "broker": "Broker Connectivity",
        "lifecycle": "Signal Lifecycle",
        "signals": "Signals Pipeline",
    }
    for name in failed:
        trace_rows.append(
            {
                "failed_check": name,
                "source": details.get(name, "—"),
                "severity": (
                    "BLOCKING" if name in ("governance", "approval", "policy") else "WARNING"
                ),
                "owner": owners.get(name, "Operations"),
                "remediation": _remediation_for_check(name, ctx),
                "impact": (
                    "Governance Authorization Block"
                    if name == "governance"
                    else "Readiness PARTIALLY_READY"
                ),
            }
        )

    return {
        "readiness_status": status,
        "checks_passing_count": readiness.get("checks_passing_count"),
        "checks_total": readiness.get("checks_total"),
        "failed_checks": failed,
        "trace_rows": trace_rows,
    }


def _remediation_for_check(check: str, ctx: Dict[str, Any]) -> str:
    cpe = ctx["capital_preservation_escalation"]
    alloc = ctx["portfolio_allocation_summary"]
    if check == "governance":
        labels = cpe.get("escalation_reason_labels") or []
        if labels:
            return f"Resolve escalation drivers: {', '.join(str(x) for x in labels)}"
        return "Reduce escalation state to YELLOW or GREEN"
    if check == "approval":
        return "Obtain operator approval in human_approval_queue.json"
    if check == "policy":
        return "Enable at least one protective policy in protective_action_policy.json"
    if check == "data_freshness":
        return "Refresh watchdog and capital preservation artifacts"
    if check == "watchdog":
        return "Resolve active watchdog alerts and restore heartbeat"
    if check == "broker":
        return "Restore broker connectivity (read-only check)"
    if check == "lifecycle":
        return "Regenerate signal_lifecycle.csv and reconcile lifecycle state"
    if check == "signals":
        return "Refresh signals.csv / signals_snapshot.json"
    if "concentration" in str(alloc.get("largest_position_weight_pct", "")):
        return "Diversify portfolio to reduce concentration below policy targets"
    return "Review failed check artifact and re-run risk_watchdog"


def _build_block_reasons(ctx: Dict[str, Any], gov_trace: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    auth = ctx["governance_authorization"]
    readiness = ctx["execution_readiness"]
    cpe = ctx["capital_preservation_escalation"]
    alloc = ctx["portfolio_allocation_summary"]
    gate = ctx["paper_submission_gate_audit"]

    # Primary governance gate block
    if not auth.get("governance_authorized"):
        rows.append(
            {
                "rule_name": "green_governance_escalation_requirement",
                "severity": "BLOCKING",
                "source_module": GOVERNANCE_SOURCE,
                "source_file": GOVERNANCE_SOURCE,
                "current_value": gov_trace.get("escalation_state"),
                "required_value": f"one of {sorted(PERMIT_ESCALATION)}",
                "failure_reason": gov_trace.get("reason"),
                "recommended_action": _remediation_for_check("governance", ctx),
            }
        )

    # Escalation drivers
    for label in cpe.get("escalation_reason_labels") or []:
        sev = "CRITICAL" if "CRITICAL" in str(label).upper() else "WARNING"
        rows.append(
            {
                "rule_name": f"escalation_driver_{str(label).lower().replace(' ', '_')}",
                "severity": sev,
                "source_module": "services/capital_preservation_escalation.py",
                "source_file": ctx["paths"]["capital_preservation_escalation"],
                "current_value": label,
                "required_value": "resolved",
                "failure_reason": f"Active escalation driver: {label}",
                "recommended_action": (
                    cpe.get("recommended_review_actions", ["Review escalation"])[0]
                    if cpe.get("recommended_review_actions")
                    else "Review escalation drivers"
                ),
            }
        )

    # Concentration
    conc = alloc.get("largest_position_weight_pct")
    if conc is not None:
        try:
            conc_f = float(conc)
            if conc_f >= 35.0:
                rows.append(
                    {
                        "rule_name": "portfolio_concentration_limit",
                        "severity": "CRITICAL" if conc_f >= 80 else "WARNING",
                        "source_module": "services/portfolio_allocation_engine.py",
                        "source_file": ctx["paths"]["portfolio_allocation_summary"],
                        "current_value": f"{conc_f:.1f}%",
                        "required_value": "<35%",
                        "failure_reason": f"Largest position weight {conc_f:.1f}% exceeds diversification target",
                        "recommended_action": "Diversify portfolio — reduce single-name concentration",
                    }
                )
        except (TypeError, ValueError):
            pass

    # Readiness failed checks
    for check in readiness.get("failed_checks") or []:
        if check == "governance":
            continue  # already captured
        detail = (readiness.get("check_details") or {}).get(check, "")
        rows.append(
            {
                "rule_name": f"readiness_check_{check}",
                "severity": "BLOCKING" if check in ("approval", "policy") else "WARNING",
                "source_module": GOVERNANCE_SOURCE,
                "source_file": ctx["paths"]["execution_readiness"],
                "current_value": detail,
                "required_value": "passing",
                "failure_reason": detail,
                "recommended_action": _remediation_for_check(check, ctx),
            }
        )

    # Paper submission gate blocks
    pg = gate.get("paper_execution_gate") or {}
    for br in pg.get("block_reasons") or []:
        if br in ("governance_authorized_false", "readiness_status_PARTIALLY_READY"):
            rows.append(
                {
                    "rule_name": f"paper_gate_{br}",
                    "severity": "BLOCKING",
                    "source_module": "services/paper_submission_gate_audit.py",
                    "source_file": ctx["paths"]["paper_submission_gate_audit"],
                    "current_value": br,
                    "required_value": "not present",
                    "failure_reason": f"Paper submission gate blocked: {br}",
                    "recommended_action": "Clear governance and readiness blocks before paper submit",
                }
            )

    # Candidate-level policy blocks (informational)
    for cand in auth.get("candidate_authorizations") or []:
        if not cand.get("policy_authorized"):
            action = cand.get("candidate_action")
            rows.append(
                {
                    "rule_name": f"protective_policy_disabled_{action}",
                    "severity": "INFO",
                    "source_module": GOVERNANCE_SOURCE,
                    "source_file": ctx["paths"]["protective_action_policy"],
                    "current_value": "disabled",
                    "required_value": "enabled",
                    "failure_reason": (cand.get("gate_reasons") or {}).get("policy", ""),
                    "recommended_action": f"Enable protective policy for {action} if trial is desired",
                }
            )

    if not rows:
        rows.append(
            {
                "rule_name": "no_blocks_detected",
                "severity": "INFO",
                "source_module": "governance_authorization_trace",
                "source_file": str(OUT_JSON),
                "current_value": "authorized",
                "required_value": "—",
                "failure_reason": "No blocking conditions found in trace",
                "recommended_action": "None — authorization gates appear open",
            }
        )

    return pd.DataFrame(rows)


def _build_dependency_tree(ctx: Dict[str, Any], traces: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["governance_authorization"]
    readiness = ctx["execution_readiness"]
    cpe = ctx["capital_preservation_escalation"]
    cp = ctx["capital_preservation"]

    return {
        "OVERALL_AUTHORIZATION": {
            "value": bool(auth.get("overall_authorization")),
            "children": {
                "Execution": {
                    "value": bool(auth.get("execution_authorized")),
                    "reason": traces["execution"]["reason"],
                    "source": GOVERNANCE_SOURCE,
                    "children": traces["execution"]["checks"],
                },
                "Operator": {
                    "value": bool(auth.get("operator_authorized")),
                    "reason": (auth.get("gate_reasons") or {}).get("operator"),
                    "source": GOVERNANCE_SOURCE,
                },
                "Policy": {
                    "value": bool(auth.get("policy_authorized")),
                    "reason": (auth.get("gate_reasons") or {}).get("policy"),
                    "source": ctx["paths"]["protective_action_policy"],
                },
                "Governance": {
                    "value": bool(auth.get("governance_authorized")),
                    "reason": traces["governance"]["reason"],
                    "source": GOVERNANCE_SOURCE,
                    "children": {
                        "Readiness": {
                            "value": str(readiness.get("readiness_status")),
                            "failed_checks": readiness.get("failed_checks"),
                            "source": ctx["paths"]["execution_readiness"],
                        },
                        "Risk": {
                            "value": ctx["governance_risk_summary"].get("governance_status"),
                            "escalation_state": traces["governance"]["escalation_state"],
                            "awareness_level": traces["governance"]["governance_awareness_level"],
                            "source": ctx["paths"]["governance_risk_summary"],
                        },
                        "Capital Preservation": {
                            "value": cp.get("preservation_status")
                            or cp.get("certification_status"),
                            "score": cpe.get("capital_preservation_score"),
                            "health_band": cpe.get("health_band"),
                            "source": ctx["paths"]["capital_preservation"],
                        },
                        "Escalation State": {
                            "value": cpe.get("escalation_state"),
                            "reasons": cpe.get("escalation_reason_labels"),
                            "source": ctx["paths"]["capital_preservation_escalation"],
                        },
                        "Portfolio Health": {
                            "value": cp.get("portfolio_health"),
                            "largest_weight_pct": ctx["portfolio_allocation_summary"].get(
                                "largest_position_weight_pct"
                            ),
                            "source": ctx["paths"]["portfolio_allocation_summary"],
                        },
                    },
                },
            },
        }
    }


def _build_authorization_trace(ctx: Dict[str, Any], traces: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["governance_authorization"]
    gate = ctx["paper_submission_gate_audit"]
    pg = gate.get("paper_execution_gate") or {}

    return {
        "execution_authorized": {
            "value": bool(auth.get("execution_authorized")),
            "why": traces["execution"]["reason"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_evaluate_execution_gate",
        },
        "operator_authorized": {
            "value": bool(auth.get("operator_authorized")),
            "why": (auth.get("gate_reasons") or {}).get("operator"),
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_approved_actions / operator gate",
        },
        "policy_authorized": {
            "value": bool(auth.get("policy_authorized")),
            "why": (auth.get("gate_reasons") or {}).get("policy"),
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "_enabled_policy_count",
        },
        "governance_authorized": {
            "value": bool(auth.get("governance_authorized")),
            "why": traces["governance"]["reason"],
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": GOVERNANCE_RULE,
            "dependency_chain": [
                "governance_authorized",
                "escalation_state",
                "GOVERNANCE_PERMIT_ESCALATION rule (GREEN/YELLOW only)",
                GOVERNANCE_SOURCE,
            ],
        },
        "overall_authorization": {
            "value": bool(auth.get("overall_authorization")),
            "why": (
                "All four gates must pass (governance AND operator AND policy AND execution)"
                if not auth.get("overall_authorization")
                else "All authorization gates passed"
            ),
            "source_module": GOVERNANCE_SOURCE,
            "source_rule": "compute_governance_authorization",
        },
        "paper_final_submit_allowed": {
            "value": bool(pg.get("final_submit_allowed")),
            "why": f"Block reasons: {pg.get('block_reasons', [])}",
            "source_module": "services/paper_submission_gate_audit.py",
        },
        "mismatch_explanation": (
            "execution_authorized=true because the paper execution policy gate passes "
            "(paper_execution_enabled, global_execution_enabled, live disabled, policies enabled). "
            "governance_authorized=false because escalation_state ORANGE is outside "
            "GOVERNANCE_PERMIT_ESCALATION (GREEN/YELLOW only) per _evaluate_governance_gate."
            if auth.get("execution_authorized") and not auth.get("governance_authorized")
            else None
        ),
    }


def _build_executive(
    ctx: Dict[str, Any],
    blocks: pd.DataFrame,
    readiness_trace: Dict[str, Any],
    traces: Dict[str, Any],
) -> Dict[str, Any]:
    auth = ctx["governance_authorization"]
    sev_order = {"CRITICAL": 0, "BLOCKING": 1, "WARNING": 2, "INFO": 3}
    blocking = blocks[blocks["severity"].isin(["CRITICAL", "BLOCKING", "WARNING"])]
    primary = blocking.iloc[0] if not blocking.empty else None
    secondary = blocking.iloc[1:4].to_dict(orient="records") if len(blocking) > 1 else []

    highest = "—"
    if not blocks.empty:
        sorted_blocks = blocks.copy()
        sorted_blocks["_rank"] = sorted_blocks["severity"].map(lambda s: sev_order.get(str(s), 9))
        top = sorted_blocks.sort_values("_rank").iloc[0]
        highest = f"{top['severity']}: {top['rule_name']}"

    return {
        "current_authorization_state": {
            "governance_authorized": auth.get("governance_authorized"),
            "execution_authorized": auth.get("execution_authorized"),
            "overall_authorization": auth.get("overall_authorization"),
            "readiness_status": readiness_trace.get("readiness_status"),
        },
        "blocking_rules_count": int(len(blocking)),
        "highest_severity_block": highest,
        "primary_cause": primary["failure_reason"] if primary is not None else "—",
        "secondary_causes": [r.get("failure_reason") for r in secondary],
        "readiness_status": readiness_trace.get("readiness_status"),
        "risk_status": ctx["governance_risk_summary"].get("governance_status", "—"),
        "capital_preservation_status": (
            ctx["capital_preservation"].get("preservation_status")
            or ctx["capital_preservation_escalation"].get("health_band")
            or "—"
        ),
        "recommended_next_action": (
            primary["recommended_action"] if primary is not None else "No blocking actions required"
        ),
        "can_execution_proceed": bool(auth.get("execution_authorized")),
        "can_paper_submit": bool(
            (ctx["paper_submission_gate_audit"].get("paper_execution_gate") or {}).get(
                "final_submit_allowed"
            )
        ),
        "what_prevents_authorization": traces["governance"]["reason"],
        "fastest_path_to_ready": _fastest_path(ctx, readiness_trace),
    }


def _fastest_path(ctx: Dict[str, Any], readiness_trace: Dict[str, Any]) -> str:
    cpe = ctx["capital_preservation_escalation"]
    alloc = ctx["portfolio_allocation_summary"]
    labels = cpe.get("escalation_reason_labels") or []
    conc = alloc.get("largest_position_weight_pct")
    steps: List[str] = []
    if labels:
        steps.append(f"Resolve escalation drivers: {', '.join(str(x) for x in labels)}")
    if conc is not None:
        try:
            if float(conc) >= 35:
                steps.append(f"Reduce concentration from {float(conc):.1f}% toward <35%")
        except (TypeError, ValueError):
            pass
    steps.append("Re-run risk_watchdog to refresh escalation_state to YELLOW or GREEN")
    steps.append("Verify governance_authorized=true and readiness_status=READY")
    return " → ".join(steps)


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ctx = _load_inputs()

    gov_trace = _trace_governance_gate(ctx)
    exec_trace = _trace_execution_gate(ctx)
    readiness_trace = _trace_readiness(ctx)
    traces = {"governance": gov_trace, "execution": exec_trace, "readiness": readiness_trace}

    blocks = _build_block_reasons(ctx, gov_trace)
    blocks.to_csv(OUT_BLOCKS, index=False, encoding="utf-8")

    tree = _build_dependency_tree(ctx, traces)
    OUT_TREE.write_text(json.dumps(tree, indent=2, default=str), encoding="utf-8")

    auth_trace = _build_authorization_trace(ctx, traces)
    executive = _build_executive(ctx, blocks, readiness_trace, traces)

    report = {
        "generated_at": _now_iso(),
        "authorization_trace": auth_trace,
        "executive_report": executive,
        "readiness_trace": readiness_trace,
        "governance_trace": gov_trace,
        "execution_trace": exec_trace,
        "risk_trace": {
            "governance_status": ctx["governance_risk_summary"].get("governance_status"),
            "escalation_state": gov_trace.get("escalation_state"),
            "escalation_reasons": ctx["capital_preservation_escalation"].get(
                "escalation_reason_labels"
            ),
            "governance_drivers": ctx["governance_risk_summary"].get("governance_drivers"),
            "source": ctx["paths"]["governance_risk_summary"],
        },
        "remediation_actions": blocks[["rule_name", "severity", "recommended_action"]].to_dict(
            orient="records"
        ),
        "input_artifacts": ctx["paths"],
        "disclaimer": (
            "Diagnostic trace only — no governance decisions, lifecycle, execution, "
            "broker, portfolio logic, or permissions were modified."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_BLOCKS.name}, {OUT_TREE.name}")
    _log(f"governance_authorized={ctx['governance_authorization'].get('governance_authorized')}")
    _log(f"execution_authorized={ctx['governance_authorization'].get('execution_authorized')}")
    _log(f"primary cause: {executive.get('primary_cause')}")
    _log(f"blocks: {len(blocks)}")
    return report


if __name__ == "__main__":
    run()
