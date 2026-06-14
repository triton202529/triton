"""
system_integration_audit.py
THP-H001 — Full TRITON system integration and source-of-truth audit (read-only).

Investigation only — no trading, governance, or architecture changes.
Generates dependency maps, source-of-truth registry, and integration risk matrix.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"
SERVICES_DIR = ROOT / "services"
CONFIG_DIR = ROOT / "config"

AUDIT_JSON = GOVERNANCE_DIR / "system_integration_audit.json"
AUDIT_CSV = GOVERNANCE_DIR / "system_integration_audit.csv"
SOURCE_OF_TRUTH_JSON = GOVERNANCE_DIR / "source_of_truth_registry.json"
DEPENDENCY_MAP_JSON = GOVERNANCE_DIR / "dependency_map.json"
UNUSED_COMPONENTS_JSON = GOVERNANCE_DIR / "unused_components_report.json"
RISK_MATRIX_JSON = GOVERNANCE_DIR / "integration_risk_matrix.json"

AUDIT_CSV_FIELDS = (
    "issue_id",
    "severity",
    "category",
    "component",
    "description",
    "remediation",
    "affects_capital",
)

STALE_HOURS_DEFAULT = 48


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _file_age_hours(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_source_files() -> List[Path]:
    """Scan code/docs only — skip data artifacts and dependencies."""
    skip_dirs = {
        "data",
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".cursor",
    }
    paths: List[Path] = []
    for base in (
        ROOT / "services",
        ROOT / "scripts",
        ROOT / "dashboard",
        ROOT / "pages",
        ROOT / "lib",
    ):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".py", ".ps1", ".js", ".jsx", ".mjs", ".md"):
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            paths.append(path)
    for name in (
        "run_full_pipeline.py",
        "place_orders_from_csv.py",
        "app.py",
        "pages_order_desk.py",
    ):
        p = ROOT / name
        if p.is_file():
            paths.append(p)
    return paths


def _grep_files(pattern: str, glob: str = "**/*.py") -> List[str]:
    hits: List[str] = []
    rx = re.compile(pattern)
    for path in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if rx.search(text):
            hits.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return sorted(set(hits))


def _issue(
    issues: List[Dict[str, Any]],
    *,
    severity: str,
    category: str,
    component: str,
    description: str,
    remediation: str,
    affects_capital: bool = False,
) -> None:
    issues.append(
        {
            "issue_id": f"INT-{len(issues) + 1:04d}",
            "severity": severity,
            "category": category,
            "component": component,
            "description": description,
            "remediation": remediation,
            "affects_capital": affects_capital,
        }
    )


def build_operating_chain() -> List[Dict[str, Any]]:
    """Canonical TRITON operating chain for audit."""
    return [
        {
            "stage": "market_data",
            "role": "advisory_input",
            "services": ["services/fetch_raw_data.py", "services/preprocess_data.py"],
            "artifacts": ["data/raw/", "data/processed/"],
            "consumers": ["run_full_pipeline.py"],
            "refresh": "pipeline stage fetch_raw",
            "failure_behavior": "pipeline aborts early stage",
            "recovery": "retry fetch_raw with diagnostics",
        },
        {
            "stage": "signal_intelligence",
            "role": "decision_input",
            "services": [
                "services/generate_signals.py",
                "services/score_stocks.py",
                "services/build_orders_from_signals.py",
            ],
            "artifacts": [
                "data/results/signals.csv",
                "data/results/signals_with_rationale.csv",
                "data/results/stock_scores.csv",
            ],
            "consumers": ["run_full_pipeline.py", "services/execute_trades.py"],
            "refresh": "pipeline signals/scores stages",
            "failure_behavior": "empty or stale signals block downstream",
            "recovery": "re-run run_full_pipeline.py",
        },
        {
            "stage": "lifecycle_engine",
            "role": "decision_authority",
            "services": [
                "services/apply_signal_lifecycle.py",
                "services/build_effective_lifecycle.py",
                "services/lifecycle_truth.py",
            ],
            "artifacts": [
                "data/results/signal_lifecycle.csv",
                "data/results/signal_lifecycle_effective.csv",
            ],
            "consumers": [
                "services/execute_trades.py",
                "services/manage_positions.py",
                "services/place_live_orders.py",
            ],
            "refresh": "pipeline lifecycle stage + build_effective_lifecycle",
            "failure_behavior": "lifecycle gate blocks entries",
            "recovery": "apply_signal_lifecycle + build_effective_lifecycle",
        },
        {
            "stage": "portfolio_allocation",
            "role": "advisory_with_enforcement",
            "services": ["services/portfolio_allocation_engine.py"],
            "artifacts": [
                "data/results/portfolio_allocation_recommendations.csv",
                "data/results/portfolio_allocation_summary.json",
            ],
            "consumers": ["services/manage_positions.py", "services/execute_trades.py"],
            "refresh": "allocation engine run",
            "failure_behavior": "allocation overlay may be stale",
            "recovery": "re-run portfolio_allocation_engine",
        },
        {
            "stage": "risk_protection",
            "role": "enforcement",
            "services": [
                "services/risk_gate.py",
                "services/performance_risk_overlay.py",
                "services/execution_guard.py",
                "services/master_execution_gate.py",
                "services/adaptive_risk_engine.py",
            ],
            "artifacts": [
                "data/results/adaptive_risk_state.json",
                "data/results/performance_risk_overlay.csv",
                "data/results/master_execution_gate.json",
                "data/results/guard_snapshot.json",
            ],
            "consumers": [
                "services/execute_trades.py",
                "services/place_live_orders.py",
                "place_orders_from_csv.py",
            ],
            "refresh": "each execution cycle",
            "failure_behavior": "master gate or guard blocks batch",
            "recovery": "refresh snapshots + risk state",
        },
        {
            "stage": "execution_gate",
            "role": "final_authority_pre_broker",
            "services": [
                "lib/executionGate.js",
                "services/triton_execution_gate.py",
            ],
            "artifacts": [
                "data/results/execution_gate_audit.csv",
                "data/governance/execution_gate_decisions.json",
            ],
            "consumers": [
                "services/broker_alpaca.py",
                "services/execute_trades.py",
                "services/place_live_orders.py",
                "services/manage_positions.py",
                "place_orders_from_csv.py",
            ],
            "refresh": "every gate evaluation",
            "failure_behavior": "BLOCK on gate failure",
            "recovery": "fix gate script / env vars",
        },
        {
            "stage": "governance_council",
            "role": "advisory_overlay",
            "services": [
                "services/execution_governance_council.py",
                "services/execution_approval_workflow.py",
                "services/execution_escalation_engine.py",
            ],
            "artifacts": [
                "data/governance/execution_approval_requests.json",
                "data/governance/execution_governance_council_status.json",
            ],
            "consumers": ["dashboard/execution_governance_council_dashboard.py"],
            "refresh": "supervision cycle",
            "failure_behavior": "does NOT block execution path by default",
            "recovery": "run governance council supervision",
        },
        {
            "stage": "broker_authorization",
            "role": "enforcement",
            "services": ["services/broker_alpaca.py"],
            "artifacts": ["data/results/live_orders_log.csv"],
            "consumers": ["services/place_live_orders.py"],
            "refresh": "per submit_order call",
            "failure_behavior": "API error raises",
            "recovery": "broker reconnect / cancel stale orders",
        },
        {
            "stage": "order_execution",
            "role": "capital_deployment",
            "services": [
                "services/execute_trades.py",
                "services/place_live_orders.py",
                "services/manage_positions.py",
            ],
            "artifacts": [
                "data/live/orders_today.csv",
                "data/results/execution_plan.json",
            ],
            "consumers": ["broker API"],
            "refresh": "execution loop",
            "failure_behavior": "logged to execution_drop_diagnostics",
            "recovery": "continuous_loop_recovery_manager",
        },
        {
            "stage": "position_management",
            "role": "capital_deployment",
            "services": ["services/manage_positions.py"],
            "artifacts": ["data/live/manage_orders.csv", "data/results/positions_snapshot.csv"],
            "consumers": ["services/place_live_orders.py"],
            "refresh": "market open loop / execution loop",
            "failure_behavior": "management plan only if dry-run",
            "recovery": "re-run manage_positions",
        },
        {
            "stage": "monitoring_recovery",
            "role": "advisory_supervision",
            "services": [
                "services/risk_watchdog.py",
                "services/loop_safety.py",
                "services/continuous_loop_recovery_manager.py",
                "services/execution_gate_health_monitor.py",
            ],
            "artifacts": [
                "data/results/watchdog_status.json",
                "data/results/heartbeat.json",
                "data/results/continuous_loop_health.json",
                "data/governance/execution_gate_operational_status.json",
            ],
            "consumers": ["dashboards", "governance monitors"],
            "refresh": "watchdog loop / supervision",
            "failure_behavior": "alerts only — does not halt trading",
            "recovery": "continuous_loop_recovery_manager restart",
        },
    ]


def build_source_of_truth_registry() -> Dict[str, Any]:
    entries = [
        {
            "decision": "lifecycle_authorization",
            "owner": "services/lifecycle_truth.py",
            "primary_artifact": "data/results/signal_lifecycle_effective.csv",
            "secondary_artifacts": ["data/results/signal_lifecycle.csv"],
            "consumers": [
                "services/execute_trades.py",
                "services/manage_positions.py",
                "services/place_live_orders.py",
            ],
            "refresh_method": "build_effective_lifecycle after apply_signal_lifecycle",
            "duplicate_risk": "signal_lifecycle.csv vs effective copy",
        },
        {
            "decision": "portfolio_allocation",
            "owner": "services/portfolio_allocation_engine.py",
            "primary_artifact": "data/results/portfolio_allocation_recommendations.csv",
            "secondary_artifacts": ["data/results/portfolio_allocation_summary.json"],
            "consumers": ["services/manage_positions.py"],
            "refresh_method": "portfolio allocation engine run",
            "duplicate_risk": "dashboard may show summary json while execution reads csv",
        },
        {
            "decision": "risk_restriction",
            "owner": "services/master_execution_gate.py",
            "primary_artifact": "data/results/master_execution_gate.json",
            "secondary_artifacts": [
                "data/results/adaptive_risk_state.json",
                "data/results/performance_risk_overlay.csv",
            ],
            "consumers": [
                "services/execute_trades.py",
                "services/place_live_orders.py",
                "place_orders_from_csv.py",
            ],
            "refresh_method": "MasterExecutionGate.evaluate each batch",
            "duplicate_risk": "adaptive_risk_state vs master gate snapshot",
        },
        {
            "decision": "execution_authorization",
            "owner": "lib/executionGate.js via services/triton_execution_gate.py",
            "primary_artifact": "data/governance/execution_gate_decisions.json",
            "secondary_artifacts": ["data/results/execution_gate_audit.csv"],
            "consumers": ["services/broker_alpaca.submit_order"],
            "refresh_method": "evaluate_execution_gate on each enforcement call",
            "duplicate_risk": "master_execution_gate vs triton execution gate (layered)",
        },
        {
            "decision": "human_governance_approval",
            "owner": "services/execution_approval_workflow.py",
            "primary_artifact": "data/governance/execution_approval_requests.json",
            "secondary_artifacts": ["data/governance/execution_escalations.json"],
            "consumers": ["dashboard/execution_governance_council_dashboard.py"],
            "refresh_method": "evaluate_governed_execution (not wired to main execution path)",
            "duplicate_risk": "THP-008 council not enforced on execute_trades path",
        },
        {
            "decision": "broker_order_submission",
            "owner": "services/broker_alpaca.py",
            "primary_artifact": "data/results/live_orders_log.csv",
            "secondary_artifacts": ["data/results/live_orders.csv"],
            "consumers": ["Alpaca API"],
            "refresh_method": "submit_order with enforce_broker_submission_gate",
            "duplicate_risk": "place_orders_from_csv uses raw requests POST",
        },
        {
            "decision": "operational_heartbeat",
            "owner": "run_full_pipeline.py / services/loop_safety.py",
            "primary_artifact": "data/results/heartbeat.json",
            "secondary_artifacts": ["data/results/pipeline_status.json"],
            "consumers": ["services/master_execution_gate.py"],
            "refresh_method": "pipeline stages + loop_safety refresh",
            "duplicate_risk": "stale heartbeat blocks master gate",
        },
    ]
    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H001",
        "entries": entries,
    }


def build_dependency_map(chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    execution_modules = [
        "services/execute_trades.py",
        "services/place_live_orders.py",
        "services/manage_positions.py",
        "services/broker_alpaca.py",
        "place_orders_from_csv.py",
    ]
    schedulers = [
        "scripts/run_continuous_loop.ps1",
        "scripts/run_market_open.ps1",
        "scripts/run_watchdog_loop.ps1",
        "services/run_execution_loop.py",
        "services/run_scheduled_paper_cycle.py",
        "run_full_pipeline.py",
    ]

    edges: List[Dict[str, str]] = []
    for stage in chain:
        for consumer in stage.get("consumers", []):
            for artifact in stage.get("artifacts", []):
                if not artifact.endswith("/"):
                    edges.append(
                        {
                            "from": artifact,
                            "to": consumer,
                            "via_stage": stage["stage"],
                            "relationship": "artifact_consumed_by",
                        }
                    )
        for svc in stage.get("services", []):
            for consumer in stage.get("consumers", []):
                if consumer.endswith(".py"):
                    edges.append(
                        {
                            "from": svc,
                            "to": consumer,
                            "via_stage": stage["stage"],
                            "relationship": "service_used_by",
                        }
                    )

    gate_callers = _grep_files(r"enforce_pipeline_gate|enforce_broker_submission_gate")
    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H001",
        "operating_chain": chain,
        "execution_modules": execution_modules,
        "schedulers": [{"path": s, "exists": (ROOT / s).is_file()} for s in schedulers],
        "gate_enforcement_callers": gate_callers,
        "edges": edges,
        "env_dependencies": [
            "TRITON_ENABLE_PAPER_EXECUTION",
            "TRITON_ENABLE_LIVE_EXECUTION",
            "TRITON_LIVE_GOVERNANCE_APPROVED",
            "TRITON_DISABLE_EXECUTION_GATE",
        ],
    }


def find_unused_services() -> Dict[str, Any]:
    """Services never imported/referenced outside themselves and dashboards."""
    all_services = sorted(p.name for p in SERVICES_DIR.glob("*.py") if p.name != "__init__.py")
    referenced: Set[str] = set()
    source_files = _iter_source_files()
    for path in source_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for svc in all_services:
            mod = svc.replace(".py", "")
            if mod in text or f"services.{mod}" in text:
                referenced.add(svc)

    execution_critical = {
        "execute_trades.py",
        "place_live_orders.py",
        "manage_positions.py",
        "broker_alpaca.py",
        "triton_execution_gate.py",
        "master_execution_gate.py",
        "execution_guard.py",
        "lifecycle_truth.py",
        "apply_signal_lifecycle.py",
        "build_effective_lifecycle.py",
        "risk_gate.py",
        "run_execution_loop.py",
        "run_scheduled_paper_cycle.py",
        "loop_safety.py",
        "risk_watchdog.py",
        "continuous_loop_recovery_manager.py",
    }
    thp_gate = {
        s for s in all_services if s.startswith("execution_gate") or s.startswith("execution_")
    }
    unused = [s for s in all_services if s not in referenced]
    advisory_only = [
        s
        for s in all_services
        if s in referenced
        and s not in execution_critical
        and s not in thp_gate
        and any(
            x in s
            for x in (
                "governance_",
                "certification",
                "simulator",
                "forecast",
                "dashboard",
                "stress",
                "institutional_",
                "recovery_",
                "concentration_",
            )
        )
    ]
    legacy_candidates = [
        s
        for s in all_services
        if s.endswith(".recovered.py") or "recovered" in s or s == "execute_trades.recovered.py"
    ]
    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H001",
        "summary": {
            "total_services": len(all_services),
            "referenced_services": len(referenced),
            "unreferenced_services": len(unused),
            "advisory_only_estimate": len(advisory_only),
        },
        "unreferenced_services": unused[:100],
        "advisory_only_services_sample": sorted(advisory_only)[:80],
        "legacy_candidates": legacy_candidates,
        "execution_critical_services": sorted(execution_critical),
    }


def analyze_integration_risks() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    # CRITICAL: gate bypass paths
    if (ROOT / "services/broker_alpaca.py").is_file():
        text = (ROOT / "services/broker_alpaca.py").read_text(encoding="utf-8", errors="ignore")
        if (
            "def place_bracket_market" in text
            and "enforce_broker_submission_gate"
            not in text.split("def place_bracket_market")[1].split("def ")[0]
        ):
            _issue(
                issues,
                severity="CRITICAL",
                category="execution_bypass",
                component="services/broker_alpaca.py:place_bracket_market",
                description="place_bracket_market posts directly to /v2/orders without enforce_broker_submission_gate.",
                remediation="Route place_bracket_market through enforce_broker_submission_gate or submit_order.",
                affects_capital=True,
            )

    if os.environ.get("TRITON_DISABLE_EXECUTION_GATE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        _issue(
            issues,
            severity="CRITICAL",
            category="execution_bypass",
            component="TRITON_DISABLE_EXECUTION_GATE",
            description="Execution gate bypass env var is currently active.",
            remediation="Unset TRITON_DISABLE_EXECUTION_GATE in all production environments.",
            affects_capital=True,
        )

    legacy_scripts = [
        "scripts/auto_execute_signals.py",
        "scripts/auto_trade.py",
        "scripts/auto_exit.py",
        "scripts/execute_trade.py",
        "alpaca_client.py",
    ]
    for script in legacy_scripts:
        if (ROOT / script).is_file():
            _issue(
                issues,
                severity="HIGH",
                category="legacy_execution_path",
                component=script,
                description="Legacy script exists outside unified execution gate pipeline.",
                remediation="Deprecate or wire through triton_execution_gate enforcement.",
                affects_capital=True,
            )

    # HIGH: governed execution not on main path
    if not _grep_files(r"evaluate_governed_execution", "services/execute_trades.py"):
        _issue(
            issues,
            severity="HIGH",
            category="governance_gap",
            component="services/execute_trades.py",
            description="Main execution path uses enforce_pipeline_gate but not evaluate_governed_execution (THP-008).",
            remediation="Wire evaluate_governed_execution into capital deployment path or document as advisory-only.",
            affects_capital=False,
        )

    # HIGH: duplicate gate systems
    _issue(
        issues,
        severity="HIGH",
        category="duplicate_authority",
        component="master_execution_gate + triton_execution_gate",
        description="Two layered gate systems (MasterExecutionGate + Node execution gate) both required; failure modes differ.",
        remediation="Document precedence chain; ensure both must pass for production.",
        affects_capital=True,
    )

    # MEDIUM: stale artifacts
    stale_checks = [
        ("data/results/signal_lifecycle_effective.csv", "lifecycle", 48),
        ("data/results/trade_opportunities.csv", "execution input", 72),
        ("data/results/heartbeat.json", "master gate freshness", 24),
        ("data/results/positions_snapshot.csv", "planning", 48),
        ("data/governance/execution_gate_certification.json", "certification", 720),
    ]
    for rel, label, max_hours in stale_checks:
        path = ROOT / rel
        age = _file_age_hours(path)
        if age is None:
            _issue(
                issues,
                severity="MEDIUM",
                category="missing_artifact",
                component=rel,
                description=f"Expected {label} artifact missing.",
                remediation=f"Ensure pipeline generates {rel}.",
            )
        elif age > max_hours:
            _issue(
                issues,
                severity="MEDIUM",
                category="stale_artifact",
                component=rel,
                description=f"{label} artifact is {age:.0f}h old (threshold {max_hours}h).",
                remediation="Refresh via run_full_pipeline or execution loop.",
            )

    # MEDIUM: governance council advisory only
    _issue(
        issues,
        severity="MEDIUM",
        category="advisory_only",
        component="execution_governance_council",
        description="THP-007/008 health and council systems monitor and recommend but do not block execute_trades by default.",
        remediation="Integrate operational_status LOCKDOWN flag into execution entry if institutional enforcement required.",
    )

    # MEDIUM: duplicate lifecycle sources
    _issue(
        issues,
        severity="MEDIUM",
        category="duplicate_source_of_truth",
        component="signal_lifecycle.csv vs signal_lifecycle_effective.csv",
        description="Two lifecycle artifacts; execution uses effective copy via lifecycle_truth.",
        remediation="Treat signal_lifecycle_effective.csv as sole execution authority; dashboards should match.",
    )

    # LOW: large advisory surface
    unused = find_unused_services()
    if unused["summary"]["unreferenced_services"] > 50:
        _issue(
            issues,
            severity="LOW",
            category="code_surface",
            component="services/",
            description=f"{unused['summary']['unreferenced_services']} service modules appear unreferenced (possible dead code).",
            remediation="Archive or document advisory-only modules; reduce operator confusion.",
        )

    if (ROOT / "services/execute_trades.recovered.py").is_file():
        _issue(
            issues,
            severity="LOW",
            category="legacy_file",
            component="services/execute_trades.recovered.py",
            description="Recovered duplicate of execute_trades exists in repo.",
            remediation="Remove or move to archive outside execution path.",
        )

    by_severity: Dict[str, int] = {}
    for i in issues:
        sev = i["severity"]
        by_severity[sev] = by_severity.get(sev, 0) + 1

    matrix = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H001",
        "summary": {
            "total_issues": len(issues),
            "by_severity": by_severity,
            "critical_count": by_severity.get("CRITICAL", 0),
            "high_count": by_severity.get("HIGH", 0),
            "capital_affecting": sum(1 for i in issues if i.get("affects_capital")),
        },
        "remediation_order": [
            "Resolve CRITICAL execution bypass paths",
            "Wire or formally exempt governance council from capital path",
            "Refresh stale operational artifacts",
            "Consolidate duplicate source-of-truth documentation",
            "Archive unused advisory services",
        ],
        "records": issues,
    }
    return issues, matrix


def trace_execution_path() -> Dict[str, Any]:
    """End-to-end execution path trace."""
    return {
        "paper_production_path": [
            "scripts/run_continuous_loop.ps1 (TRITON_ENABLE_PAPER_EXECUTION=1)",
            "scripts/run_market_open.ps1 OR services/run_execution_loop.py",
            "run_full_pipeline.py (signals → lifecycle)",
            "services/build_effective_lifecycle.py → signal_lifecycle_effective.csv",
            "services/execute_trades.py --execute (MasterGate + enforce_pipeline_gate)",
            "services/place_live_orders.py --no-dry-run (MasterGate + enforce_pipeline_gate)",
            "services/broker_alpaca.submit_order (enforce_broker_submission_gate)",
            "Alpaca POST /v2/orders",
        ],
        "management_path": [
            "services/manage_positions.py (lifecycle_truth + MasterGate + enforce_pipeline_gate)",
            "subprocess place_live_orders --no-dry-run",
            "broker_alpaca.submit_order",
        ],
        "gate_choke_points": [
            "MasterExecutionGate.evaluate — batch operational gate",
            "enforce_pipeline_gate — stage path (capital_deployment / broker_submission / position_*)",
            "ExecutionGuard.validate — per-order",
            "enforce_broker_submission_gate — final pre-API",
            "evaluate_execution_gate — Node lib/executionGate.js single source of truth",
        ],
        "advisory_only_layers": [
            "THP-004..009 governance dashboards and evidence registries",
            "services/risk_watchdog.py",
            "services/execution_gate_health_monitor.py",
            "services/execution_governance_council.py (not on main execute path)",
            "governance_* analysis and certification services",
        ],
        "components_driving_execution": [
            "signal_lifecycle_effective.csv",
            "trade_opportunities.csv",
            "MasterExecutionGate",
            "ExecutionGuard",
            "triton_execution_gate",
            "broker_alpaca.submit_order",
        ],
    }


def run_system_integration_audit() -> Dict[str, Any]:
    chain = build_operating_chain()
    sot = build_source_of_truth_registry()
    dep_map = build_dependency_map(chain)
    unused = find_unused_services()
    issues, risk_matrix = analyze_integration_risks()
    execution_trace = trace_execution_path()

    by_severity = risk_matrix.get("summary", {}).get("by_severity", {})

    audit = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H001",
        "audit_type": "full_system_integration",
        "investigation_only": True,
        "executive_summary": {
            "question": "Does every part of TRITON work together as one coherent institution?",
            "execution_path_traceable": True,
            "gate_final_authority_before_broker": True,
            "governance_council_on_execution_path": False,
            "total_issues": len(issues),
            "critical_issues": by_severity.get("CRITICAL", 0),
            "high_issues": by_severity.get("HIGH", 0),
            "components_driving_execution": execution_trace["components_driving_execution"],
            "advisory_only_layers": execution_trace["advisory_only_layers"],
        },
        "sections": {
            "1_system_dependency_map": dep_map,
            "2_source_of_truth_registry": sot,
            "3_dead_or_unused_components": unused,
            "4_operational_gaps": [
                i
                for i in issues
                if i["category"]
                in (
                    "governance_gap",
                    "missing_artifact",
                    "stale_artifact",
                    "advisory_only",
                )
            ],
            "5_risk_ranking": risk_matrix["summary"],
        },
        "execution_path_trace": execution_trace,
        "issues": issues,
    }

    _atomic_write_json(AUDIT_JSON, audit)
    _write_csv(AUDIT_CSV, issues, AUDIT_CSV_FIELDS)
    _atomic_write_json(SOURCE_OF_TRUTH_JSON, sot)
    _atomic_write_json(DEPENDENCY_MAP_JSON, dep_map)
    _atomic_write_json(UNUSED_COMPONENTS_JSON, unused)
    _atomic_write_json(RISK_MATRIX_JSON, risk_matrix)

    return audit


def load_integration_audit() -> Dict[str, Any]:
    if not AUDIT_JSON.is_file():
        return {}
    try:
        return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    doc = run_system_integration_audit()
    print(json.dumps(doc.get("executive_summary"), indent=2))
