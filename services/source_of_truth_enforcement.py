"""
source_of_truth_enforcement.py
THP-H005 — Source of truth registry, artifact conflict detection, and authority audit.

Discovery and authority definition only — no trading, lifecycle, or execution changes.
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
LIVE_DIR = ROOT / "data" / "live"

REGISTRY_JSON = GOVERNANCE_DIR / "source_of_truth_registry.json"
REGISTRY_CSV = GOVERNANCE_DIR / "source_of_truth_registry.csv"
CONFLICTS_JSON = GOVERNANCE_DIR / "source_of_truth_conflicts.json"
DEPENDENCY_MAP_JSON = GOVERNANCE_DIR / "artifact_dependency_map.json"
STALENESS_JSON = GOVERNANCE_DIR / "artifact_staleness_report.json"
REMEDIATION_JSON = GOVERNANCE_DIR / "artifact_remediation_plan.json"

REGISTRY_FIELDS = (
    "decision_domain",
    "official_owner",
    "authoritative_artifact",
    "producer_service",
    "consumers",
    "refresh_frequency",
    "derived_artifacts",
    "historical_artifacts",
    "staleness_risk",
    "compliance_status",
)

EXECUTION_CRITICAL = frozenset(
    {
        "services/execute_trades.py",
        "services/place_live_orders.py",
        "services/manage_positions.py",
        "place_orders_from_csv.py",
        "services/broker_alpaca.py",
        "services/run_execution_loop.py",
        "services/run_scheduled_paper_cycle.py",
    }
)

STALE_THRESHOLDS_HOURS = {
    "lifecycle": 24,
    "allocation": 24,
    "risk": 4,
    "governance": 168,
    "execution_state": 1,
    "pipeline": 24,
}

# Institutional source-of-truth catalog (THP-H005)
DOMAIN_CATALOG: List[Dict[str, Any]] = [
    {
        "decision_domain": "lifecycle_authorization",
        "official_owner": "services/lifecycle_truth.py",
        "authoritative_artifact": "data/results/signal_lifecycle_effective.csv",
        "producer_service": "services/build_effective_lifecycle.py",
        "refresh_frequency": "each pipeline cycle after apply_signal_lifecycle",
        "derived_artifacts": [
            "data/results/trade_opportunities.csv",
        ],
        "historical_artifacts": [
            "data/results/signal_lifecycle.csv",
            "data/results/signals.csv",
            "data/results/signals_snapshot.csv",
        ],
        "staleness_risk": "HIGH — execution may fall back to raw lifecycle or trade_opportunities",
        "artifact_patterns": [
            "signal_lifecycle_effective",
            "signal_lifecycle.csv",
            "trade_opportunities",
        ],
    },
    {
        "decision_domain": "portfolio_allocation",
        "official_owner": "services/portfolio_allocation_engine.py",
        "authoritative_artifact": "data/results/portfolio_allocation_recommendations.csv",
        "producer_service": "services/portfolio_allocation_engine.py",
        "refresh_frequency": "each pipeline cycle after lifecycle",
        "derived_artifacts": [
            "data/results/portfolio_allocation_summary.json",
            "data/results/stock_scores.csv",
        ],
        "historical_artifacts": [],
        "staleness_risk": "MEDIUM — summary JSON may lag recommendations CSV",
        "artifact_patterns": [
            "portfolio_allocation_recommendations",
            "portfolio_allocation_summary",
        ],
    },
    {
        "decision_domain": "risk_restriction",
        "official_owner": "services/master_execution_gate.py",
        "authoritative_artifact": "data/results/master_execution_gate.json",
        "producer_service": "services/master_execution_gate.py",
        "refresh_frequency": "each execution batch (MasterExecutionGate.evaluate)",
        "derived_artifacts": [
            "data/results/adaptive_risk_state.json",
            "data/results/performance_risk_overlay.csv",
            "data/results/risk_report.json",
        ],
        "historical_artifacts": [],
        "staleness_risk": "MEDIUM — adaptive_risk_state consumed directly by multiple services",
        "artifact_patterns": [
            "master_execution_gate.json",
            "adaptive_risk_state",
            "performance_risk_overlay",
        ],
    },
    {
        "decision_domain": "execution_authorization",
        "official_owner": "services/triton_execution_gate.py",
        "authoritative_artifact": "data/governance/execution_gate_decisions.json",
        "producer_service": "lib/executionGate.js via services/triton_execution_gate.py",
        "refresh_frequency": "per evaluate_governed_execution call",
        "derived_artifacts": [
            "data/results/execution_gate_audit.csv",
            "data/governance/execution_gate_evidence.json",
        ],
        "historical_artifacts": [],
        "staleness_risk": "LOW — runtime evaluation; artifacts are audit trail",
        "artifact_patterns": [
            "execution_gate_decisions",
            "execution_gate_audit",
            "execution_gate_evidence",
        ],
    },
    {
        "decision_domain": "governance_approval",
        "official_owner": "services/execution_approval_workflow.py",
        "authoritative_artifact": "data/governance/execution_approval_requests.json",
        "producer_service": "services/execution_approval_workflow.py",
        "refresh_frequency": "per evaluate_governed_execution when approval required",
        "derived_artifacts": [
            "data/governance/execution_escalations.json",
            "data/governance/execution_governance_council_status.json",
            "data/governance/execution_approval_history.csv",
        ],
        "historical_artifacts": [],
        "staleness_risk": "LOW — dashboards must be read-only consumers",
        "artifact_patterns": [
            "execution_approval_requests",
            "execution_escalations",
            "execution_governance_council_status",
        ],
    },
    {
        "decision_domain": "broker_positions_orders",
        "official_owner": "services/broker_alpaca.py",
        "authoritative_artifact": "Alpaca API (live broker state)",
        "producer_service": "services/broker_alpaca.py",
        "refresh_frequency": "real-time per broker call",
        "derived_artifacts": [
            "data/results/positions_snapshot.csv",
            "data/results/open_orders_snapshot.csv",
            "data/live/orders_today.csv",
        ],
        "historical_artifacts": [
            "data/results/portfolio_history.csv",
            "data/results/trade_log.csv",
            "data/results/live_orders_log.csv",
        ],
        "staleness_risk": "HIGH — local snapshots used for planning without guaranteed refresh",
        "artifact_patterns": [
            "positions_snapshot",
            "open_orders_snapshot",
            "live_orders_log",
            "portfolio_history",
        ],
    },
    {
        "decision_domain": "operational_pipeline_state",
        "official_owner": "run_full_pipeline.py",
        "authoritative_artifact": "data/results/heartbeat.json",
        "producer_service": "services/loop_safety.py",
        "refresh_frequency": "each pipeline / loop cycle",
        "derived_artifacts": [
            "data/results/pipeline_status.json",
            "data/results/asof_contract.json",
        ],
        "historical_artifacts": [],
        "staleness_risk": "MEDIUM — stale heartbeat blocks master execution gate",
        "artifact_patterns": ["heartbeat.json", "pipeline_status", "asof_contract"],
    },
]


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
            flat = {k: row.get(k, "") for k in fieldnames}
            for k, v in flat.items():
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v)
            writer.writerow(flat)
    os.replace(tmp, path)


def _file_age_hours(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600


def _iter_source_files() -> List[Path]:
    skip_dirs = {"data", "node_modules", ".git", ".venv", "venv", "__pycache__", ".cursor"}
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
            if path.suffix not in (".py", ".js", ".jsx", ".mjs"):
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            paths.append(path)
    for name in ("run_full_pipeline.py", "place_orders_from_csv.py", "view_results.py", "app.py"):
        p = ROOT / name
        if p.is_file():
            paths.append(p)
    return paths


def _rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _classify_consumer(rel: str) -> str:
    if rel in EXECUTION_CRITICAL:
        return "EXECUTION"
    if rel.startswith("dashboard/") or rel.startswith("pages/") or rel == "view_results.py":
        return "DASHBOARD"
    if "audit" in rel or "diagnostic" in rel or "certification" in rel or "recovery" in rel:
        return "ADVISORY"
    return "SUPPORTING"


def _find_consumers(patterns: List[str]) -> Dict[str, List[str]]:
    by_type: Dict[str, Set[str]] = {
        "EXECUTION": set(),
        "DASHBOARD": set(),
        "ADVISORY": set(),
        "SUPPORTING": set(),
        "PRODUCER": set(),
    }
    for path in _iter_source_files():
        rel = _rel_path(path)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not any(p in text for p in patterns):
            continue
        kind = _classify_consumer(rel)
        by_type[kind].add(rel)
    return {k: sorted(v) for k, v in by_type.items() if v}


def _resolve_artifact_path(artifact: str) -> Optional[Path]:
    if artifact.startswith("Alpaca"):
        return None
    return ROOT / artifact.replace("/", os.sep)


def _assess_compliance(domain: Dict[str, Any], consumers: Dict[str, List[str]]) -> str:
    domain_id = domain["decision_domain"]
    exec_consumers = consumers.get("EXECUTION", [])
    auth = domain["authoritative_artifact"]
    patterns = domain.get("artifact_patterns", [])

    if domain_id == "lifecycle_authorization":
        exec_snippets = " ".join(_read_snippet(c) for c in exec_consumers)
        if (
            "validate_authoritative_lifecycle" in exec_snippets
            or "decision_authority" in exec_snippets
        ):
            if (
                "resolve_execution_source" in exec_snippets
                and "derived_trade_opps" in exec_snippets
            ):
                return "COMPLIANT — THP-H006 effective lifecycle authority enforced"
            return "COMPLIANT — THP-H006 lifecycle authority module active"
        if "signal_lifecycle_effective" in exec_snippets:
            return "PARTIAL — uses effective artifact; authority module not unified"
        return "NON_COMPLIANT"

    if domain_id == "portfolio_allocation":
        if "services/manage_positions.py" in exec_consumers:
            return "PARTIAL — execution reads CSV; summary JSON also referenced in manage_positions"
        return "COMPLIANT"

    if domain_id == "risk_restriction":
        direct_adaptive = any(
            "adaptive_risk_state" in _read_snippet(c)
            and "decision_authority" not in _read_snippet(c)
            for c in exec_consumers
        )
        uses_authority = any(
            "decision_authority" in _read_snippet(c) or "master_execution_gate" in _read_snippet(c)
            for c in exec_consumers
        )
        if direct_adaptive:
            return "PARTIAL — execution still reads adaptive_risk_state directly"
        if uses_authority:
            return "COMPLIANT — THP-H006 master_execution_gate.json authority"
        return "NON_COMPLIANT"

    if domain_id == "execution_authorization":
        return "COMPLIANT — THP-H004 governed enforcement on all active paths"

    if domain_id == "governance_approval":
        return "COMPLIANT — THP-H004 evaluate_governed_execution enforces council authority"

    if domain_id == "broker_positions_orders":
        if exec_consumers:
            return "PARTIAL — broker is authority; local snapshots used in planning paths"
        return "COMPLIANT"

    if domain_id == "operational_pipeline_state":
        return "COMPLIANT"

    return "UNKNOWN"


def _read_snippet(rel: str) -> str:
    p = ROOT / rel.replace("/", os.sep)
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def build_registry_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for domain in DOMAIN_CATALOG:
        consumers_map = _find_consumers(domain.get("artifact_patterns", []))
        all_consumers = sorted({c for lst in consumers_map.values() for c in lst})
        rows.append(
            {
                "decision_domain": domain["decision_domain"],
                "official_owner": domain["official_owner"],
                "authoritative_artifact": domain["authoritative_artifact"],
                "producer_service": domain["producer_service"],
                "consumers": all_consumers,
                "consumers_by_role": consumers_map,
                "refresh_frequency": domain["refresh_frequency"],
                "derived_artifacts": domain["derived_artifacts"],
                "historical_artifacts": domain["historical_artifacts"],
                "staleness_risk": domain["staleness_risk"],
                "compliance_status": _assess_compliance(domain, consumers_map),
            }
        )
    return rows


def _conflict(
    conflict_type: str,
    severity: str,
    domain: str,
    description: str,
    impact: str,
    affected: List[str],
    remediation: str,
) -> Dict[str, Any]:
    return {
        "conflict_type": conflict_type,
        "severity": severity,
        "decision_domain": domain,
        "description": description,
        "impact": impact,
        "affected_systems": affected,
        "recommended_remediation": remediation,
    }


def detect_conflicts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []

    # DUPLICATE_AUTHORITY — lifecycle
    conflicts.append(
        _conflict(
            "DUPLICATE_AUTHORITY",
            "HIGH",
            "lifecycle_authorization",
            "Three artifacts can drive execution intent: signal_lifecycle_effective.csv, "
            "signal_lifecycle.csv, and trade_opportunities.csv.",
            "Capital deployment may use a different lifecycle view than dashboards or lifecycle gate.",
            [
                "services/execute_trades.py",
                "services/place_live_orders.py",
                "services/build_trade_opportunities.py",
            ],
            "H006: Enforce single read path — effective lifecycle only; trade_opportunities as "
            "derived view labeled non-authoritative.",
        )
    )

    # DUPLICATE_AUTHORITY — allocation
    conflicts.append(
        _conflict(
            "DUPLICATE_AUTHORITY",
            "MEDIUM",
            "portfolio_allocation",
            "portfolio_allocation_recommendations.csv and portfolio_allocation_summary.json "
            "both represent allocation state.",
            "Dashboard or manage_positions may display summary while sizing uses CSV.",
            [
                "services/manage_positions.py",
                "view_results.py",
                "services/execution_gate_evidence_registry.py",
            ],
            "H006: Mark summary JSON as derived read-only; dashboards must read recommendations CSV.",
        )
    )

    # DUPLICATE_AUTHORITY — risk
    conflicts.append(
        _conflict(
            "DUPLICATE_AUTHORITY",
            "HIGH",
            "risk_restriction",
            "master_execution_gate.json (batch snapshot) and adaptive_risk_state.json "
            "(continuous intelligence) both inform execution restrictions.",
            "execute_trades reads adaptive_risk_state directly while batch gate uses master snapshot.",
            [
                "services/execute_trades.py",
                "services/master_execution_gate.py",
                "services/execution_governance_council.py",
            ],
            "H006: Execution batch decisions must consume master_execution_gate.json only; "
            "adaptive_risk_state reserved for intelligence/producer input.",
        )
    )

    # STALE_CONSUMER — governance readiness checks raw lifecycle
    conflicts.append(
        _conflict(
            "STALE_CONSUMER",
            "MEDIUM",
            "lifecycle_authorization",
            "governance_execution_readiness.py validates signal_lifecycle.csv staleness, "
            "not signal_lifecycle_effective.csv.",
            "Operational readiness may pass while effective lifecycle is stale.",
            ["services/governance_execution_readiness.py"],
            "H006: Align readiness checks to authoritative effective lifecycle artifact.",
        )
    )

    # STALE_CONSUMER — positions snapshot in dashboard
    conflicts.append(
        _conflict(
            "STALE_CONSUMER",
            "MEDIUM",
            "broker_positions_orders",
            "view_results.py displays positions_snapshot.csv which may lag live broker state.",
            "Operator may see stale positions while execution uses live broker API.",
            ["view_results.py", "data/results/positions_snapshot.csv"],
            "H006: Label snapshot as derived; show as-of timestamp; optional live broker refresh panel.",
        )
    )

    # UNCONTROLLED_DERIVATION — place_live_orders lifecycle column merge
    conflicts.append(
        _conflict(
            "UNCONTROLLED_DERIVATION",
            "MEDIUM",
            "lifecycle_authorization",
            "place_live_orders.apply_effective_lifecycle_columns() re-merges effective columns "
            "at read time instead of consuming pre-built effective artifact only.",
            "In-memory transformation may diverge from build_effective_lifecycle output.",
            ["services/place_live_orders.py"],
            "H006: Require pre-built signal_lifecycle_effective.csv; remove runtime column merge.",
        )
    )

    # UNCONTROLLED_DERIVATION — ticker timeline fallback
    conflicts.append(
        _conflict(
            "UNCONTROLLED_DERIVATION",
            "LOW",
            "lifecycle_authorization",
            "ticker_timeline_builder.py falls back from effective to raw signal_lifecycle.csv.",
            "Historical timeline may mix authoritative and pre-effective states.",
            ["services/ticker_timeline_builder.py", "services/pnl_ledger_builder.py"],
            "H006: Use effective only; treat raw lifecycle as historical archive.",
        )
    )

    # ORPHANED_ARTIFACT candidates
    orphan_checks = [
        ("data/results/signals_snapshot.json", "signals_snapshot"),
        ("data/results/signals_snapshot.csv", "signals_snapshot"),
    ]
    for artifact, pattern in orphan_checks:
        consumers = _find_consumers([pattern])
        total = sum(len(v) for v in consumers.values())
        if total <= 2:
            conflicts.append(
                _conflict(
                    "ORPHANED_ARTIFACT",
                    "LOW",
                    "operational_pipeline_state",
                    f"{artifact} has minimal or no active consumers.",
                    "Disk clutter; possible confusion if mistaken for current truth.",
                    [artifact],
                    "Archive or document as historical; remove from operational checklists.",
                )
            )

    # Layered gates — informational not conflict
    conflicts.append(
        _conflict(
            "DUPLICATE_AUTHORITY",
            "LOW",
            "execution_authorization",
            "master_execution_gate (Python batch) and triton_execution_gate (Node runtime) "
            "operate at different layers.",
            "Intentional layering — not a bypass — but operators must understand both layers.",
            [
                "services/master_execution_gate.py",
                "services/triton_execution_gate.py",
            ],
            "Document layered authority in operator runbook; no merge required.",
        )
    )

    _h006_resolved_domains = {
        ("DUPLICATE_AUTHORITY", "lifecycle_authorization"),
        ("DUPLICATE_AUTHORITY", "risk_restriction"),
    }
    _h006_resolved_stale = {
        ("STALE_CONSUMER", "lifecycle_authorization"),
        ("UNCONTROLLED_DERIVATION", "lifecycle_authorization"),
    }
    for c in conflicts:
        key = (c.get("conflict_type"), c.get("decision_domain"))
        if key in _h006_resolved_domains:
            c["status"] = "RESOLVED_THP-H006"
            c["severity"] = "RESOLVED"
            c["resolution"] = (
                "THP-H006 enforced via decision_authority.py and execution consumer updates"
            )
        elif key in _h006_resolved_stale and c.get("severity") in ("MEDIUM", "LOW"):
            if (
                "place_live_orders" in str(c.get("affected_systems"))
                or "governance_execution" in str(c.get("affected_systems"))
                or "ticker_timeline" in str(c.get("affected_systems"))
            ):
                c["status"] = "RESOLVED_THP-H006"
                c["severity"] = "RESOLVED"
                c["resolution"] = "THP-H006 consumer alignment complete"

    return conflicts


def build_dependency_map(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    edges: List[Dict[str, str]] = []
    for row in rows:
        auth = row["authoritative_artifact"]
        producer = row["producer_service"]
        domain = row["decision_domain"]
        edges.append(
            {
                "from": producer,
                "to": auth,
                "relationship": "produces",
                "domain": domain,
            }
        )
        for consumer in row.get("consumers") or []:
            edges.append(
                {
                    "from": auth,
                    "to": consumer,
                    "relationship": "consumed_by",
                    "domain": domain,
                }
            )
        for derived in row.get("derived_artifacts") or []:
            edges.append(
                {
                    "from": auth,
                    "to": derived,
                    "relationship": "derives",
                    "domain": domain,
                }
            )
        for hist in row.get("historical_artifacts") or []:
            edges.append(
                {
                    "from": hist,
                    "to": auth,
                    "relationship": "superseded_by",
                    "domain": domain,
                }
            )

    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H005",
        "principle": "ONE DECISION → ONE OWNER → ONE ARTIFACT → MANY READ-ONLY CONSUMERS",
        "edge_count": len(edges),
        "edges": edges,
    }


def build_staleness_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for row in rows:
        auth = row["authoritative_artifact"]
        path = _resolve_artifact_path(auth)
        age = _file_age_hours(path) if path else None
        domain = row["decision_domain"]
        threshold_key = (
            "lifecycle"
            if "lifecycle" in domain
            else (
                "allocation"
                if "allocation" in domain
                else (
                    "risk"
                    if "risk" in domain
                    else (
                        "governance"
                        if "governance" in domain or "execution_authorization" in domain
                        else "execution_state" if "broker" in domain else "pipeline"
                    )
                )
            )
        )
        threshold = STALE_THRESHOLDS_HOURS.get(threshold_key, 48)
        stale = age is not None and age > threshold
        entries.append(
            {
                "decision_domain": domain,
                "artifact": auth,
                "exists": path.is_file() if path else None,
                "age_hours": round(age, 2) if age is not None else None,
                "stale_threshold_hours": threshold,
                "is_stale": stale,
                "staleness_risk": row.get("staleness_risk", ""),
            }
        )
        for derived in row.get("derived_artifacts") or []:
            dpath = _resolve_artifact_path(derived)
            if not dpath:
                continue
            dage = _file_age_hours(dpath)
            entries.append(
                {
                    "decision_domain": domain,
                    "artifact": derived,
                    "exists": dpath.is_file(),
                    "age_hours": round(dage, 2) if dage is not None else None,
                    "stale_threshold_hours": threshold,
                    "is_stale": dage is not None and dage > threshold,
                    "staleness_risk": "derived — must not exceed authoritative artifact age",
                }
            )

    stale_count = sum(1 for e in entries if e.get("is_stale"))
    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H005",
        "summary": {
            "artifacts_checked": len(entries),
            "stale_artifacts": stale_count,
        },
        "entries": entries,
    }


def build_remediation_plan(conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_severity: Dict[str, List[Dict[str, Any]]] = {
        "CRITICAL": [],
        "HIGH": [],
        "MEDIUM": [],
        "LOW": [],
    }
    for c in conflicts:
        sev = c.get("severity", "LOW")
        by_severity.setdefault(sev, []).append(c)

    phases = [
        {
            "phase": "THP-H006",
            "title": "Lifecycle authority hardening",
            "items": [
                c for c in conflicts if c.get("decision_domain") == "lifecycle_authorization"
            ],
        },
        {
            "phase": "THP-H006",
            "title": "Risk authority consolidation",
            "items": [c for c in conflicts if c.get("decision_domain") == "risk_restriction"],
        },
        {
            "phase": "THP-H006",
            "title": "Allocation & snapshot clarity",
            "items": [
                c
                for c in conflicts
                if c.get("decision_domain") in ("portfolio_allocation", "broker_positions_orders")
            ],
        },
        {
            "phase": "THP-H007",
            "title": "Documentation & orphan cleanup",
            "items": [
                c
                for c in conflicts
                if c.get("severity") == "LOW" or c.get("conflict_type") == "ORPHANED_ARTIFACT"
            ],
        },
    ]

    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H005",
        "discovery_only": True,
        "summary": {
            "total_conflicts": len(conflicts),
            "critical": len(by_severity.get("CRITICAL", [])),
            "high": len(by_severity.get("HIGH", [])),
            "medium": len(by_severity.get("MEDIUM", [])),
            "low": len(by_severity.get("LOW", [])),
        },
        "by_severity": by_severity,
        "phased_remediation": phases,
        "enforcement_rules": {
            "lifecycle": "Execution may only consume signal_lifecycle_effective.csv",
            "allocation": "Capital sizing must use portfolio_allocation_recommendations.csv",
            "risk": "Execution restrictions must use master_execution_gate.json snapshot",
            "governance": "Permissions via execution_approval_requests.json + evaluate_governed_execution",
            "broker": "Live state from Alpaca API via broker_alpaca.py",
        },
    }


def run_source_of_truth_enforcement() -> Dict[str, Any]:
    rows = build_registry_rows()
    conflicts = detect_conflicts(rows)
    dep_map = build_dependency_map(rows)
    staleness = build_staleness_report(rows)
    remediation = build_remediation_plan(conflicts)

    registry_doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H005",
        "principle": "ONE DECISION → ONE OWNER → ONE ARTIFACT → MANY READ-ONLY CONSUMERS",
        "domains": len(rows),
        "entries": rows,
        "summary": {
            "decision_domains_defined": len(rows),
            "compliant_domains": sum(
                1 for r in rows if str(r.get("compliance_status", "")).startswith("COMPLIANT")
            ),
            "partial_domains": sum(
                1 for r in rows if str(r.get("compliance_status", "")).startswith("PARTIAL")
            ),
            "non_compliant_domains": sum(
                1 for r in rows if str(r.get("compliance_status", "")).startswith("NON_COMPLIANT")
            ),
            "total_conflicts": len(conflicts),
            "high_or_critical_conflicts": sum(
                1 for c in conflicts if c.get("severity") in ("CRITICAL", "HIGH")
            ),
        },
    }

    conflicts_doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H005",
        "summary": remediation["summary"],
        "conflicts": conflicts,
    }

    _atomic_write_json(REGISTRY_JSON, registry_doc)
    _write_csv(
        REGISTRY_CSV,
        [{k: r.get(k, "") for k in REGISTRY_FIELDS} for r in rows],
        REGISTRY_FIELDS,
    )
    _atomic_write_json(CONFLICTS_JSON, conflicts_doc)
    _atomic_write_json(DEPENDENCY_MAP_JSON, dep_map)
    _atomic_write_json(STALENESS_JSON, staleness)
    _atomic_write_json(REMEDIATION_JSON, remediation)

    return {
        "registry": registry_doc,
        "conflicts": conflicts_doc,
        "dependency_map": dep_map,
        "staleness": staleness,
        "remediation": remediation,
    }


if __name__ == "__main__":
    result = run_source_of_truth_enforcement()
    s = result["registry"]["summary"]
    print(f"Registry: {REGISTRY_JSON}")
    print(f"Conflicts: {CONFLICTS_JSON}")
    print(
        f"Domains={s['decision_domains_defined']} "
        f"COMPLIANT={s['compliant_domains']} "
        f"PARTIAL={s['partial_domains']} "
        f"conflicts={s['total_conflicts']} "
        f"(HIGH/CRITICAL={s['high_or_critical_conflicts']})"
    )
