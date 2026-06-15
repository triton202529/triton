"""
execution_path_registry.py
THP-H003 — Execution authority registry and cleanup report generator.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

REGISTRY_JSON = GOVERNANCE_DIR / "execution_path_registry.json"
REGISTRY_CSV = GOVERNANCE_DIR / "execution_path_registry.csv"
CLEANUP_REPORT_JSON = GOVERNANCE_DIR / "execution_path_cleanup_report.json"

REGISTRY_FIELDS = (
    "file_name",
    "purpose",
    "status",
    "owner",
    "last_known_operational_use",
    "broker_access_capability",
    "execution_gate_compliance",
    "governance_compliance",
    "recommended_action",
)

# Institutional capital deployment chain (official ACTIVE path)
ORCHESTRATOR_PATHS = frozenset(
    {
        "services/execute_trades.py",
        "services/manage_positions.py",
        "services/run_execution_loop.py",
        "services/run_scheduled_paper_cycle.py",
        "services/place_live_orders.py",
    }
)

OFFICIAL_CHAIN = [
    "services/run_execution_loop.py",
    "services/run_scheduled_paper_cycle.py",
    "services/execute_trades.py",
    "services/place_live_orders.py",
    "services/manage_positions.py",
    "services/triton_execution_gate.py",
    "services/broker_alpaca.py",
]

EXECUTION_PATHS: Dict[str, Dict[str, Any]] = {
    # --- ACTIVE: institutional chain ---
    "services/run_execution_loop.py": {
        "purpose": "Continuous execution loop orchestrator",
        "status": "ACTIVE",
        "owner": "platform",
        "last_known_operational_use": "continuous loop / scheduler",
        "broker_access_capability": "indirect via place_live_orders",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — primary loop entry",
        "approved_alternative": "",
    },
    "services/run_scheduled_paper_cycle.py": {
        "purpose": "Scheduled paper trading cycle",
        "status": "ACTIVE",
        "owner": "platform",
        "last_known_operational_use": "scheduled paper runs",
        "broker_access_capability": "indirect via execute_trades",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — scheduled paper entry",
        "approved_alternative": "",
    },
    "services/execute_trades.py": {
        "purpose": "Lifecycle-aware trade execution orchestrator",
        "status": "ACTIVE",
        "owner": "execution",
        "last_known_operational_use": "pipeline execute_trades --execute",
        "broker_access_capability": "via place_live_orders / manage_positions",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — core execution orchestrator",
        "approved_alternative": "",
    },
    "services/place_live_orders.py": {
        "purpose": "Place live orders from approved signals",
        "status": "ACTIVE",
        "owner": "execution",
        "last_known_operational_use": "execute_trades capital deployment",
        "broker_access_capability": "AlpacaBroker.submit_order",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — broker submission layer",
        "approved_alternative": "",
    },
    "services/manage_positions.py": {
        "purpose": "Position management and exits",
        "status": "ACTIVE",
        "owner": "execution",
        "last_known_operational_use": "execute_trades exit path",
        "broker_access_capability": "AlpacaBroker gated methods",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — exit / rotation path",
        "approved_alternative": "",
    },
    "services/broker_alpaca.py": {
        "purpose": "Authorized Alpaca broker interface",
        "status": "ACTIVE",
        "owner": "broker",
        "last_known_operational_use": "all gated order submission",
        "broker_access_capability": "POST /v2/orders (BROKER FINAL CHECK only)",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — sole broker order POST authority",
        "approved_alternative": "",
    },
    "services/triton_execution_gate.py": {
        "purpose": "Execution gate and broker submission enforcement",
        "status": "ACTIVE",
        "owner": "governance",
        "last_known_operational_use": "evaluate_governed_execution / enforce_broker_submission_gate",
        "broker_access_capability": "none (gate only)",
        "execution_gate_compliance": "YES",
        "governance_compliance": "YES",
        "recommended_action": "KEEP — gate authority",
        "approved_alternative": "",
    },
    "place_orders_from_csv.py": {
        "purpose": "Operator CSV order workflow (view_results integration)",
        "status": "ACTIVE",
        "owner": "operator",
        "last_known_operational_use": "manual CSV batch placement",
        "broker_access_capability": "AlpacaBroker.submit_order",
        "execution_gate_compliance": "YES",
        "governance_compliance": "PARTIAL",
        "recommended_action": "KEEP — hardened to AlpacaBroker",
        "approved_alternative": "",
    },
    "pages/manual_order_desk.py": {
        "purpose": "Approved Streamlit manual order desk",
        "status": "ACTIVE",
        "owner": "operator",
        "last_known_operational_use": "view_results manual_order_desk",
        "broker_access_capability": "AlpacaBroker.submit_order",
        "execution_gate_compliance": "YES",
        "governance_compliance": "PARTIAL",
        "recommended_action": "KEEP — gated operator UI",
        "approved_alternative": "",
    },
    "pages_order_desk.py": {
        "purpose": "Legacy Streamlit order desk (gated broker wrapper)",
        "status": "ACTIVE",
        "owner": "operator",
        "last_known_operational_use": "manual operator orders",
        "broker_access_capability": "AlpacaBroker.submit_order / submit_order_notional",
        "execution_gate_compliance": "YES",
        "governance_compliance": "PARTIAL",
        "recommended_action": "KEEP — uses gated broker only",
        "approved_alternative": "",
    },
    # --- DEPRECATED: legacy bypass paths ---
    "scripts/auto_execute_signals.py": {
        "purpose": "Legacy signal CSV executor (alpaca-py direct)",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "pre-THP opening sequence / run_all",
        "broker_access_capability": "TradingClient.submit_order (un gated)",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED — use run_scheduled_paper_cycle",
        "approved_alternative": "python -m services.run_scheduled_paper_cycle",
    },
    "scripts/auto_trade.py": {
        "purpose": "Legacy live signal auto-trader",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "historical live trading",
        "broker_access_capability": "REST api.submit_order",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED",
        "approved_alternative": "python -m services.execute_trades --execute",
    },
    "scripts/auto_exit.py": {
        "purpose": "Legacy signal-driven exit script",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "historical exit automation",
        "broker_access_capability": "REST api.submit_order",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED",
        "approved_alternative": "python -m services.manage_positions",
    },
    "scripts/execute_trade.py": {
        "purpose": "Legacy single-order REST script",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "manual one-off orders",
        "broker_access_capability": "POST /v2/orders via requests",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED",
        "approved_alternative": "pages/manual_order_desk.py via view_results",
    },
    "alpaca_client.py": {
        "purpose": "Legacy Alpaca REST helper",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "imported by legacy scripts",
        "broker_access_capability": "POST /v2/orders via requests",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED — use services.broker_alpaca",
        "approved_alternative": "services.broker_alpaca.AlpacaBroker",
    },
    "ui/pages/Order_Desk.py": {
        "purpose": "Legacy Streamlit order desk (direct REST)",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "ui Order_Desk page",
        "broker_access_capability": "api_post /v2/orders",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED — use pages/manual_order_desk.py",
        "approved_alternative": "pages/manual_order_desk.py",
    },
    "scripts/postfill_attach_oco.py": {
        "purpose": "Legacy post-fill OCO attachment",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "manual OCO attachment",
        "broker_access_capability": "requests POST /v2/orders",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED — use position_protector",
        "approved_alternative": "services.position_protector",
    },
    "scripts/oco_watch.py": {
        "purpose": "Legacy OCO watch loop",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "background OCO attachment",
        "broker_access_capability": "requests POST /v2/orders",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED",
        "approved_alternative": "services.position_protector",
    },
    "scripts/attach_oco.py": {
        "purpose": "Legacy OCO attachment utility",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "manual OCO",
        "broker_access_capability": "requests POST /v2/orders",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED",
        "approved_alternative": "services.position_protector",
    },
    "scripts/chase_limit.py": {
        "purpose": "Legacy limit chase utility",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "manual limit repricing",
        "broker_access_capability": "requests POST /v2/orders",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED — use reprice_open_orders",
        "approved_alternative": "services.reprice_open_orders",
    },
    "scripts/attach_brackets_now.py": {
        "purpose": "Legacy bracket attachment utility",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "manual bracket attachment",
        "broker_access_capability": "requests POST /v2/orders",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED",
        "approved_alternative": "services.position_protector",
    },
    "scripts/opening_sequence.py": {
        "purpose": "Legacy market-open sequence (called auto_execute_signals)",
        "status": "DEPRECATED",
        "owner": "legacy",
        "last_known_operational_use": "app.py opening sequence button",
        "broker_access_capability": "indirect via deprecated auto_execute",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "RETIRED — use run_scheduled_paper_cycle",
        "approved_alternative": "python -m services.run_scheduled_paper_cycle",
    },
    # --- ARCHIVED ---
    "services/execute_trades.recovered.py": {
        "purpose": "Recovered backup of execute_trades (historical)",
        "status": "ARCHIVED",
        "owner": "legacy",
        "last_known_operational_use": "recovery reference only",
        "broker_access_capability": "unknown / historical",
        "execution_gate_compliance": "NO",
        "governance_compliance": "NO",
        "recommended_action": "ARCHIVE — reference only",
        "approved_alternative": "services/execute_trades.py",
    },
}


def get_path_status(path: str | Path) -> str:
    rel = normalize_path(path)
    entry = EXECUTION_PATHS.get(rel)
    return str(entry.get("status", "UNKNOWN")).upper() if entry else "UNKNOWN"


def normalize_path(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def path_has_deprecation_guard(full_path: Path) -> bool:
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    return "exit_if_deprecated" in text or "execution_path_guard" in text


def path_has_direct_order_post(full_path: Path) -> bool:
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    patterns = (
        r'POST.*["\']/v2/orders',
        r'["\']/v2/orders["\'].*POST',
        r"requests\.post\([^)]*v2/orders",
        r'api_post\(["\']/v2/orders',
        r'self\._post\(["\']/v2/orders',
    )
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def path_uses_gated_broker(full_path: Path) -> bool:
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    return (
        "AlpacaBroker" in text
        and "broker.submit_order" in text
        and not path_has_direct_order_post(full_path)
    )


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


def _verify_runtime(path: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    full = ROOT / path.replace("/", os.sep)
    status = str(entry.get("status", "")).upper()
    exists = full.is_file()
    guarded = path_has_deprecation_guard(full) if exists else False
    direct_post = path_has_direct_order_post(full) if exists else False
    gated_broker = path_uses_gated_broker(full) if exists else False
    has_gate_ref = False
    if exists:
        text = full.read_text(encoding="utf-8", errors="ignore")
        has_gate_ref = (
            "enforce_broker_submission_gate" in text or "evaluate_governed_execution" in text
        )

    protected = False
    risk = "NONE"
    if status == "ACTIVE":
        catalog_compliant = str(entry.get("execution_gate_compliance", "")).upper() == "YES"
        if path in ("services/broker_alpaca.py", "services/triton_execution_gate.py"):
            protected = True
        elif direct_post and path != "services/broker_alpaca.py":
            protected = False
            risk = "DIRECT_POST"
        elif gated_broker or (has_gate_ref and not direct_post):
            protected = True
        elif path in ORCHESTRATOR_PATHS and not direct_post:
            protected = True
        elif catalog_compliant and not direct_post:
            protected = True
        else:
            risk = "REVIEW"
    elif status in ("DEPRECATED", "ARCHIVED"):
        protected = guarded
        if not guarded and exists:
            risk = "UNPROTECTED_DEPRECATED"
        else:
            risk = "RESOLVED"

    return {
        "exists": exists,
        "deprecation_guard": guarded,
        "direct_order_post": direct_post,
        "uses_gated_broker": gated_broker,
        "has_gate_reference": has_gate_ref,
        "runtime_protected": protected,
        "residual_risk": risk,
    }


def build_registry_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path, meta in sorted(EXECUTION_PATHS.items()):
        runtime = _verify_runtime(path, meta)
        row = {
            "file_name": path,
            **{k: meta.get(k, "") for k in REGISTRY_FIELDS if k != "file_name"},
            **runtime,
        }
        rows.append(row)
    return rows


def build_cleanup_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_status: Dict[str, int] = {"ACTIVE": 0, "DEPRECATED": 0, "ARCHIVED": 0}
    authority_conflicts: List[str] = []
    bypass_risks: List[str] = []

    for row in rows:
        st = str(row.get("status", "")).upper()
        by_status[st] = by_status.get(st, 0) + 1
        if row.get("residual_risk") == "UNPROTECTED_DEPRECATED":
            bypass_risks.append(row["file_name"])
        if (
            st == "ACTIVE"
            and row.get("direct_order_post")
            and row["file_name"] != "services/broker_alpaca.py"
        ):
            authority_conflicts.append(row["file_name"])
        elif st == "ACTIVE" and not row.get("runtime_protected"):
            authority_conflicts.append(row["file_name"])

    unprotected_deprecated = [
        r["file_name"]
        for r in rows
        if str(r.get("status", "")).upper() in ("DEPRECATED", "ARCHIVED")
        and not r.get("deprecation_guard")
    ]

    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H003",
        "total_execution_paths_discovered": len(rows),
        "active_paths": by_status.get("ACTIVE", 0),
        "deprecated_paths": by_status.get("DEPRECATED", 0),
        "archived_paths": by_status.get("ARCHIVED", 0),
        "official_capital_chain": OFFICIAL_CHAIN,
        "capital_authority_consolidated": len(authority_conflicts) == 0
        and len(unprotected_deprecated) == 0
        and len(bypass_risks) == 0,
        "remaining_authority_conflicts": authority_conflicts,
        "remaining_bypass_risks": bypass_risks + unprotected_deprecated,
        "unprotected_deprecated_paths": unprotected_deprecated,
        "summary": {
            "single_institutional_chain": len(authority_conflicts) == 0,
            "all_deprecated_protected": len(unprotected_deprecated) == 0,
            "active_paths_gate_compliant": all(
                r.get("runtime_protected")
                for r in rows
                if str(r.get("status", "")).upper() == "ACTIVE"
            ),
        },
    }


def run_registry() -> Dict[str, Any]:
    rows = build_registry_rows()
    cleanup = build_cleanup_report(rows)

    registry_payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H003",
        "official_capital_chain": OFFICIAL_CHAIN,
        "paths": [{k: r.get(k, "") for k in REGISTRY_FIELDS} for r in rows],
        "runtime_verification": [
            {
                "file_name": r["file_name"],
                "status": r["status"],
                "deprecation_guard": r["deprecation_guard"],
                "direct_order_post": r["direct_order_post"],
                "uses_gated_broker": r["uses_gated_broker"],
                "runtime_protected": r["runtime_protected"],
                "residual_risk": r["residual_risk"],
            }
            for r in rows
        ],
        "summary": cleanup["summary"],
    }

    _atomic_write_json(REGISTRY_JSON, registry_payload)
    _write_csv(
        REGISTRY_CSV, [{k: r.get(k, "") for k in REGISTRY_FIELDS} for r in rows], REGISTRY_FIELDS
    )
    _atomic_write_json(CLEANUP_REPORT_JSON, cleanup)

    return {"registry": registry_payload, "cleanup": cleanup, "rows": rows}


if __name__ == "__main__":
    result = run_registry()
    cleanup = result["cleanup"]
    print(f"Registry written: {REGISTRY_JSON}")
    print(f"Cleanup report: {CLEANUP_REPORT_JSON}")
    print(
        f"Paths: {cleanup['total_execution_paths_discovered']} "
        f"(ACTIVE={cleanup['active_paths']} "
        f"DEPRECATED={cleanup['deprecated_paths']} "
        f"ARCHIVED={cleanup['archived_paths']})"
    )
    print(f"Capital authority consolidated: {cleanup['capital_authority_consolidated']}")
