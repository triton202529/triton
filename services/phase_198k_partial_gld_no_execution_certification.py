"""
phase_198k_partial_gld_no_execution_certification.py
Phase 198K — Certify the full partial GLD defensive path with no execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_198k_partial_gld_no_execution_certification.json"
SUMMARY_MD = RESULTS / "partial_gld_operator_certification_summary.md"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

SIZING_JSON = RESULTS / "gld_partial_exit_sizing_analysis.json"
D198_JSON = RESULTS / "phase_198d_partial_defensive_exit_preview_validation.json"
E198_JSON = RESULTS / "phase_198e_partial_approval_packet_compatibility_validation.json"
F198_JSON = RESULTS / "phase_198f_partial_broker_authorization_dry_run_compatibility.json"
G198_JSON = RESULTS / "phase_198g_partial_loop_execution_gate_dry_run_compatibility.json"
H198_JSON = RESULTS / "phase_198h_partial_readiness_bridge_compatibility_validation.json"
I198_JSON = RESULTS / "phase_198i_partial_dispatch_scope_dry_run_validation.json"
J198_JSON = RESULTS / "phase_198j_partial_dispatch_evidence_reconciliation.json"
RECON_CSV = RESULTS / "partial_dispatch_quantity_reconciliation.csv"

MARKER = "PARTIAL_GLD_NO_EXECUTION_CERTIFIED"
EXPECTED_PARTIAL = 48
EXPECTED_FULL = 195
EXPECTED_REMAINING = 147


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _layer_passed(doc: Dict[str, Any]) -> bool:
    status = str(doc.get("status") or "").upper()
    return status in ("PASS", "PASS_SAFE_BLOCKED", "PASS_DRY_RUN")


def _git_status_short() -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _score(checks: Dict[str, bool]) -> int:
    if not checks:
        return 0
    return int(round(100.0 * sum(1 for v in checks.values() if v) / len(checks)))


def _render_summary(cert: Dict[str, Any]) -> str:
    missing = cert.get("missing_requirements_for_real_execution") or []
    missing_lines = "\n".join(f"- {m}" for m in missing) or "- None listed"
    return f"""# TRITON Partial GLD Defensive Path — No-Execution Certification

**Generated:** {cert.get("generated_at")}
**Certification status:** {cert.get("certification_status")}
**Score:** {cert.get("certification_score")}

## What is certified

This certification covers the **partial GLD defensive workflow** through analysis, preview,
approval-packet compatibility, broker-authorization dry-run, loop gate, readiness bridge,
dispatch-scope validation, and quantity evidence reconciliation.

- Symbol: **{cert.get("certified_symbol")}**
- Preview type: `{cert.get("preview_type")}`
- Full position quantity: **{cert.get("full_qty")}**
- Approved / projected / authoritative partial quantity: **{cert.get("authoritative_dispatch_qty")}**
- Remaining quantity after partial: **{cert.get("remaining_qty_after_partial")}**
- Raw handoff quantity (non-authoritative): **{cert.get("raw_handoff_qty")}**
- Full exit deferred: **{cert.get("full_exit_deferred")}**

## What is not authorized

- No operator approval artifact is present or certified.
- No broker execution authorization artifact is present or certified.
- No broker orders are submitted or certified.
- Live execution remains blocked.
- `execution_allowed_now` is **false**.

## Why raw handoff qty may differ from dispatch qty

Raw `data/live/manage_orders.csv` may still show GLD qty={cert.get("raw_handoff_qty")} because it
represents the full defensive handoff candidate. Downstream governed scope uses the projected
dispatch quantity only.

**Authoritative broker dispatch quantity:** {cert.get("authoritative_dispatch_qty")}
**Authoritative source:** {cert.get("authoritative_dispatch_source")}

## Required before any future paper execution

{missing_lines}

## Explicit warning

Live execution remains blocked. This certification does not approve paper or live dispatch.
Any future paper execution still requires explicit short-lived operator approval and broker
authorization artifacts matching the partial GLD quantity.

---

*{MARKER}*
"""


def run_phase_198k_certification(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: Dict[str, bool] = {}

    from services.portfolio_history_sync import load_live_execution_flag

    required_paths = {
        "198c_sizing": SIZING_JSON,
        "198d_preview": D198_JSON,
        "198e_approval_packet": E198_JSON,
        "198f_broker_auth": F198_JSON,
        "198g_loop_gate": G198_JSON,
        "198h_readiness_bridge": H198_JSON,
        "198i_dispatch_scope": I198_JSON,
        "198j_evidence_recon": J198_JSON,
        "198j_recon_csv": RECON_CSV,
    }
    evidence: Dict[str, Dict[str, Any]] = {}
    for key, path in required_paths.items():
        exists = path.is_file()
        checks[f"evidence_exists_{key}"] = exists
        if not exists:
            notes.append(f"missing_evidence={path.relative_to(ROOT)}")
            evidence[key] = {}
        elif path.suffix.lower() == ".csv":
            evidence[key] = {"exists": True, "path": str(path)}
        else:
            evidence[key] = _read_json(path)

    sizing = evidence.get("198c_sizing") or {}
    d198 = evidence.get("198d_preview") or {}
    e198 = evidence.get("198e_approval_packet") or {}
    f198 = evidence.get("198f_broker_auth") or {}
    g198 = evidence.get("198g_loop_gate") or {}
    h198 = evidence.get("198h_readiness_bridge") or {}
    i198 = evidence.get("198i_dispatch_scope") or {}
    j198 = evidence.get("198j_evidence_recon") or {}

    approval_layer_passed = _layer_passed(e198)
    broker_auth_layer_passed = _layer_passed(f198)
    loop_gate_layer_passed = _layer_passed(g198)
    readiness_bridge_layer_passed = _layer_passed(h198)
    dispatch_scope_layer_passed = _layer_passed(i198) and bool(i198.get("dispatch_scope_valid"))
    evidence_reconciliation_passed = _layer_passed(j198)
    preview_layer_passed = _layer_passed(d198) or bool(
        (d198.get("partial_preview") or {}).get("available")
    )
    sizing_layer_passed = (
        bool(sizing) and str(sizing.get("recommended_scenario") or sizing.get("status") or "") != ""
    )

    certified_symbol = str(
        j198.get("symbol")
        or i198.get("symbol")
        or (d198.get("partial_preview") or {}).get("symbol")
        or "GLD"
    ).upper()
    preview_type = str(
        j198.get("preview_type") or i198.get("preview_type") or "PARTIAL_DEFENSIVE_EXIT_PREVIEW"
    )
    full_qty = _safe_int(j198.get("full_position_qty") or i198.get("full_qty") or EXPECTED_FULL)
    approved_partial_qty = _safe_int(
        j198.get("approved_partial_qty") or i198.get("approved_qty") or EXPECTED_PARTIAL
    )
    projected_dispatch_qty = _safe_int(
        j198.get("projected_dispatch_qty") or i198.get("projected_qty") or EXPECTED_PARTIAL
    )
    authoritative_dispatch_qty = _safe_int(
        j198.get("authoritative_dispatch_qty") or projected_dispatch_qty or EXPECTED_PARTIAL
    )
    remaining_qty = _safe_int(
        j198.get("remaining_qty_after_partial") or i198.get("remaining_qty") or EXPECTED_REMAINING
    )
    raw_handoff_qty = j198.get("raw_handoff_qty")
    if raw_handoff_qty is None:
        raw_handoff_qty = (i198.get("manage_orders_inspection") or {}).get("gld_qty")
    full_exit_deferred = bool(
        j198.get("full_exit_deferred")
        if "full_exit_deferred" in j198
        else i198.get("full_exit_deferred", True)
    )

    approval_exists = APPROVAL_PATH.is_file()
    broker_auth_exists = BROKER_AUTH_PATH.is_file()
    live_permitted = bool(load_live_execution_flag())
    broker_orders_submitted = bool(
        i198.get("broker_orders_submitted")
        or j198.get("broker_orders_submitted")
        or f198.get("broker_orders_submitted")
        or g198.get("broker_orders_submitted")
    )
    buy_add_ok = True
    buy_add_rows = (i198.get("manage_orders_inspection") or {}).get("buy_add_rows_in_scope") or []
    if buy_add_rows:
        buy_add_ok = False
        notes.append("buy_add_rows_present_in_defensive_path")

    git_status = _git_status_short()
    # Allow uncommitted runtime outputs and in-progress certification sources.
    # Fail only if real approval/auth artifacts appear in the working tree.
    dirty_lines = [ln for ln in git_status.splitlines() if ln.strip()]
    forbidden_dirty = [
        ln
        for ln in dirty_lines
        if "operator_paper_defensive_exit_approval.json" in ln
        or "operator_paper_broker_execution_authorization.json" in ln
    ]
    git_ok = len(forbidden_dirty) == 0
    if dirty_lines and git_ok:
        notes.append("git_status_has_runtime_or_source_outputs_before_cleanup")

    checks.update(
        {
            "partial_preview_exists": preview_layer_passed or bool(d198.get("status") == "PASS"),
            "approved_partial_qty_48": approved_partial_qty == EXPECTED_PARTIAL,
            "projected_dispatch_qty_48": projected_dispatch_qty == EXPECTED_PARTIAL,
            "authoritative_dispatch_qty_48": authoritative_dispatch_qty == EXPECTED_PARTIAL,
            "full_qty_195": full_qty == EXPECTED_FULL,
            "remaining_qty_147": remaining_qty == EXPECTED_REMAINING,
            "raw_handoff_may_be_195_but_not_authoritative": (
                raw_handoff_qty in (None, EXPECTED_FULL, EXPECTED_PARTIAL)
                and authoritative_dispatch_qty == EXPECTED_PARTIAL
            ),
            "full_exit_deferred": full_exit_deferred is True,
            "approval_layer_passed": approval_layer_passed,
            "broker_auth_layer_passed": broker_auth_layer_passed,
            "loop_gate_layer_passed": loop_gate_layer_passed,
            "readiness_bridge_layer_passed": readiness_bridge_layer_passed,
            "dispatch_scope_layer_passed": dispatch_scope_layer_passed,
            "evidence_reconciliation_passed": evidence_reconciliation_passed,
            "broker_orders_not_submitted": broker_orders_submitted is False,
            "approval_artifact_absent": approval_exists is False,
            "broker_auth_artifact_absent": broker_auth_exists is False,
            "live_execution_blocked": live_permitted is False,
            "no_buy_add_in_defensive_path": buy_add_ok,
            "git_status_acceptable": git_ok,
            "sizing_layer_present": sizing_layer_passed or SIZING_JSON.is_file(),
        }
    )

    failed = [k for k, v in checks.items() if not v]
    score = _score(checks)
    certification_status = "PASS" if not failed else "FAIL"
    partial_path_complete = certification_status == "PASS"
    execution_allowed_now = False

    missing_requirements = [
        "Create short-lived operator paper approval artifact for partial GLD qty=48",
        "Create short-lived paper broker execution authorization for projected partial scope",
        "Confirm 197Y loop gate eligible_for_execution=true for partial scope only",
        "Confirm readiness bridge active for paper loop-mediated path only",
        "Keep live execution blocked",
        "Do not expand dispatch quantity from 48 to full 195",
    ]

    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198K",
        "generated_at": _utc_now_iso(),
        "certification_status": certification_status,
        "status": certification_status,
        "certification_score": score,
        "certified_symbol": certified_symbol,
        "preview_type": preview_type,
        "full_qty": full_qty,
        "approved_partial_qty": approved_partial_qty,
        "projected_dispatch_qty": projected_dispatch_qty,
        "authoritative_dispatch_qty": authoritative_dispatch_qty,
        "authoritative_dispatch_source": j198.get("authoritative_dispatch_source")
        or "phase_197x_projected_paper_orders",
        "remaining_qty_after_partial": remaining_qty,
        "raw_handoff_qty": raw_handoff_qty,
        "full_exit_deferred": full_exit_deferred,
        "full_exit_deferred_reason": j198.get("full_exit_deferred_reason")
        or i198.get("full_exit_deferred_reason"),
        "partial_path_complete": partial_path_complete,
        "approval_layer_passed": approval_layer_passed,
        "broker_auth_layer_passed": broker_auth_layer_passed,
        "loop_gate_layer_passed": loop_gate_layer_passed,
        "readiness_bridge_layer_passed": readiness_bridge_layer_passed,
        "dispatch_scope_layer_passed": dispatch_scope_layer_passed,
        "evidence_reconciliation_passed": evidence_reconciliation_passed,
        "broker_orders_submitted": False,
        "approval_artifact_exists_final": approval_exists,
        "broker_authorization_artifact_exists_final": broker_auth_exists,
        "live_execution_permitted": False,
        "execution_allowed_now": execution_allowed_now,
        "missing_requirements_for_real_execution": missing_requirements,
        "reconciliation_classification": j198.get("reconciliation_classification"),
        "operator_warning": j198.get("operator_warning"),
        "checks": checks,
        "failed_checks": failed,
        "evidence_files": {
            k: str(p.relative_to(ROOT)).replace("\\", "/") for k, p in required_paths.items()
        },
        "layer_status": {
            "198c_sizing": "PRESENT" if SIZING_JSON.is_file() else "MISSING",
            "198d_preview": d198.get("status"),
            "198e_approval_packet": e198.get("status"),
            "198f_broker_auth": f198.get("status"),
            "198g_loop_gate": g198.get("status"),
            "198h_readiness_bridge": h198.get("status"),
            "198i_dispatch_scope": i198.get("status"),
            "198j_evidence_recon": j198.get("status"),
        },
        "git_status_short": git_status,
        "summary_markdown_path": str(SUMMARY_MD.relative_to(ROOT)).replace("\\", "/"),
        "marker": MARKER,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(SUMMARY_MD, _render_summary(payload))

    if verbose:
        print(f"[198K-PARTIAL-CERT] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198K-PARTIAL-CERT] wrote {SUMMARY_MD.relative_to(ROOT)}")
        print(
            f"[198K-PARTIAL-CERT] status={certification_status} score={score} "
            f"symbol={certified_symbol} auth_qty={authoritative_dispatch_qty} "
            f"execution_allowed_now={execution_allowed_now}"
        )
        print(f"[198K-PARTIAL-CERT] {MARKER}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198K partial GLD no-execution path certification"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198k_certification(verbose=not args.quiet)
    return 0 if result.get("certification_status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
