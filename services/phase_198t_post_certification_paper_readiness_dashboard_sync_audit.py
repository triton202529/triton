"""
phase_198t_post_certification_paper_readiness_dashboard_sync_audit.py
Phase 198T — Post-certification paper readiness and dashboard sync audit.

Read-only audit. No broker orders, approvals, or live enablement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"

OUTPUT_JSON = RESULTS / "phase_198t_post_certification_paper_readiness_dashboard_sync_audit.json"
OUTPUT_MD = RESULTS / "phase_198t_post_certification_paper_readiness_dashboard_sync_audit.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
PAPER_ARM_PATH = LIVE / "paper_arm_confirm.json"
LIVE_CONFIRM_PATH = LIVE / "live_confirm.json"

PAPER_CERT_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)

ARTIFACT_PATHS = {
    "certification": RESULTS / "capital_preservation_certification.json",
    "evaluation": RESULTS / "protective_action_evaluation.json",
    "governor": RESULTS / "capital_preservation_governor.json",
    "readiness": RESULTS / "execution_readiness.json",
    "governance_authorization": RESULTS / "governance_authorization.json",
    "executive_risk_summary": RESULTS / "executive_risk_summary.json",
    "watchdog_alerts": RESULTS / "watchdog_alerts.json",
    "watchdog_status": RESULTS / "watchdog_status.json",
    "positions_snapshot": RESULTS / "positions_snapshot.csv",
    "open_orders_snapshot": RESULTS / "open_orders_snapshot.csv",
    "recent_orders": RESULTS / "recent_orders.csv",
    "allocation_summary": RESULTS / "portfolio_allocation_summary.json",
    "allocation_recommendations": RESULTS / "portfolio_allocation_recommendations.csv",
    "phase_198s": RESULTS / "phase_198s_refresh_exposure_trial_weighted_drawdown_calibration.json",
    "phase_198l": RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json",
    "phase_197y": RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json",
    "phase_197z": RESULTS / "phase_197z_readiness_bridge_validation.json",
    "cpi": RESULTS / "capital_preservation_intelligence.json",
    "cert_fp_elimination": RESULTS / "certification_false_positive_elimination.json",
}

MARKER = "POST_CERTIFICATION_PAPER_READINESS_DASHBOARD_SYNC_AUDIT_ONLY"
GLD_SYMBOL = "GLD"
GLD_TARGET_WEIGHT = 0.20


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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, len(list(csv.reader(f))) - 1)
    except Exception:
        return 0


def _gld_from_positions(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"qty": 0, "weight": 0.0}
    total = 0.0
    gld: Dict[str, Any] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mv = abs(_safe_float(row.get("market_value") or row.get("value")))
            total += mv
            sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            if sym == GLD_SYMBOL:
                gld = row
    if not gld:
        return {"qty": 0, "weight": 0.0, "total_invested": total}
    qty = _safe_float(gld.get("qty") or gld.get("qty_available"))
    mv = abs(_safe_float(gld.get("market_value") or gld.get("value")))
    weight = mv / total if total > 0 else 0.0
    return {
        "qty": int(qty),
        "weight": round(weight, 6),
        "market_value": mv,
        "total_invested": total,
        "unrealized_pl": _safe_float(gld.get("unrealized_pl")),
        "unrealized_plpc": _safe_float(gld.get("unrealized_plpc")),
    }


def _allocation_counts(summary: Dict[str, Any], rec_csv: Path) -> Dict[str, Any]:
    open_new = int(summary.get("open_new_candidates") or 0)
    add = int(summary.get("add_candidates") or 0)
    exit_n = int(summary.get("exit_candidates") or 0)
    trim = int(summary.get("trim_candidates") or 0)

    top: List[Dict[str, Any]] = []
    if rec_csv.is_file():
        try:
            with rec_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ranked = sorted(
                rows,
                key=lambda r: _safe_float(r.get("allocation_score"), 0.0),
                reverse=True,
            )
            for row in ranked[:5]:
                action = str(row.get("recommended_action") or "")
                if action in ("OPEN_NEW", "ADD", "WATCH", "TRIM", "EXIT"):
                    top.append(
                        {
                            "symbol": row.get("ticker") or row.get("symbol"),
                            "recommended_action": action,
                            "allocation_score": _safe_float(row.get("allocation_score")),
                            "reason": row.get("reason"),
                        }
                    )
        except Exception:
            pass

    if not open_new and rec_csv.is_file():
        try:
            with rec_csv.open(newline="", encoding="utf-8") as f:
                open_new = sum(
                    1 for r in csv.DictReader(f) if r.get("recommended_action") == "OPEN_NEW"
                )
        except Exception:
            pass
    if not add and rec_csv.is_file():
        try:
            with rec_csv.open(newline="", encoding="utf-8") as f:
                add = sum(1 for r in csv.DictReader(f) if r.get("recommended_action") == "ADD")
        except Exception:
            pass

    return {
        "open_new_count": open_new,
        "add_count": add,
        "exit_count": exit_n,
        "trim_count": trim,
        "top_opportunities": top[:5],
    }


def _drawdown_alert_state(watchdog: Dict[str, Any], gld: Dict[str, Any]) -> Tuple[bool, bool]:
    alert = next(
        (
            a
            for a in (watchdog.get("active_alerts") or [])
            if str(a.get("alert_type") or "") == "EXCESS_POSITION_DRAWDOWN"
            and str(a.get("subject") or "").upper() == GLD_SYMBOL
        ),
        None,
    )
    if not alert:
        return False, False
    alert_pct = _safe_float((alert.get("details") or {}).get("unrealized_pl_pct"), 0.0)
    broker_pct = _safe_float(gld.get("unrealized_plpc"), 0.0) * 100.0
    supported = abs(alert_pct - broker_pct) <= 1.5 and broker_pct <= -10.0
    return True, supported


def _stale_gld_preview(phase_197y: Dict[str, Any], gld_qty: int) -> bool:
    """True when stale preview metadata could mislead runtime/dashboard, not historical audit fields."""
    if phase_197y.get("target_already_completed") and not phase_197y.get("eligible_for_execution"):
        return False
    if (
        phase_197y.get("defensive_preview_invalidated")
        and _safe_int(phase_197y.get("eligible_qty")) == 0
    ):
        return False
    preview = phase_197y.get("defensive_preview_broker_state") or {}
    preview_remaining = _safe_int(preview.get("preview_remaining_qty"))
    broker_qty = _safe_int(preview.get("broker_qty") or gld_qty)
    if (
        preview_remaining > broker_qty
        and broker_qty > 0
        and phase_197y.get("eligible_for_execution")
    ):
        return True
    if _safe_int(phase_197y.get("stale_preview_original_qty")) and phase_197y.get(
        "eligible_for_execution"
    ):
        return True
    return False


def _stale_concentration_evaluation(evaluation: Dict[str, Any]) -> bool:
    conc = next(
        (
            e
            for e in (evaluation.get("evaluations") or [])
            if e.get("trial_id") == "paper_concentration_reduction"
        ),
        {},
    )
    if not conc:
        return False
    if str(conc.get("evaluation") or "").upper() == "NEGATIVE":
        return True
    if conc.get("evaluation_mode") not in ("COMPLETED_BROKER_VERIFIED",):
        if not evaluation.get("concentration_reduction_completed"):
            return True
    return False


def _display_gate_bits() -> Tuple[Dict[str, bool], List[str]]:
    from services.dashboard_status_sync import build_dashboard_health_classification
    from services.live_gate import compute_display_gates

    live_ready, _, gate_bits, _ = compute_display_gates()
    dash = build_dashboard_health_classification(
        live_gate_bits=gate_bits,
        live_ready=live_ready,
    )
    blockers = list(dash.get("execution_block_reasons") or [])
    return gate_bits, blockers, dash


def _dashboard_sync(cert_status: str, failed: List[str]) -> Tuple[str, bool]:
    try:
        _, _, dash = _display_gate_bits()
    except Exception as exc:
        return f"ERROR:{exc}", False

    cert_norm = str(dash.get("certification_status") or "")
    blockers = list(dash.get("execution_block_reasons") or [])
    cert_layer_pass = cert_status in PAPER_CERT_STATUSES and len(failed) == 0
    stale = False
    if cert_layer_pass and "NOT_CERTIFIED" in blockers:
        stale = True
    if cert_layer_pass and cert_norm == "NOT_CERTIFIED":
        stale = True
    if cert_layer_pass and cert_norm == "CERTIFIED":
        return "SYNCED", False
    if cert_layer_pass and not stale:
        return "SYNCED", False
    if stale:
        return "STALE_CERTIFICATION_DISPLAY", True
    return f"REVIEW:{cert_norm}", stale


def _watchdog_sync(cert_status: str, cert_fp: Dict[str, Any]) -> Tuple[str, bool]:
    stale = False
    fp_status = str(cert_fp.get("certification_status") or "")
    if cert_status in PAPER_CERT_STATUSES and fp_status == "NOT_CERTIFIED":
        stale = True
    wd = _load_json(ARTIFACT_PATHS["watchdog_status"])
    if wd.get("certification_status") == "NOT_CERTIFIED" and cert_status in PAPER_CERT_STATUSES:
        stale = True
    if stale:
        return "STALE_NOT_CERTIFIED_ARTIFACT", True
    return "SYNCED", False


def _paper_entry_assessment(
    *,
    cert_status: str,
    failed: List[str],
    arm: Dict[str, Any],
    alloc: Dict[str, Any],
    dash_blockers: List[str],
    live_allowed: bool,
) -> Tuple[bool, str]:
    cert_ok = cert_status in PAPER_CERT_STATUSES and len(failed) == 0
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    paper_entry_armed = bool(arm.get("paper_entry_armed"))
    open_new = int(alloc.get("open_new_count") or 0)
    add = int(alloc.get("add_count") or 0)
    alloc_entries = open_new + add

    if live_allowed:
        return False, "live_execution_not_blocked"
    if not cert_ok:
        return False, "certification_not_ready"
    if "NOT_CERTIFIED" in dash_blockers:
        return False, "dashboard_still_shows_not_certified_block"
    if alloc_entries == 0:
        return False, "allocation_blocked_no_open_new_or_add_candidates"
    if not paper_arm_valid:
        return False, "paper_arm_confirm_not_valid"
    if not paper_entry_armed:
        return False, "paper_entry_not_armed"
    return True, (
        f"eligible_via_controlled_paper_entry_lane "
        f"(open_new={open_new}, add={add}); allocation is authority; no manual bypass"
    )


def _render_markdown(payload: Dict[str, Any]) -> str:
    audit = payload.get("audit") or {}
    sync = payload.get("sync") or {}
    safety = payload.get("safety") or {}
    alloc = audit.get("allocation_top_opportunities") or []
    alloc_lines = (
        "\n".join(
            f"- {o.get('symbol')}: {o.get('recommended_action')} (score={o.get('allocation_score')})"
            for o in alloc
        )
        or "- None in top opportunities"
    )

    return f"""# TRITON Phase 198T — Post-Certification Paper Readiness Dashboard Sync Audit

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}

## Executive summary

{payload.get("executive_summary")}

## Certification state

| Field | Value |
|-------|-------|
| Status | {audit.get("certification_status")} |
| Score | {audit.get("certification_score")} |
| Forced | {audit.get("certification_forced")} |
| Failed requirements | {audit.get("failed_requirements")} |
| Evaluation avg | {audit.get("evaluation_average_effectiveness")} |
| Beneficial / Negative | {audit.get("evaluation_beneficial_count")} / {audit.get("evaluation_negative_count")} |

## Runtime sync

| Layer | Status |
|-------|--------|
| Dashboard certification | {audit.get("dashboard_certification_sync_status")} |
| Watchdog certification | {audit.get("watchdog_certification_sync_status")} |
| Stale NOT_CERTIFIED display | {sync.get("stale_not_certified_detected")} |
| Stale GLD preview metadata | {sync.get("stale_gld_preview_detected")} |

## Paper entry posture

| Field | Value |
|-------|-------|
| Normal paper entry possible | {audit.get("normal_paper_entry_possible_now")} |
| Reason | {audit.get("reason_normal_paper_entry_blocked_or_allowed")} |
| Paper arm valid | {audit.get("paper_arm_valid")} |
| Paper entry armed | {audit.get("paper_entry_armed")} |
| Allocation open_new / add | {audit.get("allocation_open_new_count")} / {audit.get("allocation_add_count")} |

## GLD and drawdown monitoring

| Field | Value |
|-------|-------|
| GLD qty / weight | {audit.get("gld_qty")} / {audit.get("gld_weight_pct")}% |
| Target satisfied | {audit.get("gld_target_satisfied")} |
| Residual drawdown alert | {audit.get("residual_drawdown_alert_active")} |
| Alert broker-supported | {audit.get("residual_drawdown_alert_supported")} |

## Allocation top opportunities

{alloc_lines}

## Recommended runtime posture

{audit.get("recommended_runtime_posture")}

## Safety

| Check | Value |
|-------|-------|
| broker_orders_submitted | {safety.get("broker_orders_submitted")} |
| approval artifact | {audit.get("approval_artifact_present")} |
| broker auth artifact | {audit.get("broker_auth_artifact_present")} |
| live execution blocked | {audit.get("live_execution_blocked")} |

---

*{MARKER}*
"""


def run_phase_198t_audit(*, verbose: bool = True) -> Dict[str, Any]:
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    artifacts = {k: _load_json(p) for k, p in ARTIFACT_PATHS.items() if p.suffix == ".json"}
    cert = artifacts.get("certification") or {}
    evaluation = artifacts.get("evaluation") or {}
    governor = artifacts.get("governor") or {}
    readiness = artifacts.get("readiness") or {}
    gov_auth = artifacts.get("governance_authorization") or {}
    cpi = artifacts.get("cpi") or {}
    phase_198s = artifacts.get("phase_198s") or {}
    phase_198l = artifacts.get("phase_198l") or {}
    phase_197y = artifacts.get("phase_197y") or {}
    phase_197z = artifacts.get("phase_197z") or {}
    watchdog = artifacts.get("watchdog_alerts") or {}
    cert_fp = artifacts.get("cert_fp_elimination") or {}

    gld = _gld_from_positions(ARTIFACT_PATHS["positions_snapshot"])
    alloc = _allocation_counts(
        artifacts.get("allocation_summary") or {},
        ARTIFACT_PATHS["allocation_recommendations"],
    )
    open_orders = _count_csv_rows(ARTIFACT_PATHS["open_orders_snapshot"])
    positions_count = _count_csv_rows(ARTIFACT_PATHS["positions_snapshot"])

    arm = build_paper_arm_status(verbose_notes=False)
    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    live_flag = bool(load_live_execution_flag())
    live_readiness = bool(readiness.get("live_execution_permitted"))
    live_allowed = live_flag or live_readiness or bool(gov_auth.get("live_execution_permitted"))

    cert_status = str(cert.get("certification_status") or "")
    failed = cert.get("failed_requirements") or []
    cert_forced = bool((phase_198s.get("audit") or {}).get("certification_forced", False))

    dd_active, dd_supported = _drawdown_alert_state(watchdog, gld)
    stale_preview = _stale_gld_preview(phase_197y, int(gld.get("qty") or 0))
    stale_conc = _stale_concentration_evaluation(evaluation)

    dash_sync, dash_stale = _dashboard_sync(cert_status, failed)
    wd_sync, wd_stale = _watchdog_sync(cert_status, cert_fp)
    stale_not_certified = dash_stale or wd_stale

    try:
        _, dash_blockers, _ = _display_gate_bits()
    except Exception:
        dash_blockers = []

    normal_possible, entry_reason = _paper_entry_assessment(
        cert_status=cert_status,
        failed=failed,
        arm=arm,
        alloc=alloc,
        dash_blockers=dash_blockers,
        live_allowed=live_allowed,
    )

    gld_weight = float(gld.get("weight") or 0)
    gld_qty = int(gld.get("qty") or 0)
    gld_target = bool(
        phase_198l.get("target_already_satisfied") or gld_weight <= GLD_TARGET_WEIGHT + 0.005
    )

    cert_layer_pass = cert_status in PAPER_CERT_STATUSES and len(failed) == 0

    if live_allowed or approval_present or broker_auth_present:
        status = "FAIL"
    elif open_orders > 0 or gld_qty != 16 or gld_weight > GLD_TARGET_WEIGHT + 0.005:
        status = "REVIEW_REQUIRED"
    elif stale_not_certified:
        status = "REVIEW_REQUIRED"
    elif cert_layer_pass:
        status = "PASS"
    else:
        status = "REVIEW_REQUIRED"

    if cert_layer_pass and not live_allowed and not approval_present:
        posture = "PAPER_ONLY_CERTIFIED_MONITOR_DRAWDOWN" if dd_active else "PAPER_ONLY_CERTIFIED"
    elif stale_not_certified:
        posture = "REFRESH_DASHBOARD_WATCHDOG_CERTIFICATION_DISPLAY"
    else:
        posture = "REVIEW_CERTIFICATION_AND_GATES"

    audit_fields = {
        "certification_status": cert_status,
        "certification_score": cert.get("certification_score"),
        "certification_forced": cert_forced,
        "failed_requirements": failed,
        "evaluation_average_effectiveness": evaluation.get("average_effectiveness"),
        "evaluation_beneficial_count": evaluation.get("beneficial_count"),
        "evaluation_negative_count": evaluation.get("negative_count"),
        "capital_preservation_score": cpi.get("capital_preservation_score"),
        "capital_preservation_health_band": cpi.get("health_band"),
        "escalation_state": cpi.get("escalation_state")
        or governor.get("escalation_state")
        or readiness.get("governance_mode"),
        "governor_posture": governor.get("preservation_posture"),
        "execution_readiness_status": readiness.get("readiness_status"),
        "governance_authorized": bool(gov_auth.get("governance_authorized")),
        "paper_arm_valid": str(arm.get("status") or "") == "VALID",
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "live_execution_blocked": not live_allowed,
        "live_execution_allowed": live_allowed,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "open_orders_count": open_orders,
        "positions_count": positions_count,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_weight_pct": round(gld_weight * 100.0, 4),
        "gld_target_satisfied": gld_target,
        "residual_drawdown_alert_active": dd_active,
        "residual_drawdown_alert_supported": dd_supported,
        "allocation_open_new_count": alloc.get("open_new_count"),
        "allocation_add_count": alloc.get("add_count"),
        "allocation_exit_count": alloc.get("exit_count"),
        "allocation_trim_count": alloc.get("trim_count"),
        "allocation_top_opportunities": alloc.get("top_opportunities"),
        "normal_paper_entry_possible_now": normal_possible,
        "reason_normal_paper_entry_blocked_or_allowed": entry_reason,
        "dashboard_certification_sync_status": dash_sync,
        "watchdog_certification_sync_status": wd_sync,
        "stale_not_certified_detected": stale_not_certified,
        "stale_gld_preview_detected": stale_preview,
        "stale_gld_concentration_detected": stale_conc,
        "recommended_runtime_posture": posture,
    }

    checks = {
        "certification_certified_for_paper": cert_status == "CERTIFIED_FOR_PAPER_PROTECTION",
        "certification_not_forced": not cert_forced,
        "failed_requirements_empty": len(failed) == 0,
        "live_blocked": not live_allowed,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "open_orders_zero": open_orders == 0,
        "gld_qty_16": gld_qty == 16,
        "gld_weight_at_target": gld_weight <= GLD_TARGET_WEIGHT + 0.005,
        "no_broker_orders": True,
    }

    executive_summary = (
        f"Certification {cert_status} (score {cert.get('certification_score')}, forced={cert_forced}). "
        f"Dashboard sync={dash_sync}, watchdog sync={wd_sync}. "
        f"Paper entry possible={normal_possible} ({entry_reason}). "
        f"GLD {gld_qty}@{audit_fields['gld_weight_pct']}%, drawdown alert active={dd_active}."
    )

    payload: Dict[str, Any] = {
        "phase": "198T",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "executive_summary": executive_summary,
        "audit": audit_fields,
        "sync": {
            "stale_not_certified_detected": stale_not_certified,
            "stale_gld_preview_detected": stale_preview,
            "stale_gld_concentration_detected": stale_conc,
            "dashboard_execution_block_reasons": dash_blockers,
            "certification_false_positive_elimination_status": cert_fp.get("certification_status"),
            "phase_197y_defensive_preview_state": phase_197y.get("defensive_preview_state"),
            "phase_197z_bridge_active": bool(phase_197z.get("bridge_active")),
        },
        "paper_arm": {
            "status": arm.get("status"),
            "paper_entry_armed": arm.get("paper_entry_armed"),
            "paper_arm_confirm_present": PAPER_ARM_PATH.is_file(),
            "live_confirm_present": LIVE_CONFIRM_PATH.is_file(),
        },
        "safety": {
            "broker_orders_submitted": False,
            "approval_artifact_created": False,
            "broker_auth_artifact_created": False,
            "live_execution_modified": False,
        },
        "artifact_snapshot": {
            k: {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "present": p.is_file()}
            for k, p in ARTIFACT_PATHS.items()
        },
        "git_status_short": "",
        "git_diff_stat": "",
        "marker": MARKER,
    }

    try:
        payload["git_status_short"] = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        payload["git_diff_stat"] = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        pass

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(
            f"[198T] status={status} cert={cert_status} dash_sync={dash_sync} "
            f"paper_entry_possible={normal_possible} gld_qty={gld_qty}"
        )

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 198T post-certification sync audit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase_198t_audit(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    sys.exit(main())
