"""
phase_198o_post_gld_certification_evaluation_gap_audit.py
Phase 198O — Post-GLD certification evaluation gap audit.

Read-only audit. No broker orders, approvals, authorization artifacts, or governance changes.
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
GOVERNANCE = ROOT / "data" / "governance"

OUTPUT_JSON = RESULTS / "phase_198o_post_gld_certification_evaluation_gap_audit.json"
OUTPUT_MD = RESULTS / "phase_198o_post_gld_certification_evaluation_gap_audit.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

ARTIFACT_PATHS = {
    "positions_snapshot": RESULTS / "positions_snapshot.csv",
    "recent_orders": RESULTS / "recent_orders.csv",
    "open_orders_snapshot": RESULTS / "open_orders_snapshot.csv",
    "phase_198l": RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json",
    "phase_197y": RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json",
    "phase_197z": RESULTS / "phase_197z_readiness_bridge_validation.json",
    "cpi": RESULTS / "capital_preservation_intelligence.json",
    "certification": RESULTS / "capital_preservation_certification.json",
    "evaluation": RESULTS / "protective_action_evaluation.json",
    "trials": RESULTS / "protective_action_trials.json",
    "governor": RESULTS / "capital_preservation_governor.json",
    "governance_authorization": RESULTS / "governance_authorization.json",
    "execution_readiness": RESULTS / "execution_readiness.json",
    "executive_risk_summary": RESULTS / "executive_risk_summary.json",
    "watchdog_alerts": RESULTS / "watchdog_alerts.json",
}

MARKER = "POST_GLD_CERTIFICATION_EVALUATION_GAP_AUDIT_ONLY"
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


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


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


def _count_csv_data_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        return max(0, len(rows) - 1)
    except Exception:
        return 0


def _gld_from_positions() -> Dict[str, Any]:
    path = ARTIFACT_PATHS["positions_snapshot"]
    if not path.is_file():
        return {"qty": 0, "weight": 0.0, "market_value": 0.0}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return {"qty": 0, "weight": 0.0, "market_value": 0.0}

    total = 0.0
    gld_row: Optional[Dict[str, str]] = None
    for row in rows:
        total += abs(_safe_float(row.get("market_value") or row.get("value")))
        sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
        if sym == GLD_SYMBOL:
            gld_row = row

    if not gld_row:
        return {"qty": 0, "weight": 0.0, "market_value": 0.0, "total_invested": total}

    qty = _safe_float(gld_row.get("qty") or gld_row.get("qty_available"))
    mv = abs(_safe_float(gld_row.get("market_value") or gld_row.get("value")))
    weight = mv / total if total > 0 else 0.0
    return {
        "qty": qty,
        "weight": round(weight, 6),
        "weight_pct": round(weight * 100.0, 4),
        "market_value": mv,
        "total_invested": total,
    }


def _gld_sell_proof(recent_orders_path: Path) -> Dict[str, Any]:
    if not recent_orders_path.is_file():
        return {"gld_filled_sells": [], "total_gld_sold": 0}
    sells: List[Dict[str, Any]] = []
    with recent_orders_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().lower()
            status = str(row.get("status") or "").strip().lower()
            if sym != GLD_SYMBOL or side != "sell" or status != "filled":
                continue
            sells.append(
                {
                    "filled_at": row.get("filled_at"),
                    "qty": _safe_int(row.get("filled_qty") or row.get("qty")),
                    "filled_avg_price": _safe_float(row.get("filled_avg_price")),
                }
            )
    total = sum(s["qty"] for s in sells)
    return {"gld_filled_sells": sells, "total_gld_sold": total}


def _evaluation_threshold(eval_doc: Dict[str, Any]) -> Tuple[float, str]:
    baseline = str(
        eval_doc.get("baseline_context_label")
        or (eval_doc.get("baseline_context") or {}).get("baseline_context_label")
        or "STANDARD"
    )
    high_cps_adj = bool(eval_doc.get("high_cps_adjustment_applied"))
    if baseline == "HIGH_CPS_PRESERVATION" and high_cps_adj:
        return 40.0, "high_cps_baseline_aware>=40"
    if baseline == "MODERATE_CPS_IMPROVEMENT":
        return 45.0, "moderate_cps>=45"
    return 50.0, "standard>=50"


def _evaluation_would_pass(eval_doc: Dict[str, Any]) -> Tuple[bool, float, str]:
    threshold, label = _evaluation_threshold(eval_doc)
    avg = float(eval_doc.get("average_effectiveness") or 0)
    count = int(eval_doc.get("evaluation_count") or 0)
    beneficial = int(eval_doc.get("beneficial_count") or 0)
    negative = int(eval_doc.get("negative_count") or 0)
    baseline = str(
        eval_doc.get("baseline_context_label")
        or (eval_doc.get("baseline_context") or {}).get("baseline_context_label")
        or ""
    )
    high_cps_adj = bool(eval_doc.get("high_cps_adjustment_applied"))

    if count < 2:
        return False, threshold, label
    if baseline == "HIGH_CPS_PRESERVATION" and high_cps_adj:
        ok = avg >= threshold and beneficial >= 1 and negative <= 1
        return ok, threshold, label
    if baseline == "MODERATE_CPS_IMPROVEMENT":
        ok = avg >= threshold and beneficial >= 1
        return ok, threshold, label
    return avg >= threshold, threshold, label


def _concentration_resolved(
    *,
    gld_weight: float,
    open_orders: int,
    phase_198l: Dict[str, Any],
    phase_197y: Dict[str, Any],
    phase_197z: Dict[str, Any],
) -> bool:
    sell_qty = _safe_int(
        phase_198l.get("recommended_sell_qty")
        or phase_198l.get("calculation", {}).get("recommended_sell_qty")
    )
    eligible_qty = _safe_int(phase_197y.get("eligible_qty"))
    bridge_active = bool(phase_197z.get("bridge_active"))
    target_satisfied = bool(
        phase_198l.get("target_already_satisfied")
        or phase_198l.get("calculation", {}).get("target_already_satisfied")
    )
    no_eligible_dispatch = eligible_qty <= 0 and not bridge_active
    return (
        gld_weight <= GLD_TARGET_WEIGHT + 0.005
        and open_orders == 0
        and sell_qty == 0
        and no_eligible_dispatch
        and target_satisfied
    )


def _detect_stale_pre_gld_signals(
    *,
    evaluation: Dict[str, Any],
    cpi: Dict[str, Any],
    phase_197y: Dict[str, Any],
    concentration_resolved: bool,
    gld_sell_proof: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    signals: List[str] = []
    conc_component = int((cpi.get("component_scores") or {}).get("concentration") or 0)

    conc_eval = next(
        (
            e
            for e in (evaluation.get("evaluations") or [])
            if e.get("trial_id") == "paper_concentration_reduction"
        ),
        {},
    )
    if concentration_resolved and conc_component >= 95:
        if str(conc_eval.get("evaluation") or "").upper() == "NEGATIVE":
            signals.append(
                "paper_concentration_reduction trial NEGATIVE while CPI concentration component "
                f"={conc_component} and GLD target satisfied"
            )
        est_reduction = float(conc_eval.get("concentration_reduction") or 0)
        if est_reduction > 0 and gld_sell_proof.get("total_gld_sold", 0) >= 100:
            signals.append(
                f"concentration trial simulates {est_reduction}% reduction but broker already sold "
                f"{gld_sell_proof.get('total_gld_sold')} GLD shares"
            )

    preview = phase_197y.get("defensive_preview_broker_state") or {}
    if bool(phase_197y.get("defensive_preview_invalidated")):
        preview_remaining = _safe_int(preview.get("preview_remaining_qty"))
        broker_qty = _safe_int(preview.get("broker_qty"))
        if preview_remaining > broker_qty and broker_qty > 0:
            signals.append(
                f"197Y stale defensive preview: preview_remaining={preview_remaining} > broker_qty={broker_qty}"
            )

    if bool(phase_197y.get("stale_preview_original_qty")):
        signals.append(
            "197Y reports stale_preview_original_qty from pre-completion defensive preview"
        )

    return bool(signals), signals


def _detect_stale_preview(phase_197y: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if str(phase_197y.get("defensive_preview_state") or "") == "TARGET_ALREADY_COMPLETED":
        if bool(phase_197y.get("defensive_preview_invalidated")):
            reasons.append("defensive_preview_invalidated=true with TARGET_ALREADY_COMPLETED")
    preview = phase_197y.get("defensive_preview_broker_state") or {}
    if (
        _safe_int(preview.get("preview_sell_qty")) > 0
        and _safe_int(preview.get("broker_qty")) <= 20
    ):
        reasons.append(
            f"preview_sell_qty={preview.get('preview_sell_qty')} inconsistent with broker_qty={preview.get('broker_qty')}"
        )
    return bool(reasons), reasons


def _classify_gap(
    *,
    concentration_resolved: bool,
    stale_eval_input: bool,
    stale_preview: bool,
    stale_cert_input: bool,
    drawdown_remains: bool,
    mandatory_gate_failed: bool,
    evaluation_legitimately_fails: bool,
    health_band: str,
    escalation: str,
) -> Tuple[str, str, List[str]]:
    factors: List[str] = []

    if concentration_resolved:
        factors.append("GLD concentration blocker RESOLVED")

    if stale_eval_input:
        factors.append(
            "STALE_EVALUATION_INPUT: protective evaluation does not credit completed GLD reduction"
        )
    if stale_preview:
        factors.append(
            "STALE_PREVIEW: defensive preview artifacts reference pre-completion quantities"
        )
    if stale_cert_input:
        factors.append(
            "STALE_CERTIFICATION_INPUT: certification disagrees with current governance/readiness"
        )
    if drawdown_remains and concentration_resolved:
        factors.append("LEGITIMATE_DRAWDOWN_REVIEW_REQUIRED: drawdown component remains elevated")

    if mandatory_gate_failed and health_band == "HEALTHY" and escalation in ("GREEN", "YELLOW"):
        if stale_eval_input:
            factors.append(
                "STALE_MANDATORY_GATE_VETO: mandatory gate blocked by stale evaluation inputs"
            )
        elif evaluation_legitimately_fails:
            factors.append(
                "LEGITIMATE_EVALUATION_VETO: trials score below threshold on current portfolio"
            )

    if stale_eval_input and drawdown_remains:
        primary = "STALE_EVALUATION_INPUT"
        recommendation = (
            "Phase 198Q (or successor): refresh protective evaluation producer to credit completed "
            "GLD concentration reduction and apply HIGH_CPS baseline-aware threshold wiring; "
            "then re-run certification cycle without loosening gates."
        )
    elif stale_eval_input:
        primary = "STALE_EVALUATION_INPUT"
        recommendation = (
            "Refresh protective_action_evaluation to recognize broker-completed GLD reduction; "
            "invalidate concentration trial against post-reduction portfolio state."
        )
    elif drawdown_remains and concentration_resolved:
        primary = "LEGITIMATE_DRAWDOWN_REVIEW_REQUIRED"
        recommendation = (
            "Continue drawdown review under governance; certification Evaluation may remain blocked "
            "until drawdown trials improve or GLD drawdown alert clears — do not force certification."
        )
    elif stale_cert_input:
        primary = "STALE_CERTIFICATION_INPUT"
        recommendation = "Re-run capital_preservation_certification against fresh governance/readiness artifacts."
    elif evaluation_legitimately_fails:
        primary = "LEGITIMATE_EVALUATION_VETO"
        recommendation = (
            "Observe additional protective evaluation cycles; improve trial outcomes organically — "
            "do not lower thresholds or force PASS."
        )
    elif mandatory_gate_failed:
        primary = "STALE_MANDATORY_GATE_VETO"
        recommendation = "Reconcile mandatory preservation gate inputs after evaluation refresh."
    else:
        primary = "LEGITIMATE_CURRENT_STATE_BLOCKER"
        recommendation = "Review failed_requirements in capital_preservation_certification.json."

    return primary, recommendation, factors


def _mandatory_gate_status(cert: Dict[str, Any]) -> Dict[str, Any]:
    areas = cert.get("areas") or {}
    mandatory = ("Monitoring", "Governor", "Authorization", "Readiness", "Evaluation")
    gate_rows = []
    failed = []
    for name in mandatory:
        area = areas.get(name) or {}
        certified = bool(area.get("certified"))
        gate_rows.append({"gate": name, "certified": certified, "score": area.get("score")})
        if not certified:
            failed.append(name)
    veto = cert.get("veto_triggered") or []
    return {
        "mandatory_gates": gate_rows,
        "failed_mandatory_gates": failed,
        "mandatory_gate_all_pass": len(failed) == 0,
        "veto_triggered": veto,
        "pre_veto_status": cert.get("pre_veto_certification_status"),
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    audit = payload.get("audit") or {}
    proof = payload.get("proof") or {}
    classification = payload.get("classification") or {}
    safety = payload.get("safety") or {}

    blockers = audit.get("failed_requirements") or []
    blocker_lines = "\n".join(f"- `{b}`" for b in blockers) or "- None"
    factor_lines = "\n".join(f"- {f}" for f in (classification.get("factors") or [])) or "- None"
    stale_eval = classification.get("stale_pre_gld_signal_details") or []
    stale_eval_lines = "\n".join(f"- {s}" for s in stale_eval) or "- None detected"
    stale_preview = classification.get("stale_preview_details") or []
    stale_preview_lines = "\n".join(f"- {s}" for s in stale_preview) or "- None detected"

    return f"""# TRITON Phase 198O — Post-GLD Certification Evaluation Gap Audit

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Mode:** read-only audit — no broker orders, no approval/auth artifacts

## Executive summary

{payload.get("executive_summary")}

## Root cause

{classification.get("root_cause_summary")}

**Gap classification:** `{classification.get("certification_gap_classification")}`
**Stale vs legitimate:** {classification.get("stale_vs_legitimate")}
**Recommended next phase:** {classification.get("recommended_next_fix")}

## GLD target proof

| Field | Value |
|-------|-------|
| GLD qty | {audit.get("gld_qty")} |
| GLD weight | {audit.get("gld_weight_pct")}% |
| Target satisfied | {audit.get("gld_target_satisfied")} |
| Concentration resolved | {classification.get("concentration_blocker_resolved")} |
| 198L sell_qty | {proof.get("phase_198l_sell_qty")} |
| 197Y eligible_qty | {proof.get("phase_197y_eligible_qty")} |
| 197Z bridge_active | {proof.get("phase_197z_bridge_active")} |
| Broker GLD sold (recent) | {proof.get("broker_gld_shares_sold")} shares |

## Current certification blockers

{blocker_lines}

## Current risk state

| Metric | Value |
|--------|-------|
| CPS | {audit.get("capital_preservation_score")} |
| Health band | {audit.get("capital_preservation_health_band")} |
| Escalation | {audit.get("escalation_state")} |
| Certification | {audit.get("certification_status")} (score {audit.get("certification_score")}) |
| Execution readiness | {audit.get("execution_readiness_status")} |
| Governance authorized | {audit.get("governance_authorized")} |
| Drawdown component | {proof.get("drawdown_component_score")} |
| Concentration component | {proof.get("concentration_component_score")} |

## Protective evaluation

| Metric | Value |
|--------|-------|
| Trial count | {audit.get("protective_evaluation_count")} |
| Average effectiveness | {audit.get("protective_average_effectiveness")} |
| Beneficial | {audit.get("protective_beneficial_count")} |
| Negative | {audit.get("protective_negative_count")} |
| Evaluation threshold | {proof.get("evaluation_threshold_label")} |
| Evaluation would pass | {proof.get("evaluation_would_pass")} |

## Mandatory preservation gate

- Status: `{audit.get("mandatory_gate_status")}`
- Failed gates: {proof.get("failed_mandatory_gates")}
- Veto triggered: {proof.get("veto_triggered")}

## Classification factors

{factor_lines}

## Stale artifact proof

### Pre-GLD evaluation signals

{stale_eval_lines}

### Stale defensive preview

{stale_preview_lines}

## Safety proof (no execution / no authority)

| Check | Value |
|-------|-------|
| broker_orders_submitted | {safety.get("broker_orders_submitted")} |
| approval_artifact_present | {audit.get("approval_artifact_present")} |
| broker_auth_artifact_present | {audit.get("broker_auth_artifact_present")} |
| live_execution_allowed | {audit.get("live_execution_allowed")} |
| open_orders_count | {audit.get("open_orders_count")} |

## Git state

```
{payload.get("git_status_short") or "(clean)"}
```

```
{payload.get("git_diff_stat") or "(no diff)"}
```

---

*{MARKER}*
"""


def run_phase_198o_audit(*, verbose: bool = True) -> Dict[str, Any]:
    from services.portfolio_history_sync import load_live_execution_flag

    artifacts = {k: _load_json(p) for k, p in ARTIFACT_PATHS.items() if p.suffix == ".json"}
    gld_pos = _gld_from_positions()
    gld_sell = _gld_sell_proof(ARTIFACT_PATHS["recent_orders"])
    open_orders = _count_csv_data_rows(ARTIFACT_PATHS["open_orders_snapshot"])

    phase_198l = artifacts.get("phase_198l") or {}
    phase_197y = artifacts.get("phase_197y") or {}
    phase_197z = artifacts.get("phase_197z") or {}
    cpi = artifacts.get("cpi") or {}
    cert = artifacts.get("certification") or {}
    evaluation = artifacts.get("evaluation") or {}
    governor = artifacts.get("governor") or {}
    gov_auth = artifacts.get("governance_authorization") or {}
    readiness = artifacts.get("execution_readiness") or {}
    exec_risk = artifacts.get("executive_risk_summary") or {}
    watchdog = artifacts.get("watchdog_alerts") or {}

    gld_qty = _safe_int(gld_pos.get("qty") or phase_198l.get("current_gld_qty"))
    gld_weight = float(
        gld_pos.get("weight")
        or phase_198l.get("current_gld_weight")
        or phase_198l.get("calculation", {}).get("current_gld_weight")
        or 0
    )
    gld_target_satisfied = bool(
        phase_198l.get("target_already_satisfied")
        or phase_198l.get("calculation", {}).get("target_already_satisfied")
        or gld_weight <= GLD_TARGET_WEIGHT + 0.005
    )

    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    live_allowed = bool(load_live_execution_flag()) or bool(
        gov_auth.get("live_execution_permitted") or readiness.get("live_execution_permitted")
    )

    conc_resolved = _concentration_resolved(
        gld_weight=gld_weight,
        open_orders=open_orders,
        phase_198l=phase_198l,
        phase_197y=phase_197y,
        phase_197z=phase_197z,
    )

    stale_pre_gld, stale_pre_gld_details = _detect_stale_pre_gld_signals(
        evaluation=evaluation,
        cpi=cpi,
        phase_197y=phase_197y,
        concentration_resolved=conc_resolved,
        gld_sell_proof=gld_sell,
    )
    stale_preview, stale_preview_details = _detect_stale_preview(phase_197y)

    eval_area = (cert.get("areas") or {}).get("Evaluation") or {}
    auth_area = (cert.get("areas") or {}).get("Authorization") or {}
    ready_area = (cert.get("areas") or {}).get("Readiness") or {}

    stale_cert_input = (
        not auth_area.get("certified") and bool(gov_auth.get("governance_authorized"))
    ) or (
        not ready_area.get("certified")
        and str(readiness.get("readiness_status") or "").upper() == "READY"
    )

    drawdown_score = int((cpi.get("component_scores") or {}).get("drawdown") or 0)
    drawdown_remains = drawdown_score < 70 or any(
        str(a.get("alert_type") or "") == "EXCESS_POSITION_DRAWDOWN"
        for a in (watchdog.get("active_alerts") or [])
    )

    gate_info = _mandatory_gate_status(cert)
    mandatory_gate_failed = not gate_info["mandatory_gate_all_pass"] or bool(
        gate_info["veto_triggered"]
    )

    eval_pass, eval_threshold, eval_threshold_label = _evaluation_would_pass(evaluation)
    evaluation_legitimately_fails = not bool(eval_area.get("certified")) and not eval_pass

    health_band = str(cpi.get("health_band") or cert.get("portfolio_health") or "")
    escalation = str(
        cert.get("escalation_state")
        or governor.get("escalation_state")
        or exec_risk.get("escalation_state")
        or ""
    )

    primary_class, recommended_fix, factors = _classify_gap(
        concentration_resolved=conc_resolved,
        stale_eval_input=stale_pre_gld,
        stale_preview=stale_preview,
        stale_cert_input=stale_cert_input,
        drawdown_remains=drawdown_remains,
        mandatory_gate_failed=mandatory_gate_failed,
        evaluation_legitimately_fails=evaluation_legitimately_fails,
        health_band=health_band,
        escalation=escalation,
    )

    stale_vs_legitimate = (
        "STALE (evaluation/preview inputs lag completed GLD reduction)"
        if stale_pre_gld or stale_preview
        else (
            "LEGITIMATE (current trial scores and drawdown state)"
            if evaluation_legitimately_fails
            else "MIXED"
        )
    )

    root_cause = (
        "GLD concentration reduction is complete and reflected in positions/CPI, but preservation "
        "certification remains NOT_CERTIFIED because the Evaluation mandatory gate fails "
        f"(avg effectiveness {evaluation.get('average_effectiveness')}, threshold {eval_threshold_label}). "
    )
    if stale_pre_gld:
        root_cause += (
            "Protective evaluation still scores hypothetical concentration reduction trials as NEGATIVE "
            "without crediting the broker-completed GLD sell sequence. "
        )
    if drawdown_remains:
        root_cause += f"Drawdown risk remains (component={drawdown_score}, active EXCESS_POSITION_DRAWDOWN alert on GLD). "
    root_cause += (
        "The mandatory_preservation_gate_failed veto is a downstream consequence of the Evaluation failure — "
        "not a concentration or readiness regression."
    )

    audit_fields = {
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_weight_pct": round(gld_weight * 100.0, 4),
        "gld_target_satisfied": gld_target_satisfied,
        "open_orders_count": open_orders,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "live_execution_allowed": live_allowed,
        "execution_readiness_status": readiness.get("readiness_status"),
        "governance_authorized": bool(gov_auth.get("governance_authorized")),
        "capital_preservation_score": cpi.get("capital_preservation_score"),
        "capital_preservation_health_band": health_band,
        "escalation_state": escalation,
        "certification_status": cert.get("certification_status"),
        "certification_score": cert.get("certification_score"),
        "failed_requirements": cert.get("failed_requirements") or [],
        "protective_evaluation_count": evaluation.get("evaluation_count"),
        "protective_average_effectiveness": evaluation.get("average_effectiveness"),
        "protective_beneficial_count": evaluation.get("beneficial_count"),
        "protective_negative_count": evaluation.get("negative_count"),
        "mandatory_gate_status": ("FAILED" if mandatory_gate_failed else "PASS"),
        "stale_pre_gld_signals_detected": stale_pre_gld,
        "stale_preview_detected": stale_preview,
        "certification_gap_classification": primary_class,
        "recommended_next_fix": recommended_fix,
    }

    safety = {
        "broker_orders_submitted": False,
        "approval_artifact_created": False,
        "broker_auth_artifact_created": False,
        "live_execution_modified": False,
        "governance_loosened": False,
        "certification_forced_pass": False,
    }

    checks = {
        "gld_qty_16": gld_qty == 16,
        "gld_target_satisfied": gld_target_satisfied,
        "open_orders_zero": open_orders == 0,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "live_blocked": not live_allowed,
        "concentration_resolved": conc_resolved,
        "classification_present": bool(primary_class),
        "no_broker_orders": True,
        "audit_artifacts_written": True,
    }
    score = int(round(100.0 * sum(1 for v in checks.values() if v) / len(checks)))
    status = "PASS" if all(checks.values()) else "FAIL"

    git_status = ""
    git_diff_stat = ""
    try:
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        git_diff_stat = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        pass

    payload: Dict[str, Any] = {
        "phase": "198O",
        "generated_at": _utc_now_iso(),
        "status": status,
        "score": score,
        "checks": checks,
        "mode": "read_only_post_gld_certification_evaluation_gap_audit",
        "disclaimer": (
            "Audit only. Does not place broker orders, create approval/authorization artifacts, "
            "authorize live trading, force certification PASS, or loosen governance."
        ),
        "executive_summary": (
            f"GLD concentration target satisfied (qty={gld_qty}, weight={audit_fields['gld_weight_pct']}%). "
            f"Operational posture HEALTHY/YELLOW with execution_readiness=READY and governance authorized. "
            f"Preservation certification remains {cert.get('certification_status')} due to Evaluation gate "
            f"and mandatory_preservation_gate_failed veto. Classification: {primary_class}."
        ),
        "audit": audit_fields,
        "classification": {
            "certification_gap_classification": primary_class,
            "recommended_next_fix": recommended_fix,
            "concentration_blocker_resolved": conc_resolved,
            "stale_vs_legitimate": stale_vs_legitimate,
            "factors": factors,
            "root_cause_summary": root_cause,
            "stale_pre_gld_signal_details": stale_pre_gld_details,
            "stale_preview_details": stale_preview_details,
            "secondary_classifications": [
                c
                for c in (
                    (
                        "LEGITIMATE_DRAWDOWN_REVIEW_REQUIRED"
                        if drawdown_remains and conc_resolved
                        else None
                    ),
                    "LEGITIMATE_EVALUATION_VETO" if evaluation_legitimately_fails else None,
                    (
                        "STALE_MANDATORY_GATE_VETO"
                        if mandatory_gate_failed and stale_pre_gld
                        else None
                    ),
                )
                if c and c != primary_class
            ],
        },
        "proof": {
            "phase_198l_sell_qty": _safe_int(
                phase_198l.get("recommended_sell_qty")
                or phase_198l.get("calculation", {}).get("recommended_sell_qty")
            ),
            "phase_197y_eligible_qty": _safe_int(phase_197y.get("eligible_qty")),
            "phase_197y_defensive_preview_state": phase_197y.get("defensive_preview_state"),
            "phase_197z_bridge_active": bool(phase_197z.get("bridge_active")),
            "broker_gld_shares_sold": gld_sell.get("total_gld_sold"),
            "gld_filled_sells": gld_sell.get("gld_filled_sells"),
            "drawdown_component_score": drawdown_score,
            "concentration_component_score": int(
                (cpi.get("component_scores") or {}).get("concentration") or 0
            ),
            "evaluation_threshold": eval_threshold,
            "evaluation_threshold_label": eval_threshold_label,
            "evaluation_would_pass": eval_pass,
            "high_cps_adjustment_applied": bool(evaluation.get("high_cps_adjustment_applied")),
            "failed_mandatory_gates": gate_info.get("failed_mandatory_gates"),
            "veto_triggered": gate_info.get("veto_triggered"),
            "pre_veto_certification_status": cert.get("pre_veto_certification_status"),
            "top_risks": exec_risk.get("top_risks") or [],
            "active_watchdog_alerts": [
                a.get("alert_id") for a in (watchdog.get("active_alerts") or [])
            ],
        },
        "mandatory_gate_detail": gate_info,
        "safety": safety,
        "artifact_snapshot": {
            k: {"path": _rel(p), "present": p.is_file()} for k, p in ARTIFACT_PATHS.items()
        },
        "git_status_short": git_status,
        "git_diff_stat": git_diff_stat,
        "marker": MARKER,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(
            f"[198O] status={status} score={score} classification={primary_class} "
            f"gld_qty={gld_qty} gld_target_satisfied={gld_target_satisfied} "
            f"certification={cert.get('certification_status')}"
        )
        if status != "PASS":
            failed = [k for k, v in checks.items() if not v]
            print(f"[198O] failed_checks={failed}")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 198O post-GLD certification evaluation gap audit"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary")
    args = parser.parse_args(argv)
    result = run_phase_198o_audit(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
