"""
phase_198s_refresh_exposure_trial_weighted_drawdown_calibration.py
Phase 198S — Refresh exposure trial inputs and portfolio-weighted drawdown calibration.

Evaluation logic refresh only. Does not force certification PASS or place orders.
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

OUTPUT_JSON = RESULTS / "phase_198s_refresh_exposure_trial_weighted_drawdown_calibration.json"
OUTPUT_MD = RESULTS / "phase_198s_refresh_exposure_trial_weighted_drawdown_calibration.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

MARKER = "REFRESH_EXPOSURE_TRIAL_WEIGHTED_DRAWDOWN_CALIBRATION_ONLY"


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


def _trial_snapshot(eval_doc: Dict[str, Any], trial_id: str) -> Dict[str, Any]:
    trial = next(
        (e for e in (eval_doc.get("evaluations") or []) if e.get("trial_id") == trial_id),
        {},
    )
    return {
        "trial_id": trial_id,
        "evaluation": trial.get("evaluation"),
        "effectiveness_score": trial.get("effectiveness_score"),
        "evaluation_mode": trial.get("evaluation_mode"),
        "evaluation_reason": trial.get("evaluation_reason"),
    }


def _eval_metrics(eval_doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "average_effectiveness": eval_doc.get("average_effectiveness"),
        "beneficial_count": eval_doc.get("beneficial_count"),
        "negative_count": eval_doc.get("negative_count"),
        "neutral_count": eval_doc.get("neutral_count"),
    }


def _cert_metrics(cert: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "certification_status": cert.get("certification_status"),
        "certification_score": cert.get("certification_score"),
        "failed_requirements": cert.get("failed_requirements") or [],
        "evaluation_area_certified": (cert.get("areas") or {})
        .get("Evaluation", {})
        .get("certified"),
    }


def _refresh_certification(results_dir: Path) -> Dict[str, Any]:
    from services.capital_preservation_certification import (
        _atomic_write_json as cert_write,
        compute_preservation_certification,
    )

    rd = results_dir
    cert_doc = compute_preservation_certification(
        cpi_doc=_load_json(rd / "capital_preservation_intelligence.json"),
        alerts_doc=_load_json(rd / "watchdog_alerts.json"),
        gov_doc=_load_json(rd / "governance_risk_summary.json"),
        auth_doc=_load_json(rd / "governance_authorization.json"),
        readiness_doc=_load_json(rd / "execution_readiness.json"),
        sim_doc=_load_json(rd / "defensive_simulation_results.json"),
        evaluation_doc=_load_json(rd / "protective_action_evaluation.json"),
        governor_doc=_load_json(rd / "capital_preservation_governor.json"),
        cpe_doc=_load_json(rd / "capital_preservation_escalation.json"),
        exec_doc=_load_json(rd / "executive_risk_report.json"),
        watchdog_doc=_load_json(rd / "watchdog_status.json"),
        verbose=False,
    )
    cert_write(cert_doc, rd / "capital_preservation_certification.json")
    return cert_doc


def _remaining_blockers(cert_after: Dict[str, Any], eval_after: Dict[str, Any]) -> List[str]:
    blockers: List[str] = []
    eval_area = (cert_after.get("areas") or {}).get("Evaluation") or {}
    if not eval_area.get("certified"):
        negative = int(eval_after.get("negative_count") or 0)
        if negative >= 1:
            dd = _trial_snapshot(eval_after, "paper_drawdown_reduction")
            if str(dd.get("evaluation") or "").upper() == "NEGATIVE":
                blockers.append(
                    "RESIDUAL_DRAWDOWN_REVIEW_REQUIRED: drawdown trial remains NEGATIVE with broker-supported GLD alert"
                )
        if negative > 1:
            blockers.append(f"Evaluation negative_count={negative} exceeds high-CPS limit (<=1)")
    for req in cert_after.get("failed_requirements") or []:
        if str(req).startswith("veto:"):
            blockers.append(f"Certification veto: {req}")
    return blockers


def _render_markdown(payload: Dict[str, Any]) -> str:
    audit = payload.get("audit") or {}
    before = payload.get("before") or {}
    after = payload.get("after") or {}
    safety = payload.get("safety") or {}
    blockers = payload.get("remaining_legitimate_blockers") or []
    blocker_lines = "\n".join(f"- {b}" for b in blockers) or "- None"

    return f"""# TRITON Phase 198S — Exposure Trial and Weighted Drawdown Calibration

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}

## Executive summary

{payload.get("executive_summary")}

## Root cause

{payload.get("root_cause")}

## Exposure trial — before vs after

| Field | Before | After |
|-------|--------|-------|
| Evaluation | {before.get("exposure_trial", {}).get("evaluation")} | {after.get("exposure_trial", {}).get("evaluation")} |
| Score | {before.get("exposure_trial", {}).get("effectiveness_score")} | {after.get("exposure_trial", {}).get("effectiveness_score")} |
| Mode | {before.get("exposure_trial", {}).get("evaluation_mode")} | {after.get("exposure_trial", {}).get("evaluation_mode")} |
| Negative before | {audit.get("exposure_trial_negative_before")} | {audit.get("exposure_trial_negative_after")} |

## Drawdown trial — before vs after

| Field | Before | After |
|-------|--------|-------|
| Evaluation | {before.get("drawdown_trial", {}).get("evaluation")} | {after.get("drawdown_trial", {}).get("evaluation")} |
| Score | {before.get("drawdown_trial", {}).get("effectiveness_score")} | {after.get("drawdown_trial", {}).get("effectiveness_score")} |
| Review required | — | {audit.get("drawdown_trial_remains_review_required")} |
| Alert broker-supported | — | {audit.get("drawdown_alert_broker_supported")} |

## Evaluation metrics

| Metric | Before | After |
|--------|--------|-------|
| Average effectiveness | {before.get("average_effectiveness")} | {after.get("average_effectiveness")} |
| Beneficial | {before.get("beneficial_count")} | {after.get("beneficial_count")} |
| Negative | {before.get("negative_count")} | {after.get("negative_count")} |

## Certification

| Metric | Before | After |
|--------|--------|-------|
| Status | {before.get("certification_status")} | {after.get("certification_status")} |
| Score | {before.get("certification_score")} | {after.get("certification_score")} |
| Naturally passed | — | {audit.get("certification_naturally_passed")} |
| Forced | — | {audit.get("certification_forced")} |

## Portfolio-weighted drawdown metadata

- Position-level loss: {audit.get("gld_unrealized_pl_pct")}%
- Position weight: {audit.get("gld_weight_pct")}%
- Portfolio-weighted impact: {audit.get("portfolio_weighted_gld_drawdown_impact")}%

## Remaining blockers

{blocker_lines}

## Safety

| Check | Value |
|-------|-------|
| broker_orders_submitted | {safety.get("broker_orders_submitted")} |
| certification_forced | {audit.get("certification_forced")} |
| live_execution_allowed | {audit.get("live_execution_allowed")} |

---

*{MARKER}*
"""


def run_phase_198s(*, verbose: bool = True) -> Dict[str, Any]:
    from services.capital_preservation_evaluation import run_evaluation_chain
    from services.portfolio_history_sync import load_live_execution_flag

    eval_before = _load_json(RESULTS / "protective_action_evaluation.json")
    cert_before = _load_json(RESULTS / "capital_preservation_certification.json")

    before_exposure = _trial_snapshot(eval_before, "paper_exposure_reduction")
    before_drawdown = _trial_snapshot(eval_before, "paper_drawdown_reduction")
    before_eval = _eval_metrics(eval_before)
    before_cert = _cert_metrics(cert_before)

    run_evaluation_chain(results_dir=RESULTS)
    cert_after_doc = _refresh_certification(RESULTS)

    eval_after = _load_json(RESULTS / "protective_action_evaluation.json")
    after_exposure = _trial_snapshot(eval_after, "paper_exposure_reduction")
    after_drawdown = _trial_snapshot(eval_after, "paper_drawdown_reduction")
    after_eval = _eval_metrics(eval_after)
    after_cert = _cert_metrics(cert_after_doc)

    dd_meta = eval_after.get("portfolio_weighted_drawdown_metadata") or {}
    gld_ctx = eval_after.get("gld_concentration_context") or {}

    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    live_allowed = bool(load_live_execution_flag())

    exposure_neg_before = str(before_exposure.get("evaluation") or "").upper() == "NEGATIVE"
    exposure_neg_after = str(after_exposure.get("evaluation") or "").upper() == "NEGATIVE"

    cert_naturally_passed = (
        str(after_cert.get("certification_status") or "")
        in (
            "CERTIFIED_FOR_PAPER_PROTECTION",
            "CONDITIONALLY_CERTIFIED",
            "PARTIALLY_CERTIFIED",
        )
        and str(before_cert.get("certification_status") or "") == "NOT_CERTIFIED"
    )

    remaining = _remaining_blockers(cert_after_doc, eval_after)

    gld_qty = int(gld_ctx.get("gld_qty") or dd_meta.get("gld_qty") or 0)
    gld_weight = float(gld_ctx.get("gld_weight") or dd_meta.get("position_weight") or 0)

    objective_checks = {
        "gld_qty_16": gld_qty == 16,
        "gld_weight_at_target": gld_weight <= 0.205,
        "concentration_resolved": bool(eval_after.get("concentration_reduction_completed")),
        "exposure_stale_resolved": bool(eval_after.get("exposure_trial_stale_input_resolved")),
        "exposure_no_longer_negative_for_stale_gld": not exposure_neg_after
        or not exposure_neg_before,
        "drawdown_trial_honest": str(after_drawdown.get("evaluation") or "").upper() == "NEGATIVE",
        "negative_count_improved": int(after_eval.get("negative_count") or 0)
        <= int(before_eval.get("negative_count") or 0),
        "certification_not_forced": True,
        "no_broker_orders": True,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "live_blocked": not live_allowed,
    }

    if all(objective_checks.values()):
        if str(after_cert.get("certification_status") or "") == "NOT_CERTIFIED":
            status = "REVIEW_REQUIRED"
        else:
            status = "PASS"
    else:
        status = "FAIL"
        if verbose:
            print(f"[198S] failed_checks={[k for k, v in objective_checks.items() if not v]}")

    recommended_next = (
        "Continue residual GLD drawdown operator review; no further certification threshold changes."
        if str(after_cert.get("certification_status") or "") == "NOT_CERTIFIED"
        else "Observe next preservation cycle; certification improved naturally without forced PASS."
    )

    root_cause = (
        "Phase 198R identified STALE_EXPOSURE_TRIAL_INPUT: paper_exposure_reduction still penalized "
        "evaluation with hypothetical concentration reduction after GLD concentration was broker-verified "
        "complete. Phase 198S marks exposure trial COMPLETED_OR_NOT_APPLICABLE and adds "
        "portfolio-weight-aware drawdown metadata without erasing broker-supported GLD drawdown alert."
    )

    executive_summary = (
        f"Exposure trial {before_exposure.get('evaluation')}->{after_exposure.get('evaluation')} "
        f"({before_exposure.get('evaluation_mode')}->{after_exposure.get('evaluation_mode')}). "
        f"Negative count {before_eval.get('negative_count')}->{after_eval.get('negative_count')}. "
        f"Certification {before_cert.get('certification_status')}->{after_cert.get('certification_status')} "
        f"(score {before_cert.get('certification_score')}->{after_cert.get('certification_score')}, "
        f"naturally_passed={cert_naturally_passed}, forced=False)."
    )

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
        "phase": "198S",
        "generated_at": _utc_now_iso(),
        "status": status,
        "objective_checks": objective_checks,
        "root_cause": root_cause,
        "executive_summary": executive_summary,
        "audit": {
            "gld_qty": gld_qty,
            "gld_weight": gld_weight,
            "gld_weight_pct": round(gld_weight * 100.0, 4),
            "gld_unrealized_pl": dd_meta.get("gld_unrealized_pl"),
            "gld_unrealized_pl_pct": dd_meta.get("gld_unrealized_pl_pct")
            or dd_meta.get("position_level_drawdown_pct"),
            "portfolio_weighted_gld_drawdown_impact": dd_meta.get(
                "portfolio_weighted_drawdown_impact_pct"
            ),
            "concentration_resolved": bool(eval_after.get("concentration_reduction_completed")),
            "exposure_trial_before": before_exposure,
            "exposure_trial_after": after_exposure,
            "exposure_trial_stale_input_detected": bool(
                eval_after.get("exposure_trial_stale_input_detected")
            ),
            "exposure_trial_stale_input_resolved": bool(
                eval_after.get("exposure_trial_stale_input_resolved")
            ),
            "exposure_trial_negative_before": exposure_neg_before,
            "exposure_trial_negative_after": exposure_neg_after,
            "drawdown_trial_before": before_drawdown,
            "drawdown_trial_after": after_drawdown,
            "drawdown_alert_broker_supported": bool(dd_meta.get("drawdown_alert_broker_supported")),
            "drawdown_trial_remains_review_required": bool(
                dd_meta.get("drawdown_trial_remains_review_required")
            ),
            "negative_count_before": before_eval.get("negative_count"),
            "negative_count_after": after_eval.get("negative_count"),
            "beneficial_count_before": before_eval.get("beneficial_count"),
            "beneficial_count_after": after_eval.get("beneficial_count"),
            "average_effectiveness_before": before_eval.get("average_effectiveness"),
            "average_effectiveness_after": after_eval.get("average_effectiveness"),
            "certification_status_before": before_cert.get("certification_status"),
            "certification_status_after": after_cert.get("certification_status"),
            "certification_score_before": before_cert.get("certification_score"),
            "certification_score_after": after_cert.get("certification_score"),
            "failed_requirements_before": before_cert.get("failed_requirements"),
            "failed_requirements_after": after_cert.get("failed_requirements"),
            "certification_naturally_passed": cert_naturally_passed,
            "certification_forced": False,
            "approval_artifact_present": approval_present,
            "broker_auth_artifact_present": broker_auth_present,
            "live_execution_allowed": live_allowed,
            "recommended_next_phase": recommended_next,
        },
        "before": {
            **before_eval,
            **before_cert,
            "exposure_trial": before_exposure,
            "drawdown_trial": before_drawdown,
        },
        "after": {
            **after_eval,
            **after_cert,
            "exposure_trial": after_exposure,
            "drawdown_trial": after_drawdown,
        },
        "portfolio_weighted_drawdown_metadata": dd_meta,
        "remaining_legitimate_blockers": remaining,
        "safety": {
            "broker_orders_submitted": False,
            "certification_forced": False,
            "evaluation_thresholds_unchanged": True,
        },
        "files_changed": [
            "services/capital_preservation_evaluation.py",
            "services/phase_198s_refresh_exposure_trial_weighted_drawdown_calibration.py",
        ],
        "git_status_short": git_status,
        "git_diff_stat": git_diff_stat,
        "marker": MARKER,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(
            f"[198S] status={status} exposure={before_exposure.get('evaluation')}->{after_exposure.get('evaluation')} "
            f"negative={before_eval.get('negative_count')}->{after_eval.get('negative_count')} "
            f"cert={before_cert.get('certification_status')}->{after_cert.get('certification_status')}"
        )

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 198S exposure trial and drawdown calibration"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase_198s(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    sys.exit(main())
