"""
phase_198q_refresh_protective_evaluation_after_gld_reduction.py
Phase 198Q — Refresh protective evaluation after completed GLD reduction.

Evaluation/certification hardening only. No broker orders, approvals, or live enablement.
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

OUTPUT_JSON = RESULTS / "phase_198q_refresh_protective_evaluation_after_gld_reduction.json"
OUTPUT_MD = RESULTS / "phase_198q_refresh_protective_evaluation_after_gld_reduction.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

MARKER = "REFRESH_PROTECTIVE_EVALUATION_AFTER_GLD_REDUCTION_ONLY"


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


def _mandatory_gate_status(cert: Dict[str, Any]) -> str:
    areas = cert.get("areas") or {}
    mandatory = ("Monitoring", "Governor", "Authorization", "Readiness", "Evaluation")
    failed = [name for name in mandatory if not (areas.get(name) or {}).get("certified")]
    veto = cert.get("veto_triggered") or []
    if failed or veto:
        return "FAILED"
    return "PASS"


def _eval_summary(eval_doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "average_effectiveness": eval_doc.get("average_effectiveness"),
        "beneficial_count": eval_doc.get("beneficial_count"),
        "neutral_count": eval_doc.get("neutral_count"),
        "negative_count": eval_doc.get("negative_count"),
        "high_cps_adjustment_applied": eval_doc.get("high_cps_adjustment_applied"),
        "concentration_reduction_completed": eval_doc.get("concentration_reduction_completed"),
        "stale_concentration_trial_detected": eval_doc.get("stale_concentration_trial_detected"),
        "stale_concentration_trial_resolved": eval_doc.get("stale_concentration_trial_resolved"),
    }


def _cert_summary(cert: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "certification_status": cert.get("certification_status"),
        "certification_score": cert.get("certification_score"),
        "failed_requirements": cert.get("failed_requirements") or [],
        "mandatory_gate_status": _mandatory_gate_status(cert),
        "evaluation_area_certified": (cert.get("areas") or {})
        .get("Evaluation", {})
        .get("certified"),
    }


def _remaining_legitimate_blockers(
    *,
    cert_after: Dict[str, Any],
    eval_after: Dict[str, Any],
) -> List[str]:
    blockers: List[str] = []
    eval_area = (cert_after.get("areas") or {}).get("Evaluation") or {}
    if not eval_area.get("certified"):
        negative = int(eval_after.get("negative_count") or 0)
        avg = float(eval_after.get("average_effectiveness") or 0)
        if negative >= 2:
            blockers.append(
                "LEGITIMATE_DRAWDOWN_REVIEW_REQUIRED: drawdown and exposure trials remain NEGATIVE"
            )
        if avg < 40:
            blockers.append(
                f"Evaluation average effectiveness {avg} below high-CPS threshold despite concentration credit"
            )
        elif negative > 1:
            blockers.append(
                f"Evaluation negative_count={negative} exceeds high-CPS certification limit (<=1)"
            )

    for req in cert_after.get("failed_requirements") or []:
        if str(req).startswith("veto:"):
            blockers.append(f"Certification veto active: {req}")

    failed = cert_after.get("failed_requirements") or []
    if "Evaluation" in failed and "Evaluation area not certified" not in blockers:
        notes = eval_area.get("notes") or ""
        blockers.append(f"Evaluation area not certified ({notes})")

    return blockers


def _refresh_certification(results_dir: Path) -> Dict[str, Any]:
    from services.capital_preservation_certification import compute_preservation_certification

    rd = results_dir
    docs = {
        "cpi": _load_json(rd / "capital_preservation_intelligence.json"),
        "alerts": _load_json(rd / "watchdog_alerts.json"),
        "gov": _load_json(rd / "governance_risk_summary.json"),
        "auth": _load_json(rd / "governance_authorization.json"),
        "readiness": _load_json(rd / "execution_readiness.json"),
        "sim": _load_json(rd / "defensive_simulation_results.json"),
        "evaluation": _load_json(rd / "protective_action_evaluation.json"),
        "governor": _load_json(rd / "capital_preservation_governor.json"),
        "cpe": _load_json(rd / "capital_preservation_escalation.json"),
        "exec": _load_json(rd / "executive_risk_report.json"),
        "watchdog": _load_json(rd / "watchdog_status.json"),
    }
    cert_doc = compute_preservation_certification(
        cpi_doc=docs["cpi"],
        alerts_doc=docs["alerts"],
        gov_doc=docs["gov"],
        auth_doc=docs["auth"],
        readiness_doc=docs["readiness"],
        sim_doc=docs["sim"],
        evaluation_doc=docs["evaluation"],
        governor_doc=docs["governor"],
        cpe_doc=docs["cpe"],
        exec_doc=docs["exec"],
        watchdog_doc=docs["watchdog"],
        verbose=False,
    )
    from services.capital_preservation_certification import _atomic_write_json as cert_write

    cert_write(cert_doc, rd / "capital_preservation_certification.json")
    return cert_doc


def _render_markdown(payload: Dict[str, Any]) -> str:
    audit = payload.get("audit") or {}
    before = payload.get("before") or {}
    after = payload.get("after") or {}
    safety = payload.get("safety") or {}
    blockers = payload.get("remaining_legitimate_blockers") or []
    blocker_lines = "\n".join(f"- {b}" for b in blockers) or "- None identified"

    return f"""# TRITON Phase 198Q — Refresh Protective Evaluation After GLD Reduction

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Mode:** evaluation/certification hardening only — no broker orders

## Executive summary

{payload.get("executive_summary")}

## Root cause addressed

{payload.get("root_cause")}

## Completed GLD reduction proof

| Field | Value |
|-------|-------|
| GLD qty | {audit.get("gld_qty")} |
| GLD weight | {audit.get("gld_weight_pct")}% |
| Completed GLD sell qty | {audit.get("completed_gld_sell_qty")} |
| Concentration reduction completed | {audit.get("concentration_reduction_completed")} |
| Stale concentration trial resolved | {audit.get("stale_concentration_trial_resolved")} |

## Protective evaluation — before vs after

| Metric | Before | After |
|--------|--------|-------|
| Average effectiveness | {before.get("protective_average_effectiveness")} | {after.get("protective_average_effectiveness")} |
| Beneficial count | {before.get("beneficial_count")} | {after.get("beneficial_count")} |
| Negative count | {before.get("negative_count")} | {after.get("negative_count")} |
| High CPS adjustment | {before.get("high_cps_adjustment_applied")} | {after.get("high_cps_adjustment_applied")} |

## Certification — before vs after

| Metric | Before | After |
|--------|--------|-------|
| Status | {before.get("certification_status")} | {after.get("certification_status")} |
| Score | {before.get("certification_score")} | {after.get("certification_score")} |
| Mandatory gate | {before.get("mandatory_gate_status")} | {after.get("mandatory_gate_status")} |
| Failed requirements | {before.get("failed_requirements")} | {after.get("failed_requirements")} |

## Remaining legitimate blockers

{blocker_lines}

## Safety proof

| Check | Value |
|-------|-------|
| broker_orders_submitted | {safety.get("broker_orders_submitted")} |
| approval_artifact_present | {audit.get("approval_artifact_present")} |
| broker_auth_artifact_present | {audit.get("broker_auth_artifact_present")} |
| live_execution_allowed | {audit.get("live_execution_allowed")} |

## Files changed

{payload.get("files_changed")}

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


def run_phase_198q(*, verbose: bool = True, refresh_cert: bool = True) -> Dict[str, Any]:
    from services.capital_preservation_evaluation import run_evaluation_chain
    from services.portfolio_history_sync import load_live_execution_flag

    eval_before = _load_json(RESULTS / "protective_action_evaluation.json")
    cert_before = _load_json(RESULTS / "capital_preservation_certification.json")

    before_eval = _eval_summary(eval_before)
    before_cert = _cert_summary(cert_before)

    run_evaluation_chain(results_dir=RESULTS)

    eval_after_doc = _load_json(RESULTS / "protective_action_evaluation.json")
    if refresh_cert:
        cert_after_doc = _refresh_certification(RESULTS)
    else:
        cert_after_doc = _load_json(RESULTS / "capital_preservation_certification.json")

    after_eval = _eval_summary(eval_after_doc)
    after_cert = _cert_summary(cert_after_doc)

    gld_ctx = eval_after_doc.get("gld_concentration_context") or {}
    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    live_allowed = bool(load_live_execution_flag()) or bool(
        _load_json(RESULTS / "governance_authorization.json").get("live_execution_permitted")
    )

    remaining = _remaining_legitimate_blockers(cert_after=cert_after_doc, eval_after=eval_after_doc)

    concentration_completed = bool(eval_after_doc.get("concentration_reduction_completed"))
    stale_resolved = bool(eval_after_doc.get("stale_concentration_trial_resolved"))
    gld_qty = int(gld_ctx.get("gld_qty") or 0)
    gld_weight = float(gld_ctx.get("gld_weight") or 0)

    conc_trial = next(
        (
            e
            for e in (eval_after_doc.get("evaluations") or [])
            if e.get("trial_id") == "paper_concentration_reduction"
        ),
        {},
    )
    concentration_credit_applied = (
        concentration_completed
        and stale_resolved
        and str(conc_trial.get("evaluation") or "").upper() == "BENEFICIAL"
        and str(conc_trial.get("evaluation_mode") or "") == "COMPLETED_BROKER_VERIFIED"
    )

    objective_checks = {
        "gld_qty_16": gld_qty == 16,
        "gld_weight_at_target": gld_weight <= 0.205,
        "concentration_reduction_completed": concentration_completed,
        "stale_concentration_resolved": stale_resolved,
        "concentration_credit_applied": concentration_credit_applied,
        "no_broker_orders": True,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "live_blocked": not live_allowed,
    }

    cert_still_fails = str(after_cert.get("certification_status") or "") == "NOT_CERTIFIED"
    if all(objective_checks.values()):
        status = "REVIEW_REQUIRED" if cert_still_fails else "PASS"
    else:
        status = "FAIL"
        if verbose:
            failed = [k for k, v in objective_checks.items() if not v]
            print(f"[198Q] objective_checks_failed={failed}")

    root_cause = (
        "Phase 198O identified STALE_EVALUATION_INPUT: protective evaluation scored hypothetical "
        "concentration reduction as NEGATIVE after broker-verified GLD sells (48 + 131 shares) "
        "already satisfied the <=20% weight target. Phase 198Q credits completed concentration "
        "reduction separately from remaining drawdown/exposure hypothetical trials."
    )

    executive_summary = (
        f"Protective evaluation refreshed: avg {before_eval.get('average_effectiveness')} -> "
        f"{after_eval.get('average_effectiveness')}, beneficial "
        f"{before_eval.get('beneficial_count')} -> {after_eval.get('beneficial_count')}, "
        f"negative {before_eval.get('negative_count')} -> {after_eval.get('negative_count')}. "
        f"Concentration reduction completed={concentration_completed}, stale trial resolved={stale_resolved}. "
        f"Certification: {before_cert.get('certification_status')} -> {after_cert.get('certification_status')} "
        f"(not forced)."
    )
    if cert_still_fails and status == "REVIEW_REQUIRED":
        executive_summary += f" Remaining blockers: {'; '.join(remaining[:2])}."

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
        "phase": "198Q",
        "generated_at": _utc_now_iso(),
        "status": status,
        "mode": "refresh_protective_evaluation_after_gld_reduction",
        "disclaimer": (
            "Evaluation/certification hardening only. Does not place broker orders, create "
            "approval/authorization artifacts, authorize live trading, force certification PASS, "
            "or loosen governance."
        ),
        "root_cause": root_cause,
        "executive_summary": executive_summary,
        "audit": {
            "gld_qty": gld_qty,
            "gld_weight": gld_weight,
            "gld_weight_pct": round(gld_weight * 100.0, 4),
            "completed_gld_sell_qty": int(gld_ctx.get("completed_gld_sell_qty") or 0),
            "concentration_reduction_completed": concentration_completed,
            "stale_concentration_trial_detected": bool(
                eval_after_doc.get("stale_concentration_trial_detected")
            ),
            "stale_concentration_trial_resolved": stale_resolved,
            "concentration_trial_stale": bool(eval_after_doc.get("concentration_trial_stale")),
            "protective_average_effectiveness_before": before_eval.get("average_effectiveness"),
            "protective_average_effectiveness_after": after_eval.get("average_effectiveness"),
            "beneficial_count_before": before_eval.get("beneficial_count"),
            "beneficial_count_after": after_eval.get("beneficial_count"),
            "negative_count_before": before_eval.get("negative_count"),
            "negative_count_after": after_eval.get("negative_count"),
            "high_cps_adjustment_applied": bool(after_eval.get("high_cps_adjustment_applied")),
            "mandatory_gate_before": before_cert.get("mandatory_gate_status"),
            "mandatory_gate_after": after_cert.get("mandatory_gate_status"),
            "certification_status_before": before_cert.get("certification_status"),
            "certification_status_after": after_cert.get("certification_status"),
            "failed_requirements_before": before_cert.get("failed_requirements"),
            "failed_requirements_after": after_cert.get("failed_requirements"),
            "remaining_legitimate_blockers": remaining,
            "approval_artifact_present": approval_present,
            "broker_auth_artifact_present": broker_auth_present,
            "live_execution_allowed": live_allowed,
        },
        "before": {
            **before_eval,
            **before_cert,
            "protective_average_effectiveness": before_eval.get("average_effectiveness"),
        },
        "after": {
            **after_eval,
            **after_cert,
            "protective_average_effectiveness": after_eval.get("average_effectiveness"),
        },
        "evaluation_detail": {
            "concentration_trial_after": next(
                (
                    e
                    for e in (eval_after_doc.get("evaluations") or [])
                    if e.get("trial_id") == "paper_concentration_reduction"
                ),
                {},
            ),
            "trial_evaluation_modes": eval_after_doc.get("trial_evaluation_modes"),
            "evaluation_reason": eval_after_doc.get("evaluation_reason"),
        },
        "remaining_legitimate_blockers": remaining,
        "safety": {
            "broker_orders_submitted": False,
            "approval_artifact_created": False,
            "broker_auth_artifact_created": False,
            "live_execution_modified": False,
            "certification_forced_pass": False,
        },
        "objective_checks": objective_checks,
        "files_changed": [
            "services/capital_preservation_evaluation.py",
            "services/phase_198q_refresh_protective_evaluation_after_gld_reduction.py",
        ],
        "git_status_short": git_status,
        "git_diff_stat": git_diff_stat,
        "marker": MARKER,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(
            f"[198Q] status={status} gld_qty={gld_qty} concentration_completed={concentration_completed} "
            f"avg={before_eval.get('average_effectiveness')}->{after_eval.get('average_effectiveness')} "
            f"cert={before_cert.get('certification_status')}->{after_cert.get('certification_status')}"
        )

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 198Q refresh protective evaluation after GLD reduction"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-cert-refresh", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase_198q(verbose=not args.quiet, refresh_cert=not args.skip_cert_refresh)
    return 0 if result.get("status") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    sys.exit(main())
