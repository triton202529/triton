"""
TRITON Phase 159 — Escalation Consistency Hardening.

Audits escalation transitions, committee alignment, certification coupling,
de-escalation integrity, and adversarial stress scenarios. Simulation only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.capital_preservation_certification import (
    ESCALATION_VETO_STATES,
    compute_preservation_certification,
)
from services.institutional_intelligence import compute_decision_quality_assessment
from services.institutional_protection import compute_risk_committee_oversight
from services.risk_watchdog import (
    ESCALATION_LADDER,
    validate_deescalation_path,
    validate_escalation_transition,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "escalation_consistency_hardening.json"
OUTPUT_CSV = RESULTS / "escalation_consistency_scorecard.csv"

SCORECARD_COLUMNS = [
    "section",
    "scenario",
    "status",
    "detail",
    "generated_at",
]

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_scorecard_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log_tag(tag: str, fields: Dict[str, Any], verbose: bool) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _aggregate_status(statuses: List[str]) -> str:
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "WARN" for s in statuses):
        return "WARN"
    return "PASS"


def _merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_baseline(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    rd = Path(results_dir)
    return {
        "cpi": _read_json(rd / "capital_preservation_intelligence.json"),
        "cpe": _read_json(rd / "capital_preservation_escalation.json"),
        "cpa": _read_json(rd / "capital_preservation_advisory.json"),
        "gov": _read_json(rd / "governance_risk_summary.json"),
        "auth": _read_json(rd / "governance_authorization.json"),
        "readiness": _read_json(rd / "execution_readiness.json"),
        "governor": _read_json(rd / "capital_preservation_governor.json"),
        "adaptive": _read_json(rd / "adaptive_capital_preservation.json"),
        "evaluation": _read_json(rd / "protective_action_evaluation.json"),
        "alerts": _read_json(rd / "watchdog_alerts.json"),
        "exec": _read_json(rd / "executive_risk_summary.json"),
        "sim": _read_json(rd / "defensive_simulation_results.json"),
        "accountability": _read_json(rd / "accountability_registry.json"),
        "audit": _read_json(rd / "capital_preservation_audit.json"),
        "policy": _read_json(rd / "protective_action_policy.json"),
        "queue": _read_json(rd / "human_approval_queue.json"),
        "candidates": _read_json(rd / "defensive_action_candidates.json"),
        "cert": _read_json(rd / "capital_preservation_certification.json"),
    }


def _committee_stack(
    baseline: Dict[str, Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cpi = _merge(baseline["cpi"], overrides.get("cpi", {}))
    cpe = _merge(baseline["cpe"], overrides.get("cpe", {}))
    gov = _merge(baseline["gov"], overrides.get("gov", {}))
    auth = _merge(baseline["auth"], overrides.get("auth", {}))
    governor = _merge(baseline["governor"], overrides.get("governor", {}))
    readiness = _merge(baseline["readiness"], overrides.get("readiness", {}))
    cert = compute_preservation_certification(
        cpi_doc=cpi,
        alerts_doc=_merge(baseline["alerts"], overrides.get("alerts", {})),
        gov_doc=gov,
        auth_doc=auth,
        readiness_doc=readiness,
        sim_doc=_merge(baseline["sim"], overrides.get("sim", {})),
        evaluation_doc=_merge(baseline["evaluation"], overrides.get("evaluation", {})),
        governor_doc=governor,
        cpe_doc=cpe,
    )
    committee = compute_risk_committee_oversight(
        cpi_doc=cpi,
        gov_doc=gov,
        auth_doc=auth,
        governor_doc=governor,
        cpe_doc=cpe,
        readiness_doc=readiness,
        cert_doc=cert,
    )
    return {"cpi": cpi, "cpe": cpe, "gov": gov, "cert": cert, "committee": committee}


def _audit_transition_integrity(*, verbose: bool) -> Tuple[str, List[Dict[str, str]]]:
    """Part A — escalation transition audit."""
    rows: List[Dict[str, str]] = []
    statuses: List[str] = []

    forward_pairs = list(zip(ESCALATION_LADDER[:-1], ESCALATION_LADDER[1:]))
    reverse_pairs = list(zip(reversed(ESCALATION_LADDER[1:]), reversed(ESCALATION_LADDER[:-1])))

    for from_state, to_state in forward_pairs + reverse_pairs:
        status, reason = validate_escalation_transition(from_state, to_state, verbose=verbose)
        statuses.append(status)
        rows.append(
            {
                "section": "transition_integrity",
                "scenario": f"{from_state}_to_{to_state}",
                "status": status,
                "detail": reason,
            }
        )

    invalid_cases = [
        ("GREEN", "ORANGE"),
        ("GREEN", "RED"),
        ("RED", "GREEN"),
        ("RED", "YELLOW"),
        ("ORANGE", "GREEN"),
    ]
    for from_state, to_state in invalid_cases:
        status, reason = validate_escalation_transition(from_state, to_state, verbose=verbose)
        expected = "FAIL"
        ok = status == expected
        statuses.append("PASS" if ok else "FAIL")
        rows.append(
            {
                "section": "transition_integrity",
                "scenario": f"reject_{from_state}_to_{to_state}",
                "status": "PASS" if ok else "FAIL",
                "detail": f"expected={expected} got={status} reason={reason}",
            }
        )

    result = _aggregate_status(statuses)
    _log_tag("ESCALATION_TRANSITION_AUDIT", {"status": result, "cases": len(rows)}, verbose)
    return result, rows


def _audit_committee_consistency(
    *,
    baseline: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    """Part B — committee consistency across oversight layers."""
    rows: List[Dict[str, str]] = []
    statuses: List[str] = []

    scenarios = [
        (
            "green_healthy_cps",
            {
                "cpi": {"capital_preservation_score": 85, "health_band": "HEALTHY"},
                "cpe": {"escalation_state": "GREEN"},
                "governor": {"preservation_posture": "GREEN"},
            },
            ("CLEAR", "MONITOR"),
        ),
        (
            "yellow_elevated",
            {
                "cpi": {"capital_preservation_score": 65, "health_band": "ELEVATED_RISK"},
                "cpe": {"escalation_state": "YELLOW"},
                "governor": {"preservation_posture": "YELLOW"},
            },
            ("MONITOR", "REVIEW_REQUIRED"),
        ),
        (
            "red_critical",
            {
                "cpi": {"capital_preservation_score": 28, "health_band": "CRITICAL"},
                "cpe": {
                    "escalation_state": "RED",
                    "escalation_reason": ["EXCESS_POSITION_DRAWDOWN"],
                },
                "governor": {"preservation_posture": "RED"},
            },
            ("REVIEW_REQUIRED", "ESCALATE"),
        ),
    ]

    for name, overrides, allowed in scenarios:
        stack = _committee_stack(baseline, overrides)
        committee_status = str(stack["committee"].get("committee_status") or "")
        aligned = committee_status in allowed
        status = "PASS" if aligned else "WARN"
        statuses.append(status)
        _log_tag(
            "COMMITTEE_CONSISTENCY",
            {
                "committee": "risk_committee",
                "recommendation": committee_status,
                "status": status,
                "scenario": name,
            },
            verbose,
        )
        rows.append(
            {
                "section": "committee_consistency",
                "scenario": name,
                "status": status,
                "detail": f"committee={committee_status} allowed={allowed}",
            }
        )

    result = _aggregate_status(statuses)
    return result, rows


def _audit_certification_alignment(
    *,
    baseline: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    """Part C — certification must not certify at RED; may certify at GREEN."""
    rows: List[Dict[str, str]] = []
    statuses: List[str] = []

    red_stack = _committee_stack(
        baseline,
        {
            "cpi": {"capital_preservation_score": 30, "health_band": "CRITICAL"},
            "cpe": {"escalation_state": "RED"},
            "governor": {"preservation_posture": "RED"},
        },
    )
    red_cert = str(red_stack["cert"].get("certification_status") or "")
    red_ok = red_cert != "CERTIFIED_FOR_PAPER_PROTECTION"
    statuses.append("PASS" if red_ok else "FAIL")
    rows.append(
        {
            "section": "certification_alignment",
            "scenario": "red_not_certified",
            "status": "PASS" if red_ok else "FAIL",
            "detail": f"certification_status={red_cert} veto_states={sorted(ESCALATION_VETO_STATES)}",
        }
    )

    green_stack = _committee_stack(
        baseline,
        {
            "cpi": {"capital_preservation_score": 88, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
            "gov": {"governance_awareness_level": "GREEN"},
            "auth": {
                "overall_authorization": True,
                "governance_authorized": True,
                "paper_execution_permitted": True,
                "live_execution_permitted": False,
            },
            "readiness": {
                "readiness_status": "READY",
                "checks_passing_count": 8,
                "checks_total": 8,
                "failed_checks": [],
                "live_execution_permitted": False,
            },
            "governor": {"preservation_posture": "GREEN", "governor_confidence": 95},
            "evaluation": {
                "average_effectiveness": 75.0,
                "evaluation_count": 5,
                "beneficial_count": 4,
                "negative_count": 0,
                "neutral_count": 1,
            },
        },
    )
    green_cert = str(green_stack["cert"].get("certification_status") or "")
    green_ok = green_cert in (
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "PARTIALLY_CERTIFIED",
        "NOT_CERTIFIED",
    )
    statuses.append("PASS" if green_ok else "WARN")
    rows.append(
        {
            "section": "certification_alignment",
            "scenario": "green_may_certify",
            "status": "PASS" if green_ok else "WARN",
            "detail": f"certification_status={green_cert}",
        }
    )

    result = _aggregate_status(statuses)
    _log_tag("CERTIFICATION_ESCALATION_ALIGNMENT", {"status": result}, verbose)
    return result, rows


def _audit_deescalation_integrity(*, verbose: bool) -> Tuple[str, List[Dict[str, str]]]:
    """Part D — de-escalation hardening."""
    rows: List[Dict[str, str]] = []
    statuses: List[str] = []

    valid_path = ["RED", "ORANGE", "YELLOW", "GREEN"]
    invalid_path = ["RED", "GREEN"]
    for label, path in (("orderly", valid_path), ("direct_red_green", invalid_path)):
        status, reason = validate_deescalation_path(path)
        expected = "PASS" if label == "orderly" else "FAIL"
        ok = status == expected
        statuses.append("PASS" if ok else "FAIL")
        rows.append(
            {
                "section": "de_escalation_integrity",
                "scenario": label,
                "status": "PASS" if ok else "FAIL",
                "detail": f"path={'->'.join(path)} result={status} reason={reason}",
            }
        )

    result = _aggregate_status(statuses)
    _log_tag("DE_ESCALATION_AUDIT", {"status": result}, verbose)
    return result, rows


def _audit_stress_suite(
    *,
    baseline: Dict[str, Dict[str, Any]],
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    """Part E — escalation stress scenarios."""
    rows: List[Dict[str, str]] = []
    statuses: List[str] = []

    def run_case(
        name: str,
        overrides: Dict[str, Dict[str, Any]],
        check,
    ) -> None:
        stack = _committee_stack(baseline, overrides)
        decision = compute_decision_quality_assessment(
            results_dir=results_dir,
            cpa_doc=baseline["cpa"],
            cpe_doc=stack["cpe"],
            cpi_doc=stack["cpi"],
            alerts_doc=baseline["alerts"],
            gov_doc=stack["gov"],
            auth_doc=_merge(baseline["auth"], overrides.get("auth", {})),
            readiness_doc=_merge(baseline["readiness"], overrides.get("readiness", {})),
            governor_doc=_merge(baseline["governor"], overrides.get("governor", {})),
            evaluation_doc=_merge(baseline["evaluation"], overrides.get("evaluation", {})),
            adaptive_doc=baseline["adaptive"],
        )
        ok, detail = check(stack, decision)
        status = "PASS" if ok else "WARN"
        statuses.append(status)
        _log_tag("ESCALATION_STRESS", {"scenario": name, "result": status}, verbose)
        rows.append(
            {
                "section": "stress_suite",
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )

    run_case(
        "false_crisis",
        {
            "cpi": {"capital_preservation_score": 88, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "RED"},
            "governor": {"preservation_posture": "GREEN"},
        },
        lambda stack, decision: (
            str(stack["committee"]["committee_status"])
            in ("REVIEW_REQUIRED", "ESCALATE", "MONITOR"),
            f"committee={stack['committee']['committee_status']}",
        ),
    )

    run_case(
        "true_crisis",
        {
            "cpi": {
                "capital_preservation_score": 25,
                "health_band": "CRITICAL",
                "component_scores": {"drawdown": 15, "concentration": 20},
            },
            "cpe": {
                "escalation_state": "RED",
                "escalation_reason": ["EXCESS_POSITION_DRAWDOWN"],
            },
            "governor": {"preservation_posture": "RED"},
        },
        lambda stack, decision: (
            str(stack["committee"]["committee_status"]) in ("REVIEW_REQUIRED", "ESCALATE"),
            f"committee={stack['committee']['committee_status']}",
        ),
    )

    run_case(
        "optimistic_governance_mismatch",
        {
            "cpi": {"capital_preservation_score": 32, "health_band": "CRITICAL"},
            "cpe": {"escalation_state": "RED"},
            "gov": {"governance_awareness_level": "GREEN"},
            "governor": {"preservation_posture": "GREEN", "governor_confidence": 98},
        },
        lambda stack, decision: (
            next(
                int(m.get("score") or 0)
                for m in decision.get("metrics", [])
                if m.get("metric") == "Escalation Consistency"
            )
            < 70,
            f"escalation_consistency="
            f"{next(int(m.get('score') or 0) for m in decision.get('metrics', []) if m.get('metric') == 'Escalation Consistency')}",
        ),
    )

    run_case(
        "green_recovery",
        {
            "cpi": {"capital_preservation_score": 86, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
            "governor": {"preservation_posture": "GREEN"},
        },
        lambda stack, decision: (
            str(stack["committee"]["committee_status"]) in ("CLEAR", "MONITOR"),
            f"committee={stack['committee']['committee_status']}",
        ),
    )

    deesc_status, deesc_reason = validate_deescalation_path(["RED", "ORANGE", "YELLOW", "GREEN"])
    statuses.append("PASS" if deesc_status == "PASS" else "FAIL")
    rows.append(
        {
            "section": "stress_suite",
            "scenario": "improving_recovery_path",
            "status": "PASS" if deesc_status == "PASS" else "FAIL",
            "detail": f"deescalation={deesc_status} reason={deesc_reason}",
        }
    )
    _log_tag(
        "ESCALATION_STRESS",
        {"scenario": "improving_recovery_path", "result": deesc_status},
        verbose,
    )

    result = _aggregate_status(statuses)
    return result, rows


def _revalidate_institutional_intelligence(
    *,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, str]:
    """Part G — re-run institutional intelligence stress test."""
    prior = _read_json(results_dir / "institutional_intelligence_stress_test.json")
    before = str(prior.get("overall_result") or "UNKNOWN")

    from services.institutional_intelligence_stress_test import (
        compute_institutional_intelligence_stress_test,
    )

    report = compute_institutional_intelligence_stress_test(
        results_dir=results_dir, verbose=verbose
    )
    after = str(report.get("overall_result") or "UNKNOWN")
    _log_tag(
        "INSTITUTIONAL_REVALIDATION",
        {"before": before, "after": after},
        verbose,
    )

    decision = (
        report.get("part_i_final_decision")
        or report.get("institutional_intelligence_certification")
        or {}
    )
    new_status = str(
        decision.get("institutional_intelligence_certification_status")
        or decision.get("status")
        or after
    )
    _log_tag(
        "INSTITUTIONAL_INTELLIGENCE_CERTIFICATION",
        {
            "previous_status": before,
            "new_status": new_status,
            "reason": str(decision.get("recommendation", ""))[:120],
        },
        verbose,
    )
    if new_status == "STRESS_TESTED_CERTIFIED":
        _log_tag("INSTITUTIONAL_INTELLIGENCE_CERTIFIED", {"status": new_status}, verbose)

    return before, after


def compute_escalation_consistency_hardening(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
    skip_revalidation: bool = False,
) -> Dict[str, Any]:
    """Run Phase 159 escalation consistency hardening audits."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    baseline = _load_baseline(rd)
    csv_rows: List[Dict[str, str]] = []

    part_a, rows_a = _audit_transition_integrity(verbose=verbose)
    part_b, rows_b = _audit_committee_consistency(baseline=baseline, verbose=verbose)
    part_c, rows_c = _audit_certification_alignment(baseline=baseline, verbose=verbose)
    part_d, rows_d = _audit_deescalation_integrity(verbose=verbose)
    part_e, rows_e = _audit_stress_suite(baseline=baseline, results_dir=rd, verbose=verbose)

    for block in (rows_a, rows_b, rows_c, rows_d, rows_e):
        for row in block:
            row["generated_at"] = ts
            csv_rows.append(row)

    overall = _aggregate_status([part_a, part_b, part_c, part_d, part_e])

    revalidation: Dict[str, str] = {}
    if not skip_revalidation:
        before, after = _revalidate_institutional_intelligence(results_dir=rd, verbose=verbose)
        revalidation = {"before": before, "after": after}

    payload = {
        "generated_at": ts,
        "phase": "159",
        "mode": "simulation_only_hardening",
        "transition_integrity": part_a,
        "committee_consistency": part_b,
        "certification_alignment": part_c,
        "de_escalation_integrity": part_d,
        "stress_suite": part_e,
        "overall_result": overall,
        "institutional_revalidation": revalidation,
    }

    _atomic_write_json(payload, OUTPUT_JSON)
    _write_scorecard_csv(csv_rows, OUTPUT_CSV)

    _log_tag(
        "ESCALATION_CONSISTENCY_HARDENING",
        {"overall_result": overall, "path": str(OUTPUT_JSON)},
        verbose,
    )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 159 — Escalation Consistency Hardening (simulation only)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print audit tags to stdout")
    parser.add_argument(
        "--skip-revalidation",
        action="store_true",
        help="Skip embedded institutional intelligence stress re-run",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing institutional result artifacts",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = compute_escalation_consistency_hardening(
        results_dir=args.results_dir,
        verbose=args.verbose,
        skip_revalidation=args.skip_revalidation,
    )
    return 0 if report.get("overall_result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
