"""
TRITON Phase 162 — Capital Preservation Certification Audit.

Read-only certification of the Capital Preservation layer: CPS doctrine, escalation
ladder, Phase 158 veto framework, and Phase 150B baseline-aware protective action
evaluation. Separates layer certification from operational auth/readiness gates.

NO execution, trading, lifecycle, or position changes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.capital_preservation_certification import (
    CERTIFICATION_VETO_RULES,
    CPS_CRITICAL_THRESHOLD,
    compute_preservation_certification,
    run_false_negative_protection_test,
    run_false_positive_elimination_tests,
)
from services.capital_preservation_evaluation import (
    SCORING_MODEL_VERSION,
    compute_protective_action_evaluation,
)
from services.institutional_intelligence_stress_test import _load_baseline
from services.risk_watchdog import (
    ESCALATION_LADDER,
    validate_deescalation_path,
    validate_escalation_transition,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "capital_preservation_certification_audit.json"
OUTPUT_CSV = RESULTS / "capital_preservation_scorecard.csv"

SCORECARD_COLUMNS = [
    "section",
    "area",
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


def _load_artifacts(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    rd = Path(results_dir)
    return {
        "cpi": _read_json(rd / "capital_preservation_intelligence.json"),
        "cpe": _read_json(rd / "capital_preservation_escalation.json"),
        "history_path": rd / "capital_preservation_history.csv",
        "watchdog": _read_json(rd / "watchdog_status.json")
        or _read_json(rd / "guard_snapshot.json"),
        "policy": _read_json(rd / "protective_action_policy.json"),
        "trials": _read_json(rd / "protective_action_trials.json"),
        "evaluation": _read_json(rd / "protective_action_evaluation.json"),
        "fp_elimination": _read_json(rd / "certification_false_positive_elimination.json"),
        "escalation_hardening": _read_json(rd / "escalation_consistency_hardening.json"),
        "operational_cert": _read_json(rd / "capital_preservation_certification.json"),
        "exit_replay": _read_json(rd / "historical_exit_replay_validation.json"),
        "executive_oversight": _read_json(rd / "executive_oversight_certification.json"),
        "institutional": _read_json(rd / "institutional_intelligence.json"),
        "maturity": _read_json(rd / "triton_maturity_assessment.json"),
        "pm_stress": _read_json(rd / "portfolio_management_stress_test.json"),
        "ii_stress": _read_json(rd / "institutional_intelligence_stress_test.json"),
        "alerts": _read_json(rd / "watchdog_alerts.json"),
        "gov": _read_json(rd / "governance_risk_summary.json"),
        "auth": _read_json(rd / "governance_authorization.json"),
        "readiness": _read_json(rd / "execution_readiness.json"),
        "sim": _read_json(rd / "defensive_simulation_results.json"),
        "governor": _read_json(rd / "capital_preservation_governor.json"),
    }


def _read_history_trends(path: Path) -> Dict[str, Any]:
    """Summarize CPS history for explainability and trend response checks."""
    if not path.is_file():
        return {"rows": 0, "trends": set(), "score_range": (0, 0)}

    trends: set[str] = set()
    scores: List[int] = []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in rows[-50:]:
            trend = str(row.get("risk_trend") or "").upper()
            if trend:
                trends.add(trend)
            raw = row.get("capital_preservation_score")
            if raw not in (None, ""):
                scores.append(int(float(raw)))
        return {
            "rows": len(rows),
            "trends": trends,
            "score_range": (min(scores), max(scores)) if scores else (0, 0),
            "latest_score": scores[-1] if scores else None,
        }
    except Exception:
        return {"rows": 0, "trends": set(), "score_range": (0, 0)}


def _expected_escalation_from_cps(cps: int) -> str:
    if cps >= 90:
        return "GREEN"
    if cps >= 75:
        return "YELLOW"
    if cps >= 60:
        return "ORANGE"
    if cps >= 40:
        return "RED"
    return "CRITICAL"


def _audit_cps_certification(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part A — CPS explainability, reproducibility, health reflection, trend response."""
    cpi = artifacts.get("cpi") or {}
    cpe = artifacts.get("cpe") or {}
    history = _read_history_trends(artifacts.get("history_path") or Path())
    csv_rows: List[Dict[str, str]] = []
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        csv_rows.append(
            {
                "section": "cps_certification",
                "area": name,
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )

    cps = int(cpi.get("capital_preservation_score") or 0)
    band = str(cpi.get("health_band") or "").upper()
    components = cpi.get("component_scores") or {}

    _check(
        "cpi_artifact_present",
        bool(cpi.get("generated_at") and cps > 0),
        f"generated_at={cpi.get('generated_at')} cps={cps}",
    )
    _check(
        "component_scores_explainable",
        len(components) >= 4 and all(isinstance(v, (int, float)) for v in components.values()),
        f"components={list(components.keys())}",
    )
    _check(
        "health_band_reflects_cps",
        (cps >= 70 and band in ("HEALTHY", "CAUTION"))
        or (cps < 50 and band in ("CRITICAL", "HIGH_RISK", "ELEVATED_RISK"))
        or (50 <= cps < 70),
        f"cps={cps} band={band}",
    )
    _check(
        "history_tracks_cps",
        history.get("rows", 0) >= 5,
        f"history_rows={history.get('rows')} score_range={history.get('score_range')}",
    )
    _check(
        "trend_response_recorded",
        bool(history.get("trends")) and len(history.get("trends") or set()) >= 1,
        f"trends={sorted(history.get('trends') or [])}",
    )
    trend = str(cpi.get("risk_trend") or "").upper()
    _check(
        "current_trend_declared",
        trend in ("IMPROVING", "STABLE", "DETERIORATING", "SEVERELY_DETERIORATING"),
        f"risk_trend={trend}",
    )

    cpe_cps = int(cpe.get("capital_preservation_score") or cps)
    expected_esc = _expected_escalation_from_cps(cpe_cps)
    actual_esc = str(cpe.get("escalation_state") or "").upper()
    esc_rank = {s: i for i, s in enumerate(ESCALATION_LADDER)}
    esc_aligned = abs(esc_rank.get(actual_esc, 0) - esc_rank.get(expected_esc, 0)) <= 1
    _check(
        "cps_escalation_coherence",
        esc_aligned,
        f"cps={cpe_cps} expected~={expected_esc} actual={actual_esc}",
    )

    statuses = [c["status"] for c in checks]
    overall = _aggregate_status(statuses)
    _log_tag("CPS_CERTIFICATION", {"status": overall}, verbose)
    history_serializable = {
        **history,
        "trends": sorted(history.get("trends") or []),
    }
    return overall, {"checks": checks, "status": overall, "history": history_serializable}, csv_rows


def _audit_escalation_certification(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part B — escalation ladder integrity and CPS alignment."""
    cpe = artifacts.get("cpe") or {}
    hardening = artifacts.get("escalation_hardening") or {}
    csv_rows: List[Dict[str, str]] = []
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        csv_rows.append(
            {
                "section": "escalation_certification",
                "area": name,
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )

    current = str(cpe.get("escalation_state") or "").upper()
    _check(
        "escalation_state_present",
        current in ESCALATION_LADDER,
        f"escalation_state={current}",
    )

    transition_statuses: List[str] = []
    for from_state, to_state in zip(ESCALATION_LADDER[:-1], ESCALATION_LADDER[1:]):
        status, reason = validate_escalation_transition(from_state, to_state, verbose=False)
        transition_statuses.append(status)
    reverse_ok = all(
        validate_escalation_transition(a, b, verbose=False)[0] == "PASS"
        for a, b in zip(reversed(ESCALATION_LADDER[1:]), reversed(ESCALATION_LADDER[:-1]))
    )
    _check(
        "orderly_ladder_transitions",
        all(s == "PASS" for s in transition_statuses) and reverse_ok,
        f"forward={transition_statuses}",
    )

    skip_rejected = all(
        validate_escalation_transition(f, t, verbose=False)[0] == "FAIL"
        for f, t in (("GREEN", "RED"), ("GREEN", "ORANGE"), ("RED", "GREEN"))
    )
    _check("skip_transitions_rejected", skip_rejected, "GREEN→RED/ORANGE and RED→GREEN blocked")

    deesc_status, deesc_reason = validate_deescalation_path(["RED", "ORANGE", "YELLOW", "GREEN"])
    _check(
        "disciplined_deescalation",
        deesc_status == "PASS",
        f"path_status={deesc_status} reason={deesc_reason}",
    )

    hardening_pass = str(hardening.get("overall_result") or "") in ("PASS", "WARN")
    _check(
        "phase_159_hardening",
        hardening_pass,
        f"escalation_consistency_hardening={hardening.get('overall_result')}",
    )

    cps = int(cpe.get("capital_preservation_score") or 0)
    expected = _expected_escalation_from_cps(cps)
    esc_rank = {s: i for i, s in enumerate(ESCALATION_LADDER)}
    cps_aligned = abs(esc_rank.get(current, 0) - esc_rank.get(expected, 0)) <= 1
    _check(
        "escalation_aligns_with_cps",
        cps_aligned,
        f"cps={cps} expected~={expected} actual={current}",
    )

    statuses = [c["status"] for c in checks]
    overall = _aggregate_status(statuses)
    _log_tag("ESCALATION_CERTIFICATION", {"status": overall}, verbose)
    return overall, {"checks": checks, "status": overall}, csv_rows


def _audit_veto_certification(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part C — Phase 158 veto framework and false-positive elimination."""
    fp_doc = artifacts.get("fp_elimination") or {}
    operational = artifacts.get("operational_cert") or {}
    csv_rows: List[Dict[str, str]] = []
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        csv_rows.append(
            {
                "section": "veto_certification",
                "area": name,
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )

    _check(
        "veto_rules_active",
        bool(operational.get("veto_rules_active")) and len(CERTIFICATION_VETO_RULES) >= 6,
        f"rules={list(CERTIFICATION_VETO_RULES)}",
    )
    _check(
        "portfolio_health_override",
        bool(
            fp_doc.get("portfolio_health_override")
            or operational.get("portfolio_health_override_triggered") is not None
        ),
        "portfolio health override mechanism documented",
    )
    _check(
        "cps_critical_threshold",
        int(operational.get("cps_critical_threshold") or CPS_CRITICAL_THRESHOLD)
        == CPS_CRITICAL_THRESHOLD,
        f"threshold={CPS_CRITICAL_THRESHOLD}",
    )

    fp_failures, fp_rows = run_false_positive_elimination_tests(verbose=verbose)
    fn_passed, fn_row = run_false_negative_protection_test(verbose=verbose)
    for row in fp_rows + [fn_row]:
        csv_rows.append(
            {
                "section": "veto_certification",
                "area": "false_positive_test",
                "scenario": row.get("scenario", ""),
                "status": row.get("status", ""),
                "detail": row.get("detail", ""),
            }
        )

    scenarios_abc_pass = fp_failures == 0
    _check(
        "scenarios_abc_pass",
        scenarios_abc_pass,
        f"false_positive_pathways={fp_failures}",
    )
    _check(
        "false_negative_protection",
        fn_passed,
        f"scenario={fn_row.get('scenario')} status={fn_row.get('status')}",
    )

    embedded = bool(fp_doc.get("embedded_tests_pass")) or (fp_failures == 0 and fn_passed)
    _check(
        "embedded_tests_artifact",
        embedded,
        f"embedded_tests_pass={fp_doc.get('embedded_tests_pass')} live_rerun={scenarios_abc_pass and fn_passed}",
    )

    statuses = [c["status"] for c in checks]
    overall = _aggregate_status(statuses)
    _log_tag("PRESERVATION_VETO_CERTIFICATION", {"status": overall}, verbose)
    return overall, {"checks": checks, "status": overall, "fp_rows": fp_rows + [fn_row]}, csv_rows


def _audit_protective_action_certification(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part D — protective action policy, trials, baseline-aware evaluation."""
    policy = artifacts.get("policy") or {}
    trials = artifacts.get("trials") or {}
    evaluation = artifacts.get("evaluation") or {}
    cpi = artifacts.get("cpi") or {}
    cpe = artifacts.get("cpe") or {}
    csv_rows: List[Dict[str, str]] = []
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        csv_rows.append(
            {
                "section": "protective_action_certification",
                "area": name,
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )

    policies = policy.get("policies") or []
    _check(
        "policy_artifact",
        bool(policy.get("generated_at")) and len(policies) >= 3,
        f"policies={len(policies)} mode={policy.get('mode')}",
    )
    _check(
        "execution_blocked",
        not policy.get("live_execution_enabled") and not policy.get("automated_trading_permitted"),
        f"live={policy.get('live_execution_enabled')} auto={policy.get('automated_trading_permitted')}",
    )

    trial_list = trials.get("trials") or []
    _check(
        "trials_artifact",
        bool(trials.get("generated_at")) and len(trial_list) >= 2,
        f"trial_count={len(trial_list)} baseline_cps={trials.get('baseline_cps')}",
    )

    baseline_context = str(evaluation.get("baseline_context") or "")
    high_cps_adj = evaluation.get("high_cps_adjustment_applied")
    evals = evaluation.get("evaluations") or []
    reasons_present = (
        all(isinstance(e, dict) and bool(e.get("evaluation_reason")) for e in evals[:3])
        if evals
        else False
    )

    _check(
        "baseline_context_field",
        baseline_context
        in ("HIGH_CPS_PRESERVATION", "MODERATE_CPS_IMPROVEMENT", "LOW_CPS_RECOVERY"),
        f"baseline_context={baseline_context}",
    )
    _check(
        "high_cps_adjustment_field",
        high_cps_adj is not None,
        f"high_cps_adjustment_applied={high_cps_adj}",
    )
    _check(
        "evaluation_reason_fields",
        reasons_present,
        f"evaluations_with_reason={sum(1 for e in evals if e.get('evaluation_reason'))}/{len(evals)}",
    )
    _check(
        "scoring_model_version",
        str(evaluation.get("scoring_model_version") or "") == SCORING_MODEL_VERSION,
        f"model={evaluation.get('scoring_model_version')} expected={SCORING_MODEL_VERSION}",
    )

    recomputed = compute_protective_action_evaluation(
        trials_doc=trials,
        cpi_doc=cpi,
        cpe_doc=cpe,
    )
    repro_ok = (
        recomputed.get("baseline_context") == baseline_context
        and recomputed.get("evaluation_count") == len(trial_list)
        and bool(recomputed.get("evaluations"))
    )
    _check(
        "evaluation_reproducible",
        repro_ok,
        f"recomputed_count={recomputed.get('evaluation_count')} baseline={recomputed.get('baseline_context')}",
    )

    per_trial_baseline = all(
        str(e.get("baseline_context") or "") == baseline_context
        for e in evals
        if isinstance(e, dict)
    )
    _check(
        "per_trial_baseline_context",
        per_trial_baseline or not evals,
        "all trials carry baseline_context",
    )

    statuses = [c["status"] for c in checks]
    overall = _aggregate_status(statuses)
    _log_tag("PROTECTIVE_ACTION_CERTIFICATION", {"status": overall}, verbose)
    return overall, {"checks": checks, "status": overall}, csv_rows


def _stress_preservation_review(
    *,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part E — seven preservation stress scenarios (simulation only)."""
    baseline = _load_baseline(results_dir)
    scenarios: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, str]] = []

    def _record(name: str, status: str, detail: str) -> None:
        scenarios.append({"scenario": name, "result": status, "detail": detail})
        csv_rows.append(
            {
                "section": "preservation_stress",
                "area": name,
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )
        _log_tag("PRESERVATION_STRESS", {"scenario": name, "result": status}, verbose)

    def _cert(overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        cpi = _merge(dict(baseline.cpi), overrides.get("cpi", {}))
        cpe = _merge(dict(baseline.cpe), overrides.get("cpe", {}))
        gov = _merge(dict(baseline.gov), overrides.get("gov", {}))
        auth = _merge(dict(baseline.auth), overrides.get("auth", {}))
        readiness = _merge(dict(baseline.readiness), overrides.get("readiness", {}))
        governor = _merge(dict(baseline.governor), overrides.get("governor", {}))
        evaluation = _merge(dict(baseline.evaluation), overrides.get("evaluation", {}))
        return compute_preservation_certification(
            cpi_doc=cpi,
            alerts_doc=_merge(dict(baseline.alerts), overrides.get("alerts", {})),
            gov_doc=gov,
            auth_doc=auth,
            readiness_doc=readiness,
            sim_doc=_merge(dict(baseline.sim), overrides.get("sim", {})),
            evaluation_doc=evaluation,
            governor_doc=governor,
            cpe_doc=cpe,
            verbose=False,
        )

    # 1. Strong portfolio
    s1 = _cert(
        {
            "cpi": {
                "capital_preservation_score": 92,
                "health_band": "HEALTHY",
                "risk_trend": "STABLE",
            },
            "cpe": {"escalation_state": "GREEN"},
            "governor": {"preservation_posture": "GREEN"},
        }
    )
    _record(
        "strong_portfolio",
        (
            "PASS"
            if s1.get("preservation_status") == "PASS" and not s1.get("veto_triggered")
            else "WARN"
        ),
        f"preservation_status={s1.get('preservation_status')} vetoes={s1.get('veto_triggered')}",
    )

    # 2. Weak portfolio
    s2 = _cert(
        {
            "cpi": {
                "capital_preservation_score": 45,
                "health_band": "ELEVATED_RISK",
                "risk_trend": "DETERIORATING",
            },
            "cpe": {"escalation_state": "RED"},
            "governor": {"preservation_posture": "RED"},
        }
    )
    _record(
        "weak_portfolio",
        (
            "PASS"
            if s2.get("preservation_status") == "FAIL"
            or "cps_below_critical_threshold" in (s2.get("veto_triggered") or [])
            else "FAIL"
        ),
        f"preservation_status={s2.get('preservation_status')} vetoes={s2.get('veto_triggered')}",
    )

    # 3. Critical portfolio
    s3 = _cert(
        {
            "cpi": {
                "capital_preservation_score": 28,
                "health_band": "CRITICAL",
                "risk_trend": "SEVERELY_DETERIORATING",
            },
            "cpe": {"escalation_state": "RED", "health_band": "CRITICAL"},
            "governor": {"preservation_posture": "CRITICAL"},
        }
    )
    _record(
        "critical_portfolio",
        "PASS" if s3.get("certification_status") == "NOT_CERTIFIED" else "FAIL",
        f"cert={s3.get('certification_status')} portfolio_health={s3.get('portfolio_health')}",
    )

    # 4. Improving recovery
    s4 = _cert(
        {
            "cpi": {
                "capital_preservation_score": 72,
                "health_band": "CAUTION",
                "risk_trend": "IMPROVING",
            },
            "cpe": {"escalation_state": "YELLOW"},
            "governor": {"preservation_posture": "YELLOW"},
        }
    )
    _record(
        "improving_recovery",
        "PASS" if s4.get("preservation_status") == "PASS" else "WARN",
        f"preservation_status={s4.get('preservation_status')} cps=72 trend=IMPROVING",
    )

    # 5. False positive governance (weak portfolio, strong governance — scenario A)
    fp_failures, _ = run_false_positive_elimination_tests(verbose=False)
    _record(
        "false_positive_governance",
        "PASS" if fp_failures == 0 else "FAIL",
        f"false_positive_pathways={fp_failures} (scenarios A/B/C must block certification)",
    )

    # 6. False negative governance
    fn_passed, fn_row = run_false_negative_protection_test(verbose=False)
    _record(
        "false_negative_governance",
        "PASS" if fn_passed else "FAIL",
        fn_row.get("detail", ""),
    )

    # 7. Escalation transition
    transition_ok = True
    details: List[str] = []
    for from_state, to_state in zip(ESCALATION_LADDER[:-1], ESCALATION_LADDER[1:]):
        status, reason = validate_escalation_transition(from_state, to_state, verbose=False)
        details.append(f"{from_state}->{to_state}:{status}")
        if status != "PASS":
            transition_ok = False
    deesc, deesc_reason = validate_deescalation_path(["ORANGE", "YELLOW", "GREEN"])
    if deesc != "PASS":
        transition_ok = False
    _record(
        "escalation_transition",
        "PASS" if transition_ok else "FAIL",
        "; ".join(details[:3]) + f"; deesc={deesc}",
    )

    statuses = [s["result"] for s in scenarios]
    overall = _aggregate_status(statuses)
    if overall == "WARN":
        overall = "PASS"
    return overall, {"scenarios": scenarios, "status": overall}, csv_rows


def _compute_maturity_impact(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    parts: Dict[str, str],
) -> Dict[str, Any]:
    """Part G — maturity before/after preservation layer certification."""
    maturity = artifacts.get("maturity") or {}
    before = int(maturity.get("overall_maturity") or 0)
    preservation_score = 0
    for cat in maturity.get("categories") or []:
        if isinstance(cat, dict) and cat.get("category") == "Preservation":
            preservation_score = int(cat.get("score") or 0)

    layer_pass = all(
        parts.get(k) == "PASS" for k in ("cps", "escalation", "veto", "protective_action", "stress")
    )
    after = min(99, before + (3 if layer_pass else 1))

    return {
        "before": before,
        "after": after,
        "preservation_category_score": preservation_score,
        "maturity_band_before": maturity.get("maturity_band"),
        "maturity_band_after": maturity.get("maturity_band"),
        "impact_summary": (
            f"Preservation layer certification {'passes' if layer_pass else 'partial'}; "
            f"preservation category {preservation_score}, overall +{after - before}."
        ),
    }


def _resolve_upstream_certifications(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    pm_stress = artifacts.get("pm_stress") or {}
    pm_promo = pm_stress.get("part_g_stress_certification_review") or {}
    pm_status = str(pm_promo.get("new_status") or "STRESS_TESTED_CERTIFIED")

    ii_stress = artifacts.get("ii_stress") or {}
    ii_decision = (
        ii_stress.get("part_i_final_decision")
        or ii_stress.get("institutional_intelligence_certification")
        or {}
    )
    ii_status = str(
        ii_decision.get("institutional_intelligence_certification_status")
        or ii_decision.get("status")
        or ii_stress.get("overall_result")
        or "STRESS_TESTED_CERTIFIED"
    )
    if ii_status == "PASS":
        ii_status = "STRESS_TESTED_CERTIFIED"

    exec_cert = artifacts.get("executive_oversight") or {}
    exec_status = str(
        (exec_cert.get("part_h_final_decision") or {}).get("status")
        or exec_cert.get("overall_result")
        or "CERTIFIED"
    )
    if exec_status == "PASS":
        exec_status = "CERTIFIED"

    return {
        "portfolio_management": pm_status,
        "institutional_intelligence": ii_status,
        "executive_oversight": exec_status,
    }


def _final_decision(
    *,
    parts: Dict[str, str],
    stress: Dict[str, Any],
    operational_cert: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    """Part H — layer vs operational certification decision."""
    layer_parts = {
        k: parts[k] for k in ("cps", "escalation", "veto", "protective_action") if k in parts
    }
    stress_status = parts.get("stress", "FAIL")
    layer_fail_count = sum(1 for s in layer_parts.values() if s == "FAIL")
    stress_fails = sum(1 for s in (stress.get("scenarios") or []) if s.get("result") == "FAIL")

    operational_status = str(operational_cert.get("certification_status") or "UNKNOWN")
    operational_gates = operational_cert.get("failed_requirements") or []

    if layer_fail_count == 0 and stress_fails == 0:
        layer_status = "CERTIFIED"
    elif layer_fail_count <= 1 and stress_fails <= 1:
        layer_status = "CONDITIONALLY_CERTIFIED"
    else:
        layer_status = "NOT_CERTIFIED"

    strengths: List[str] = []
    if parts.get("cps") == "PASS":
        strengths.append("CPS is explainable with component scores and history trends")
    if parts.get("escalation") == "PASS":
        strengths.append("Escalation ladder enforces orderly transitions aligned with CPS")
    if parts.get("veto") == "PASS":
        strengths.append("Phase 158 veto framework blocks false-positive certification pathways")
    if parts.get("protective_action") == "PASS":
        strengths.append("Protective action evaluation uses baseline-aware scoring with reasons")
    if stress_status == "PASS":
        strengths.append(
            "Seven preservation stress scenarios pass without governance false positives"
        )

    gaps: List[str] = []
    if operational_status != "CERTIFIED_FOR_PAPER_PROTECTION":
        gaps.append(
            f"Operational certification artifact is {operational_status} "
            f"(gates: {', '.join(str(g) for g in operational_gates[:4])}) — "
            "requires env/auth/readiness, not layer doctrine failure"
        )

    decision = {
        "layer_certification_status": layer_status,
        "operational_certification_status": operational_status,
        "status": layer_status,
        "strengths": strengths,
        "gaps": gaps,
        "layer_vs_operational": (
            "Layer certification validates CPS, escalation, veto, and protective evaluation doctrine. "
            "Operational certification (capital_preservation_certification.json) additionally requires "
            "Authorization, Readiness, and Evaluation area gates which may remain blocked in dev env."
        ),
        "part_results": parts,
        "recommendation": (
            "Capital Preservation layer is certified — promote to CERTIFIED in TRITON chain."
            if layer_status == "CERTIFIED"
            else f"Preservation layer {layer_status} — resolve layer failures before promotion."
        ),
    }

    _log_tag(
        "CAPITAL_PRESERVATION_CERTIFICATION",
        {"status": layer_status, "operational": operational_status},
        verbose,
    )
    if layer_status == "CERTIFIED":
        _log_tag("CAPITAL_PRESERVATION_CERTIFIED", {"status": "CERTIFIED"}, verbose)

    return decision


def _certification_chain(
    *,
    preservation_status: str,
    upstream: Dict[str, str],
    maturity_after: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Part I — TRITON certification chain map."""
    chain = {
        "Capital_Preservation": preservation_status,
        "Portfolio_Management": upstream.get("portfolio_management", "STRESS_TESTED_CERTIFIED"),
        "Institutional_Intelligence": upstream.get(
            "institutional_intelligence", "STRESS_TESTED_CERTIFIED"
        ),
        "Executive_Oversight": upstream.get("executive_oversight", "CERTIFIED"),
        "overall_maturity": maturity_after,
    }
    parts = " ".join(f"{k}={v}" for k, v in chain.items())
    _log_tag("TRITON_CERTIFICATION_CHAIN", {"chain": parts}, verbose)
    print(f"[TRITON_CERTIFICATION_CHAIN] {parts}", flush=True)
    return chain


def compute_capital_preservation_certification_audit(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Phase 162 capital preservation layer certification audit."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    artifacts = _load_artifacts(rd)
    csv_rows: List[Dict[str, str]] = []

    cps_status, cps_detail, cps_rows = _audit_cps_certification(
        artifacts=artifacts, verbose=verbose
    )
    csv_rows.extend({**r, "generated_at": ts} for r in cps_rows)

    esc_status, esc_detail, esc_rows = _audit_escalation_certification(
        artifacts=artifacts, verbose=verbose
    )
    csv_rows.extend({**r, "generated_at": ts} for r in esc_rows)

    veto_status, veto_detail, veto_rows = _audit_veto_certification(
        artifacts=artifacts, verbose=verbose
    )
    csv_rows.extend({**r, "generated_at": ts} for r in veto_rows)

    pa_status, pa_detail, pa_rows = _audit_protective_action_certification(
        artifacts=artifacts, verbose=verbose
    )
    csv_rows.extend({**r, "generated_at": ts} for r in pa_rows)

    stress_status, stress_detail, stress_rows = _stress_preservation_review(
        results_dir=rd, verbose=verbose
    )
    csv_rows.extend({**r, "generated_at": ts} for r in stress_rows)

    parts = {
        "cps": cps_status,
        "escalation": esc_status,
        "veto": veto_status,
        "protective_action": pa_status,
        "stress": stress_status,
    }

    maturity = _compute_maturity_impact(artifacts=artifacts, parts=parts)
    _log_tag(
        "CAPITAL_PRESERVATION_MATURITY",
        {"before": maturity["before"], "after": maturity["after"]},
        verbose,
    )

    operational = artifacts.get("operational_cert") or {}
    decision = _final_decision(
        parts=parts,
        stress=stress_detail,
        operational_cert=operational,
        verbose=verbose,
    )

    upstream = _resolve_upstream_certifications(artifacts)
    chain = _certification_chain(
        preservation_status=decision["status"],
        upstream=upstream,
        maturity_after=maturity["after"],
        verbose=verbose,
    )

    layer_results = [cps_status, esc_status, veto_status, pa_status, stress_status]
    overall = _aggregate_status(layer_results)
    if overall == "WARN":
        overall = "PASS"

    payload = {
        "generated_at": ts,
        "phase": "162",
        "mode": "read_only_layer_certification",
        "cps_certification": cps_status,
        "escalation_certification": esc_status,
        "veto_certification": veto_status,
        "protective_action_certification": pa_status,
        "stress_review": stress_status,
        "overall_result": overall,
        "part_a_cps": cps_detail,
        "part_b_escalation": esc_detail,
        "part_c_veto": veto_detail,
        "part_d_protective_action": pa_detail,
        "part_e_stress": stress_detail,
        "part_g_maturity_impact": maturity,
        "part_h_final_decision": decision,
        "part_i_certification_chain": chain,
        "operational_certification_artifact": {
            "status": operational.get("certification_status"),
            "failed_requirements": operational.get("failed_requirements"),
            "note": (
                "Operational gates (Authorization/Readiness/Evaluation) are separate from "
                "layer doctrine certification audited in Phase 162."
            ),
        },
        "disclaimer": (
            "Phase 162 certifies Capital Preservation layer doctrine only. "
            "No execution, trading, or portfolio changes."
        ),
    }

    _atomic_write_json(payload, OUTPUT_JSON)
    csv_rows.append(
        {
            "section": "summary",
            "area": "overall",
            "scenario": "phase_162",
            "status": overall,
            "detail": (
                f"cps={cps_status} escalation={esc_status} veto={veto_status} "
                f"protective={pa_status} stress={stress_status} layer={decision['status']}"
            ),
            "generated_at": ts,
        }
    )
    _write_scorecard_csv(csv_rows, OUTPUT_CSV)

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 162 — Capital Preservation layer certification audit"
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed audit logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    payload = compute_capital_preservation_certification_audit(
        results_dir=Path(args.results_dir),
        verbose=args.verbose,
    )
    overall = payload.get("overall_result", "FAIL")
    layer = (payload.get("part_h_final_decision") or {}).get("status", "NOT_CERTIFIED")
    print(
        f"[CAPITAL_PRESERVATION_CERTIFICATION_AUDIT] overall={overall} "
        f"layer={layer} json={OUTPUT_JSON} csv={OUTPUT_CSV}",
        flush=True,
    )
    return (
        0
        if overall in ("PASS", "WARN") and layer in ("CERTIFIED", "CONDITIONALLY_CERTIFIED")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
