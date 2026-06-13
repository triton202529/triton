"""
TRITON Phase 198 — Execution Readiness Stabilization & Certification Integrity.

Separates intentional governance restrictions from true operational failures and
temporary monitoring anomalies. Read-only intelligence — does NOT modify trading,
governance rules, lifecycle, allocation, execution, broker, or risk controls.

Outputs (data/results/):
    execution_readiness_stability.json
    readiness_failure_history.csv
    readiness_failure_patterns.csv

Run:
    python -m services.execution_readiness_stability
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "execution_readiness_stability.json"
OUT_HISTORY = RESULTS / "readiness_failure_history.csv"
OUT_PATTERNS = RESULTS / "readiness_failure_patterns.csv"

READINESS_HISTORY = RESULTS / "execution_readiness_history.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
AUTH_JSON = RESULTS / "governance_authorization.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
GOV_SUMMARY = RESULTS / "governance_risk_summary.json"
RECOVERY_STAB_HISTORY = RESULTS / "recovery_stabilization_history.csv"

CHECK_COLS = [
    "governance",
    "approval",
    "policy",
    "data_freshness",
    "watchdog",
    "broker",
    "lifecycle",
    "signals",
]
OPERATIONAL_CHECKS = ("data_freshness", "watchdog", "broker", "lifecycle", "signals")
AUTHORIZATION_CHECKS = ("governance", "approval", "policy")
TEMPORARY_THRESHOLD_MINUTES = 15.0


def _log(msg: str) -> None:
    print(f"[execution_readiness_stability] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _derive_readiness_status(checks: Dict[str, bool]) -> str:
    if not all(checks.get(k) for k in OPERATIONAL_CHECKS):
        return "NOT_READY"
    if all(checks.get(k) for k in AUTHORIZATION_CHECKS):
        return "READY"
    return "PARTIALLY_READY"


def _load_history() -> pd.DataFrame:
    if not READINESS_HISTORY.is_file():
        return pd.DataFrame()
    df = pd.read_csv(READINESS_HISTORY)
    if df.empty or "timestamp" not in df.columns:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in CHECK_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda x: bool(x) if pd.notna(x) else False)
    return df


def _operational_all_pass(row: pd.Series) -> bool:
    return all(bool(row.get(c, False)) for c in OPERATIONAL_CHECKS)


def _extract_episodes(df: pd.DataFrame, check: str) -> List[Dict[str, Any]]:
    if df.empty or check not in df.columns:
        return []
    episodes: List[Dict[str, Any]] = []
    in_episode = False
    start_ts: Optional[str] = None
    start_idx = 0

    for idx, row in df.iterrows():
        failed = not bool(row.get(check, True))
        ts = str(row.get("timestamp", ""))
        if failed and not in_episode:
            in_episode = True
            start_ts = ts
            start_idx = idx
        elif not failed and in_episode:
            end_ts = ts
            duration = _episode_minutes(start_ts, end_ts)
            episodes.append(
                _episode_row(
                    check,
                    start_ts,
                    end_ts,
                    duration,
                    df.iloc[start_idx:idx],
                    recovered=True,
                )
            )
            in_episode = False
            start_ts = None

    if in_episode and start_ts:
        last_ts = str(df.iloc[-1].get("timestamp", start_ts))
        duration = _episode_minutes(start_ts, last_ts)
        episodes.append(
            _episode_row(
                check,
                start_ts,
                last_ts,
                duration,
                df.iloc[start_idx:],
                recovered=False,
            )
        )
    return episodes


def _episode_minutes(start: Optional[str], end: Optional[str]) -> Optional[float]:
    a, b = _parse_iso(start), _parse_iso(end)
    if not a or not b:
        return None
    return round(max(0.0, (b - a).total_seconds() / 60.0), 2)


def _episode_row(
    check: str,
    start: str,
    end: str,
    duration: Optional[float],
    slice_df: pd.DataFrame,
    *,
    recovered: bool,
) -> Dict[str, Any]:
    if len(slice_df):
        op_rate = float(slice_df.apply(_operational_all_pass, axis=1).mean())
        operational_ok = op_rate >= 0.75
        check_fail_rate = (~slice_df[check].astype(bool)).mean()
        other_pass = all(
            slice_df[c].astype(bool).mean() >= 0.75
            for c in CHECK_COLS
            if c != check and c in slice_df
        )
        only_this_failed = check_fail_rate >= 0.75 and other_pass
    else:
        operational_ok = False
        only_this_failed = False
        op_rate = 0.0

    return {
        "check_name": check,
        "first_seen": start,
        "last_seen": end,
        "duration_minutes": duration,
        "occurrence_count": len(slice_df),
        "operational_checks_passed": bool(operational_ok),
        "operational_pass_rate": round(op_rate, 3),
        "isolated_failure": bool(only_this_failed),
        "recovered": recovered,
    }


def _classify_episode(ep: Dict[str, Any], auth: Dict[str, Any]) -> str:
    check = ep["check_name"]
    duration = ep.get("duration_minutes") or 0.0

    if check == "governance" and ep.get("operational_checks_passed"):
        return "EXPECTED_GOVERNANCE_BLOCK"

    if check in ("approval", "policy") and ep.get("operational_checks_passed"):
        if check == "policy" and auth.get("execution_authorized") is False:
            return "EXPECTED_MARKET_CONDITION"
        return "EXPECTED_GOVERNANCE_BLOCK"

    if ep.get("recovered"):
        if check == "watchdog" and duration <= TEMPORARY_THRESHOLD_MINUTES:
            return "RECOVERED_EVENT"
        if check in OPERATIONAL_CHECKS and duration <= TEMPORARY_THRESHOLD_MINUTES:
            return "TEMPORARY_ANOMALY"

    if check == "watchdog":
        if ep.get("isolated_failure") or (
            ep.get("operational_checks_passed") and duration <= TEMPORARY_THRESHOLD_MINUTES
        ):
            return "TEMPORARY_ANOMALY"
        if ep.get("operational_checks_passed") and duration <= 60:
            return "RECOVERED_EVENT"
        return "CRITICAL_FAILURE"

    if check in OPERATIONAL_CHECKS:
        if duration <= TEMPORARY_THRESHOLD_MINUTES and ep.get("recovered"):
            return "TEMPORARY_ANOMALY"
        return "CRITICAL_FAILURE"

    return "CRITICAL_FAILURE"


def _score_episode(ep: Dict[str, Any], failure_type: str) -> Dict[str, float]:
    duration = ep.get("duration_minutes") or 0.0
    count = ep.get("occurrence_count") or 1
    severity_map = {
        "CRITICAL_FAILURE": 90.0,
        "TEMPORARY_ANOMALY": 25.0,
        "EXPECTED_GOVERNANCE_BLOCK": 10.0,
        "EXPECTED_MARKET_CONDITION": 15.0,
        "RECOVERED_EVENT": 20.0,
    }
    severity = severity_map.get(failure_type, 50.0)
    frequency = min(100.0, count * 5.0)
    impact = severity
    if failure_type in ("EXPECTED_GOVERNANCE_BLOCK", "EXPECTED_MARKET_CONDITION"):
        impact = 5.0
    elif failure_type == "TEMPORARY_ANOMALY":
        impact = min(40.0, 15.0 + duration)
    return {
        "frequency_score": round(frequency, 1),
        "severity_score": round(severity, 1),
        "operational_impact": round(impact, 1),
    }


def _analyze_failures(
    df: pd.DataFrame, auth: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    history_rows: List[Dict[str, Any]] = []
    pattern_agg: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for check in CHECK_COLS:
        if check not in df.columns:
            continue
        for ep in _extract_episodes(df, check):
            failure_type = _classify_episode(ep, auth)
            scores = _score_episode(ep, failure_type)
            recovery_behavior = (
                "SELF_RECOVERED"
                if ep.get("recovered") and failure_type in ("TEMPORARY_ANOMALY", "RECOVERED_EVENT")
                else "ONGOING" if not ep.get("recovered") else "CLEARED"
            )
            row = {
                **ep,
                "failure_type": failure_type,
                **scores,
                "recovery_behavior": recovery_behavior,
            }
            history_rows.append(row)

            key = (check, failure_type)
            if key not in pattern_agg:
                pattern_agg[key] = {
                    "check_name": check,
                    "failure_type": failure_type,
                    "occurrence_count": 0,
                    "total_duration_minutes": 0.0,
                    "first_seen": ep["first_seen"],
                    "last_seen": ep["last_seen"],
                    "avg_duration_minutes": 0.0,
                    "max_duration_minutes": 0.0,
                    "avg_severity_score": 0.0,
                    "avg_operational_impact": 0.0,
                }
            pat = pattern_agg[key]
            pat["occurrence_count"] += 1
            dur = ep.get("duration_minutes") or 0.0
            pat["total_duration_minutes"] += dur
            pat["last_seen"] = max(pat["last_seen"], ep["last_seen"])
            pat["first_seen"] = min(pat["first_seen"], ep["first_seen"])
            pat["max_duration_minutes"] = max(pat["max_duration_minutes"], dur)
            pat["avg_severity_score"] += scores["severity_score"]
            pat["avg_operational_impact"] += scores["operational_impact"]

    patterns: List[Dict[str, Any]] = []
    for pat in pattern_agg.values():
        n = pat["occurrence_count"]
        pat["avg_duration_minutes"] = round(pat["total_duration_minutes"] / n, 2) if n else 0.0
        pat["avg_severity_score"] = round(pat["avg_severity_score"] / n, 1) if n else 0.0
        pat["avg_operational_impact"] = round(pat["avg_operational_impact"] / n, 1) if n else 0.0
        del pat["total_duration_minutes"]
        patterns.append(pat)

    patterns.sort(key=lambda p: (-p["avg_operational_impact"], -p["occurrence_count"]))
    history_rows.sort(key=lambda r: r.get("last_seen") or "", reverse=True)
    return history_rows, patterns


def _watchdog_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "watchdog" not in df.columns:
        return {"watchdog_stability_score": None, "note": "No history"}

    wd_fail = ~df["watchdog"]
    total_fail = int(wd_fail.sum())
    other_ops_cols = [c for c in OPERATIONAL_CHECKS if c != "watchdog"]

    only_watchdog_rows = 0
    gaps: List[float] = []
    for _, row in df.iterrows():
        if not bool(row.get("watchdog", True)):
            if all(bool(row.get(c, False)) for c in other_ops_cols):
                only_watchdog_rows += 1

    episodes = _extract_episodes(df, "watchdog")
    durations = [e["duration_minutes"] for e in episodes if e.get("duration_minutes") is not None]
    recovery_times = durations

    consecutive = 0
    max_consecutive = 0
    for val in df["watchdog"]:
        if not bool(val):
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    false_positive_prob = round(only_watchdog_rows / total_fail, 3) if total_fail else 0.0

    # Heartbeat gap proxy: time between readiness samples when watchdog flips
    prev_ts = None
    prev_ok = True
    for _, row in df.iterrows():
        ts = _parse_iso(str(row.get("timestamp")))
        ok = bool(row.get("watchdog", True))
        if prev_ts and ts and prev_ok and not ok:
            gaps.append((ts - prev_ts).total_seconds() / 60.0)
        if ts:
            prev_ts = ts
        prev_ok = ok

    avg_gap = round(sum(gaps) / len(gaps), 2) if gaps else None
    max_gap = round(max(gaps), 2) if gaps else None
    avg_recovery = round(sum(recovery_times) / len(recovery_times), 2) if recovery_times else None

    stability = 100.0
    stability -= false_positive_prob * 40.0
    stability -= min(30.0, max_consecutive * 5.0)
    if avg_recovery and avg_recovery > TEMPORARY_THRESHOLD_MINUTES:
        stability -= min(20.0, avg_recovery)
    stability = round(max(0.0, min(100.0, stability)), 1)

    return {
        "false_positive_probability": false_positive_prob,
        "average_heartbeat_gap_minutes": avg_gap,
        "largest_heartbeat_gap_minutes": max_gap,
        "heartbeat_recovery_time_minutes": avg_recovery,
        "consecutive_failures_max": max_consecutive,
        "watchdog_failure_episodes": len(episodes),
        "watchdog_isolated_failures": only_watchdog_rows,
        "watchdog_stability_score": stability,
        "assessment": (
            "Watchdog reporting temporary timing noise — not systemic failure"
            if false_positive_prob >= 0.5 and max_consecutive <= 2
            else (
                "Watchdog shows intermittent anomalies with self-recovery"
                if total_fail and false_positive_prob >= 0.3
                else (
                    "Watchdog appears stable"
                    if total_fail == 0
                    else "Watchdog may indicate genuine monitoring gaps"
                )
            )
        ),
    }


def _governance_awareness(
    readiness: Dict[str, Any], auth: Dict[str, Any], gov: Dict[str, Any]
) -> Dict[str, Any]:
    checks = readiness.get("checks") or {}
    operational_ok = all(checks.get(c) for c in OPERATIONAL_CHECKS)
    gov_fail = not checks.get("governance", True)

    escalation = (
        gov.get("escalation_state")
        or gov.get("current_escalation")
        or _read_json(WATCHDOG_STATUS)
        .get("capital_preservation_escalation", {})
        .get("escalation_state")
    )

    if gov_fail and operational_ok:
        classification = "EXPECTED_GOVERNANCE_RESTRICTION"
        case = "A"
        detail = auth.get("gate_reasons", {}).get(
            "governance", readiness.get("check_details", {}).get("governance", "")
        )
    elif not operational_ok:
        classification = "CRITICAL_OPERATIONAL_FAILURE"
        case = "B"
        failed_ops = [c for c in OPERATIONAL_CHECKS if not checks.get(c)]
        detail = f"Operational checks failing: {', '.join(failed_ops)}"
    else:
        classification = "OPERATIONAL_CLEAR"
        case = "NONE"
        detail = "All operational and governance checks passing"

    return {
        "classification": classification,
        "case": case,
        "detail": detail,
        "governance_authorized": auth.get("governance_authorized"),
        "escalation_state": escalation,
        "operational_checks_passing": operational_ok,
        "governance_reduces_reliability_score": False,
        "note": (
            "Governance restrictions remain enforced but do not reduce operational reliability scoring"
        ),
    }


def _recovery_success_rate() -> Optional[float]:
    if not RECOVERY_STAB_HISTORY.is_file():
        return None
    df = pd.read_csv(RECOVERY_STAB_HISTORY)
    if df.empty or "recovery_success" not in df.columns:
        return None
    obs = df[df["recovery_success"].notna()]
    if obs.empty:
        return None
    successes = obs["recovery_success"].astype(bool).sum()
    return round(successes / len(obs) * 100, 1)


def _stability_score(
    df: pd.DataFrame,
    wd: Dict[str, Any],
    patterns: List[Dict[str, Any]],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    recent = df.tail(500) if len(df) > 500 else df

    monitoring = wd.get("watchdog_stability_score") or 70.0
    broker_pct = (
        recent["broker"].mean() * 100 if "broker" in recent.columns and len(recent) else 80.0
    )
    recovery_pct = _recovery_success_rate() or 50.0

    critical_impacts = [
        p["avg_operational_impact"] for p in patterns if p["failure_type"] == "CRITICAL_FAILURE"
    ]
    recurrence_penalty = min(25.0, sum(critical_impacts) / max(len(critical_impacts), 1) * 0.2)

    fp = wd.get("false_positive_probability") or 0.0
    false_pos_component = max(0.0, 100.0 - fp * 100.0)

    if "readiness_status" in recent.columns and len(recent) > 1:
        status_changes = (recent["readiness_status"] != recent["readiness_status"].shift()).sum()
        consistency = max(0.0, 100.0 - (status_changes / len(recent)) * 200.0)
    else:
        consistency = 75.0

    score = (
        monitoring * 0.22
        + broker_pct * 0.18
        + recovery_pct * 0.15
        + false_pos_component * 0.15
        + consistency * 0.15
        + max(0.0, 100.0 - recurrence_penalty) * 0.15
    )
    score = round(max(0.0, min(100.0, score)), 1)

    if score >= 90:
        band = "INSTITUTIONAL_GRADE"
    elif score >= 75:
        band = "STABLE"
    elif score >= 50:
        band = "DEVELOPING"
    else:
        band = "UNSTABLE"

    return {
        "stability_score": score,
        "classification": band,
        "factors": {
            "monitoring_reliability": round(monitoring, 1),
            "broker_connectivity": round(broker_pct, 1),
            "recovery_success_rate": recovery_pct,
            "failure_recurrence_penalty": round(recurrence_penalty, 1),
            "false_positive_component": round(false_pos_component, 1),
            "certification_consistency": round(consistency, 1),
        },
    }


def _certification_impact(
    readiness: Dict[str, Any],
    failure_history: List[Dict[str, Any]],
    gov: Dict[str, Any],
) -> Dict[str, Any]:
    checks = {k: bool(v) for k, v in (readiness.get("checks") or {}).items()}
    actual = readiness.get("readiness_status") or _derive_readiness_status(checks)

    temporary_checks = {
        e["check_name"]
        for e in failure_history
        if e.get("failure_type") == "TEMPORARY_ANOMALY"
        and e.get("last_seen", "") >= (readiness.get("generated_at") or "")[:10]
    }
    adjusted = dict(checks)
    for c in temporary_checks:
        adjusted[c] = True
    status_if_temporary_removed = _derive_readiness_status(adjusted)

    operational_only = dict(checks)
    for c in AUTHORIZATION_CHECKS:
        operational_only[c] = True
    operational_ready_status = _derive_readiness_status(operational_only)

    gov_restricted = dict(checks)
    for c in AUTHORIZATION_CHECKS:
        if c == "governance":
            gov_restricted[c] = False
        else:
            gov_restricted[c] = checks.get(c, False)
    status_under_governance_only = _derive_readiness_status(gov_restricted)

    technical_failures = [c for c in OPERATIONAL_CHECKS if not checks.get(c)]
    governance_failures = [c for c in AUTHORIZATION_CHECKS if not checks.get(c)]

    return {
        "actual_readiness_status": actual,
        "status_if_temporary_anomalies_removed": status_if_temporary_removed,
        "operational_readiness_status": operational_ready_status,
        "status_with_intentional_governance_block": status_under_governance_only,
        "technical_failures": technical_failures,
        "governance_restrictions": governance_failures,
        "true_operational_weakness": bool(technical_failures),
        "intentional_governance_block_only": bool(governance_failures) and not technical_failures,
        "would_be_ready_without_governance": operational_ready_status == "READY",
        "certification_summary": (
            "Operational infrastructure is healthy; certification blocked by governance only"
            if not technical_failures and governance_failures
            else (
                "Technical operational failures present — not merely governance"
                if technical_failures
                else "All checks passing — ready for certification"
            )
        ),
    }


def _executive_report(
    stability: Dict[str, Any],
    readiness: Dict[str, Any],
    gov_aware: Dict[str, Any],
    wd: Dict[str, Any],
    cert: Dict[str, Any],
    patterns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    recommendations: List[str] = []
    if cert.get("intentional_governance_block_only"):
        recommendations.append(
            "Resolve governance escalation (ORANGE→GREEN) — operational stack is healthy"
        )
    if wd.get("false_positive_probability", 0) > 0.3:
        recommendations.append(
            "Tune watchdog heartbeat grace window to reduce false-positive readiness drops"
        )
    if cert.get("technical_failures"):
        recommendations.append(
            f"Restore operational checks: {', '.join(cert['technical_failures'])}"
        )
    critical = [p for p in patterns if p["failure_type"] == "CRITICAL_FAILURE"]
    if critical:
        recommendations.append(
            f"Investigate critical failure pattern: {critical[0]['check_name']} "
            f"({critical[0]['occurrence_count']} episodes)"
        )
    if stability.get("classification") == "UNSTABLE":
        recommendations.append("Stability score below 50 — prioritize monitoring reliability")
    if not recommendations:
        recommendations.append(
            "Continue monitoring — distinguish governance blocks from infra failures"
        )

    return {
        "report_type": "EXECUTION_READINESS_STABILITY_REPORT",
        "current_stability_score": stability.get("stability_score"),
        "stability_classification": stability.get("classification"),
        "readiness_status": readiness.get("readiness_status"),
        "governance_vs_technical": {
            "governance_classification": gov_aware.get("classification"),
            "technical_failures": cert.get("technical_failures"),
            "governance_restrictions": cert.get("governance_restrictions"),
        },
        "watchdog_reliability_summary": wd.get("assessment"),
        "false_positive_analysis": {
            "probability": wd.get("false_positive_probability"),
            "isolated_watchdog_failures": wd.get("watchdog_isolated_failures"),
        },
        "certification_impact_summary": cert.get("certification_summary"),
        "institutional_learning": (
            "Governance blocks are discipline, not dysfunction. "
            f"{gov_aware.get('detail', '')} "
            "Temporary watchdog failures must not be scored equivalently to broker or lifecycle outages."
        ),
        "recommended_actions": recommendations,
        "executive_questions": {
            "is_triton_actually_unhealthy": cert.get("true_operational_weakness", True),
            "are_failures_temporary_or_critical": (
                "Mostly governance-intentional; watchdog anomalies are temporary"
                if cert.get("intentional_governance_block_only")
                else (
                    "Critical operational failures detected"
                    if cert.get("technical_failures")
                    else "Mixed — review failure patterns"
                )
            ),
            "is_governance_blocking_intentionally": gov_aware.get("classification")
            == "EXPECTED_GOVERNANCE_RESTRICTION",
            "true_operational_readiness_score": stability.get("factors", {}).get(
                "monitoring_reliability"
            ),
            "improvement_before_certification": recommendations[:3],
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    readiness = _read_json(READINESS_JSON)
    auth = _read_json(AUTH_JSON)
    gov = _read_json(GOV_SUMMARY)
    alerts = _read_json(WATCHDOG_ALERTS)
    df = _load_history()

    failure_history, patterns = _analyze_failures(df, auth)
    wd = _watchdog_reliability(df)
    gov_aware = _governance_awareness(readiness, auth, gov)
    stability = _stability_score(df, wd, patterns, readiness)
    cert = _certification_impact(readiness, failure_history, gov)
    executive = _executive_report(stability, readiness, gov_aware, wd, cert, patterns)

    # Recent failure timeline (last 20 episodes)
    timeline = failure_history[:20]

    report = {
        "generated_at": ts,
        "phase": "198",
        "mode": "execution_readiness_stability",
        "disclaimer": (
            "Readiness stabilization intelligence only — does NOT modify governance rules, "
            "execution permissions, trading strategy, or risk controls."
        ),
        "current_readiness": {
            "status": readiness.get("readiness_status"),
            "checks": readiness.get("checks"),
            "failed_checks": readiness.get("failed_checks"),
            "check_details": readiness.get("check_details"),
        },
        "stability_score": stability,
        "governance_awareness": gov_aware,
        "watchdog_reliability": wd,
        "certification_impact": cert,
        "failure_timeline": timeline,
        "failure_patterns": patterns,
        "active_watchdog_alerts": len(alerts.get("active_alerts") or []),
        "executive_recommendation": executive,
        "summary": {
            "stability_score": stability.get("stability_score"),
            "stability_classification": stability.get("classification"),
            "readiness_status": readiness.get("readiness_status"),
            "operational_readiness_status": cert.get("operational_readiness_status"),
            "intentional_governance_only": cert.get("intentional_governance_block_only"),
            "autonomous_recovery_certifiable": False,
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if failure_history:
        pd.DataFrame(failure_history).to_csv(OUT_HISTORY, index=False, encoding="utf-8")
    else:
        OUT_HISTORY.write_text(
            "check_name,first_seen,last_seen,duration_minutes,failure_type\n",
            encoding="utf-8",
        )

    if patterns:
        pd.DataFrame(patterns).to_csv(OUT_PATTERNS, index=False, encoding="utf-8")
    else:
        OUT_PATTERNS.write_text(
            "check_name,failure_type,occurrence_count,first_seen,last_seen\n", encoding="utf-8"
        )

    _log(
        f"score={stability.get('stability_score')} band={stability.get('classification')} "
        f"readiness={readiness.get('readiness_status')} gov_only={cert.get('intentional_governance_block_only')}"
    )
    return report


def main() -> int:
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
