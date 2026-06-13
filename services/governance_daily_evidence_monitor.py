"""
TRITON Phase 201 — Governance Recovery Daily Evidence Monitor.

Continuous institutional evidence monitoring for governance recovery.
Read-only — does NOT modify governance, execution, trading, or risk controls.

Outputs (data/results/):
    governance_daily_evidence.json
    governance_daily_evidence_history.csv
    governance_recovery_velocity.csv
    governance_recovery_trend.csv

Run:
    python -m services.governance_daily_evidence_monitor
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_daily_evidence.json"
OUT_HISTORY = RESULTS / "governance_daily_evidence_history.csv"
OUT_VELOCITY = RESULTS / "governance_recovery_velocity.csv"
OUT_TREND = RESULTS / "governance_recovery_trend.csv"

ROADMAP_JSON = RESULTS / "governance_recovery_roadmap.json"
PROGRESS_HISTORY = RESULTS / "governance_recovery_progress_history.csv"
EVIDENCE_LOG = RESULTS / "governance_recovery_evidence_log.csv"


def _log(msg: str) -> None:
    print(f"[governance_daily_evidence_monitor] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _recovery_score_band(score: float) -> str:
    if score >= 90:
        return "AUTHORIZATION_READY"
    if score >= 75:
        return "NEAR_YELLOW"
    if score >= 50:
        return "RECOVERING"
    if score >= 25:
        return "STALLED"
    return "CRITICAL"


def _load_snapshot() -> Dict[str, Any]:
    roadmap = _read_json(ROADMAP_JSON)
    metrics = roadmap.get("metrics") or {}
    readiness = roadmap.get("governance_readiness_score") or {}
    prob = roadmap.get("authorization_probability") or {}
    evidence = roadmap.get("evidence_summary") or {}
    timeline = (
        _read_json(RESULTS / "governance_recovery_timeline.json").get("executive_report") or {}
    )
    current = _read_json(RESULTS / "governance_recovery_timeline.json").get("current_state") or {}
    stability = _read_json(RESULTS / "execution_readiness_stability.json")
    recovery_stab = _read_json(RESULTS / "recovery_stabilization_status.json")

    cps = float(metrics.get("cps") or current.get("current_cps") or 66)
    qqq = float(metrics.get("qqq_concentration_pct") or current.get("qqq_concentration_pct") or 100)
    dist_y = float(metrics.get("distance_to_yellow") or max(0, CPS_YELLOW - cps))
    dist_g = float(metrics.get("distance_to_green") or max(0, CPS_GREEN - cps))

    collected = int(evidence.get("collected") or 0)
    total_ev = int(evidence.get("total") or 1)
    evidence_pct = round(collected / max(total_ev, 1) * 100, 1)

    milestones = roadmap.get("milestones") or []
    completed_ms = sum(1 for m in milestones if m.get("status") == "COMPLETE")

    wd_score = (stability.get("watchdog_reliability") or {}).get("watchdog_stability_score") or 0
    stab_score = (stability.get("stability_score") or {}).get("stability_score") or 0

    return {
        "date": _today(),
        "timestamp": _now_iso(),
        "escalation_state": metrics.get("escalation") or "ORANGE",
        "cps": cps,
        "distance_to_yellow": dist_y,
        "distance_to_green": dist_g,
        "qqq_concentration_pct": qqq,
        "target_qqq_pct": float(metrics.get("target_qqq_pct") or 60),
        "cps_momentum": metrics.get("cps_momentum") or "STABLE",
        "recovery_progress_pct": float(roadmap.get("recovery_progress_pct") or 0),
        "readiness_score": float(readiness.get("governance_readiness_score") or 0),
        "evidence_completion_pct": evidence_pct,
        "milestones_completed": completed_ms,
        "milestones_total": len(milestones),
        "probability_yellow_pct": float(prob.get("probability_yellow_pct") or 75),
        "expected_yellow_date": metrics.get("expected_yellow_date")
        or timeline.get("expected_yellow_date"),
        "expected_green_date": metrics.get("expected_green_date")
        or timeline.get("expected_green_date"),
        "base_days_to_yellow": int(
            metrics.get("days_to_yellow") or timeline.get("days_to_yellow") or 20
        ),
        "base_days_to_green": int(
            metrics.get("days_to_green") or timeline.get("days_to_green") or 60
        ),
        "hd_status": metrics.get("hd_status") or current.get("hd_status"),
        "xlk_status": metrics.get("xlk_status") or current.get("xlk_status"),
        "bito_status": metrics.get("bito_status") or current.get("bito_status"),
        "broker_uptime_proxy": bool(
            _read_json(RESULTS / "watchdog_status.json").get("broker_connected")
        ),
        "watchdog_stability_score": float(wd_score),
        "execution_stability_score": float(stab_score),
        "loop_recovery_state": (recovery_stab.get("summary") or {}).get("current_state"),
    }


def _prior_snapshot(history: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if history.empty:
        ph = _read_csv(PROGRESS_HISTORY)
        if len(ph) >= 2:
            row = ph.iloc[-2].to_dict()
            return {k: (None if pd.isna(v) else v) for k, v in row.items()}
        if len(ph) == 1:
            row = ph.iloc[-1].to_dict()
            return {k: (None if pd.isna(v) else v) for k, v in row.items()}
        return None
    if len(history) >= 1:
        row = history.iloc[-1].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return None


def _daily_deltas(cur: Dict[str, Any], prior: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not prior:
        return {
            "cps_change": 0.0,
            "distance_to_yellow_change": 0.0,
            "distance_to_green_change": 0.0,
            "qqq_concentration_change": 0.0,
            "evidence_completion_change": 0.0,
            "readiness_score_change": 0.0,
            "days_since_prior": 0,
        }

    prior_ts = _parse_ts(str(prior.get("timestamp") or prior.get("date", "")))
    cur_ts = _parse_ts(cur["timestamp"])
    days = 1.0
    if prior_ts and cur_ts:
        days = max(1.0, (cur_ts - prior_ts).total_seconds() / 86400.0)

    def _delta(key: str, cur_key: Optional[str] = None) -> float:
        ck = cur_key or key
        a = float(cur.get(ck) or 0)
        b = float(prior.get(key) or prior.get(ck) or 0)
        return round(a - b, 2)

    return {
        "cps_change": _delta("cps"),
        "distance_to_yellow_change": _delta("distance_to_yellow"),
        "distance_to_green_change": _delta("distance_to_green"),
        "qqq_concentration_change": _delta("qqq_concentration_pct"),
        "evidence_completion_change": _delta("evidence_completion_pct"),
        "readiness_score_change": _delta("readiness_score"),
        "days_since_prior": round(days, 2),
    }


def _velocity_engine(cur: Dict[str, Any], deltas: Dict[str, Any]) -> Dict[str, Any]:
    cps_per_day = deltas["cps_change"] / max(deltas["days_since_prior"], 1)
    conc_per_day = deltas["qqq_concentration_change"] / max(deltas["days_since_prior"], 1)

    if cps_per_day > 0.1 or conc_per_day < -0.5:
        direction = "IMPROVING"
    elif cps_per_day < -0.1 or conc_per_day > 0.5:
        direction = "DETERIORATING"
    else:
        direction = "STABLE"

    if abs(cps_per_day) > 0.5 or abs(conc_per_day) > 2:
        velocity = (
            "ACCELERATING"
            if direction == "IMPROVING"
            else "SLOWING" if direction == "DETERIORATING" else "NORMAL"
        )
    elif direction == "STABLE":
        velocity = "NORMAL"
    else:
        velocity = "NORMAL"

    days_gained_yellow = (
        round(
            max(0, deltas["cps_change"])
            / max(CPS_YELLOW - cur["cps"] + deltas["cps_change"], 1)
            * deltas["days_since_prior"],
            1,
        )
        if deltas["cps_change"] > 0
        else 0
    )
    days_lost_yellow = (
        round(abs(min(0, deltas["cps_change"])) * 2, 1) if deltas["cps_change"] < 0 else 0
    )

    base_y = int(cur.get("base_days_to_yellow") or 20)
    base_g = int(cur.get("base_days_to_green") or 60)
    adj_y = max(1, base_y - int(days_gained_yellow) + int(days_lost_yellow))
    adj_g = max(adj_y + 1, base_g - int(days_gained_yellow * 1.5) + int(days_lost_yellow * 2))

    now = datetime.now(timezone.utc)
    updated_yellow = (now + timedelta(days=adj_y)).strftime("%Y-%m-%d")
    updated_green = (now + timedelta(days=adj_g)).strftime("%Y-%m-%d")

    return {
        "recovery_direction": direction,
        "recovery_velocity": velocity,
        "cps_change_per_day": round(cps_per_day, 3),
        "concentration_change_per_day": round(conc_per_day, 3),
        "days_gained_toward_yellow": days_gained_yellow,
        "days_lost_toward_yellow": days_lost_yellow,
        "updated_yellow_forecast_date": updated_yellow,
        "updated_green_forecast_date": updated_green,
        "updated_days_to_yellow": adj_y,
        "updated_days_to_green": adj_g,
        "baseline_yellow_date": cur.get("expected_yellow_date"),
        "baseline_green_date": cur.get("expected_green_date"),
    }


def _momentum_analysis(
    cur: Dict[str, Any], deltas: Dict[str, Any], velocity: Dict[str, Any]
) -> Dict[str, Any]:
    positive: List[str] = []
    negative: List[str] = []

    if deltas["cps_change"] > 0:
        positive.append(f"CPS increased by {deltas['cps_change']:.1f}")
    elif deltas["cps_change"] < 0:
        negative.append(f"CPS declined by {abs(deltas['cps_change']):.1f}")

    if deltas["qqq_concentration_change"] < 0:
        positive.append(
            f"QQQ concentration reduced by {abs(deltas['qqq_concentration_change']):.1f}%"
        )
    elif deltas["qqq_concentration_change"] > 0:
        negative.append(f"QQQ concentration increased by {deltas['qqq_concentration_change']:.1f}%")

    if deltas["evidence_completion_change"] > 0:
        positive.append(f"Evidence completion +{deltas['evidence_completion_change']:.1f}%")
    if cur.get("broker_uptime_proxy"):
        positive.append("Broker connectivity stable")
    if (cur.get("watchdog_stability_score") or 0) >= 50:
        positive.append("Watchdog stability acceptable")
    else:
        negative.append("Watchdog stability below threshold")

    if cur.get("hd_status") == "OPEN_NEW" and cur.get("xlk_status") == "OPEN_NEW":
        positive.append("HD/XLK recovery candidates available")
    if cur.get("bito_status") == "WATCH":
        negative.append("BITO still WATCH — GREEN path blocked")

    if velocity["recovery_direction"] == "IMPROVING":
        momentum_class = (
            "STRONG_POSITIVE" if velocity["recovery_velocity"] == "ACCELERATING" else "POSITIVE"
        )
    elif velocity["recovery_direction"] == "DETERIORATING":
        momentum_class = "CRITICAL" if deltas["cps_change"] < -1 else "NEGATIVE"
    else:
        momentum_class = "NEUTRAL"

    return {
        "classification": momentum_class,
        "positive_evidence": positive,
        "negative_evidence": negative,
    }


def _governance_recovery_score(
    cur: Dict[str, Any], deltas: Dict[str, Any], velocity: Dict[str, Any]
) -> Dict[str, Any]:
    cps_progress = min(100.0, (cur["cps"] / CPS_GREEN) * 100)
    div_progress = max(
        0.0,
        min(
            100.0, (100 - cur["qqq_concentration_pct"]) / max(100 - cur["target_qqq_pct"], 1) * 100
        ),
    )
    ops = min(
        100.0,
        (cur.get("watchdog_stability_score") or 0) * 0.5
        + (cur.get("execution_stability_score") or 0) * 0.5,
    )
    vel_bonus = {"IMPROVING": 10, "STABLE": 0, "DETERIORATING": -15}.get(
        velocity["recovery_direction"], 0
    )
    evidence = cur.get("evidence_completion_pct") or 0

    score = round(
        cps_progress * 0.28
        + div_progress * 0.27
        + ops * 0.15
        + evidence * 0.15
        + cur.get("recovery_progress_pct", 0) * 0.15
        + vel_bonus,
        1,
    )
    score = max(0.0, min(100.0, score))

    return {
        "governance_recovery_score": score,
        "classification": _recovery_score_band(score),
        "factors": {
            "cps_progress": round(cps_progress, 1),
            "diversification_evidence": round(div_progress, 1),
            "operational_reliability": round(ops, 1),
            "recovery_velocity_bonus": vel_bonus,
            "evidence_completion": round(evidence, 1),
        },
    }


def _early_warnings(
    cur: Dict[str, Any], deltas: Dict[str, Any], velocity: Dict[str, Any], momentum: Dict[str, Any]
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    def _warn(code: str, cause: str, y_impact: str, g_impact: str, conf: str, action: str) -> None:
        warnings.append(
            {
                "warning_code": code,
                "root_cause": cause,
                "impact_yellow_timeline": y_impact,
                "impact_green_timeline": g_impact,
                "confidence": conf,
                "recommended_action": action,
            }
        )

    if (
        velocity["recovery_direction"] == "IMPROVING"
        and velocity["recovery_velocity"] == "ACCELERATING"
    ):
        _warn(
            "RECOVERY_ACCELERATING",
            "CPS and/or concentration metrics improving faster than baseline",
            f"May advance YELLOW forecast toward {velocity['updated_yellow_forecast_date']}",
            "GREEN timeline may compress if momentum sustained",
            "MODERATE",
            "Continue evidence collection; document daily improvements",
        )
    if velocity["recovery_direction"] == "STABLE" and cur["qqq_concentration_pct"] >= 95:
        _warn(
            "RECOVERY_STALLED",
            "No measurable CPS or concentration progress — QQQ remains at 100%",
            f"YELLOW date may slip beyond {cur.get('expected_yellow_date')}",
            "GREEN path unchanged without diversification action",
            "HIGH",
            "Prioritize concentration reduction milestone M2",
        )
    if velocity["recovery_direction"] == "DETERIORATING":
        _warn(
            "RECOVERY_REVERSING",
            f"CPS change {deltas['cps_change']:.1f} — metrics moving away from thresholds",
            f"+{velocity['days_lost_toward_yellow']:.0f} days to YELLOW",
            "GREEN timeline extended",
            "HIGH",
            "Investigate concentration and operational regressions immediately",
        )
    if (
        deltas["cps_change"] == 0
        and deltas["qqq_concentration_change"] == 0
        and cur["distance_to_yellow"] > 0
    ):
        _warn(
            "RECOVERY_DELAYED",
            "Zero daily progress on primary recovery metrics",
            "Baseline forecast unchanged — no days gained",
            "Evidence accumulation not advancing authorization clock",
            "MODERATE",
            "Execute recovery basket plan; avoid passive waiting",
        )

    if not warnings:
        _warn(
            "RECOVERY_MONITORING",
            "Continuous monitoring active — no critical alerts",
            "Tracking against baseline forecast",
            "Tracking against baseline forecast",
            "LOW",
            "Run daily monitor and update evidence log",
        )

    return warnings


def _executive_report(
    cur: Dict[str, Any],
    deltas: Dict[str, Any],
    velocity: Dict[str, Any],
    momentum: Dict[str, Any],
    score: Dict[str, Any],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    closer_yellow = deltas["distance_to_yellow_change"] < 0 or deltas["cps_change"] > 0
    return {
        "report_type": "GOVERNANCE_DAILY_EVIDENCE_REPORT",
        "report_date": cur["date"],
        "current_governance_position": cur["escalation_state"],
        "current_cps": cur["cps"],
        "daily_cps_change": deltas["cps_change"],
        "recovery_direction": velocity["recovery_direction"],
        "recovery_velocity": velocity["recovery_velocity"],
        "distance_to_yellow": cur["distance_to_yellow"],
        "distance_to_green": cur["distance_to_green"],
        "updated_authorization_forecast": {
            "yellow_date": velocity["updated_yellow_forecast_date"],
            "green_date": velocity["updated_green_forecast_date"],
            "baseline_yellow": cur.get("expected_yellow_date"),
            "baseline_green": cur.get("expected_green_date"),
        },
        "evidence_completion_pct": cur["evidence_completion_pct"],
        "governance_recovery_score": score["governance_recovery_score"],
        "governance_recovery_classification": score["classification"],
        "top_positive_drivers": momentum["positive_evidence"][:5],
        "top_negative_drivers": momentum["negative_evidence"][:5],
        "recommended_focus": [
            "Reduce QQQ concentration toward 60% YELLOW target",
            "Close remaining 9 CPS points to YELLOW threshold",
            "Complete operator risk reviews for evidence package",
            "Promote BITO lifecycle status for GREEN path",
        ],
        "executive_questions": {
            "closer_to_yellow_than_yesterday": closer_yellow,
            "accelerating_or_slowing": velocity["recovery_velocity"],
            "concentration_improving": deltas["qqq_concentration_change"] < 0,
            "requirements_being_satisfied": deltas["evidence_completion_change"] >= 0,
            "authorization_date_changed": velocity["updated_yellow_forecast_date"]
            != cur.get("expected_yellow_date"),
            "biggest_obstacle": (
                "QQQ 100% concentration"
                if cur["qqq_concentration_pct"] >= 95
                else "CPS gap to YELLOW"
            ),
            "fastest_improving_evidence": (
                momentum["positive_evidence"][0]
                if momentum["positive_evidence"]
                else "None yet — baseline day"
            ),
        },
        "active_warnings": [w["warning_code"] for w in warnings],
    }


def _append_history(row: Dict[str, Any]) -> None:
    if OUT_HISTORY.is_file():
        df = _read_csv(OUT_HISTORY)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUT_HISTORY, index=False, encoding="utf-8")


def _build_trend(history: pd.DataFrame, cur: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in history.iterrows():
        rows.append(
            {
                "date": r.get("date"),
                "cps": r.get("cps"),
                "distance_to_yellow": r.get("distance_to_yellow"),
                "distance_to_green": r.get("distance_to_green"),
                "qqq_concentration_pct": r.get("qqq_concentration_pct"),
                "governance_recovery_score": r.get("governance_recovery_score"),
                "recovery_direction": r.get("recovery_direction"),
            }
        )
    if not rows or rows[-1].get("date") != cur["date"]:
        rows.append(
            {
                "date": cur["date"],
                "cps": cur["cps"],
                "distance_to_yellow": cur["distance_to_yellow"],
                "distance_to_green": cur["distance_to_green"],
                "qqq_concentration_pct": cur["qqq_concentration_pct"],
                "governance_recovery_score": None,
                "recovery_direction": None,
            }
        )
    return pd.DataFrame(rows)


def run_monitor() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    history = _read_csv(OUT_HISTORY)
    cur = _load_snapshot()
    prior = _prior_snapshot(history)

    deltas = _daily_deltas(cur, prior)
    velocity = _velocity_engine(cur, deltas)
    momentum = _momentum_analysis(cur, deltas, velocity)
    score = _governance_recovery_score(cur, deltas, velocity)
    warnings = _early_warnings(cur, deltas, velocity, momentum)
    executive = _executive_report(cur, deltas, velocity, momentum, score, warnings)

    daily_evidence = {
        "capital_preservation": {
            "cps": cur["cps"],
            "cps_change_daily": deltas["cps_change"],
            "cps_momentum": cur["cps_momentum"],
            "distance_to_yellow": cur["distance_to_yellow"],
            "distance_to_green": cur["distance_to_green"],
        },
        "portfolio_recovery": {
            "qqq_concentration_pct": cur["qqq_concentration_pct"],
            "qqq_concentration_change": deltas["qqq_concentration_change"],
            "hd_status": cur["hd_status"],
            "xlk_status": cur["xlk_status"],
            "bito_status": cur["bito_status"],
        },
        "operational_stability": {
            "broker_uptime_proxy": cur["broker_uptime_proxy"],
            "watchdog_stability_score": cur["watchdog_stability_score"],
            "execution_stability_score": cur["execution_stability_score"],
            "loop_recovery_state": cur["loop_recovery_state"],
        },
        "governance_evidence": {
            "milestones_completed": cur["milestones_completed"],
            "milestones_total": cur["milestones_total"],
            "evidence_completion_pct": cur["evidence_completion_pct"],
            "readiness_score": cur["readiness_score"],
        },
    }

    report = {
        "generated_at": ts,
        "phase": "201",
        "mode": "governance_daily_evidence_monitor",
        "disclaimer": (
            "Daily governance recovery monitoring only — does NOT modify governance rules, "
            "execution permissions, trading, allocation, or risk controls."
        ),
        "daily_evidence": daily_evidence,
        "daily_deltas": deltas,
        "recovery_velocity": velocity,
        "evidence_momentum": momentum,
        "governance_recovery_score": score,
        "early_warnings": warnings,
        "executive_recommendation": executive,
        "summary": {
            "report_date": cur["date"],
            "cps": cur["cps"],
            "daily_cps_change": deltas["cps_change"],
            "recovery_direction": velocity["recovery_direction"],
            "recovery_score": score["governance_recovery_score"],
            "momentum": momentum["classification"],
            "updated_yellow_forecast": velocity["updated_yellow_forecast_date"],
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    hist_row = {
        "timestamp": ts,
        "date": cur["date"],
        "cps": cur["cps"],
        "cps_change_daily": deltas["cps_change"],
        "distance_to_yellow": cur["distance_to_yellow"],
        "distance_to_green": cur["distance_to_green"],
        "qqq_concentration_pct": cur["qqq_concentration_pct"],
        "evidence_completion_pct": cur["evidence_completion_pct"],
        "governance_recovery_score": score["governance_recovery_score"],
        "recovery_direction": velocity["recovery_direction"],
        "recovery_velocity": velocity["recovery_velocity"],
        "momentum_classification": momentum["classification"],
        "updated_yellow_forecast": velocity["updated_yellow_forecast_date"],
        "updated_green_forecast": velocity["updated_green_forecast_date"],
    }
    _append_history(hist_row)

    vel_row = {
        "timestamp": ts,
        "date": cur["date"],
        **velocity,
        **deltas,
    }
    if OUT_VELOCITY.is_file():
        vdf = _read_csv(OUT_VELOCITY)
        vdf = pd.concat([vdf, pd.DataFrame([vel_row])], ignore_index=True)
    else:
        vdf = pd.DataFrame([vel_row])
    vdf.to_csv(OUT_VELOCITY, index=False, encoding="utf-8")

    trend_df = _build_trend(_read_csv(OUT_HISTORY), cur)
    for i, r in trend_df.iterrows():
        if r["date"] == cur["date"]:
            trend_df.at[i, "governance_recovery_score"] = score["governance_recovery_score"]
            trend_df.at[i, "recovery_direction"] = velocity["recovery_direction"]
    trend_df.to_csv(OUT_TREND, index=False, encoding="utf-8")

    _log(
        f"date={cur['date']} cps={cur['cps']} delta={deltas['cps_change']} "
        f"direction={velocity['recovery_direction']} score={score['governance_recovery_score']}"
    )
    return report


def main() -> int:
    run_monitor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
