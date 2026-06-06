"""
TRITON Risk Watchdog — monitoring through executive risk intelligence (Phases 1–12).

Read-only: capital preservation stack, governance, simulation, prediction, executive
briefings. No trading or portfolio actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "data" / "results"
LIVE_DIR = ROOT / "data" / "live"
STATUS_PATH = RESULTS_DIR / "watchdog_status.json"
ALERTS_PATH = RESULTS_DIR / "watchdog_alerts.json"
ALERTS_LOG_PATH = RESULTS_DIR / "watchdog_alerts.log"
CPI_PATH = RESULTS_DIR / "capital_preservation_intelligence.json"
CPI_HISTORY_PATH = RESULTS_DIR / "capital_preservation_history.csv"
CPE_PATH = RESULTS_DIR / "capital_preservation_escalation.json"
CPE_HISTORY_PATH = RESULTS_DIR / "capital_preservation_escalation_history.csv"
CPA_PATH = RESULTS_DIR / "capital_preservation_advisory.json"
CPA_HISTORY_PATH = RESULTS_DIR / "capital_preservation_advisory_history.csv"
CPD_PATH = RESULTS_DIR / "capital_preservation_decision_support.json"
CPD_HISTORY_PATH = RESULTS_DIR / "capital_preservation_decision_history.csv"
GOV_RISK_PATH = RESULTS_DIR / "governance_risk_summary.json"
STATE_PATH = LIVE_DIR / "watchdog_alert_state.json"
LOGS_DIR = RESULTS_DIR / "watchdog_logs"

CPI_WEIGHTS = {
    "drawdown": 0.30,
    "concentration": 0.25,
    "exposure": 0.20,
    "operational": 0.15,
    "execution": 0.10,
}
TREND_DELTA_POINTS = 3.0
TREND_LOOKBACK_ROWS = 8

ALERT_ESCALATION_REASON: Dict[str, str] = {
    "EXCESS_CONCENTRATION": "Concentration Risk",
    "EXCESS_POSITION_DRAWDOWN": "Drawdown Risk",
    "BROKER_DISCONNECTED": "Operational Risk",
    "STALE_HEARTBEAT": "Monitoring Risk",
    "OPEN_ORDER_AGING": "Execution Risk",
}

ALERT_REVIEW_ACTIONS: Dict[str, str] = {
    "EXCESS_CONCENTRATION": "Review concentration risk",
    "EXCESS_POSITION_DRAWDOWN": "Review losing positions",
    "BROKER_DISCONNECTED": "Review broker connectivity",
    "STALE_HEARTBEAT": "Review monitoring health",
    "OPEN_ORDER_AGING": "Review stale orders",
}

ALERT_ADVISORY_LEVEL: Dict[str, str] = {
    "BROKER_DISCONNECTED": "LEVEL_1",
    "EXCESS_CONCENTRATION": "LEVEL_1",
    "STALE_HEARTBEAT": "LEVEL_1",
    "LOW_CAPITAL_PRESERVATION_SCORE": "LEVEL_1",
    "EXCESS_POSITION_DRAWDOWN": "LEVEL_2",
    "OPEN_ORDER_AGING": "LEVEL_3",
}

ALERT_ADVISORY_REVIEW: Dict[str, str] = {
    "EXCESS_CONCENTRATION": "Review concentration exposure",
    "EXCESS_POSITION_DRAWDOWN": "Review losing position",
    "BROKER_DISCONNECTED": "Review broker connectivity",
    "STALE_HEARTBEAT": "Review monitoring health",
    "OPEN_ORDER_AGING": "Review stale orders",
    "LOW_CAPITAL_PRESERVATION_SCORE": "Review overall portfolio health",
}

LEVEL_BASE_URGENCY: Dict[str, int] = {
    "LEVEL_1": 92,
    "LEVEL_2": 80,
    "LEVEL_3": 60,
    "LEVEL_4": 35,
}

PRIORITY_RANK: Dict[str, int] = {
    "LEVEL_1": 1,
    "LEVEL_2": 2,
    "LEVEL_3": 3,
    "LEVEL_4": 4,
}

# Discussion options only — not trade signals or execution instructions.
DSE_OPTION_LIBRARY: Dict[str, List[Dict[str, Any]]] = {
    "Concentration Risk": [
        {
            "option": "Maintain Current Exposure",
            "potential_benefits": ["Retain upside participation"],
            "potential_risks": ["Concentration remains elevated"],
        },
        {
            "option": "Review Position Sizing",
            "potential_benefits": ["May improve diversification"],
            "potential_risks": ["Requires further analysis"],
        },
        {
            "option": "Review Diversification",
            "potential_benefits": ["Broader risk distribution across holdings"],
            "potential_risks": ["May alter portfolio construction assumptions"],
        },
    ],
    "Drawdown Risk": [
        {
            "option": "Continue Monitoring",
            "potential_benefits": ["Allows time for thesis to play out"],
            "potential_risks": ["Drawdown may deepen while monitoring"],
        },
        {
            "option": "Review Position Thesis",
            "potential_benefits": ["Clarifies whether loss is temporary or structural"],
            "potential_risks": ["May surface conflicting fundamental views"],
        },
        {
            "option": "Review Risk Exposure",
            "potential_benefits": ["Better understanding of position-level risk"],
            "potential_risks": ["Requires additional research time"],
        },
    ],
    "Operational Risk": [
        {
            "option": "Review Connectivity",
            "potential_benefits": ["May restore broker visibility"],
            "potential_risks": ["Underlying network issue may persist"],
        },
        {
            "option": "Review Broker Status",
            "potential_benefits": ["Confirms external service health"],
            "potential_risks": ["May require vendor-side resolution"],
        },
        {
            "option": "Review Monitoring Health",
            "potential_benefits": ["Restores observability confidence"],
            "potential_risks": ["Gap in monitoring may have hidden issues"],
        },
    ],
    "Execution Risk": [
        {
            "option": "Review Open Orders",
            "potential_benefits": ["Clarifies outstanding execution state"],
            "potential_risks": ["May reveal unintended order state"],
        },
        {
            "option": "Review Order Aging",
            "potential_benefits": ["Identifies stale workflow items"],
            "potential_risks": ["Aged orders may no longer reflect intent"],
        },
        {
            "option": "Review Execution Health",
            "potential_benefits": ["Broader execution process check"],
            "potential_risks": ["May require manual follow-up"],
        },
    ],
    "Monitoring Risk": [
        {
            "option": "Review Heartbeat Status",
            "potential_benefits": ["Confirms watchdog cycle continuity"],
            "potential_risks": ["Gap period may have missed events"],
        },
        {
            "option": "Review Monitoring Coverage",
            "potential_benefits": ["Validates observability scope"],
            "potential_risks": ["May reveal monitoring blind spots"],
        },
    ],
    "Portfolio Health": [
        {
            "option": "Review Overall Portfolio Health",
            "potential_benefits": ["Holistic view of capital preservation posture"],
            "potential_risks": ["Multiple issues may compound"],
        },
        {
            "option": "Review Risk Metrics",
            "potential_benefits": ["Quantifies deterioration drivers"],
            "potential_risks": ["Metrics alone do not resolve issues"],
        },
        {
            "option": "Continue Monitoring",
            "potential_benefits": ["Tracks whether score stabilizes"],
            "potential_risks": ["Score may continue to decline"],
        },
    ],
}

DSE_MONITORING_BY_ISSUE: Dict[str, List[str]] = {
    "Concentration Risk": [
        "Track largest position weight each watchdog cycle",
        "Watch for new concentration alerts",
    ],
    "Drawdown Risk": [
        "Monitor unrealized P/L percent on affected symbols",
        "Watch for additional drawdown alerts",
    ],
    "Operational Risk": [
        "Confirm broker connectivity on next cycle",
        "Review incident intelligence for disconnect patterns",
    ],
    "Execution Risk": [
        "Track open order count and aging alerts",
        "Verify order status on broker dashboard",
    ],
    "Monitoring Risk": [
        "Confirm heartbeat timestamps update each cycle",
        "Review watchdog loop process is running",
    ],
    "Portfolio Health": [
        "Track capital preservation score trend",
        "Review escalation state changes",
    ],
}

PENDING_ORDER_STATUSES = frozenset(
    {
        "pending_new",
        "pending_replace",
        "pending_cancel",
        "accepted",
        "new",
        "held",
    }
)

DEFAULT_DRAWDOWN_PCT = -10.0
DEFAULT_CONCENTRATION_PCT = 25.0
DEFAULT_ORDER_AGE_MINUTES = 60
DEFAULT_STALE_HEARTBEAT_MULTIPLIER = 2.0
DEFAULT_HEARTBEAT_GRACE_MINUTES = 55.0
BROKER_DISCONNECT_CYCLES = 2
INCIDENT_WINDOW_HOURS = 24


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _minutes_between(start: Optional[datetime], end: Optional[datetime]) -> float:
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 60.0)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _append_alerts_log(block: List[str]) -> None:
    ALERTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")


def _write_watchdog_log(lines: List[str]) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"watchdog_{stamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return log_path


def check_broker_connectivity(mode: str) -> Tuple[bool, str, Optional[str]]:
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode=mode)
        broker.get_account()
        return True, "connected", None
    except Exception as e:
        return False, "disconnected", f"{type(e).__name__}: {e}"


def fetch_account_snapshot(
    mode: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read-only account fields for CPI exposure scoring."""
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode=mode)
        acct = broker.get_account() or {}
        equity = _safe_float(acct.get("equity") or acct.get("portfolio_value"))
        cash = _safe_float(acct.get("cash"))
        return {
            "equity": equity,
            "cash": cash,
            "buying_power": _safe_float(acct.get("buying_power")),
        }, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _health_band(score: int) -> str:
    if score >= 90:
        return "EXCELLENT"
    if score >= 75:
        return "HEALTHY"
    if score >= 60:
        return "CAUTION"
    if score >= 40:
        return "ELEVATED_RISK"
    if score >= 20:
        return "HIGH_RISK"
    return "CRITICAL"


def _active_alerts_by_type(active_alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in active_alerts:
        if not isinstance(a, dict):
            continue
        t = str(a.get("alert_type") or "")
        counts[t] = counts.get(t, 0) + 1
    return counts


def _score_drawdown(
    positions: List[Dict[str, Any]],
    alert_counts: Dict[str, int],
) -> int:
    score = 100.0
    pcts = [p.get("unrealized_pl_pct") for p in positions if p.get("unrealized_pl_pct") is not None]
    if pcts:
        worst = min(pcts)
        if worst < 0:
            score = 100.0 + worst * 2.5
    elif not positions:
        score = 92.0
    score -= 12.0 * alert_counts.get("EXCESS_POSITION_DRAWDOWN", 0)
    return _clamp_score(score)


def _score_concentration(
    positions: List[Dict[str, Any]],
    alert_counts: Dict[str, int],
    *,
    concentration_threshold_pct: float,
) -> int:
    mvs = [abs(_safe_float(p.get("market_value"), 0.0) or 0.0) for p in positions]
    total_mv = sum(mvs)
    if total_mv <= 0:
        return 88 if not alert_counts.get("EXCESS_CONCENTRATION") else 65

    largest_pct = (max(mvs) / total_mv) * 100.0
    over = max(0.0, largest_pct - concentration_threshold_pct)
    score = 100.0 - over * 1.5
    score -= 10.0 * alert_counts.get("EXCESS_CONCENTRATION", 0)
    return _clamp_score(score)


def _score_exposure(
    positions: List[Dict[str, Any]],
    account: Optional[Dict[str, Any]],
) -> int:
    n_pos = len(positions)
    mvs = [abs(_safe_float(p.get("market_value"), 0.0) or 0.0) for p in positions]
    total_mv = sum(mvs)

    equity = _safe_float((account or {}).get("equity"))
    cash = _safe_float((account or {}).get("cash"))

    if equity and equity > 0:
        cash_ratio = (cash or 0.0) / equity
        invested_ratio = total_mv / equity
    elif total_mv > 0:
        cash_ratio = 0.0
        invested_ratio = 1.0
    else:
        return 90

    if 0.10 <= cash_ratio <= 0.65:
        cash_score = 100.0
    elif cash_ratio < 0.05:
        cash_score = 45.0
    elif cash_ratio < 0.10:
        cash_score = 70.0
    elif cash_ratio > 0.85:
        cash_score = 72.0
    else:
        cash_score = 85.0

    if invested_ratio > 0.98:
        invested_score = 40.0
    elif invested_ratio > 0.90:
        invested_score = 65.0
    elif invested_ratio < 0.05:
        invested_score = 75.0
    else:
        invested_score = 95.0

    if n_pos == 0:
        div_score = 70.0
    elif n_pos == 1:
        div_score = 45.0
    elif n_pos <= 3:
        div_score = 65.0
    elif n_pos <= 8:
        div_score = 85.0
    else:
        div_score = 100.0

    return _clamp_score((cash_score + invested_score + div_score) / 3.0)


def _score_operational(
    alert_counts: Dict[str, int],
    incident_intelligence: Dict[str, Any],
    *,
    broker_connected: bool,
) -> int:
    score = 100.0
    score -= 40.0 * alert_counts.get("BROKER_DISCONNECTED", 0)
    score -= 35.0 * alert_counts.get("STALE_HEARTBEAT", 0)
    score -= 5.0 * int(incident_intelligence.get("incident_count_24h") or 0)
    score -= 8.0 * int(incident_intelligence.get("broker_disconnect_count") or 0)
    if not broker_connected:
        score = min(score, 25.0)
    return _clamp_score(score)


def _score_execution(
    alert_counts: Dict[str, int],
    *,
    open_orders_count: int,
    pending_orders_count: int,
) -> int:
    score = 100.0
    score -= 20.0 * alert_counts.get("OPEN_ORDER_AGING", 0)
    score -= min(15.0, 2.0 * pending_orders_count)
    if open_orders_count > 5:
        score -= 10.0
    return _clamp_score(score)


def _compute_risk_trend(current_score: int) -> str:
    if not CPI_HISTORY_PATH.is_file():
        return "STABLE"
    try:
        prior_scores: List[int] = []
        with open(CPI_HISTORY_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("capital_preservation_score")
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    prior_scores.append(int(float(raw)))
                except ValueError:
                    continue
        if not prior_scores:
            return "STABLE"
        window = prior_scores[-TREND_LOOKBACK_ROWS:]
        baseline = sum(window) / len(window)
        if current_score >= baseline + TREND_DELTA_POINTS:
            return "IMPROVING"
        if current_score <= baseline - TREND_DELTA_POINTS:
            return "DETERIORATING"
        return "STABLE"
    except OSError:
        return "STABLE"


def _append_cpi_history(
    timestamp: str,
    score: int,
    health_band: str,
    active_alerts: int,
    risk_trend: str,
) -> None:
    CPI_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CPI_HISTORY_PATH.exists() or CPI_HISTORY_PATH.stat().st_size == 0
    with open(CPI_HISTORY_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "capital_preservation_score",
                "health_band",
                "active_alerts",
                "risk_trend",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "capital_preservation_score": score,
                "health_band": health_band,
                "active_alerts": active_alerts,
                "risk_trend": risk_trend,
            }
        )


def compute_capital_preservation_intelligence(
    *,
    timestamp: str,
    positions: List[Dict[str, Any]],
    active_alerts: List[Dict[str, Any]],
    incident_intelligence: Dict[str, Any],
    account: Optional[Dict[str, Any]],
    broker_connected: bool,
    open_orders_count: int,
    pending_orders_count: int,
    concentration_threshold_pct: float,
) -> Dict[str, Any]:
    """Read-only Capital Preservation Score (CPS) and health classification."""
    alert_counts = _active_alerts_by_type(active_alerts)
    components = {
        "drawdown": _score_drawdown(positions, alert_counts),
        "concentration": _score_concentration(
            positions,
            alert_counts,
            concentration_threshold_pct=concentration_threshold_pct,
        ),
        "exposure": _score_exposure(positions, account),
        "operational": _score_operational(
            alert_counts,
            incident_intelligence,
            broker_connected=broker_connected,
        ),
        "execution": _score_execution(
            alert_counts,
            open_orders_count=open_orders_count,
            pending_orders_count=pending_orders_count,
        ),
    }
    cps = _clamp_score(sum(components[k] * CPI_WEIGHTS[k] for k in CPI_WEIGHTS))
    band = _health_band(cps)
    trend = _compute_risk_trend(cps)
    active_n = len(active_alerts)

    doc = {
        "generated_at": timestamp,
        "capital_preservation_score": cps,
        "health_band": band,
        "risk_trend": trend,
        "component_scores": components,
        "active_alerts": active_n,
        "weights": CPI_WEIGHTS,
    }
    _atomic_write_json(doc, CPI_PATH)
    _append_cpi_history(timestamp, cps, band, active_n, trend)
    return doc


def _escalation_state_from_cps(cps: int) -> str:
    if cps >= 90:
        return "GREEN"
    if cps >= 75:
        return "YELLOW"
    if cps >= 60:
        return "ORANGE"
    if cps >= 40:
        return "RED"
    return "CRITICAL"


def _escalation_reasons_from_alerts(active_alerts: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    reasons: List[str] = []
    for alert in active_alerts:
        if not isinstance(alert, dict):
            continue
        alert_type = str(alert.get("alert_type") or "").strip()
        if not alert_type or alert_type in seen:
            continue
        seen.add(alert_type)
        reasons.append(alert_type)
    return sorted(reasons)


def _recommended_review_actions(active_alerts: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    actions: List[str] = []
    for alert in active_alerts:
        if not isinstance(alert, dict):
            continue
        alert_type = str(alert.get("alert_type") or "").strip()
        action = ALERT_REVIEW_ACTIONS.get(alert_type)
        if action and action not in seen:
            seen.add(action)
            actions.append(action)
    return actions


def _append_escalation_history(
    timestamp: str,
    score: int,
    health_band: str,
    escalation_state: str,
    active_alerts: int,
) -> None:
    CPE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CPE_HISTORY_PATH.exists() or CPE_HISTORY_PATH.stat().st_size == 0
    with open(CPE_HISTORY_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "capital_preservation_score",
                "health_band",
                "escalation_state",
                "active_alerts",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "capital_preservation_score": score,
                "health_band": health_band,
                "escalation_state": escalation_state,
                "active_alerts": active_alerts,
            }
        )


def compute_capital_preservation_escalation(
    cpi_doc: Dict[str, Any],
    active_alerts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Read-only CPEF: escalation state and review recommendations (no actions)."""
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    health_band = str(cpi_doc.get("health_band") or "")
    timestamp = str(cpi_doc.get("generated_at") or _iso_utc())
    escalation_state = _escalation_state_from_cps(cps)

    escalation_reason = _escalation_reasons_from_alerts(active_alerts)
    recommended = _recommended_review_actions(active_alerts)

    if not escalation_reason and escalation_state in ("ORANGE", "RED", "CRITICAL"):
        escalation_reason = ["LOW_CAPITAL_PRESERVATION_SCORE"]
    if not recommended and escalation_state in ("ORANGE", "RED", "CRITICAL"):
        recommended = ["Review overall portfolio health"]

    reason_labels = [ALERT_ESCALATION_REASON.get(r, r) for r in escalation_reason]

    doc = {
        "generated_at": timestamp,
        "capital_preservation_score": cps,
        "health_band": health_band,
        "risk_trend": cpi_doc.get("risk_trend"),
        "escalation_state": escalation_state,
        "escalation_reason": escalation_reason,
        "escalation_reason_labels": reason_labels,
        "recommended_review_actions": recommended,
        "active_alerts": len(active_alerts),
    }
    _atomic_write_json(doc, CPE_PATH)
    _append_escalation_history(timestamp, cps, health_band, escalation_state, len(active_alerts))
    return doc


def _advisory_reason_from_alert(alert: Dict[str, Any]) -> str:
    alert_type = str(alert.get("alert_type") or "")
    subject = str(alert.get("subject") or "").strip()
    details = alert.get("details") if isinstance(alert.get("details"), dict) else {}

    if alert_type == "EXCESS_CONCENTRATION" and subject:
        pct = details.get("portfolio_pct")
        if pct is not None:
            return f"{subject} exceeds concentration threshold ({pct}% of portfolio)"
        return f"{subject} exceeds concentration threshold"

    if alert_type == "EXCESS_POSITION_DRAWDOWN" and subject:
        pct = details.get("unrealized_pl_pct")
        if pct is not None:
            return f"{subject} exceeds drawdown threshold ({pct}% unrealized)"
        return f"{subject} exceeds drawdown threshold"

    if alert_type == "BROKER_DISCONNECTED":
        return "Broker connectivity unavailable"

    if alert_type == "STALE_HEARTBEAT":
        age = details.get("age_minutes")
        if age is not None:
            return f"Watchdog heartbeat stale ({age} minutes since last beat)"
        return "Watchdog heartbeat is stale"

    if alert_type == "OPEN_ORDER_AGING":
        sym = subject or details.get("order_id") or "order"
        age = details.get("age_minutes")
        if age is not None:
            return f"Open order for {sym} exceeds age threshold ({age} minutes)"
        return f"Open order for {sym} exceeds age threshold"

    if alert_type == "LOW_CAPITAL_PRESERVATION_SCORE":
        return "Capital preservation score below acceptable range"

    return ALERT_ESCALATION_REASON.get(alert_type, alert_type or "Portfolio concern")


def _urgency_score_for_advisory(
    priority: str,
    alert: Dict[str, Any],
    *,
    cps: int,
    escalation_state: str,
) -> int:
    base = LEVEL_BASE_URGENCY.get(priority, 50)
    score = float(base)
    severity = str(alert.get("severity") or "").upper()
    if severity == "HIGH":
        score += 8.0
    elif severity == "MEDIUM":
        score += 3.0

    details = alert.get("details") if isinstance(alert.get("details"), dict) else {}
    alert_type = str(alert.get("alert_type") or "")

    if alert_type == "EXCESS_CONCENTRATION":
        pct = _safe_float(details.get("portfolio_pct"), 0.0) or 0.0
        thresh = _safe_float(details.get("threshold_pct"), 25.0) or 25.0
        score += min(8.0, max(0.0, (pct - thresh) * 0.15))

    if alert_type == "EXCESS_POSITION_DRAWDOWN":
        pct = _safe_float(details.get("unrealized_pl_pct"), 0.0) or 0.0
        thresh = _safe_float(details.get("threshold_pct"), -10.0) or -10.0
        if pct < thresh:
            score += min(8.0, abs(pct - thresh) * 0.25)

    if alert_type == "STALE_HEARTBEAT":
        age = _safe_float(details.get("age_minutes"), 0.0) or 0.0
        score += min(6.0, age * 0.5)

    if alert_type == "OPEN_ORDER_AGING":
        age = _safe_float(details.get("age_minutes"), 0.0) or 0.0
        score += min(5.0, age * 0.05)

    if escalation_state in ("RED", "CRITICAL"):
        score += 4.0
    if cps < 40:
        score += 3.0

    dur = _safe_float(alert.get("duration_minutes"), 0.0) or 0.0
    score += min(5.0, dur * 0.02)

    return _clamp_score(score)


def _advisory_from_alert(
    alert: Dict[str, Any],
    *,
    cps: int,
    escalation_state: str,
) -> Dict[str, Any]:
    alert_type = str(alert.get("alert_type") or "UNKNOWN")
    priority = ALERT_ADVISORY_LEVEL.get(alert_type, "LEVEL_3")
    title = ALERT_ESCALATION_REASON.get(alert_type, alert_type)
    return {
        "priority": priority,
        "title": title,
        "urgency_score": _urgency_score_for_advisory(
            priority, alert, cps=cps, escalation_state=escalation_state
        ),
        "reason": _advisory_reason_from_alert(alert),
        "recommended_review": ALERT_ADVISORY_REVIEW.get(
            alert_type, ALERT_REVIEW_ACTIONS.get(alert_type, "Review portfolio concern")
        ),
        "alert_type": alert_type,
        "subject": alert.get("subject"),
    }


def _append_advisory_history(
    timestamp: str,
    advisories: List[Dict[str, Any]],
    escalation_state: str,
) -> None:
    if not advisories:
        return
    CPA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CPA_HISTORY_PATH.exists() or CPA_HISTORY_PATH.stat().st_size == 0
    with open(CPA_HISTORY_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "priority",
                "title",
                "urgency_score",
                "escalation_state",
            ],
        )
        if write_header:
            writer.writeheader()
        for adv in advisories:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "priority": adv.get("priority"),
                    "title": adv.get("title"),
                    "urgency_score": adv.get("urgency_score"),
                    "escalation_state": escalation_state,
                }
            )


def compute_capital_preservation_advisory(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    active_alerts: List[Dict[str, Any]],
    incident_intelligence: Dict[str, Any],
) -> Dict[str, Any]:
    """Read-only CPAE: ranked review advisories (no actions)."""
    timestamp = str(cpi_doc.get("generated_at") or cpe_doc.get("generated_at") or _iso_utc())
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation_state = str(cpe_doc.get("escalation_state") or "GREEN")

    advisories: List[Dict[str, Any]] = []
    seen_types: set[str] = set()

    for alert in active_alerts:
        if not isinstance(alert, dict):
            continue
        alert_type = str(alert.get("alert_type") or "")
        advisories.append(_advisory_from_alert(alert, cps=cps, escalation_state=escalation_state))
        seen_types.add(alert_type)

    for reason in cpe_doc.get("escalation_reason") or []:
        if reason == "LOW_CAPITAL_PRESERVATION_SCORE" and reason not in seen_types:
            pseudo = {
                "alert_type": reason,
                "severity": "HIGH",
                "subject": None,
                "details": {"capital_preservation_score": cps},
                "duration_minutes": 0.0,
            }
            advisories.append(
                _advisory_from_alert(pseudo, cps=cps, escalation_state=escalation_state)
            )

    if int(incident_intelligence.get("broker_disconnect_count") or 0) > 0:
        has_broker = any(a.get("alert_type") == "BROKER_DISCONNECTED" for a in advisories)
        if not has_broker:
            pseudo = {
                "alert_type": "BROKER_DISCONNECTED",
                "severity": "HIGH",
                "subject": None,
                "details": incident_intelligence,
                "duration_minutes": incident_intelligence.get("max_disconnect_duration_minutes", 0),
            }
            advisories.append(
                _advisory_from_alert(pseudo, cps=cps, escalation_state=escalation_state)
            )

    advisories.sort(
        key=lambda a: (
            PRIORITY_RANK.get(str(a.get("priority")), 9),
            -int(a.get("urgency_score") or 0),
        ),
    )

    if advisories:
        highest_priority = min(
            (str(a.get("priority")) for a in advisories),
            key=lambda p: PRIORITY_RANK.get(p, 9),
        )
        top_issue = str(advisories[0].get("title"))
    else:
        highest_priority = "LEVEL_4"
        top_issue = "No active concerns"

    doc = {
        "generated_at": timestamp,
        "capital_preservation_score": cps,
        "escalation_state": escalation_state,
        "risk_trend": cpi_doc.get("risk_trend"),
        "advisories": advisories,
        "summary": {
            "highest_priority": highest_priority,
            "advisory_count": len(advisories),
            "top_issue": top_issue,
        },
    }
    _atomic_write_json(doc, CPA_PATH)
    _append_advisory_history(timestamp, advisories, escalation_state)
    return doc


def _decision_support_item_from_advisory(advisory: Dict[str, Any]) -> Dict[str, Any]:
    issue = str(advisory.get("title") or "Portfolio concern")
    options_template = DSE_OPTION_LIBRARY.get(issue, DSE_OPTION_LIBRARY["Portfolio Health"])
    options = [dict(o) for o in options_template]
    monitoring = list(
        DSE_MONITORING_BY_ISSUE.get(issue, DSE_MONITORING_BY_ISSUE["Portfolio Health"])
    )

    reason = str(advisory.get("reason") or "")
    subject = advisory.get("subject")
    if subject:
        issue_summary = f"{issue}: {reason} (subject: {subject})"
    else:
        issue_summary = f"{issue}: {reason}" if reason else issue

    return {
        "issue": issue,
        "priority": advisory.get("priority"),
        "urgency_score": advisory.get("urgency_score"),
        "issue_summary": issue_summary,
        "available_review_options": [o["option"] for o in options],
        "monitoring_considerations": monitoring,
        "options": options,
        "alert_type": advisory.get("alert_type"),
        "subject": subject,
    }


def _append_decision_history(
    timestamp: str,
    items: List[Dict[str, Any]],
    escalation_state: str,
) -> None:
    if not items:
        return
    CPD_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CPD_HISTORY_PATH.exists() or CPD_HISTORY_PATH.stat().st_size == 0
    with open(CPD_HISTORY_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "issue",
                "priority",
                "urgency_score",
                "escalation_state",
            ],
        )
        if write_header:
            writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "issue": item.get("issue"),
                    "priority": item.get("priority"),
                    "urgency_score": item.get("urgency_score"),
                    "escalation_state": escalation_state,
                }
            )


def compute_capital_preservation_decision_support(
    cpa_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Read-only DSE: structured review options per advisory (not trade instructions)."""
    timestamp = str(cpa_doc.get("generated_at") or _iso_utc())
    cps = int(cpa_doc.get("capital_preservation_score") or 0)
    escalation_state = str(cpa_doc.get("escalation_state") or "GREEN")

    items = [
        _decision_support_item_from_advisory(adv)
        for adv in (cpa_doc.get("advisories") or [])
        if isinstance(adv, dict)
    ]

    if items:
        highest_priority = min(
            (str(i.get("priority")) for i in items),
            key=lambda p: PRIORITY_RANK.get(p, 9),
        )
        top_issue = str(items[0].get("issue"))
    else:
        highest_priority = "LEVEL_4"
        top_issue = "No active concerns"

    doc = {
        "generated_at": timestamp,
        "capital_preservation_score": cps,
        "escalation_state": escalation_state,
        "risk_trend": cpa_doc.get("risk_trend"),
        "decision_support_items": items,
        "summary": {
            "issue_count": len(items),
            "highest_priority": highest_priority,
            "top_issue": top_issue,
        },
    }
    _atomic_write_json(doc, CPD_PATH)
    _append_decision_history(timestamp, items, escalation_state)
    return doc


GOVERNANCE_AWARENESS_BY_ESCALATION: Dict[str, Tuple[str, str]] = {
    "GREEN": ("ROUTINE_OVERSIGHT", "Routine Oversight"),
    "YELLOW": ("INCREASED_MONITORING", "Increased Monitoring"),
    "ORANGE": ("GOVERNANCE_REVIEW_RECOMMENDED", "Governance Review Recommended"),
    "RED": ("MANAGEMENT_REVIEW_REQUIRED", "Management Review Required"),
    "CRITICAL": ("GOVERNANCE_ESCALATION_REQUIRED", "Governance Escalation Required"),
}


def compute_governance_risk_summary(
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Read-only governance awareness mapping from capital preservation posture."""
    timestamp = str(cpi_doc.get("generated_at") or cpe_doc.get("generated_at") or _iso_utc())
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation_state = str(cpe_doc.get("escalation_state") or "GREEN")
    level_code, level_label = GOVERNANCE_AWARENESS_BY_ESCALATION.get(
        escalation_state,
        ("INCREASED_MONITORING", "Increased Monitoring"),
    )

    drivers: List[str] = []
    for adv in cpa_doc.get("advisories") or []:
        if isinstance(adv, dict):
            title = str(adv.get("title") or "").strip()
            if title and title not in drivers:
                drivers.append(title)
    if not drivers:
        for label in cpe_doc.get("escalation_reason_labels") or []:
            s = str(label).strip()
            if s and s not in drivers:
                drivers.append(s)

    doc = {
        "generated_at": timestamp,
        "capital_preservation_score": cps,
        "escalation_state": escalation_state,
        "health_band": cpi_doc.get("health_band"),
        "risk_trend": cpi_doc.get("risk_trend"),
        "governance_awareness_level": level_code,
        "governance_awareness_label": level_label,
        "governance_summary": drivers,
        "governance_drivers": drivers,
        "governance_status": level_label,
    }
    _atomic_write_json(doc, GOV_RISK_PATH)
    return doc


def normalize_position(raw: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(raw.get("symbol") or raw.get("ticker") or "").strip().upper()
    qty = _safe_float(raw.get("qty") or raw.get("quantity"), 0.0) or 0.0
    market_value = _safe_float(raw.get("market_value"))
    unrealized_pl = _safe_float(raw.get("unrealized_pl"))
    ulpc = _safe_float(raw.get("unrealized_plpc") or raw.get("unrealized_pl_pct"))
    if ulpc is not None and abs(ulpc) <= 1.0:
        unrealized_pl_pct = ulpc * 100.0
    else:
        unrealized_pl_pct = ulpc
    return {
        "symbol": sym,
        "quantity": qty,
        "market_value": market_value,
        "unrealized_pl": unrealized_pl,
        "unrealized_pl_pct": unrealized_pl_pct,
    }


def check_positions_health(mode: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode=mode)
        raw = broker.get_positions() or []
        return [normalize_position(p) for p in raw if isinstance(p, dict)], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def check_open_orders(
    mode: str,
) -> Tuple[int, int, List[Dict[str, Any]], Optional[str]]:
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode=mode)
        orders = broker.list_orders(status="open", nested=True, limit=500) or []
        pending_count = 0
        normalized: List[Dict[str, Any]] = []
        for o in orders:
            if not isinstance(o, dict):
                continue
            st = str(o.get("status") or "").lower()
            if st in PENDING_ORDER_STATUSES:
                pending_count += 1
            submitted = o.get("submitted_at") or o.get("created_at") or o.get("updated_at")
            normalized.append(
                {
                    "id": o.get("id"),
                    "symbol": str(o.get("symbol") or "").strip().upper(),
                    "status": st,
                    "submitted_at": str(submitted) if submitted else None,
                }
            )
        return len(normalized), pending_count, normalized, None
    except Exception as e:
        return 0, 0, [], f"{type(e).__name__}: {e}"


def _largest_unrealized_extremes(
    positions: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    pl_values = [p["unrealized_pl"] for p in positions if p.get("unrealized_pl") is not None]
    if not pl_values:
        return None, None
    losses = [v for v in pl_values if v < 0]
    gains = [v for v in pl_values if v > 0]
    largest_loss = min(losses) if losses else None
    largest_gain = max(gains) if gains else None
    return largest_loss, largest_gain


def _overall_watchdog_status(
    broker_connected: bool,
    positions_error: Optional[str],
    orders_error: Optional[str],
) -> str:
    if not broker_connected:
        return "ERROR"
    if positions_error or orders_error:
        return "DEGRADED"
    return "OK"


def _load_previous_heartbeat_ts() -> Optional[str]:
    prev = _read_json(STATUS_PATH)
    hb = prev.get("heartbeat") if isinstance(prev.get("heartbeat"), dict) else {}
    return hb.get("timestamp") or prev.get("timestamp")


def _alert_key(alert_type: str, subject: str = "") -> str:
    return f"{alert_type}:{subject or 'global'}"


def _new_alert(
    alert_type: str,
    severity: str,
    first_seen: str,
    *,
    subject: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "alert_id": _alert_key(alert_type, subject),
        "alert_type": alert_type,
        "severity": severity,
        "first_seen": first_seen,
        "duration_minutes": 0.0,
        "status": "ACTIVE",
        "subject": subject or None,
        "details": details or {},
    }


def _update_duration(alert: Dict[str, Any], now: datetime) -> None:
    first = _parse_iso(alert.get("first_seen"))
    alert["duration_minutes"] = round(_minutes_between(first, now), 2)


def _log_alert_event(alert: Dict[str, Any], *, resolved: bool = False) -> None:
    tag = "[RESOLVED]" if resolved else "[ALERT]"
    lines = [
        tag,
        f"type={alert.get('alert_type')}",
        f"severity={alert.get('severity')}",
    ]
    if resolved:
        lines.append(f"resolved_at={alert.get('resolved_at')}")
    else:
        lines.append(f"first_seen={alert.get('first_seen')}")
    lines.append(f"duration={alert.get('duration_minutes')}m")
    if alert.get("subject"):
        lines.append(f"subject={alert.get('subject')}")
    _append_alerts_log(lines)


def _prune_incident_events(events: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
    cutoff = now - timedelta(hours=INCIDENT_WINDOW_HOURS)
    kept: List[Dict[str, Any]] = []
    for ev in events:
        started = _parse_iso(ev.get("started"))
        if started and started >= cutoff:
            kept.append(ev)
    return kept


def _compute_incident_intelligence(
    events: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    events = _prune_incident_events(events, now)
    disconnect_events = [e for e in events if e.get("type") == "BROKER_DISCONNECTED"]
    max_dur = 0.0
    last_disconnect: Optional[str] = None
    for ev in disconnect_events:
        started = _parse_iso(ev.get("started"))
        ended = _parse_iso(ev.get("ended")) or now
        dur = _minutes_between(started, ended)
        if dur > max_dur:
            max_dur = dur
        if started:
            iso = _iso_utc(started)
            if last_disconnect is None or iso > last_disconnect:
                last_disconnect = iso

    return {
        "incident_count_24h": len(events),
        "broker_disconnect_count": len(disconnect_events),
        "max_disconnect_duration_minutes": round(max_dur, 2),
        "last_disconnect": last_disconnect,
    }


class WatchdogAlertEngine:
    """Stateful read-only alert evaluation across watchdog cycles."""

    def __init__(
        self,
        *,
        drawdown_pct: float,
        concentration_pct: float,
        order_age_minutes: float,
        stale_heartbeat_minutes: float,
    ) -> None:
        self.drawdown_pct = drawdown_pct
        self.concentration_pct = concentration_pct
        self.order_age_minutes = order_age_minutes
        self.stale_heartbeat_minutes = stale_heartbeat_minutes
        self.now = _utc_now()
        self.ts = _iso_utc(self.now)

        self.state = _read_json(STATE_PATH)
        self.alerts_doc = _read_json(ALERTS_PATH)
        self.active: Dict[str, Dict[str, Any]] = {}
        for a in self.alerts_doc.get("active_alerts") or []:
            if isinstance(a, dict) and a.get("alert_id"):
                self.active[str(a["alert_id"])] = dict(a)

        self.resolved: List[Dict[str, Any]] = list(self.alerts_doc.get("resolved_alerts") or [])
        self.incident_events: List[Dict[str, Any]] = list(self.state.get("incident_events") or [])
        self.broker_disconnect_streak = int(self.state.get("broker_disconnect_streak") or 0)
        self.newly_raised: List[str] = []
        self.newly_resolved: List[str] = []

    def _activate(self, alert: Dict[str, Any]) -> None:
        aid = str(alert["alert_id"])
        if aid not in self.active:
            self.newly_raised.append(aid)
            _log_alert_event(alert, resolved=False)
        else:
            alert["first_seen"] = self.active[aid].get("first_seen", alert["first_seen"])
        _update_duration(alert, self.now)
        self.active[aid] = alert

    def _resolve(self, alert_id: str) -> None:
        if alert_id not in self.active:
            return
        alert = dict(self.active.pop(alert_id))
        alert["status"] = "RESOLVED"
        alert["resolved_at"] = self.ts
        _update_duration(alert, self.now)
        self.resolved.append(alert)
        self.newly_resolved.append(alert_id)
        _log_alert_event(alert, resolved=True)

        if alert.get("alert_type") == "BROKER_DISCONNECTED":
            started = _parse_iso(alert.get("first_seen"))
            self.incident_events.append(
                {
                    "type": "BROKER_DISCONNECTED",
                    "started": alert.get("first_seen"),
                    "ended": self.ts,
                    "duration_minutes": alert.get("duration_minutes"),
                }
            )
            if started:
                dur = _minutes_between(started, self.now)
                if dur > 0:
                    self.incident_events[-1]["duration_minutes"] = round(dur, 2)

    def _sync_type(self, alert_type: str, desired: Dict[str, Dict[str, Any]]) -> None:
        desired_ids = set(desired.keys())
        for aid, alert in list(self.active.items()):
            if alert.get("alert_type") == alert_type and aid not in desired_ids:
                self._resolve(aid)
        for aid, alert in desired.items():
            self._activate(alert)

    def evaluate_broker_disconnect(self, broker_connected: bool) -> None:
        if broker_connected:
            self.broker_disconnect_streak = 0
            self._sync_type("BROKER_DISCONNECTED", {})
            return

        self.broker_disconnect_streak += 1
        if self.broker_disconnect_streak < BROKER_DISCONNECT_CYCLES:
            self._sync_type("BROKER_DISCONNECTED", {})
            return

        aid = _alert_key("BROKER_DISCONNECTED")
        first_seen = self.ts
        if aid in self.active:
            first_seen = self.active[aid].get("first_seen", first_seen)
        elif self.state.get("broker_disconnect_first_seen"):
            first_seen = str(self.state["broker_disconnect_first_seen"])

        alert = _new_alert("BROKER_DISCONNECTED", "HIGH", first_seen)
        self._sync_type("BROKER_DISCONNECTED", {aid: alert})
        self.state["broker_disconnect_first_seen"] = first_seen

    def evaluate_stale_heartbeat(self, previous_hb_ts: Optional[str]) -> None:
        grace_until = _parse_iso(self.state.get("heartbeat_grace_until"))
        if grace_until is not None and self.now < grace_until:
            self._sync_type("STALE_HEARTBEAT", {})
            return

        desired: Dict[str, Dict[str, Any]] = {}
        prev_dt = _parse_iso(previous_hb_ts)
        if prev_dt is None:
            self._sync_type("STALE_HEARTBEAT", {})
            return

        age_min = _minutes_between(prev_dt, self.now)
        if age_min > self.stale_heartbeat_minutes:
            aid = _alert_key("STALE_HEARTBEAT")
            first_seen = self.active.get(aid, {}).get("first_seen", self.ts)
            alert = _new_alert(
                "STALE_HEARTBEAT",
                "HIGH",
                first_seen,
                details={
                    "previous_heartbeat": previous_hb_ts,
                    "age_minutes": round(age_min, 2),
                    "threshold_minutes": self.stale_heartbeat_minutes,
                },
            )
            desired[aid] = alert
        self._sync_type("STALE_HEARTBEAT", desired)

    def evaluate_drawdown(self, positions: List[Dict[str, Any]]) -> None:
        desired: Dict[str, Dict[str, Any]] = {}
        for p in positions:
            sym = p.get("symbol") or ""
            pct = p.get("unrealized_pl_pct")
            if not sym or pct is None or pct > self.drawdown_pct:
                continue
            aid = _alert_key("EXCESS_POSITION_DRAWDOWN", sym)
            first_seen = self.active.get(aid, {}).get("first_seen", self.ts)
            desired[aid] = _new_alert(
                "EXCESS_POSITION_DRAWDOWN",
                "MEDIUM",
                first_seen,
                subject=sym,
                details={"unrealized_pl_pct": pct, "threshold_pct": self.drawdown_pct},
            )
        self._sync_type("EXCESS_POSITION_DRAWDOWN", desired)

    def evaluate_concentration(self, positions: List[Dict[str, Any]]) -> None:
        desired: Dict[str, Dict[str, Any]] = {}
        mvs = [abs(_safe_float(p.get("market_value"), 0.0) or 0.0) for p in positions]
        total_mv = sum(mvs)
        if total_mv <= 0:
            self._sync_type("EXCESS_CONCENTRATION", {})
            return

        for p in positions:
            sym = p.get("symbol") or ""
            mv = abs(_safe_float(p.get("market_value"), 0.0) or 0.0)
            if not sym or mv <= 0:
                continue
            pct_of_portfolio = (mv / total_mv) * 100.0
            if pct_of_portfolio <= self.concentration_pct:
                continue
            aid = _alert_key("EXCESS_CONCENTRATION", sym)
            first_seen = self.active.get(aid, {}).get("first_seen", self.ts)
            desired[aid] = _new_alert(
                "EXCESS_CONCENTRATION",
                "MEDIUM",
                first_seen,
                subject=sym,
                details={
                    "portfolio_pct": round(pct_of_portfolio, 2),
                    "threshold_pct": self.concentration_pct,
                    "market_value": mv,
                    "portfolio_market_value": round(total_mv, 2),
                },
            )
        self._sync_type("EXCESS_CONCENTRATION", desired)

    def evaluate_order_aging(self, orders: List[Dict[str, Any]]) -> None:
        desired: Dict[str, Dict[str, Any]] = {}
        for o in orders:
            oid = str(o.get("id") or o.get("symbol") or "unknown")
            submitted = _parse_iso(o.get("submitted_at"))
            if submitted is None:
                continue
            age_min = _minutes_between(submitted, self.now)
            if age_min <= self.order_age_minutes:
                continue
            aid = _alert_key("OPEN_ORDER_AGING", oid)
            first_seen = self.active.get(aid, {}).get("first_seen", self.ts)
            desired[aid] = _new_alert(
                "OPEN_ORDER_AGING",
                "LOW",
                first_seen,
                subject=str(o.get("symbol") or oid),
                details={
                    "order_id": o.get("id"),
                    "age_minutes": round(age_min, 2),
                    "threshold_minutes": self.order_age_minutes,
                    "submitted_at": o.get("submitted_at"),
                },
            )
        self._sync_type("OPEN_ORDER_AGING", desired)

    def persist(self) -> Dict[str, Any]:
        self.incident_events = _prune_incident_events(self.incident_events, self.now)
        intelligence = _compute_incident_intelligence(self.incident_events, self.now)

        active_list = sorted(self.active.values(), key=lambda a: a.get("alert_type", ""))
        resolved_tail = self.resolved[-200:]
        out = {
            "generated_at": self.ts,
            "active_alerts": active_list,
            "resolved_alerts": resolved_tail,
            "incident_intelligence": intelligence,
        }
        _atomic_write_json(out, ALERTS_PATH)

        self.state["broker_disconnect_streak"] = self.broker_disconnect_streak
        self.state["last_cycle_ts"] = self.ts
        self.state["incident_events"] = self.incident_events
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.state, STATE_PATH)
        return out


def run_watchdog(
    mode: str = "paper",
    *,
    drawdown_pct: float = DEFAULT_DRAWDOWN_PCT,
    concentration_pct: float = DEFAULT_CONCENTRATION_PCT,
    order_age_minutes: float = DEFAULT_ORDER_AGE_MINUTES,
    expected_interval_minutes: float = 1.0,
) -> Dict[str, Any]:
    ts = _iso_utc()
    previous_hb_ts = _load_previous_heartbeat_ts()
    stale_minutes = max(
        1.0,
        expected_interval_minutes * DEFAULT_STALE_HEARTBEAT_MULTIPLIER,
    )

    broker_connected, broker_status, broker_error = check_broker_connectivity(mode)

    positions: List[Dict[str, Any]] = []
    positions_error: Optional[str] = None
    if broker_connected:
        positions, positions_error = check_positions_health(mode)

    open_orders_count = 0
    pending_orders_count = 0
    open_orders: List[Dict[str, Any]] = []
    orders_error: Optional[str] = None
    if broker_connected:
        open_orders_count, pending_orders_count, open_orders, orders_error = check_open_orders(mode)

    account: Optional[Dict[str, Any]] = None
    account_error: Optional[str] = None
    if broker_connected:
        account, account_error = fetch_account_snapshot(mode)

    largest_loss, largest_gain = _largest_unrealized_extremes(positions)
    watchdog_status = _overall_watchdog_status(broker_connected, positions_error, orders_error)

    heartbeat = {
        "timestamp": ts,
        "status": watchdog_status,
        "positions_count": len(positions),
        "open_orders_count": open_orders_count,
    }

    engine = WatchdogAlertEngine(
        drawdown_pct=drawdown_pct,
        concentration_pct=concentration_pct,
        order_age_minutes=order_age_minutes,
        stale_heartbeat_minutes=stale_minutes,
    )
    engine.evaluate_broker_disconnect(broker_connected)
    engine.evaluate_stale_heartbeat(previous_hb_ts)
    if broker_connected and not positions_error:
        engine.evaluate_drawdown(positions)
        engine.evaluate_concentration(positions)
    if broker_connected and not orders_error:
        engine.evaluate_order_aging(open_orders)
    alerts_doc = engine.persist()
    active_alerts = list(alerts_doc.get("active_alerts") or [])
    incident_intel = alerts_doc.get("incident_intelligence") or {}

    cpi_doc = compute_capital_preservation_intelligence(
        timestamp=ts,
        positions=positions,
        active_alerts=active_alerts,
        incident_intelligence=incident_intel,
        account=account,
        broker_connected=broker_connected,
        open_orders_count=open_orders_count,
        pending_orders_count=pending_orders_count,
        concentration_threshold_pct=concentration_pct,
    )

    cpe_doc = compute_capital_preservation_escalation(cpi_doc, active_alerts)

    cpa_doc = compute_capital_preservation_advisory(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        active_alerts=active_alerts,
        incident_intelligence=incident_intel,
    )
    cpa_summary = cpa_doc.get("summary") or {}

    cpd_doc = compute_capital_preservation_decision_support(cpa_doc)
    cpd_summary = cpd_doc.get("summary") or {}

    gov_doc = compute_governance_risk_summary(cpi_doc, cpe_doc, cpa_doc)

    from services.risk_intelligence import persist_advanced_risk_intelligence

    advanced = persist_advanced_risk_intelligence(
        results_dir=RESULTS_DIR,
        positions=positions,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        cpa_doc=cpa_doc,
        gov_doc=gov_doc,
        alerts_doc=alerts_doc,
        drawdown_threshold_pct=drawdown_pct,
    )
    pred_doc = advanced["predictive_risk_intelligence"]
    exec_summary = advanced["executive_risk_summary"]
    sim_doc = advanced["defensive_simulation"]

    from services.activation_safety import persist_activation_safety

    activation = persist_activation_safety(
        results_dir=RESULTS_DIR,
        positions=positions,
        active_alerts=active_alerts,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        sim_doc=sim_doc,
    )
    candidates_doc = activation["defensive_action_candidates"]
    queue_doc = activation["human_approval_queue"]
    policy_doc = activation["protective_action_policy"]

    from services.governance_execution_readiness import persist_governance_execution_readiness

    governance_exec = persist_governance_execution_readiness(
        results_dir=RESULTS_DIR,
        positions=positions,
        active_alerts=active_alerts,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        gov_doc=gov_doc,
        candidates_doc=candidates_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        sim_doc=sim_doc,
        watchdog_status=watchdog_status,
        watchdog_ts=ts,
        broker_connected=broker_connected,
        broker_error=broker_error,
    )
    auth_doc = governance_exec["governance_authorization"]
    readiness_doc = governance_exec["execution_readiness"]
    trials_doc = governance_exec["protective_action_trials"]

    from services.capital_preservation_evaluation import persist_capital_preservation_evaluation

    preservation_eval = persist_capital_preservation_evaluation(
        results_dir=RESULTS_DIR,
        trials_doc=trials_doc,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        gov_doc=gov_doc,
    )
    evaluation_doc = preservation_eval["protective_action_evaluation"]
    adaptive_doc = preservation_eval["adaptive_capital_preservation"]
    governor_doc = preservation_eval["capital_preservation_governor"]

    from services.institutional_autonomy import persist_institutional_autonomy

    institutional = persist_institutional_autonomy(
        results_dir=RESULTS_DIR,
        positions=positions,
        active_alerts=active_alerts,
        alerts_doc=alerts_doc,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        cpa_doc=cpa_doc,
        cpd_doc=cpd_doc,
        sim_doc=sim_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        evaluation_doc=evaluation_doc,
        adaptive_doc=adaptive_doc,
        governor_doc=governor_doc,
        trials_doc=trials_doc,
        broker_connected=broker_connected,
        watchdog_status=watchdog_status,
    )
    audit_doc = institutional["capital_preservation_audit"]
    stress_doc = institutional["stress_test_results"]
    cert_doc = institutional["capital_preservation_certification"]

    from services.institutional_protection import persist_institutional_protection

    protection = persist_institutional_protection(
        results_dir=RESULTS_DIR,
        cpi_doc=cpi_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        governor_doc=governor_doc,
        cpe_doc=cpe_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        queue_doc=queue_doc,
        candidates_doc=candidates_doc,
        evaluation_doc=evaluation_doc,
        audit_doc=audit_doc,
        policy_doc=policy_doc,
    )
    committee_doc = protection["risk_committee_oversight"]
    accountability_doc = protection["accountability_registry"]
    board_doc = protection["preservation_governance_board"]

    from services.institutional_operations import persist_institutional_operations

    operations = persist_institutional_operations(
        results_dir=RESULTS_DIR,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        alerts_doc=alerts_doc,
        exec_doc=exec_summary,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        cert_doc=cert_doc,
        readiness_doc=readiness_doc,
        governor_doc=governor_doc,
        adaptive_doc=adaptive_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        accountability_doc=accountability_doc,
    )
    investment_doc = operations["investment_committee_review"]
    maturity_doc = operations["triton_maturity_assessment"]
    strategic_doc = operations["strategic_oversight"]

    from services.institutional_intelligence import persist_institutional_intelligence

    intelligence = persist_institutional_intelligence(
        results_dir=RESULTS_DIR,
        cpa_doc=cpa_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        alerts_doc=alerts_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        governor_doc=governor_doc,
        evaluation_doc=evaluation_doc,
        adaptive_doc=adaptive_doc,
        cert_doc=cert_doc,
        exec_doc=exec_summary,
        committee_doc=committee_doc,
        board_doc=board_doc,
        accountability_doc=accountability_doc,
        audit_doc=audit_doc,
        maturity_doc=maturity_doc,
        strategic_doc=strategic_doc,
    )
    decision_doc = intelligence["decision_quality_assessment"]
    intelligence_doc = intelligence["institutional_intelligence"]
    improvement_doc = intelligence["strategic_self_improvement"]

    from services.institutional_memory import persist_institutional_memory

    memory = persist_institutional_memory(
        results_dir=RESULTS_DIR,
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpa_doc=cpa_doc,
        governor_doc=governor_doc,
        cert_doc=cert_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        strategic_doc=strategic_doc,
        audit_doc=audit_doc,
        readiness_doc=readiness_doc,
        decision_doc=decision_doc,
        intelligence_doc=intelligence_doc,
        improvement_doc=improvement_doc,
    )
    memory_doc = memory["institutional_memory"]
    graph_doc = memory["institutional_knowledge_graph"]
    learning_doc = memory["organizational_learning"]

    from services.institutional_reasoning import persist_institutional_reasoning

    reasoning = persist_institutional_reasoning(
        results_dir=RESULTS_DIR,
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        auth_doc=auth_doc,
        strategic_doc=strategic_doc,
        board_doc=board_doc,
        exec_doc=exec_summary,
        intelligence_doc=intelligence_doc,
        memory_doc=memory_doc,
        learning_doc=learning_doc,
        improvement_doc=improvement_doc,
    )
    causal_doc = reasoning["causal_reasoning"]
    explanations_doc = reasoning["institutional_explanations"]
    insights_doc = reasoning["institutional_insights"]

    from services.strategic_intelligence import persist_strategic_intelligence

    strategic_intel = persist_strategic_intelligence(
        results_dir=RESULTS_DIR,
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        auth_doc=auth_doc,
        strategic_doc=strategic_doc,
        exec_doc=exec_summary,
        pred_doc=pred_doc,
        learning_doc=learning_doc,
        memory_doc=memory_doc,
        graph_doc=graph_doc,
        improvement_doc=improvement_doc,
        intelligence_doc=intelligence_doc,
        maturity_doc=maturity_doc,
        investment_doc=investment_doc,
        decision_doc=decision_doc,
        causal_doc=causal_doc,
        insights_doc=insights_doc,
    )
    strategic_reasoning_doc = strategic_intel["strategic_reasoning"]
    consequence_doc = strategic_intel["consequence_forecasts"]
    wisdom_doc = strategic_intel["institutional_wisdom"]

    from services.institutional_planning import persist_institutional_planning

    planning = persist_institutional_planning(
        results_dir=RESULTS_DIR,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        pred_doc=pred_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
        consequence_doc=consequence_doc,
        insights_doc=insights_doc,
        maturity_doc=maturity_doc,
        improvement_doc=improvement_doc,
        wisdom_doc=wisdom_doc,
        learning_doc=learning_doc,
    )
    scenario_doc = planning["scenario_planning"]
    future_paths_doc = planning["future_path_analysis"]
    priorities_doc = planning["strategic_priorities"]

    from services.institutional_optimization import persist_institutional_optimization

    optimization = persist_institutional_optimization(
        results_dir=RESULTS_DIR,
        priorities_doc=priorities_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
        consequence_doc=consequence_doc,
        cpe_doc=cpe_doc,
        strategic_doc=strategic_doc,
        insights_doc=insights_doc,
        graph_doc=graph_doc,
        future_paths_doc=future_paths_doc,
        intelligence_doc=intelligence_doc,
        maturity_doc=maturity_doc,
        improvement_doc=improvement_doc,
    )
    attention_doc = optimization["attention_allocation"]
    coordination_doc = optimization["system_coordination"]
    optimization_doc = optimization["institutional_optimization"]

    summary: Dict[str, Any] = {
        "timestamp": ts,
        "watchdog_status": watchdog_status,
        "broker_connected": broker_connected,
        "broker_status": broker_status,
        "broker_error": broker_error,
        "positions_count": len(positions),
        "open_orders_count": open_orders_count,
        "pending_orders_count": pending_orders_count,
        "largest_unrealized_loss": largest_loss,
        "largest_unrealized_gain": largest_gain,
        "heartbeat": heartbeat,
        "positions": positions,
        "checks": {
            "positions_error": positions_error,
            "orders_error": orders_error,
            "account_error": account_error,
        },
        "capital_preservation": {
            "score": cpi_doc["capital_preservation_score"],
            "health_band": cpi_doc["health_band"],
            "risk_trend": cpi_doc["risk_trend"],
            "escalation_state": cpe_doc["escalation_state"],
        },
        "capital_preservation_escalation": {
            "escalation_state": cpe_doc["escalation_state"],
            "escalation_reason": cpe_doc["escalation_reason"],
            "recommended_review_actions": cpe_doc["recommended_review_actions"],
        },
        "capital_preservation_advisory": {
            "highest_priority": cpa_summary.get("highest_priority"),
            "advisory_count": cpa_summary.get("advisory_count", 0),
            "top_issue": cpa_summary.get("top_issue"),
        },
        "capital_preservation_decision_support": {
            "issue_count": cpd_summary.get("issue_count", 0),
            "highest_priority": cpd_summary.get("highest_priority"),
            "top_issue": cpd_summary.get("top_issue"),
        },
        "governance_risk_summary": {
            "governance_awareness_level": gov_doc.get("governance_awareness_level"),
            "governance_awareness_label": gov_doc.get("governance_awareness_label"),
            "governance_status": gov_doc.get("governance_status"),
            "governance_summary": gov_doc.get("governance_summary"),
            "escalation_state": gov_doc.get("escalation_state"),
        },
        "predictive_risk_intelligence": {
            "risk_direction": pred_doc.get("risk_direction"),
            "risk_momentum": pred_doc.get("risk_momentum"),
            "forecast_confidence": pred_doc.get("forecast_confidence"),
            "projected_escalation": pred_doc.get("escalation_forecast", {}).get(
                "projected_escalation"
            ),
            "projected_cps": pred_doc.get("portfolio_health_forecast", {}).get("projected_cps"),
            "alert_velocity": pred_doc.get("alert_velocity"),
            "alert_acceleration": pred_doc.get("alert_acceleration"),
            "estimated_days_to_threshold": pred_doc.get("estimated_days_to_threshold"),
        },
        "executive_risk_summary": {
            "portfolio_health": exec_summary.get("executive_summary", {}).get("portfolio_health"),
            "escalation_state": exec_summary.get("escalation_state"),
            "top_risks": exec_summary.get("top_risks"),
            "governance_awareness": exec_summary.get("executive_summary", {}).get(
                "governance_awareness"
            ),
            "risk_direction": exec_summary.get("predictive_outlook", {}).get("risk_direction"),
        },
        "defensive_sandbox": {
            "candidate_count": candidates_doc.get("candidate_count", 0),
            "top_candidate": (
                (candidates_doc.get("candidates") or [{}])[0].get("candidate_action")
                if candidates_doc.get("candidates")
                else None
            ),
            "simulation_only": True,
            "execution_permitted": False,
        },
        "human_approval_center": {
            "pending_review": (queue_doc.get("counts") or {}).get("PENDING_REVIEW", 0),
            "approved": (queue_doc.get("counts") or {}).get("APPROVED", 0),
            "rejected": (queue_doc.get("counts") or {}).get("REJECTED", 0),
            "expired": (queue_doc.get("counts") or {}).get("EXPIRED", 0),
            "execution_permitted": False,
        },
        "protective_action_policy": {
            "global_execution_enabled": policy_doc.get("global_execution_enabled", False),
            "automated_trading_permitted": policy_doc.get("automated_trading_permitted", False),
            "policy_count": len(policy_doc.get("policies") or []),
            "enabled_policy_count": sum(
                1
                for p in (policy_doc.get("policies") or [])
                if isinstance(p, dict) and p.get("enabled")
            ),
        },
        "governance_authorization": {
            "overall_authorization": auth_doc.get("overall_authorization", False),
            "governance_authorized": auth_doc.get("governance_authorized", False),
            "operator_authorized": auth_doc.get("operator_authorized", False),
            "policy_authorized": auth_doc.get("policy_authorized", False),
            "execution_authorized": auth_doc.get("execution_authorized", False),
        },
        "execution_readiness": {
            "readiness_status": readiness_doc.get("readiness_status", "NOT_READY"),
            "checks_passing_count": readiness_doc.get("checks_passing_count", 0),
            "checks_total": readiness_doc.get("checks_total", 0),
            "failed_checks": readiness_doc.get("failed_checks") or [],
            "live_execution_permitted": readiness_doc.get("live_execution_permitted", False),
        },
        "protective_action_trials": {
            "trial_count": trials_doc.get("trial_count", 0),
            "baseline_cps": trials_doc.get("baseline_cps"),
            "simulation_only": True,
            "execution_permitted": False,
        },
        "protective_action_evaluation": {
            "evaluation_count": evaluation_doc.get("evaluation_count", 0),
            "average_effectiveness": evaluation_doc.get("average_effectiveness"),
            "beneficial_count": evaluation_doc.get("beneficial_count", 0),
            "negative_count": evaluation_doc.get("negative_count", 0),
            "top_trial": (evaluation_doc.get("top_trial") or {}).get("trial_name"),
            "top_score": (evaluation_doc.get("top_trial") or {}).get("effectiveness_score"),
        },
        "adaptive_capital_preservation": {
            "best_protection": adaptive_doc.get("best_protection"),
            "average_effectiveness": adaptive_doc.get("average_effectiveness"),
            "confidence": adaptive_doc.get("confidence"),
            "best_trial_name": adaptive_doc.get("best_trial_name"),
        },
        "capital_preservation_governor": {
            "preservation_posture": governor_doc.get("preservation_posture"),
            "governor_confidence": governor_doc.get("governor_confidence"),
            "top_drivers": governor_doc.get("top_drivers") or [],
            "live_execution_permitted": governor_doc.get("live_execution_permitted", False),
        },
        "preservation_audit": {
            "event_count": audit_doc.get("event_count", 0),
            "latest_event_type": audit_doc.get("latest_event_type"),
            "latest_event_result": audit_doc.get("latest_event_result"),
            "latest_event_source": audit_doc.get("latest_event_source"),
            "summary_by_event_type": audit_doc.get("summary_by_event_type") or {},
        },
        "stress_test_results": {
            "scenario_count": stress_doc.get("scenario_count", 0),
            "pass_count": stress_doc.get("pass_count", 0),
            "warn_count": stress_doc.get("warn_count", 0),
            "fail_count": stress_doc.get("fail_count", 0),
            "average_survivability": stress_doc.get("average_survivability"),
        },
        "preservation_certification": {
            "certification_status": cert_doc.get("certification_status"),
            "certification_score": cert_doc.get("certification_score"),
            "certified_area_count": cert_doc.get("certified_area_count"),
            "failed_requirements": cert_doc.get("failed_requirements") or [],
        },
        "risk_committee_oversight": {
            "committee_status": committee_doc.get("committee_status"),
            "overall_assessment": committee_doc.get("overall_assessment"),
            "average_domain_score": committee_doc.get("average_domain_score"),
            "top_concerns": committee_doc.get("top_concerns") or [],
        },
        "accountability_registry": {
            "entry_count": accountability_doc.get("entry_count", 0),
            "not_certified_count": accountability_doc.get("not_certified_count", 0),
            "summary_by_origin": accountability_doc.get("summary_by_origin") or {},
        },
        "preservation_governance_board": {
            "board_status": board_doc.get("board_status"),
            "governance_confidence": board_doc.get("governance_confidence"),
            "preservation_authority": board_doc.get("preservation_authority"),
            "automation_authorized": board_doc.get("automation_authorized", False),
            "board_recommendations": board_doc.get("board_recommendations") or [],
        },
        "investment_committee_review": {
            "committee_recommendation": investment_doc.get("committee_recommendation"),
            "confidence": investment_doc.get("confidence"),
            "top_concerns": investment_doc.get("top_concerns") or [],
            "average_area_score": investment_doc.get("average_area_score"),
        },
        "triton_maturity_assessment": {
            "overall_maturity": maturity_doc.get("overall_maturity"),
            "maturity_band": maturity_doc.get("maturity_band"),
            "strongest_area": maturity_doc.get("strongest_area"),
            "weakest_area": maturity_doc.get("weakest_area"),
        },
        "strategic_oversight": {
            "oversight_status": strategic_doc.get("oversight_status"),
            "strategic_confidence": strategic_doc.get("strategic_confidence"),
            "institutional_readiness": strategic_doc.get("institutional_readiness"),
            "automation_status": strategic_doc.get("automation_status"),
            "top_strategic_concerns": strategic_doc.get("top_strategic_concerns") or [],
        },
        "decision_quality": {
            "decision_quality_score": decision_doc.get("decision_quality_score"),
            "quality_band": decision_doc.get("quality_band"),
            "strongest_area": decision_doc.get("strongest_area"),
            "weakest_area": decision_doc.get("weakest_area"),
        },
        "institutional_intelligence": {
            "institutional_intelligence_score": intelligence_doc.get(
                "institutional_intelligence_score"
            ),
            "institutional_band": intelligence_doc.get("institutional_band"),
            "coordination_score": intelligence_doc.get("coordination_score"),
            "strongest_area": intelligence_doc.get("strongest_area"),
            "weakest_area": intelligence_doc.get("weakest_area"),
        },
        "strategic_self_improvement": {
            "top_priority": improvement_doc.get("top_priority"),
            "improvement_score": improvement_doc.get("improvement_score"),
            "recommended_focus": improvement_doc.get("recommended_focus") or [],
            "weakest_systems": improvement_doc.get("weakest_systems") or [],
        },
        "institutional_memory": {
            "memory_entries": memory_doc.get("memory_entries"),
            "last_major_event": memory_doc.get("last_major_event"),
            "retention_status": memory_doc.get("retention_status"),
        },
        "institutional_knowledge_graph": {
            "nodes": graph_doc.get("nodes"),
            "relationships": graph_doc.get("relationships"),
            "most_connected_area": graph_doc.get("most_connected_area"),
        },
        "organizational_learning": {
            "top_lesson": learning_doc.get("top_lesson"),
            "confidence": learning_doc.get("confidence"),
            "learning_status": learning_doc.get("learning_status"),
            "top_priority": learning_doc.get("top_priority"),
        },
        "causal_reasoning": {
            "reasoning_count": causal_doc.get("reasoning_count"),
            "issues": [a.get("issue") for a in (causal_doc.get("analyses") or [])],
        },
        "institutional_explanations": {
            "explanation_count": explanations_doc.get("explanation_count"),
            "topics": [e.get("topic") for e in (explanations_doc.get("explanations") or [])],
        },
        "institutional_insights": {
            "top_insight": insights_doc.get("top_insight"),
            "insight_confidence": insights_doc.get("insight_confidence"),
            "most_important_risk": insights_doc.get("most_important_risk"),
            "most_important_weakness": insights_doc.get("most_important_weakness"),
        },
        "strategic_reasoning": {
            "top_strategic_issue": strategic_reasoning_doc.get("top_strategic_issue"),
            "strategic_importance": strategic_reasoning_doc.get("strategic_importance"),
            "impact_scope": strategic_reasoning_doc.get("impact_scope"),
            "issue_count": strategic_reasoning_doc.get("issue_count"),
        },
        "consequence_forecasts": {
            "forecast_count": consequence_doc.get("forecast_count"),
            "forecast_horizon_days": consequence_doc.get("forecast_horizon_days"),
            "high_severity_count": sum(
                1 for f in (consequence_doc.get("forecasts") or []) if f.get("severity") == "HIGH"
            ),
            "top_forecast_issue": (
                (consequence_doc.get("forecasts") or [{}])[0].get("issue")
                if consequence_doc.get("forecasts")
                else None
            ),
        },
        "institutional_wisdom": {
            "confidence": wisdom_doc.get("confidence"),
            "supporting_systems": wisdom_doc.get("supporting_systems"),
            "guidance_count": len(wisdom_doc.get("guidance_items") or []),
            "wisdom_preview": (wisdom_doc.get("wisdom_statement") or "")[:120],
        },
        "scenario_planning": {
            "scenario_count": scenario_doc.get("scenario_count"),
            "probability_sum": scenario_doc.get("probability_sum"),
            "top_scenario": max(
                (scenario_doc.get("scenarios") or []),
                key=lambda s: s.get("probability") or 0,
                default={},
            ).get("scenario"),
        },
        "future_path_analysis": {
            "path_count": future_paths_doc.get("path_count"),
            "recommended_path": future_paths_doc.get("recommended_path"),
            "top_benefit": (
                (future_paths_doc.get("paths") or [{}])[0].get("expected_benefit")
                if future_paths_doc.get("paths")
                else None
            ),
        },
        "strategic_priorities": {
            "top_priority": priorities_doc.get("top_priority"),
            "highest_priority_issue": priorities_doc.get("highest_priority_issue"),
            "priority_count": len(priorities_doc.get("priorities") or []),
            "recommended_path": priorities_doc.get("recommended_path"),
        },
        "attention_allocation": {
            "highest_attention_area": attention_doc.get("highest_attention_area"),
            "attention_score": attention_doc.get("attention_score"),
            "recommended_focus_percent": attention_doc.get("recommended_focus_percent"),
            "allocation_count": len(attention_doc.get("allocations") or []),
        },
        "system_coordination": {
            "coordination_score": coordination_doc.get("coordination_score"),
            "strongest_connection": coordination_doc.get("strongest_connection"),
            "weakest_connection": coordination_doc.get("weakest_connection"),
        },
        "institutional_optimization": {
            "top_optimization": optimization_doc.get("top_optimization"),
            "optimization_score": optimization_doc.get("optimization_score"),
            "expected_system_benefit": optimization_doc.get("expected_system_benefit"),
            "highest_roi_improvement": optimization_doc.get("highest_roi_improvement"),
        },
        "alerts": {
            "active_count": len(alerts_doc.get("active_alerts") or []),
            "resolved_count": len(alerts_doc.get("resolved_alerts") or []),
            "newly_raised": engine.newly_raised,
            "newly_resolved": engine.newly_resolved,
            "incident_intelligence": alerts_doc.get("incident_intelligence"),
        },
    }

    _atomic_write_json(summary, STATUS_PATH)

    broker_log = "OK" if broker_connected else "FAIL"
    log_lines = [
        "[WATCHDOG]",
        f"timestamp={ts}",
        f"broker={broker_log}",
        f"positions={len(positions)}",
        f"open_orders={open_orders_count}",
        f"pending_orders={pending_orders_count}",
        f"largest_loss={largest_loss if largest_loss is not None else 'n/a'}",
        f"largest_gain={largest_gain if largest_gain is not None else 'n/a'}",
        f"status={watchdog_status}",
        f"active_alerts={summary['alerts']['active_count']}",
        f"cps={cpi_doc['capital_preservation_score']}",
        f"health_band={cpi_doc['health_band']}",
        f"risk_trend={cpi_doc['risk_trend']}",
        f"escalation_state={cpe_doc['escalation_state']}",
        f"advisory_count={cpa_summary.get('advisory_count', 0)}",
        f"top_issue={cpa_summary.get('top_issue')}",
        f"decision_support_issues={cpd_summary.get('issue_count', 0)}",
        f"governance_level={gov_doc.get('governance_awareness_level')}",
        f"risk_direction={pred_doc.get('risk_direction')}",
        f"defensive_candidates={candidates_doc.get('candidate_count', 0)}",
        f"approval_pending={(queue_doc.get('counts') or {}).get('PENDING_REVIEW', 0)}",
        f"policies_enabled={sum(1 for p in (policy_doc.get('policies') or []) if isinstance(p, dict) and p.get('enabled'))}",
        f"overall_authorization={auth_doc.get('overall_authorization', False)}",
        f"readiness_status={readiness_doc.get('readiness_status', 'NOT_READY')}",
        f"protective_trials={trials_doc.get('trial_count', 0)}",
        f"eval_avg_effectiveness={evaluation_doc.get('average_effectiveness')}",
        f"adaptive_best={adaptive_doc.get('best_protection')}",
        f"preservation_posture={governor_doc.get('preservation_posture')}",
        f"audit_events={audit_doc.get('event_count', 0)}",
        f"stress_pass={stress_doc.get('pass_count', 0)}",
        f"stress_fail={stress_doc.get('fail_count', 0)}",
        f"certification_status={cert_doc.get('certification_status')}",
        f"certification_score={cert_doc.get('certification_score')}",
        f"committee_status={committee_doc.get('committee_status')}",
        f"overall_assessment={committee_doc.get('overall_assessment')}",
        f"accountability_entries={accountability_doc.get('entry_count', 0)}",
        f"board_status={board_doc.get('board_status')}",
        f"governance_confidence={board_doc.get('governance_confidence')}",
        f"preservation_authority={board_doc.get('preservation_authority')}",
        f"automation_authorized={board_doc.get('automation_authorized', False)}",
        f"committee_recommendation={investment_doc.get('committee_recommendation')}",
        f"investment_confidence={investment_doc.get('confidence')}",
        f"overall_maturity={maturity_doc.get('overall_maturity')}",
        f"maturity_band={maturity_doc.get('maturity_band')}",
        f"oversight_status={strategic_doc.get('oversight_status')}",
        f"strategic_confidence={strategic_doc.get('strategic_confidence')}",
        f"automation_status={strategic_doc.get('automation_status')}",
        f"decision_quality_score={decision_doc.get('decision_quality_score')}",
        f"quality_band={decision_doc.get('quality_band')}",
        f"institutional_intelligence_score={intelligence_doc.get('institutional_intelligence_score')}",
        f"institutional_band={intelligence_doc.get('institutional_band')}",
        f"coordination_score={intelligence_doc.get('coordination_score')}",
        f"improvement_top_priority={improvement_doc.get('top_priority')}",
        f"improvement_score={improvement_doc.get('improvement_score')}",
        f"memory_entries={memory_doc.get('memory_entries')}",
        f"last_major_event={memory_doc.get('last_major_event')}",
        f"graph_nodes={graph_doc.get('nodes')}",
        f"graph_relationships={graph_doc.get('relationships')}",
        f"most_connected_area={graph_doc.get('most_connected_area')}",
        f"top_lesson={learning_doc.get('top_lesson')}",
        f"learning_confidence={learning_doc.get('confidence')}",
        f"learning_status={learning_doc.get('learning_status')}",
        f"causal_reasoning_count={causal_doc.get('reasoning_count')}",
        f"explanation_count={explanations_doc.get('explanation_count')}",
        f"top_insight={insights_doc.get('top_insight')}",
        f"insight_confidence={insights_doc.get('insight_confidence')}",
        f"top_strategic_issue={strategic_reasoning_doc.get('top_strategic_issue')}",
        f"strategic_importance={strategic_reasoning_doc.get('strategic_importance')}",
        f"impact_scope={strategic_reasoning_doc.get('impact_scope')}",
        f"forecast_count={consequence_doc.get('forecast_count')}",
        f"forecast_horizon_days={consequence_doc.get('forecast_horizon_days')}",
        f"wisdom_confidence={wisdom_doc.get('confidence')}",
        f"supporting_systems={wisdom_doc.get('supporting_systems')}",
        f"scenario_count={scenario_doc.get('scenario_count')}",
        f"scenario_probability_sum={scenario_doc.get('probability_sum')}",
        f"recommended_path={future_paths_doc.get('recommended_path')}",
        f"top_priority={priorities_doc.get('top_priority')}",
        f"strategic_priority_count={len(priorities_doc.get('priorities') or [])}",
        f"highest_attention_area={attention_doc.get('highest_attention_area')}",
        f"attention_score={attention_doc.get('attention_score')}",
        f"system_coordination_score={coordination_doc.get('coordination_score')}",
        f"strongest_connection={coordination_doc.get('strongest_connection')}",
        f"top_optimization={optimization_doc.get('top_optimization')}",
        f"optimization_score={optimization_doc.get('optimization_score')}",
        f"expected_system_benefit={optimization_doc.get('expected_system_benefit')}",
    ]
    if broker_error:
        log_lines.append(f"broker_error={broker_error}")
    log_path = _write_watchdog_log(log_lines)
    summary["log_path"] = str(log_path)
    summary["alerts_path"] = str(ALERTS_PATH)
    summary["capital_preservation_path"] = str(CPI_PATH)
    summary["capital_preservation_escalation_path"] = str(CPE_PATH)
    summary["capital_preservation_advisory_path"] = str(CPA_PATH)
    summary["capital_preservation_decision_support_path"] = str(CPD_PATH)
    summary["governance_risk_summary_path"] = str(GOV_RISK_PATH)
    summary["defensive_simulation_path"] = str(RESULTS_DIR / "defensive_simulation_results.json")
    summary["predictive_risk_intelligence_path"] = str(
        RESULTS_DIR / "predictive_risk_intelligence.json"
    )
    summary["executive_risk_summary_path"] = str(RESULTS_DIR / "executive_risk_summary.json")
    summary["defensive_action_candidates_path"] = str(
        RESULTS_DIR / "defensive_action_candidates.json"
    )
    summary["human_approval_queue_path"] = str(RESULTS_DIR / "human_approval_queue.json")
    summary["protective_action_policy_path"] = str(RESULTS_DIR / "protective_action_policy.json")
    summary["governance_authorization_path"] = str(RESULTS_DIR / "governance_authorization.json")
    summary["execution_readiness_path"] = str(RESULTS_DIR / "execution_readiness.json")
    summary["protective_action_trials_path"] = str(RESULTS_DIR / "protective_action_trials.json")
    summary["protective_action_evaluation_path"] = str(
        RESULTS_DIR / "protective_action_evaluation.json"
    )
    summary["adaptive_capital_preservation_path"] = str(
        RESULTS_DIR / "adaptive_capital_preservation.json"
    )
    summary["capital_preservation_governor_path"] = str(
        RESULTS_DIR / "capital_preservation_governor.json"
    )
    summary["capital_preservation_audit_path"] = str(
        RESULTS_DIR / "capital_preservation_audit.json"
    )
    summary["stress_test_results_path"] = str(RESULTS_DIR / "stress_test_results.json")
    summary["capital_preservation_certification_path"] = str(
        RESULTS_DIR / "capital_preservation_certification.json"
    )
    summary["risk_committee_oversight_path"] = str(RESULTS_DIR / "risk_committee_oversight.json")
    summary["accountability_registry_path"] = str(RESULTS_DIR / "accountability_registry.json")
    summary["preservation_governance_board_path"] = str(
        RESULTS_DIR / "preservation_governance_board.json"
    )
    summary["investment_committee_review_path"] = str(
        RESULTS_DIR / "investment_committee_review.json"
    )
    summary["triton_maturity_assessment_path"] = str(
        RESULTS_DIR / "triton_maturity_assessment.json"
    )
    summary["strategic_oversight_path"] = str(RESULTS_DIR / "strategic_oversight.json")
    summary["decision_quality_assessment_path"] = str(
        RESULTS_DIR / "decision_quality_assessment.json"
    )
    summary["institutional_intelligence_path"] = str(
        RESULTS_DIR / "institutional_intelligence.json"
    )
    summary["strategic_self_improvement_path"] = str(
        RESULTS_DIR / "strategic_self_improvement.json"
    )
    summary["institutional_memory_path"] = str(RESULTS_DIR / "institutional_memory.json")
    summary["institutional_knowledge_graph_path"] = str(
        RESULTS_DIR / "institutional_knowledge_graph.json"
    )
    summary["organizational_learning_path"] = str(RESULTS_DIR / "organizational_learning.json")
    summary["causal_reasoning_path"] = str(RESULTS_DIR / "causal_reasoning.json")
    summary["institutional_explanations_path"] = str(
        RESULTS_DIR / "institutional_explanations.json"
    )
    summary["institutional_insights_path"] = str(RESULTS_DIR / "institutional_insights.json")
    summary["strategic_reasoning_path"] = str(RESULTS_DIR / "strategic_reasoning.json")
    summary["consequence_forecasts_path"] = str(RESULTS_DIR / "consequence_forecasts.json")
    summary["institutional_wisdom_path"] = str(RESULTS_DIR / "institutional_wisdom.json")
    summary["scenario_planning_path"] = str(RESULTS_DIR / "scenario_planning.json")
    summary["future_path_analysis_path"] = str(RESULTS_DIR / "future_path_analysis.json")
    summary["strategic_priorities_path"] = str(RESULTS_DIR / "strategic_priorities.json")
    summary["attention_allocation_path"] = str(RESULTS_DIR / "attention_allocation.json")
    summary["system_coordination_path"] = str(RESULTS_DIR / "system_coordination.json")
    summary["institutional_optimization_path"] = str(
        RESULTS_DIR / "institutional_optimization.json"
    )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRITON Risk Watchdog — read-only monitoring, CPI, CPEF, CPAE, DSE, governance"
    )
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument(
        "--drawdown-pct",
        type=float,
        default=DEFAULT_DRAWDOWN_PCT,
        help="Alert when position unrealized P/L %% is at or below this (default -10)",
    )
    parser.add_argument(
        "--concentration-pct",
        type=float,
        default=DEFAULT_CONCENTRATION_PCT,
        help="Alert when single position exceeds this %% of portfolio MV (default 25)",
    )
    parser.add_argument(
        "--order-age-minutes",
        type=float,
        default=DEFAULT_ORDER_AGE_MINUTES,
        help="Alert when open order age exceeds this many minutes (default 60)",
    )
    parser.add_argument(
        "--expected-interval-minutes",
        type=float,
        default=1.0,
        help="Expected watchdog cycle interval for stale-heartbeat detection",
    )
    args = parser.parse_args()

    summary = run_watchdog(
        mode=args.mode,
        drawdown_pct=args.drawdown_pct,
        concentration_pct=args.concentration_pct,
        order_age_minutes=args.order_age_minutes,
        expected_interval_minutes=args.expected_interval_minutes,
    )
    print(json.dumps(summary, indent=2))
    # Alerts do not stop the watchdog process; always exit 0 if the run completed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
