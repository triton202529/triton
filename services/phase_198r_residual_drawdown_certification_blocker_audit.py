"""
phase_198r_residual_drawdown_certification_blocker_audit.py
Phase 198R — Residual drawdown certification blocker audit.

Read-only audit. Does not re-score evaluation, modify certification, or place orders.
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

OUTPUT_JSON = RESULTS / "phase_198r_residual_drawdown_certification_blocker_audit.json"
OUTPUT_MD = RESULTS / "phase_198r_residual_drawdown_certification_blocker_audit.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

ARTIFACT_PATHS = {
    "positions_snapshot": RESULTS / "positions_snapshot.csv",
    "recent_orders": RESULTS / "recent_orders.csv",
    "open_orders_snapshot": RESULTS / "open_orders_snapshot.csv",
    "cpi": RESULTS / "capital_preservation_intelligence.json",
    "cpe": RESULTS / "capital_preservation_escalation.json",
    "advisory": RESULTS / "capital_preservation_advisory.json",
    "decision_support": RESULTS / "capital_preservation_decision_support.json",
    "evaluation": RESULTS / "protective_action_evaluation.json",
    "trials": RESULTS / "protective_action_trials.json",
    "certification": RESULTS / "capital_preservation_certification.json",
    "governor": RESULTS / "capital_preservation_governor.json",
    "executive_risk_summary": RESULTS / "executive_risk_summary.json",
    "watchdog_alerts": RESULTS / "watchdog_alerts.json",
    "phase_198q": RESULTS / "phase_198q_refresh_protective_evaluation_after_gld_reduction.json",
}

MARKER = "RESIDUAL_DRAWDOWN_CERTIFICATION_BLOCKER_AUDIT_ONLY"
GLD_SYMBOL = "GLD"
GLD_TARGET_WEIGHT = 0.20
DRAWDOWN_THRESHOLD_PCT = -10.0


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


def _count_csv_data_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, len(list(csv.reader(f))) - 1)
    except Exception:
        return 0


def _parse_portfolio(positions_path: Path) -> Dict[str, Any]:
    if not positions_path.is_file():
        return {"positions": [], "total_invested": 0.0, "total_unrealized_pl": 0.0}

    positions: List[Dict[str, Any]] = []
    total_invested = 0.0
    total_unrealized = 0.0

    with positions_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            mv = abs(_safe_float(row.get("market_value") or row.get("value")))
            upl = _safe_float(row.get("unrealized_pl"), 0.0)
            uplpc = _safe_float(row.get("unrealized_plpc"), 0.0)
            qty = _safe_float(row.get("qty") or row.get("qty_available"))
            total_invested += mv
            total_unrealized += upl
            positions.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "market_value": mv,
                    "unrealized_pl": upl,
                    "unrealized_plpc": uplpc,
                    "cost_basis": _safe_float(row.get("cost_basis"), 0.0),
                }
            )

    losers = sorted(positions, key=lambda p: p.get("unrealized_pl") or 0.0)
    largest_loss = losers[0] if losers else {}

    gld = next((p for p in positions if p.get("symbol") == GLD_SYMBOL), {})
    gld_weight = (gld.get("market_value") or 0) / total_invested if total_invested > 0 else 0.0

    return {
        "positions": positions,
        "total_invested": round(total_invested, 2),
        "total_unrealized_pl": round(total_unrealized, 2),
        "gld": gld,
        "gld_weight": round(gld_weight, 6),
        "largest_unrealized_loss_symbol": largest_loss.get("symbol"),
        "largest_unrealized_loss_value": round(_safe_float(largest_loss.get("unrealized_pl")), 2),
    }


def _find_drawdown_alert(watchdog: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    for alert in watchdog.get("active_alerts") or []:
        if str(alert.get("alert_type") or "") != "EXCESS_POSITION_DRAWDOWN":
            continue
        if str(alert.get("subject") or "").upper() == symbol.upper():
            return alert
    return None


def _alert_supported_by_broker(
    alert: Optional[Dict[str, Any]],
    gld: Dict[str, Any],
) -> Tuple[bool, str]:
    if not alert or not gld:
        return False, "alert_or_gld_position_missing"
    alert_pct = _safe_float((alert.get("details") or {}).get("unrealized_pl_pct"), 0.0)
    broker_pct = _safe_float(gld.get("unrealized_plpc"), 0.0) * 100.0
    if broker_pct == 0 and alert_pct == 0:
        return False, "broker_unrealized_pct_zero"
    delta = abs(alert_pct - broker_pct)
    if broker_pct <= DRAWDOWN_THRESHOLD_PCT and delta <= 1.5:
        return True, f"alert_pct={alert_pct:.3f} broker_pct={broker_pct:.3f} delta={delta:.3f}"
    if broker_pct > DRAWDOWN_THRESHOLD_PCT:
        return False, f"broker_pct={broker_pct:.3f} above_threshold={DRAWDOWN_THRESHOLD_PCT}"
    return False, f"alert_broker_mismatch alert_pct={alert_pct:.3f} broker_pct={broker_pct:.3f}"


def _stale_alert_on_pre_reduction_notional(
    alert: Optional[Dict[str, Any]],
    gld: Dict[str, Any],
    *,
    concentration_completed: bool,
) -> Tuple[bool, str]:
    """Detect alerts that still imply pre-reduction GLD notional rather than residual qty."""
    if not alert or not concentration_completed:
        return False, "not_applicable"
    qty = _safe_float(gld.get("qty"))
    mv = _safe_float(gld.get("market_value"))
    if qty <= 20 and mv < 8000:
        alert_pct = _safe_float((alert.get("details") or {}).get("unrealized_pl_pct"), 0.0)
        broker_pct = _safe_float(gld.get("unrealized_plpc"), 0.0) * 100.0
        if abs(alert_pct - broker_pct) <= 1.5:
            return False, "alert_pct_matches_residual_position_not_pre_reduction_notional"
    return True, "possible_pre_reduction_notional_reference"


def _exposure_trial_stale(
    *,
    evaluation: Dict[str, Any],
    trials: Dict[str, Any],
    cpi: Dict[str, Any],
    gld_weight: float,
) -> Tuple[bool, str]:
    conc_component = int((cpi.get("component_scores") or {}).get("concentration") or 0)
    conc_completed = bool(evaluation.get("concentration_reduction_completed"))
    exposure_eval = next(
        (
            e
            for e in (evaluation.get("evaluations") or [])
            if e.get("trial_id") == "paper_exposure_reduction"
        ),
        {},
    )
    exposure_trial = next(
        (
            t
            for t in (trials.get("trials") or [])
            if t.get("trial_id") == "paper_exposure_reduction"
        ),
        {},
    )
    est_conc = _safe_float(exposure_trial.get("estimated_concentration_reduction"), 0.0)
    if (
        conc_completed
        and conc_component >= 95
        and est_conc >= 4.0
        and gld_weight <= GLD_TARGET_WEIGHT + 0.005
    ):
        return True, (
            f"exposure trial still assumes {est_conc}% concentration reduction while "
            f"CPI concentration={conc_component} and GLD weight={gld_weight * 100:.2f}%"
        )
    if (
        str(exposure_eval.get("evaluation") or "").upper() == "NEGATIVE"
        and int((cpi.get("component_scores") or {}).get("exposure") or 0) >= 85
    ):
        return True, "exposure trial NEGATIVE despite strong CPI exposure component"
    return False, "exposure trial inputs appear current"


def _severity_calibration_issue(
    *,
    gld: Dict[str, Any],
    gld_weight: float,
    total_invested: float,
    alert_supported: bool,
) -> Tuple[bool, str]:
    upl = _safe_float(gld.get("unrealized_pl"))
    position_pct = abs(_safe_float(gld.get("unrealized_plpc"))) * 100.0
    portfolio_impact_pct = (abs(upl) / total_invested * 100.0) if total_invested > 0 else 0.0
    if (
        alert_supported
        and position_pct >= 10.0
        and portfolio_impact_pct <= 3.5
        and gld_weight <= 0.22
    ):
        return True, (
            f"position-level loss {position_pct:.2f}% vs portfolio-weighted impact "
            f"{portfolio_impact_pct:.2f}% on {gld_weight * 100:.2f}% weight — trials/alerts not weight-aware"
        )
    return False, "severity appears proportionate to portfolio impact"


def _classify_blocker(
    *,
    stale_alert: bool,
    alert_supported: bool,
    gld_weight: float,
    concentration_completed: bool,
    stale_exposure: bool,
    legitimate_exposure: bool,
    severity_calibration: bool,
    drawdown_trial_negative: bool,
) -> Tuple[str, str, List[str], bool, bool]:
    factors: List[str] = []
    stale_alert_detected = stale_alert and not alert_supported
    stale_trial_detected = stale_exposure

    if concentration_completed and gld_weight <= GLD_TARGET_WEIGHT + 0.005:
        factors.append("GLD concentration resolved; residual position under 20% weight")

    if alert_supported:
        factors.append(
            "EXCESS_POSITION_DRAWDOWN alert supported by current broker GLD unrealized pct"
        )
    if stale_alert_detected:
        factors.append("STALE_DRAWDOWN_ALERT: alert may reference pre-reduction notional")

    if drawdown_trial_negative:
        factors.append("paper_drawdown_reduction trial remains NEGATIVE under HIGH_CPS baseline")
    if stale_exposure:
        factors.append(
            "STALE_EXPOSURE_TRIAL_INPUT: exposure trial assumes concentration reduction already completed"
        )
    if legitimate_exposure:
        factors.append(
            "LEGITIMATE_EXPOSURE_REVIEW_REQUIRED: exposure trial weak despite adequate exposure component"
        )
    if severity_calibration:
        factors.append(
            "SEVERITY_CALIBRATION_REQUIRED: position-level drawdown severity exceeds portfolio-weight impact"
        )

    if stale_alert_detected and not alert_supported:
        primary = "STALE_DRAWDOWN_ALERT"
        fix = "Phase 198S: refresh drawdown alert subject/notional against post-reduction GLD qty before certification review."
    elif stale_exposure and not legitimate_exposure:
        primary = "STALE_EXPOSURE_TRIAL_INPUT"
        fix = (
            "Phase 198S: refresh exposure trial inputs to exclude completed concentration reduction; "
            "do not loosen certification thresholds."
        )
    elif severity_calibration and concentration_completed:
        primary = "SEVERITY_CALIBRATION_REQUIRED"
        fix = (
            "Phase 198S: add portfolio-weight-aware drawdown severity to protective trials and alert escalation; "
            "audit-only calibration review — no auto-certify."
        )
    elif concentration_completed and gld_weight <= GLD_TARGET_WEIGHT + 0.005 and alert_supported:
        primary = "RESIDUAL_DRAWDOWN_REVIEW_REQUIRED"
        fix = (
            "Continue operator drawdown review on residual GLD (16 shares, ~20% weight); "
            "monitor EXCESS_POSITION_DRAWDOWN alert; do not force certification PASS."
        )
    elif legitimate_exposure:
        primary = "LEGITIMATE_EXPOSURE_REVIEW_REQUIRED"
        fix = "Review portfolio exposure quality via governance-approved monitoring; no certification threshold changes."
    else:
        primary = "LEGITIMATE_CURRENT_STATE_BLOCKER"
        fix = "Continue monitoring and targeted drawdown/exposure review under governance."

    if drawdown_trial_negative and primary != "STALE_DRAWDOWN_ALERT":
        if "RESIDUAL" not in primary:
            primary = "RESIDUAL_DRAWDOWN_REVIEW_REQUIRED"

    return primary, fix, factors, stale_alert_detected, stale_trial_detected


def _render_markdown(payload: Dict[str, Any]) -> str:
    audit = payload.get("audit") or {}
    classification = payload.get("classification") or {}
    safety = payload.get("safety") or {}
    negative_trials = audit.get("negative_trials") or []
    neg_lines = (
        "\n".join(
            f"- `{t.get('trial_id')}` score={t.get('effectiveness_score')} — {t.get('evaluation_reason', '')[:100]}"
            for t in negative_trials
        )
        or "- None"
    )
    factor_lines = "\n".join(f"- {f}" for f in (classification.get("factors") or [])) or "- None"

    return f"""# TRITON Phase 198R — Residual Drawdown Certification Blocker Audit

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Mode:** read-only audit — no re-scoring, no certification changes

## Executive summary

{payload.get("executive_summary")}

## Root cause

{classification.get("root_cause_summary")}

**Blocker classification:** `{audit.get("certification_blocker_classification")}`
**Stale vs legitimate:** {classification.get("stale_vs_legitimate")}
**Recommended next phase:** {audit.get("recommended_next_fix")}

## GLD residual drawdown

| Field | Value |
|-------|-------|
| GLD qty | {audit.get("gld_qty")} |
| GLD market value | {audit.get("gld_market_value")} |
| GLD weight | {audit.get("gld_weight_pct")}% |
| GLD unrealized P/L | {audit.get("gld_unrealized_pl")} |
| GLD unrealized P/L % | {audit.get("gld_unrealized_pl_pct")}% |
| Largest unrealized loss | {audit.get("largest_unrealized_loss_symbol")} ({audit.get("largest_unrealized_loss_value")}) |
| Portfolio total invested | {audit.get("total_invested_value")} |
| Portfolio total unrealized P/L | {audit.get("total_unrealized_pl")} |

## Drawdown alert

| Field | Value |
|-------|-------|
| Active | {audit.get("drawdown_alert_active")} |
| Symbol | {audit.get("drawdown_alert_symbol")} |
| Source | {audit.get("drawdown_alert_source")} |
| Age (minutes) | {audit.get("drawdown_alert_age")} |
| Supported by broker state | {audit.get("drawdown_alert_currently_supported_by_broker_state")} |
| Residual exposure severity | {audit.get("residual_exposure_severity")} |
| Position-weight-adjusted severity | {audit.get("position_weight_adjusted_drawdown_severity")} |

## Certification state (unchanged by this audit)

| Field | Value |
|-------|-------|
| Status | {audit.get("certification_status")} |
| Score | {audit.get("certification_score")} |
| Failed requirements | {audit.get("failed_requirements")} |
| Avg effectiveness | {audit.get("protective_average_effectiveness")} |
| Beneficial / Negative | {audit.get("beneficial_count")} / {audit.get("negative_count")} |

## Negative trials

{neg_lines}

### Drawdown trial reason

{audit.get("drawdown_trial_reason")}

### Exposure trial reason

{audit.get("exposure_trial_reason")}

## Classification factors

{factor_lines}

## Safety proof

| Check | Value |
|-------|-------|
| broker_orders_submitted | {safety.get("broker_orders_submitted")} |
| approval_artifact_present | {audit.get("approval_artifact_present")} |
| broker_auth_artifact_present | {audit.get("broker_auth_artifact_present")} |
| live_execution_allowed | {audit.get("live_execution_allowed")} |

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


def run_phase_198r_audit(*, verbose: bool = True) -> Dict[str, Any]:
    from services.portfolio_history_sync import load_live_execution_flag

    artifacts = {k: _load_json(p) for k, p in ARTIFACT_PATHS.items() if p.suffix == ".json"}
    portfolio = _parse_portfolio(ARTIFACT_PATHS["positions_snapshot"])

    gld = portfolio.get("gld") or {}
    gld_qty = _safe_int(gld.get("qty"))
    gld_mv = round(_safe_float(gld.get("market_value")), 2)
    gld_weight = float(portfolio.get("gld_weight") or 0)
    gld_upl = round(_safe_float(gld.get("unrealized_pl")), 2)
    gld_uplpc = round(_safe_float(gld.get("unrealized_plpc")) * 100.0, 4)

    evaluation = artifacts.get("evaluation") or {}
    trials = artifacts.get("trials") or {}
    cpi = artifacts.get("cpi") or {}
    cert = artifacts.get("certification") or {}
    watchdog = artifacts.get("watchdog_alerts") or {}
    phase_198q = artifacts.get("phase_198q") or {}

    drawdown_alert = _find_drawdown_alert(watchdog, GLD_SYMBOL)
    alert_supported, alert_support_reason = _alert_supported_by_broker(drawdown_alert, gld)
    stale_alert_flag, stale_alert_reason = _stale_alert_on_pre_reduction_notional(
        drawdown_alert,
        gld,
        concentration_completed=bool(evaluation.get("concentration_reduction_completed")),
    )

    concentration_completed = bool(evaluation.get("concentration_reduction_completed"))
    stale_exposure, stale_exposure_reason = _exposure_trial_stale(
        evaluation=evaluation,
        trials=trials,
        cpi=cpi,
        gld_weight=gld_weight,
    )
    exposure_component = int((cpi.get("component_scores") or {}).get("exposure") or 0)
    legitimate_exposure = (
        not stale_exposure
        and exposure_component < 85
        and any(
            e.get("trial_id") == "paper_exposure_reduction" and e.get("evaluation") == "NEGATIVE"
            for e in (evaluation.get("evaluations") or [])
        )
    )

    severity_cal, severity_reason = _severity_calibration_issue(
        gld=gld,
        gld_weight=gld_weight,
        total_invested=_safe_float(portfolio.get("total_invested")),
        alert_supported=alert_supported,
    )

    drawdown_eval = next(
        (
            e
            for e in (evaluation.get("evaluations") or [])
            if e.get("trial_id") == "paper_drawdown_reduction"
        ),
        {},
    )
    exposure_eval = next(
        (
            e
            for e in (evaluation.get("evaluations") or [])
            if e.get("trial_id") == "paper_exposure_reduction"
        ),
        {},
    )
    negative_trials = [
        e
        for e in (evaluation.get("evaluations") or [])
        if str(e.get("evaluation") or "").upper() == "NEGATIVE"
    ]

    portfolio_impact_pct = (
        abs(gld_upl) / _safe_float(portfolio.get("total_invested")) * 100.0
        if portfolio.get("total_invested")
        else 0.0
    )
    if portfolio_impact_pct <= 2.0:
        pw_severity = "LOW"
    elif portfolio_impact_pct <= 3.5:
        pw_severity = "MODERATE"
    else:
        pw_severity = "ELEVATED"

    residual_severity = str((drawdown_alert or {}).get("severity") or "UNKNOWN")

    primary, recommended_fix, factors, stale_alert_detected, stale_trial_detected = (
        _classify_blocker(
            stale_alert=stale_alert_flag,
            alert_supported=alert_supported,
            gld_weight=gld_weight,
            concentration_completed=concentration_completed,
            stale_exposure=stale_exposure,
            legitimate_exposure=legitimate_exposure,
            severity_calibration=severity_cal,
            drawdown_trial_negative=str(drawdown_eval.get("evaluation") or "").upper()
            == "NEGATIVE",
        )
    )

    secondary = [
        c
        for c in (
            "STALE_DRAWDOWN_ALERT" if stale_alert_detected else None,
            "STALE_EXPOSURE_TRIAL_INPUT" if stale_trial_detected else None,
            "SEVERITY_CALIBRATION_REQUIRED" if severity_cal else None,
            "LEGITIMATE_EXPOSURE_REVIEW_REQUIRED" if legitimate_exposure else None,
        )
        if c and c != primary
    ]

    stale_vs_legitimate = (
        "MIXED (residual drawdown legitimate; exposure trial inputs partially stale)"
        if stale_trial_detected and alert_supported
        else (
            "STALE"
            if stale_alert_detected or stale_trial_detected
            else "LEGITIMATE (current broker-supported drawdown on residual GLD)"
        )
    )

    root_cause = (
        f"After Phase 198Q, certification remains NOT_CERTIFIED because Evaluation fails high-CPS "
        f"criteria (avg={evaluation.get('average_effectiveness')}, beneficial={evaluation.get('beneficial_count')}, "
        f"negative={evaluation.get('negative_count')}, limit negative<=1). "
        f"GLD residual position ({gld_qty} shares, {gld_weight * 100:.2f}% weight) carries "
        f"{gld_uplpc:.2f}% unrealized loss (${gld_upl}), portfolio-weighted impact ~{portfolio_impact_pct:.2f}%. "
    )
    if alert_supported:
        root_cause += f"EXCESS_POSITION_DRAWDOWN alert is current ({alert_support_reason}). "
    if stale_exposure:
        root_cause += f"Exposure trial input appears stale: {stale_exposure_reason}. "
    if severity_cal:
        root_cause += f"Severity calibration gap: {severity_reason}. "
    root_cause += "This audit does not modify evaluation or certification scores."

    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    live_allowed = bool(load_live_execution_flag()) or bool(
        _load_json(RESULTS / "governance_authorization.json").get("live_execution_permitted")
    )

    audit_fields = {
        "gld_qty": gld_qty,
        "gld_market_value": gld_mv,
        "gld_weight": gld_weight,
        "gld_weight_pct": round(gld_weight * 100.0, 4),
        "gld_unrealized_pl": gld_upl,
        "gld_unrealized_pl_pct": gld_uplpc,
        "total_invested_value": portfolio.get("total_invested"),
        "total_unrealized_pl": portfolio.get("total_unrealized_pl"),
        "largest_unrealized_loss_symbol": portfolio.get("largest_unrealized_loss_symbol"),
        "largest_unrealized_loss_value": portfolio.get("largest_unrealized_loss_value"),
        "drawdown_alert_active": bool(drawdown_alert),
        "drawdown_alert_symbol": (drawdown_alert or {}).get("subject"),
        "drawdown_alert_source": "watchdog_alerts.json",
        "drawdown_alert_age": (drawdown_alert or {}).get("duration_minutes"),
        "drawdown_alert_currently_supported_by_broker_state": alert_supported,
        "residual_exposure_severity": residual_severity,
        "position_weight_adjusted_drawdown_severity": pw_severity,
        "certification_status": cert.get("certification_status"),
        "certification_score": cert.get("certification_score"),
        "failed_requirements": cert.get("failed_requirements") or [],
        "protective_average_effectiveness": evaluation.get("average_effectiveness"),
        "beneficial_count": evaluation.get("beneficial_count"),
        "negative_count": evaluation.get("negative_count"),
        "negative_trials": negative_trials,
        "exposure_trial_reason": exposure_eval.get("evaluation_reason"),
        "drawdown_trial_reason": drawdown_eval.get("evaluation_reason"),
        "certification_blocker_classification": primary,
        "stale_alert_detected": stale_alert_detected,
        "stale_trial_detected": stale_trial_detected,
        "recommended_next_fix": recommended_fix,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "live_execution_allowed": live_allowed,
    }

    checks = {
        "gld_qty_16": gld_qty == 16,
        "gld_weight_at_target": gld_weight <= GLD_TARGET_WEIGHT + 0.005,
        "classification_present": bool(primary),
        "certification_not_forced": str(cert.get("certification_status") or "") == "NOT_CERTIFIED",
        "no_broker_orders": True,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "live_blocked": not live_allowed,
    }
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
        "phase": "198R",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "mode": "read_only_residual_drawdown_certification_blocker_audit",
        "disclaimer": (
            "Audit only. Does not re-score evaluation, modify certification thresholds, place broker "
            "orders, create approval/authorization artifacts, or force certification PASS."
        ),
        "executive_summary": (
            f"Residual certification blocker classified as {primary}. GLD qty={gld_qty}, "
            f"weight={audit_fields['gld_weight_pct']}%, unrealized={gld_uplpc:.2f}%. "
            f"Certification remains {cert.get('certification_status')} (score {cert.get('certification_score')}). "
            f"Negative trials: {len(negative_trials)}. Alert broker-supported={alert_supported}."
        ),
        "audit": audit_fields,
        "classification": {
            "certification_blocker_classification": primary,
            "secondary_classifications": secondary,
            "recommended_next_fix": recommended_fix,
            "stale_vs_legitimate": stale_vs_legitimate,
            "factors": factors,
            "root_cause_summary": root_cause,
            "alert_support_reason": alert_support_reason,
            "stale_alert_reason": stale_alert_reason,
            "stale_exposure_reason": stale_exposure_reason,
            "severity_calibration_reason": severity_reason,
        },
        "phase_198q_reference": {
            "status": phase_198q.get("status"),
            "concentration_reduction_completed": (phase_198q.get("audit") or {}).get(
                "concentration_reduction_completed"
            ),
            "remaining_blockers": phase_198q.get("remaining_legitimate_blockers"),
        },
        "proof": {
            "open_orders_count": _count_csv_data_rows(ARTIFACT_PATHS["open_orders_snapshot"]),
            "drawdown_component_score": (cpi.get("component_scores") or {}).get("drawdown"),
            "exposure_component_score": exposure_component,
            "concentration_component_score": (cpi.get("component_scores") or {}).get(
                "concentration"
            ),
            "portfolio_weighted_gld_loss_pct": round(portfolio_impact_pct, 4),
            "phase_198q_concentration_resolved": concentration_completed,
        },
        "safety": {
            "broker_orders_submitted": False,
            "approval_artifact_created": False,
            "broker_auth_artifact_created": False,
            "live_execution_modified": False,
            "evaluation_rescored": False,
            "certification_modified": False,
        },
        "artifact_snapshot": {
            k: {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "present": p.is_file()}
            for k, p in ARTIFACT_PATHS.items()
        },
        "git_status_short": git_status,
        "git_diff_stat": git_diff_stat,
        "marker": MARKER,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(
            f"[198R] status={status} classification={primary} gld_qty={gld_qty} "
            f"gld_weight={audit_fields['gld_weight_pct']}% cert={cert.get('certification_status')} "
            f"negative_trials={len(negative_trials)}"
        )
        if status != "PASS":
            print(f"[198R] failed_checks={[k for k, v in checks.items() if not v]}")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 198R residual drawdown certification blocker audit"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase_198r_audit(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
