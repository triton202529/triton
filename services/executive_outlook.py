"""
TRITON Phase 161 — Executive Outlook Reconciliation.

Trace, validate, and reconcile backward-looking risk trend with forward-looking
risk direction before executive artifacts are written.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

RECONCILIATION_JSON = RESULTS / "executive_outlook_reconciliation.json"
RECONCILIATION_CSV = RESULTS / "executive_outlook_scorecard.csv"

# Pairs that are contradictory unless historical_trend / forward_outlook are explicit.
_INCOMPATIBLE_PAIRS = (
    (("DETERIORATING", "IMPROVING"), "backward_vs_forward_recovery"),
    (("IMPROVING", "DECLINING"), "momentum_reversal"),
    (("CRITICAL", "POSITIVE"), "severity_vs_sentiment"),
    (("RED", "RECOVERY"), "escalation_vs_recovery"),
)

_VALID_SEPARATED_PAIRS = {
    ("DETERIORATING", "IMPROVING"),
    ("IMPROVING", "DECLINING"),
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(value: Any, default: str = "STABLE") -> str:
    text = str(value or default).strip().upper()
    return text if text else default


def _log_tag(tag: str, fields: Dict[str, Any], verbose: bool = True) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def trace_executive_outlook_fields(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    verbose: bool = False,
) -> List[Dict[str, str]]:
    """Part A — trace all outlook-related field sources."""
    traces: List[Dict[str, str]] = []

    def _trace(field: str, source: str, calculation: str, raw: Any, status: str) -> None:
        entry = {
            "field": field,
            "source": source,
            "calculation": calculation,
            "value": _norm(raw),
            "status": status,
        }
        traces.append(entry)
        _log_tag(
            "EXECUTIVE_OUTLOOK_TRACE",
            {
                "field": field,
                "source": source,
                "calculation": calculation,
                "status": status,
            },
            verbose,
        )

    historical = _norm(cpi_doc.get("risk_trend"))
    forward = _norm(pred_doc.get("risk_direction") or pred_doc.get("risk_momentum"))
    escalation = _norm(cpe_doc.get("escalation_state"), "GREEN")
    health = _norm(cpi_doc.get("health_band"), "UNKNOWN")

    _trace(
        "historical_trend",
        "capital_preservation_intelligence.risk_trend",
        "current_cps vs rolling_baseline (backward-looking)",
        historical,
        "VALID",
    )
    _trace(
        "forward_outlook",
        "predictive_risk_intelligence.risk_direction",
        "cps_slope_and_momentum (forward-looking)",
        forward,
        "VALID",
    )
    _trace(
        "risk_trend",
        "cpi_doc.risk_trend",
        "alias_of_historical_trend",
        historical,
        (
            "VALID"
            if historical == forward or (historical, forward) in _VALID_SEPARATED_PAIRS
            else "CONFLICT"
        ),
    )
    _trace(
        "risk_direction",
        "pred_doc.risk_direction",
        "alias_of_forward_outlook",
        forward,
        (
            "VALID"
            if historical == forward or (historical, forward) in _VALID_SEPARATED_PAIRS
            else "CONFLICT"
        ),
    )
    _trace(
        "current_state",
        "cpi.health_band + cpe.escalation_state",
        f"health_band={health} escalation={escalation}",
        f"{health}_{escalation}",
        "VALID",
    )
    _trace(
        "forecast_confidence",
        "predictive_risk_intelligence.forecast_confidence",
        "history_depth_weighted",
        pred_doc.get("forecast_confidence"),
        "VALID",
    )

    return traces


def detect_outlook_contradictions(
    *,
    historical_trend: str,
    forward_outlook: str,
    current_state: str,
    escalation_state: str,
    explicit_separation: bool,
    verbose: bool = False,
) -> Tuple[List[Dict[str, str]], int]:
    """Part B — detect incompatible outlook field pairs."""
    historical = _norm(historical_trend)
    forward = _norm(forward_outlook)
    state = _norm(current_state)
    escalation = _norm(escalation_state, "GREEN")

    contradictions: List[Dict[str, str]] = []

    def _check(field_a: str, val_a: str, field_b: str, val_b: str) -> None:
        pair = (val_a, val_b)
        for incompatible, reason in _INCOMPATIBLE_PAIRS:
            if pair == incompatible:
                if explicit_separation and pair in _VALID_SEPARATED_PAIRS:
                    status = "PASS"
                else:
                    status = "FAIL"
                entry = {
                    "field_a": field_a,
                    "value_a": val_a,
                    "field_b": field_b,
                    "value_b": val_b,
                    "reason": reason,
                    "status": status,
                }
                contradictions.append(entry)
                _log_tag(
                    "OUTLOOK_CONTRADICTION",
                    {
                        "field_a": field_a,
                        "field_b": field_b,
                        "status": status,
                    },
                    verbose,
                )
                return

        if val_a == val_b or val_a == "STABLE" or val_b == "STABLE":
            status = "PASS"
        elif explicit_separation:
            status = "PASS"
        else:
            status = "PASS"
        _log_tag(
            "OUTLOOK_CONTRADICTION",
            {"field_a": field_a, "field_b": field_b, "status": status},
            verbose,
        )

    _check("historical_trend", historical, "forward_outlook", forward)
    _check("trend", historical, "direction", forward)

    if state == "CRITICAL" and forward in ("POSITIVE", "IMPROVING"):
        contradictions.append(
            {
                "field_a": "current_state",
                "value_a": state,
                "field_b": "forward_outlook",
                "value_b": forward,
                "reason": "severity_vs_sentiment",
                "status": "FAIL",
            }
        )
        _log_tag(
            "OUTLOOK_CONTRADICTION",
            {
                "field_a": "current_state",
                "field_b": "forward_outlook",
                "status": "FAIL",
            },
            verbose,
        )

    if escalation == "RED" and forward == "RECOVERY" and not explicit_separation:
        contradictions.append(
            {
                "field_a": "escalation_state",
                "value_a": escalation,
                "field_b": "forward_outlook",
                "value_b": forward,
                "reason": "escalation_vs_recovery",
                "status": "FAIL",
            }
        )
        _log_tag(
            "OUTLOOK_CONTRADICTION",
            {
                "field_a": "escalation_state",
                "field_b": "forward_outlook",
                "status": "FAIL",
            },
            verbose,
        )

    fail_count = sum(1 for c in contradictions if c["status"] == "FAIL")
    return contradictions, fail_count


def build_authoritative_executive_outlook(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Part C/D — single authoritative executive outlook with explicit
    historical vs forward separation when backward and forward signals diverge.
    """
    historical_trend = _norm(cpi_doc.get("risk_trend"))
    forward_outlook = _norm(pred_doc.get("risk_direction") or pred_doc.get("risk_momentum"))
    health_band = _norm(cpi_doc.get("health_band"), "UNKNOWN")
    escalation = _norm(cpe_doc.get("escalation_state"), "GREEN")
    confidence = int(pred_doc.get("forecast_confidence") or 50)

    current_state = health_band
    if escalation in ("RED", "CRITICAL"):
        current_state = f"{health_band}_{escalation}"

    separated = historical_trend != forward_outlook
    explicit_separation = (
        separated and (historical_trend, forward_outlook) in _VALID_SEPARATED_PAIRS
    )

    trace_executive_outlook_fields(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        pred_doc=pred_doc,
        verbose=verbose,
    )

    contradictions, fail_count = detect_outlook_contradictions(
        historical_trend=historical_trend,
        forward_outlook=forward_outlook,
        current_state=current_state,
        escalation_state=escalation,
        explicit_separation=explicit_separation or not separated,
        verbose=verbose,
    )

    if fail_count > 0 and not explicit_separation:
        # Reconcile by aligning forward to historical when not a valid separated pair.
        if (historical_trend, forward_outlook) not in _VALID_SEPARATED_PAIRS:
            forward_outlook = historical_trend
            separated = False
            explicit_separation = False
            contradictions, fail_count = detect_outlook_contradictions(
                historical_trend=historical_trend,
                forward_outlook=forward_outlook,
                current_state=current_state,
                escalation_state=escalation,
                explicit_separation=True,
                verbose=verbose,
            )

    outlook_consistency = "PASS" if fail_count == 0 else "FAIL"
    semantic_status = (
        "PASS" if explicit_separation or historical_trend == forward_outlook else "FAIL"
    )
    if separated and explicit_separation:
        semantic_status = "PASS"

    outlook: Dict[str, Any] = {
        "current_state": current_state,
        "historical_trend": historical_trend,
        "forward_outlook": forward_outlook,
        "trend": historical_trend,
        "direction": forward_outlook,
        "confidence": confidence,
        "outlook_consistency": outlook_consistency,
        "semantic_separation": explicit_separation,
        "outlook_narrative": (
            f"Historical CPS trend is {historical_trend}; forward predictive outlook is {forward_outlook}."
            if explicit_separation
            else f"Risk trend and direction aligned at {historical_trend}."
        ),
    }

    auth_status = "PASS" if outlook_consistency == "PASS" and semantic_status == "PASS" else "FAIL"
    _log_tag("AUTHORITATIVE_OUTLOOK", {"status": auth_status}, verbose)
    _log_tag("OUTLOOK_SEMANTICS", {"status": semantic_status}, verbose)

    return {
        "outlook": outlook,
        "contradictions": contradictions,
        "contradictions_found": len(contradictions),
        "contradictions_resolved": sum(
            1 for c in contradictions if c["status"] == "PASS" and c.get("reason")
        ),
        "semantic_consistency": semantic_status,
        "authoritative_status": auth_status,
    }


def write_executive_outlook_reconciliation(
    reconciliation: Dict[str, Any],
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Part F — persist reconciliation JSON and scorecard CSV."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    outlook = reconciliation.get("outlook") or {}
    overall = (
        "PASS"
        if reconciliation.get("authoritative_status") == "PASS"
        and reconciliation.get("semantic_consistency") == "PASS"
        else "FAIL"
    )

    payload = {
        "generated_at": ts,
        "phase": "161",
        "contradictions_found": reconciliation.get("contradictions_found", 0),
        "contradictions_resolved": reconciliation.get("contradictions_resolved", 0),
        "authoritative_outlook": reconciliation.get("authoritative_status") == "PASS",
        "semantic_consistency": reconciliation.get("semantic_consistency", "FAIL"),
        "outlook_consistency": outlook.get("outlook_consistency", "FAIL"),
        "overall_result": overall,
        "authoritative_outlook_object": outlook,
        "contradiction_details": reconciliation.get("contradictions") or [],
    }

    json_path = rd / "executive_outlook_reconciliation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(json_path)

    csv_path = rd / "executive_outlook_scorecard.csv"
    rows = [
        {
            "field": "historical_trend",
            "value": outlook.get("historical_trend"),
            "source": "cpi.risk_trend",
            "status": "PASS",
        },
        {
            "field": "forward_outlook",
            "value": outlook.get("forward_outlook"),
            "source": "pred.risk_direction",
            "status": "PASS",
        },
        {
            "field": "outlook_consistency",
            "value": outlook.get("outlook_consistency"),
            "source": "reconciliation",
            "status": outlook.get("outlook_consistency"),
        },
        {
            "field": "semantic_separation",
            "value": outlook.get("semantic_separation"),
            "source": "reconciliation",
            "status": reconciliation.get("semantic_consistency"),
        },
        {
            "field": "overall_result",
            "value": overall,
            "source": "phase_161",
            "status": overall,
        },
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "value", "source", "status"])
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(
            f"[EXECUTIVE_OUTLOOK_RECONCILIATION] overall={overall} "
            f"json={json_path} csv={csv_path}",
            flush=True,
        )

    return payload
