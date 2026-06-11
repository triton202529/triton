"""
TRITON Phase 177 — Recovery Basket Monitoring & Promotion Tracking (read-only).

Continuously monitors the diversification recovery basket and tracks progress
toward ORANGE → YELLOW → GREEN governance status.

SAFETY: monitoring and intelligence only — no governance, execution, lifecycle,
allocation, broker, or authorization changes.

Outputs (data/results/):
    recovery_basket_monitor.json
    recovery_basket_status.csv
    recovery_basket_promotion_history.csv

Run:
    python -m services.recovery_basket_monitor
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    CPI_WEIGHTS,
    _estimate_cps,
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.diversification_feasibility_analysis import (
    MIN_OPEN_NEW_CONFIDENCE,
    _classify_feasibility,
    _equal_weight,
    _lifecycle_row,
)
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_basket_monitor.json"
OUT_STATUS = RESULTS / "recovery_basket_status.csv"
OUT_HISTORY = RESULTS / "recovery_basket_promotion_history.csv"

CPS_YELLOW = 75
CPS_GREEN = 90
PERMIT_ESCALATION = {"GREEN", "YELLOW"}
DEFAULT_BASKET = ["QQQ", "HD", "XLK", "BITO"]


def _log(msg: str) -> None:
    print(f"[recovery_basket_monitor] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _basket_components(recovery: Dict[str, Any], feasibility: Dict[str, Any]) -> List[str]:
    members = list(
        recovery.get("executive_report", {}).get("best_recovery_basket") or DEFAULT_BASKET
    )
    adds = list((recovery.get("fastest_path_to_green") or {}).get("green_tickers") or [])
    extra = [
        _norm_sym(r.get("ticker"))
        for r in (feasibility.get("feasibility_candidates") or [])
        if r.get("in_recovery_basket")
    ]
    return list(dict.fromkeys(members + adds + extra))


def _build_status_rows(
    tickers: List[str],
    recommendations: pd.DataFrame,
    lifecycle: pd.DataFrame,
    alloc_summary: Dict[str, Any],
    recovery_rankings: Dict[str, float],
    held: Set[str],
    ts: str,
) -> List[Dict[str, Any]]:
    blocked = set(alloc_summary.get("blocked_new_buy_symbols") or [])
    rows: List[Dict[str, Any]] = []
    for sym in tickers:
        sym = _norm_sym(sym)
        rec_row = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
        rec = rec_row.iloc[0].to_dict() if not rec_row.empty else {"ticker": sym}
        lc = _lifecycle_row(lifecycle, sym)
        in_basket = True
        feasibility, blockers = _classify_feasibility(rec, lc, blocked, in_basket)
        conf = _to_float(rec.get("confidence") or lc.get("confidence"), np.nan)
        edge = _to_float(rec.get("edge_score"), np.nan)
        delta = _to_float(rec.get("delta_pct") or lc.get("delta_pct"), np.nan)

        rows.append(
            {
                "generated_at": ts,
                "ticker": sym,
                "is_held": sym in held,
                "lifecycle_status": str(
                    lc.get("lifecycle_action") or rec.get("lifecycle_action") or "—"
                ),
                "effective_stance": str(
                    lc.get("effective_stance") or rec.get("effective_stance") or "—"
                ),
                "allocation_status": str(rec.get("recommended_action") or "—"),
                "confidence": _f(conf, 4),
                "edge_score": _f(edge, 4),
                "delta_pct": _f(delta, 4),
                "risk_flags": str(rec.get("risk_flag") or "OK"),
                "governance_recovery_contribution": _f(recovery_rankings.get(sym), 1),
                "feasibility_status": feasibility,
                "blockers": "; ".join(blockers) if blockers else "",
                "sector": get_sector(sym),
            }
        )
    return rows


def _previous_status_by_ticker() -> Dict[str, Dict[str, Any]]:
    hist = _read_csv(OUT_STATUS)
    if hist.empty or "ticker" not in hist.columns:
        return {}
    hist = hist.sort_values("generated_at")
    out: Dict[str, Dict[str, Any]] = {}
    for sym, grp in hist.groupby("ticker"):
        out[str(sym)] = grp.iloc[-1].to_dict()
    return out


def _detect_promotions(
    current: List[Dict[str, Any]],
    previous: Dict[str, Dict[str, Any]],
    ts: str,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    watch_pairs = [("WATCH", "OPEN_NEW"), ("OPEN_NEW", "WATCH")]
    lifecycle_pairs = [("BUY", "HOLD"), ("HOLD", "EXIT")]

    for row in current:
        sym = row["ticker"]
        prev = previous.get(sym)
        if not prev:
            continue

        def _event(field: str, old: str, new: str, reason: str) -> None:
            if old != new and old != "—" and new != "—":
                events.append(
                    {
                        "promotion_timestamp": ts,
                        "ticker": sym,
                        "field": field,
                        "old_status": old,
                        "new_status": new,
                        "reason": reason,
                        "confidence_change": _f(
                            _to_float(row.get("confidence"), np.nan)
                            - _to_float(prev.get("confidence"), np.nan),
                            4,
                        ),
                        "edge_change": _f(
                            _to_float(row.get("edge_score"), np.nan)
                            - _to_float(prev.get("edge_score"), np.nan),
                            4,
                        ),
                    }
                )

        old_alloc = str(prev.get("allocation_status") or "—")
        new_alloc = str(row.get("allocation_status") or "—")
        if (old_alloc, new_alloc) in watch_pairs:
            _event("allocation_status", old_alloc, new_alloc, f"{sym} {old_alloc} → {new_alloc}")

        old_lc = str(prev.get("lifecycle_status") or "—")
        new_lc = str(row.get("lifecycle_status") or "—")
        if (old_lc, new_lc) in lifecycle_pairs:
            _event("lifecycle_status", old_lc, new_lc, f"{sym} lifecycle {old_lc} → {new_lc}")

        old_feas = str(prev.get("feasibility_status") or "—")
        new_feas = str(row.get("feasibility_status") or "—")
        if old_feas != new_feas:
            _event(
                "feasibility_status",
                old_feas,
                new_feas,
                f"{sym} feasibility {old_feas} → {new_feas}",
            )

    return events


def _append_history(events: List[Dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return _read_csv(OUT_HISTORY)
    hist = _read_csv(OUT_HISTORY)
    new_df = pd.DataFrame(events)
    combined = pd.concat([hist, new_df], ignore_index=True) if not hist.empty else new_df
    combined.to_csv(OUT_HISTORY, index=False, encoding="utf-8")
    return combined


def _append_status(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    hist = _read_csv(OUT_STATUS)
    combined = pd.concat([hist, new_df], ignore_index=True) if not hist.empty else new_df
    combined.to_csv(OUT_STATUS, index=False, encoding="utf-8")
    return combined


def _recovery_probability(
    status_rows: List[Dict[str, Any]],
    recovery_adds: List[str],
    distance_yellow: float,
    distance_green: float,
) -> Tuple[str, float]:
    immediately = sum(
        1
        for r in status_rows
        if r.get("feasibility_status") == "IMMEDIATELY_FEASIBLE" and not r.get("is_held")
    )
    required = max(len(recovery_adds), 1)
    avail = immediately / required

    conf_scores = []
    for r in status_rows:
        if r.get("is_held"):
            continue
        c = _to_float(r.get("confidence"), np.nan)
        if np.isfinite(c):
            conf_scores.append(min(c / MIN_OPEN_NEW_CONFIDENCE, 1.0))
    conf_factor = float(np.mean(conf_scores)) if conf_scores else 0.0

    edge_scores = []
    for r in status_rows:
        if r.get("is_held"):
            continue
        e = _to_float(r.get("edge_score"), np.nan)
        if np.isfinite(e):
            edge_scores.append(min(e / 0.65, 1.0))
    edge_factor = float(np.mean(edge_scores)) if edge_scores else 0.0

    dist_factor = max(0.0, 1.0 - distance_yellow / 30.0)
    score = 0.40 * avail + 0.25 * conf_factor + 0.20 * edge_factor + 0.15 * dist_factor
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.80:
        label = "VERY_HIGH"
    elif score >= 0.60:
        label = "HIGH"
    elif score >= 0.35:
        label = "MODERATE"
    else:
        label = "LOW"
    return label, score


def _project_cps(
    held: List[str], feasible_adds: List[str], total_mv: float, base_components: Dict[str, int]
) -> int:
    positions = _equal_weight(held, feasible_adds, total_mv)
    cps, _ = _estimate_cps(positions, base_components)
    return cps


def _generate_alerts(
    events: List[Dict[str, Any]],
    prev_monitor: Dict[str, Any],
    current_cps: int,
    projected_cps: int,
    projected_esc: str,
    prob_label: str,
    prev_prob: str,
) -> List[str]:
    alerts: List[str] = []
    for e in events:
        if e.get("field") == "allocation_status":
            alerts.append(
                f"[RECOVERY_ALERT] {e['ticker']} promoted {e['old_status']} → {e['new_status']}"
            )
        elif (
            e.get("field") == "feasibility_status" and e.get("new_status") == "IMMEDIATELY_FEASIBLE"
        ):
            alerts.append(f"[RECOVERY_ALERT] {e['ticker']} now IMMEDIATELY_FEASIBLE")

    prev_cps = _to_float((prev_monitor.get("executive_report") or {}).get("current_cps"), np.nan)
    if np.isfinite(prev_cps):
        if prev_cps < CPS_YELLOW <= current_cps:
            alerts.append("[RECOVERY_ALERT] CPS crossed YELLOW threshold (current snapshot)")
        if prev_cps < CPS_GREEN <= projected_cps:
            alerts.append("[RECOVERY_ALERT] Projected CPS now GREEN-capable")

    if projected_esc == "GREEN":
        alerts.append("[RECOVERY_ALERT] Recovery basket now GREEN-capable (full basket simulation)")
    elif projected_esc == "YELLOW" and (prev_monitor.get("governance_tracker") or {}).get(
        "projected_escalation"
    ) not in (
        "YELLOW",
        "GREEN",
    ):
        alerts.append("[RECOVERY_ALERT] Projected governance recovery reaches YELLOW")

    if prob_label != prev_prob and prev_prob:
        alerts.append(f"[RECOVERY_ALERT] Recovery probability changed {prev_prob} → {prob_label}")

    return alerts


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    prev_monitor = _read_json(OUT_JSON)

    recovery = _read_json(RESULTS / "diversification_recovery_basket.json")
    feasibility = _read_json(RESULTS / "diversification_feasibility_analysis.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")

    positions = _load_positions()
    held_list = [p["symbol"] for p in positions]
    held_set = set(held_list)
    total_mv = sum(p["market_value"] for p in positions)

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)

    rankings: Dict[str, float] = {}
    for row in recovery.get("best_diversification_candidates") or []:
        sym = _norm_sym(row.get("ticker"))
        if sym:
            rankings[sym] = _to_float(row.get("governance_improvement_score"), 0.0)

    tickers = _basket_components(recovery, feasibility)
    status_rows = _build_status_rows(
        tickers, recommendations, lifecycle, alloc_summary, rankings, held_set, ts
    )

    previous = _previous_status_by_ticker()
    events = _detect_promotions(status_rows, previous, ts)
    history_df = _append_history(events)
    status_df = _append_status(status_rows)

    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    current_esc = str(cpe.get("escalation_state") or _escalation_from_cps(current_cps))

    recovery_adds = [
        s
        for s in (recovery.get("fastest_path_to_green") or {}).get("green_tickers") or []
        if s not in held_set
    ]
    immediately = [
        r["ticker"]
        for r in status_rows
        if r.get("feasibility_status") == "IMMEDIATELY_FEASIBLE" and not r.get("is_held")
    ]
    feasible_adds = [s for s in recovery_adds if s in immediately]
    all_feasible_sim = immediately  # best available now

    projected_cps_partial = _project_cps(held_list, feasible_adds, total_mv, base_components)
    projected_cps_full = _project_cps(held_list, recovery_adds, total_mv, base_components)
    projected_esc_partial = _escalation_from_cps(projected_cps_partial)
    projected_esc_full = _escalation_from_cps(projected_cps_full)

    dist_yellow = max(0, CPS_YELLOW - current_cps)
    dist_green = max(0, CPS_GREEN - current_cps)

    prob_label, prob_score = _recovery_probability(
        status_rows, recovery_adds, dist_yellow, dist_green
    )
    prev_prob = str((prev_monitor.get("executive_report") or {}).get("recovery_probability") or "")

    prev_cps = _to_float((prev_monitor.get("executive_report") or {}).get("current_cps"), np.nan)
    if np.isfinite(prev_cps):
        if current_cps > prev_cps + 1:
            trend = "IMPROVING"
        elif current_cps < prev_cps - 1:
            trend = "DETERIORATING"
        else:
            trend = "UNCHANGED"
    else:
        trend = "UNCHANGED"

    blocking = [
        r["ticker"]
        for r in status_rows
        if not r.get("is_held") and r.get("feasibility_status") != "IMMEDIATELY_FEASIBLE"
    ]
    non_held = [r for r in status_rows if not r.get("is_held")]
    best = max(non_held, key=lambda r: _to_float(r.get("confidence"), 0), default=None)
    weakest = min(non_held, key=lambda r: _to_float(r.get("confidence"), 999), default=None)

    alerts = _generate_alerts(
        events,
        prev_monitor,
        current_cps,
        projected_cps_full,
        projected_esc_full,
        prob_label,
        prev_prob,
    )

    readiness = (
        "GREEN_READY"
        if projected_esc_full == "GREEN" and len(feasible_adds) == len(recovery_adds)
        else (
            "YELLOW_READY"
            if projected_esc_partial == "YELLOW" or projected_esc_full == "YELLOW"
            else "ORANGE — awaiting promotion"
        )
    )

    executive = {
        "current_cps": current_cps,
        "current_governance_state": (
            "AUTHORIZED" if auth.get("governance_authorized") else "BLOCKED"
        ),
        "current_escalation_state": current_esc,
        "recovery_progress": f"{len(feasible_adds)}/{len(recovery_adds)} additions immediately feasible",
        "distance_to_yellow": dist_yellow,
        "distance_to_green": dist_green,
        "target_cps_yellow": CPS_YELLOW,
        "target_cps_green": CPS_GREEN,
        "best_candidate": best["ticker"] if best else "—",
        "weakest_candidate": weakest["ticker"] if weakest else "—",
        "promotion_events_this_run": len(events),
        "recovery_probability": prob_label,
        "recovery_probability_score": _f(prob_score * 100, 1),
        "estimated_recovery_readiness": readiness,
        "blocking_candidates": blocking,
    }

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "recovery_basket_status": status_rows,
        "promotion_events": events,
        "promotion_history_count": int(len(history_df)),
        "governance_tracker": {
            "current_governance_status": executive["current_governance_state"],
            "current_escalation_state": current_esc,
            "current_cps": current_cps,
            "projected_cps_partial": projected_cps_partial,
            "projected_cps_full": projected_cps_full,
            "projected_escalation_partial": projected_esc_partial,
            "projected_escalation_full": projected_esc_full,
            "projected_governance_status_partial": (
                "AUTHORIZED" if projected_esc_partial in PERMIT_ESCALATION else "BLOCKED"
            ),
            "projected_governance_status_full": (
                "AUTHORIZED" if projected_esc_full in PERMIT_ESCALATION else "BLOCKED"
            ),
            "recovery_trend": trend,
        },
        "distance_to_recovery": {
            "current_cps": current_cps,
            "target_cps_yellow": CPS_YELLOW,
            "target_cps_green": CPS_GREEN,
            "distance_to_yellow": dist_yellow,
            "distance_to_green": dist_green,
        },
        "recovery_probability": {
            "classification": prob_label,
            "score_pct": _f(prob_score * 100, 1),
            "factors": {
                "immediately_feasible": f"{len(feasible_adds)}/{len(recovery_adds)}",
                "distance_to_yellow": dist_yellow,
            },
        },
        "alerts": alerts,
        "certification": {
            "status": "CERTIFIED" if status_rows else "NOT_CERTIFIED",
            "components_tracked": len(status_rows),
            "safety": "monitoring only — no governance, execution, lifecycle, or allocation changes",
        },
        "disclaimer": "Recovery basket monitor — observability only, no operational changes.",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_STATUS.name}, {OUT_HISTORY.name}")
    _log(f"CPS={current_cps} dist_yellow={dist_yellow} dist_green={dist_green} prob={prob_label}")
    _log(f"events={len(events)} alerts={len(alerts)} trend={trend}")
    return report


if __name__ == "__main__":
    run()
