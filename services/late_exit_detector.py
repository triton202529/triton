"""
TRITON Phase 172 — Late Exit Detector (OBSERVE_ONLY).

Governance-approved observe-only detector for open positions. Scores, logs,
and tracks potential late exits while TRITON continues normal operation.

SAFETY — HARD GOVERNANCE ENFORCEMENT:
    mode = OBSERVE_ONLY (immutable)
    NO actions, orders, exits, lifecycle changes, or execution modifications.
    Output artifacts are evidence-only and MUST NOT be routed to execution.

Run:
    python -m services.late_exit_detector
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

import numpy as np
import pandas as pd

from services.pnl_ledger_builder import _load_price_series, _norm_sym, _to_float
from services.ticker_timeline_builder import build_ticker_timelines

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "late_exit_detector.json"
OUT_CANDIDATES = RESULTS / "late_exit_candidates.csv"
OUT_HISTORY = RESULTS / "late_exit_tracking_history.csv"

# ── Governance: immutable observe-only mode ──────────────────────────────────
MODE = "OBSERVE_ONLY"
ACTIONS_ALLOWED = False
EXECUTION_ROUTING_ALLOWED = False

GOVERNANCE_BANNER = {
    "mode": MODE,
    "actions_allowed": ACTIONS_ALLOWED,
    "execution_routing_allowed": EXECUTION_ROUTING_ALLOWED,
    "no_actions_generated": True,
    "no_orders_generated": True,
    "no_lifecycle_modifications": True,
    "no_execution_modifications": True,
    "notice": (
        "This detector is OBSERVE_ONLY. Outputs are evidence artifacts only "
        "and must never be connected to lifecycle rules, execution, or broker logic."
    ),
}


def _log(msg: str) -> None:
    print(f"[late_exit_detector] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any, nd: int = 2) -> Optional[float]:
    x = _to_float(v, np.nan)
    return round(float(x), nd) if np.isfinite(x) else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _classify_score(score: float) -> str:
    s = score if np.isfinite(score) else 0.0
    if s >= 90:
        return "EXIT_CANDIDATE"
    if s >= 75:
        return "ESCALATE"
    if s >= 50:
        return "REVIEW"
    if s >= 25:
        return "WATCH"
    return "NORMAL"


def _flag_count(flags: Any) -> int:
    if flags is None or (isinstance(flags, float) and np.isnan(flags)):
        return 0
    s = str(flags).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return 0
    if s.startswith("["):
        try:
            parsed = json.loads(s.replace("'", '"'))
            return len(parsed) if isinstance(parsed, list) else 1
        except Exception:
            pass
    return len([p for p in s.split(",") if p.strip()])


def _nearest_metric(
    tl_sym: pd.DataFrame, when: pd.Timestamp, col: str, days: int = 7
) -> Optional[float]:
    if tl_sym.empty or pd.isna(when) or col not in tl_sym.columns:
        return None
    sub = tl_sym.dropna(subset=[col])
    if sub.empty:
        return None
    deltas = (sub["timestamp"] - when).abs()
    idx = deltas.idxmin()
    if deltas.loc[idx] > pd.Timedelta(days=days):
        return None
    return float(sub.loc[idx, col])


def _nearest_flags(tl_sym: pd.DataFrame, when: pd.Timestamp, days: int = 7) -> int:
    if tl_sym.empty or pd.isna(when) or "risk_flags" not in tl_sym.columns:
        return 0
    sub = tl_sym[tl_sym["risk_flags"].notna()]
    if sub.empty:
        return 0
    deltas = (sub["timestamp"] - when).abs()
    idx = deltas.idxmin()
    if deltas.loc[idx] > pd.Timedelta(days=days):
        return 0
    return _flag_count(sub.loc[idx, "risk_flags"])


def _score_components(
    days_since_peak: float,
    drawdown_from_peak: float,
    confidence_decay: float,
    expected_return_decay: float,
    risk_deterioration: float,
    position_age: float,
) -> Tuple[float, Dict[str, float]]:
    """Return composite late_exit_score (0-100) and per-component contributions."""
    c_days = float(np.clip(days_since_peak / 15.0 * 100.0, 0.0, 100.0))
    c_dd = float(np.clip(drawdown_from_peak * 100.0, 0.0, 100.0))
    c_conf = float(np.clip(confidence_decay / 0.30 * 100.0, 0.0, 100.0))
    c_er = float(np.clip(expected_return_decay / 0.05 * 100.0, 0.0, 100.0))
    c_risk = float(np.clip(risk_deterioration * 25.0, 0.0, 100.0))
    c_age = float(np.clip(position_age / 60.0 * 100.0, 0.0, 100.0))

    score = 0.20 * c_days + 0.25 * c_dd + 0.15 * c_conf + 0.15 * c_er + 0.15 * c_risk + 0.10 * c_age
    parts = {
        "days_since_peak": c_days,
        "drawdown_from_peak": c_dd,
        "confidence_decay": c_conf,
        "expected_return_decay": c_er,
        "risk_deterioration": c_risk,
        "position_age": c_age,
    }
    return float(np.clip(score, 0.0, 100.0)), parts


def _load_open_positions() -> pd.DataFrame:
    snap = _read_csv(RESULTS / "positions_snapshot.csv")
    if snap.empty:
        return pd.DataFrame()
    snap.columns = [str(c).strip().lower() for c in snap.columns]
    sym_col = "ticker" if "ticker" in snap.columns else "symbol"
    rows: List[Dict[str, Any]] = []
    for _, r in snap.iterrows():
        sym = _norm_sym(r.get(sym_col))
        qty = _to_float(r.get("qty") or r.get("qty_available"))
        side = str(r.get("side") or "long").lower()
        if not sym or not np.isfinite(qty) or abs(qty) < 1e-9 or side not in ("long", ""):
            continue
        entry_px = _to_float(r.get("avg_entry_price"))
        current_px = _to_float(r.get("current_price") or r.get("lastday_price"))
        cost = _to_float(r.get("cost_basis"))
        if not np.isfinite(cost) and np.isfinite(entry_px):
            cost = entry_px * abs(qty)
        rows.append(
            {
                "ticker": sym,
                "qty": abs(qty),
                "avg_entry_price": entry_px,
                "current_price": current_px,
                "cost_basis": cost,
                "snapshot_ts": r.get("snapshot_ts") or r.get("date"),
            }
        )
    return pd.DataFrame(rows)


def _entry_for_position(tl_sym: pd.DataFrame, asof: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    buys = tl_sym[
        (tl_sym["action"].isin(["BUY", "ADD"]))
        & (tl_sym["event_source"] == "fill_ledger")
        & (tl_sym["timestamp"] <= asof)
    ]
    if buys.empty:
        buys = tl_sym[(tl_sym["action"] == "BUY") & (tl_sym["timestamp"] <= asof)]
    if buys.empty:
        return pd.NaT, np.nan
    last = buys.iloc[-1]
    return pd.Timestamp(last["timestamp"]), _to_float(last.get("price"), np.nan)


def _analyze_open_position(
    sym: str,
    pos: pd.Series,
    tl: pd.DataFrame,
    lifecycle: pd.DataFrame,
    asof: pd.Timestamp,
) -> Optional[Dict[str, Any]]:
    tl_sym = tl[tl["ticker"] == sym].copy()
    if not tl_sym.empty:
        tl_sym["confidence"] = pd.to_numeric(tl_sym["confidence"], errors="coerce")
        tl_sym["expected_return"] = pd.to_numeric(tl_sym["expected_return"], errors="coerce")

    entry_ts, entry_px = _entry_for_position(tl_sym, asof)
    if pd.isna(entry_ts):
        lc = lifecycle[lifecycle["ticker"] == sym]
        if not lc.empty and "date" in lc.columns:
            entry_ts = pd.Timestamp(lc["date"].iloc[0])
        else:
            entry_ts = asof - pd.Timedelta(days=30)

    ref_px = (
        entry_px
        if np.isfinite(entry_px) and entry_px > 0
        else _to_float(pos.get("avg_entry_price"))
    )
    current_px = _to_float(pos.get("current_price"))
    qty = _to_float(pos.get("qty"), np.nan)
    if not np.isfinite(current_px) or current_px <= 0:
        prices = _load_price_series(sym)
        if not prices.empty:
            current_px = float(prices["close"].iloc[-1])

    prices = _load_price_series(sym)
    if prices.empty or not np.isfinite(current_px):
        return None

    base = asof.tz_localize(None).normalize()
    start = (
        entry_ts.tz_localize(None).normalize()
        if not pd.isna(entry_ts)
        else base - pd.Timedelta(days=30)
    )
    window = prices[(prices["date"] >= start) & (prices["date"] <= base)]
    if window.empty:
        return None

    peak_idx = window["close"].idxmax()
    peak_px = float(window.loc[peak_idx, "close"])
    peak_date = pd.Timestamp(window.loc[peak_idx, "date"])
    days_since_peak = float((base - peak_date).days)
    position_age = float((base - start).days)

    runup = max(peak_px - (ref_px if np.isfinite(ref_px) else current_px), 0.0)
    drawdown_from_peak = (peak_px - current_px) / runup if runup > 1e-9 else 0.0
    drawdown_from_peak = float(np.clip(drawdown_from_peak, 0.0, 1.0))

    peak_ts = peak_date.tz_localize("UTC")
    conf_peak = _nearest_metric(tl_sym, peak_ts, "confidence")
    conf_now = _nearest_metric(tl_sym, asof, "confidence")
    if conf_now is None and not lifecycle.empty:
        lc = lifecycle[lifecycle["ticker"] == sym]
        if not lc.empty:
            conf_now = _to_float(lc["confidence"].iloc[0], np.nan)
    confidence_change = (
        (conf_peak - conf_now) if conf_peak is not None and conf_now is not None else np.nan
    )
    confidence_decay = max(confidence_change, 0.0) if np.isfinite(confidence_change) else 0.0

    er_peak = _nearest_metric(tl_sym, peak_ts, "expected_return")
    er_now = _nearest_metric(tl_sym, asof, "expected_return")
    if er_now is None and not lifecycle.empty:
        lc = lifecycle[lifecycle["ticker"] == sym]
        if not lc.empty and "predicted_close" in lc.columns and "close" in lc.columns:
            close = _to_float(lc["close"].iloc[0], np.nan)
            pred = _to_float(lc["predicted_close"].iloc[0], np.nan)
            if np.isfinite(close) and close > 0 and np.isfinite(pred):
                er_now = (pred / close) - 1.0
    er_decay_raw = (er_peak - er_now) if er_peak is not None and er_now is not None else np.nan
    expected_return_decay = max(er_decay_raw, 0.0) if np.isfinite(er_decay_raw) else 0.0

    flags_peak = _nearest_flags(tl_sym, peak_ts)
    flags_now = _nearest_flags(tl_sym, asof)
    risk_change = flags_now - flags_peak
    risk_deterioration = max(float(risk_change), 0.0)

    score, _parts = _score_components(
        days_since_peak,
        drawdown_from_peak,
        confidence_decay,
        expected_return_decay,
        risk_deterioration,
        position_age,
    )
    classification = _classify_score(score)
    opp_cost = max((peak_px - current_px) * qty, 0.0) if np.isfinite(qty) else 0.0

    return {
        "ticker": sym,
        "score": _f(score, 1),
        "classification": classification,
        "days_since_peak": _f(days_since_peak, 1),
        "drawdown_from_peak": _f(drawdown_from_peak, 4),
        "confidence_change": _f(confidence_change, 4),
        "risk_change": int(risk_change),
        "position_age": _f(position_age, 1),
        "estimated_opportunity_cost": _f(opp_cost),
        "generated_at": _now_iso(),
        "confidence_decay": _f(confidence_decay, 4),
        "expected_return_decay": _f(expected_return_decay, 4),
        "risk_deterioration": _f(risk_deterioration, 1),
        "peak_price": _f(peak_px),
        "current_price": _f(current_px),
    }


def detect_open_positions(
    tl: pd.DataFrame,
    lifecycle: pd.DataFrame,
    asof: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    positions = _load_open_positions()
    if positions.empty:
        return pd.DataFrame()
    asof = asof or pd.Timestamp.now(tz=timezone.utc)
    rows: List[Dict[str, Any]] = []
    for _, pos in positions.iterrows():
        sym = str(pos["ticker"])
        row = _analyze_open_position(sym, pos, tl, lifecycle, asof)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def append_tracking_history(candidates: pd.DataFrame) -> pd.DataFrame:
    """Append current snapshot; return full history."""
    registry_cols = [
        "ticker",
        "score",
        "classification",
        "days_since_peak",
        "drawdown_from_peak",
        "confidence_change",
        "risk_change",
        "position_age",
        "estimated_opportunity_cost",
        "generated_at",
    ]
    hist = _read_csv(OUT_HISTORY)
    snap = (
        candidates[registry_cols].copy()
        if not candidates.empty
        else pd.DataFrame(columns=registry_cols)
    )
    if hist.empty:
        combined = snap
    else:
        combined = pd.concat([hist, snap], ignore_index=True)
    if not combined.empty:
        combined.to_csv(OUT_HISTORY, index=False, encoding="utf-8")
    return combined


def _score_from_catalog_row(row: pd.Series) -> Tuple[float, str]:
    days = _to_float(row.get("days_late"), 0.0)
    # Catalog giveback_pct is price giveback; scale so 8%+ maps to full drawdown component.
    giveback = min(_to_float(row.get("giveback_pct"), 0.0) / 8.0, 1.0)
    conf_decay = max(_to_float(row.get("confidence_decay"), 0.0), 0.0)
    score, _ = _score_components(days, giveback, conf_decay, 0.0, 0.0, days)
    return score, _classify_score(score)


def _catalog_signal_positive(row: pd.Series, score: float) -> bool:
    band = str(row.get("escalation_band_at_exit") or "")
    return score >= 50.0 or band in ("REVIEW", "ESCALATE", "EXIT_CANDIDATE")


def build_accuracy(catalog: pd.DataFrame) -> Dict[str, Any]:
    """Compare retrospective detector signals to actual exit outcomes."""
    rows: List[Dict[str, Any]] = []
    if catalog.empty:
        return {
            "detector_accuracy": None,
            "precision": None,
            "recall": None,
            "historical_hit_rate": None,
            "samples": 0,
            "rows": [],
        }

    for _, r in catalog.iterrows():
        actual = str(r.get("classification") or "")
        actual_late = actual in ("LATE_EXIT", "SEVERE_LATE_EXIT")
        score, det_class = _score_from_catalog_row(r)
        signal_positive = _catalog_signal_positive(r, score)

        if signal_positive and actual_late:
            label = "TRUE_POSITIVE"
        elif signal_positive and not actual_late:
            label = "FALSE_POSITIVE"
        elif not signal_positive and not actual_late:
            label = "TRUE_NEGATIVE"
        else:
            label = "FALSE_NEGATIVE"

        rows.append(
            {
                "ticker": r.get("ticker"),
                "exit_date": r.get("exit_date"),
                "detector_score": _f(score, 1),
                "detector_classification": det_class,
                "actual_classification": actual,
                "accuracy_label": label,
            }
        )

    df = pd.DataFrame(rows)
    tp = int((df["accuracy_label"] == "TRUE_POSITIVE").sum())
    fp = int((df["accuracy_label"] == "FALSE_POSITIVE").sum())
    tn = int((df["accuracy_label"] == "TRUE_NEGATIVE").sum())
    fn = int((df["accuracy_label"] == "FALSE_NEGATIVE").sum())
    total = len(df)
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp > 0 else None)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    hit_rate = tp / (tp + fp + fn) if (tp + fp + fn) else None

    return {
        "detector_accuracy": _f(accuracy * 100 if accuracy is not None else np.nan, 1),
        "precision": _f(precision * 100 if precision is not None else np.nan, 1),
        "recall": _f(recall * 100 if recall is not None else np.nan, 1),
        "historical_hit_rate": _f(hit_rate * 100 if hit_rate is not None else np.nan, 1),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "samples": total,
        "rows": rows,
    }


def build_certification(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    accuracy: Dict[str, Any],
) -> Dict[str, Any]:
    reasons: List[str] = []
    inputs = [
        "positions_snapshot.csv",
        "ticker_decision_timelines.parquet",
        "exit_timing_framework.json",
        "late_exit_catalog.csv",
        "signal_lifecycle_effective.csv",
    ]
    present = sum(1 for p in inputs if (RESULTS / p).is_file() or (ROOT / "data" / p).is_file())
    data_quality = present / len(inputs) * 100.0

    n_hist = int(len(history))
    n_acc = int(accuracy.get("samples") or 0)
    acc_val = _to_float(accuracy.get("detector_accuracy"), np.nan)
    tickers_tracked = (
        history["ticker"].nunique() if not history.empty and "ticker" in history.columns else 0
    )

    if data_quality < 80:
        reasons.append(f"only {present}/{len(inputs)} input artifacts present")
    if n_acc < 30:
        reasons.append(f"accuracy sample is {n_acc} exits — directional only until 30+")
    if n_hist < 5:
        reasons.append("tracking history has fewer than 5 snapshots — coverage building")
    if not np.isfinite(acc_val):
        reasons.append("accuracy metrics not yet computable")

    if (
        data_quality >= 80
        and n_acc >= 30
        and np.isfinite(acc_val)
        and acc_val >= 70
        and n_hist >= 10
    ):
        status = "CERTIFIED"
    elif data_quality >= 60 and n_acc >= 10:
        status = "PARTIALLY_CERTIFIED"
    else:
        status = "NOT_CERTIFIED"

    return {
        "detector_certification_status": status,
        "data_quality_score": _f(data_quality, 1),
        "accuracy_sample_size": n_acc,
        "tracking_snapshots": n_hist,
        "tickers_tracked": int(tickers_tracked),
        "detector_accuracy": accuracy.get("detector_accuracy"),
        "reasons": reasons,
    }


def build_executive(
    candidates: pd.DataFrame,
    accuracy: Dict[str, Any],
    positions_count: int,
) -> Dict[str, Any]:
    if candidates.empty:
        return {
            "open_positions_reviewed": positions_count,
            "candidate_count": 0,
            "highest_risk_position": "—",
            "average_score": None,
            "average_opportunity_cost": None,
            "detector_accuracy": accuracy.get("detector_accuracy"),
            "precision": accuracy.get("precision"),
            "recall": accuracy.get("recall"),
            "governance_mode": MODE,
        }

    scores = pd.to_numeric(candidates["score"], errors="coerce")
    opp = pd.to_numeric(candidates["estimated_opportunity_cost"], errors="coerce")
    top = candidates.iloc[0]
    watch_plus = candidates[
        candidates["classification"].isin(["WATCH", "REVIEW", "ESCALATE", "EXIT_CANDIDATE"])
    ]

    return {
        "open_positions_reviewed": positions_count,
        "candidate_count": int(len(watch_plus)),
        "highest_risk_position": f"{top['ticker']} (score {top['score']}, {top['classification']})",
        "average_score": _f(scores.mean(), 1),
        "average_opportunity_cost": _f(opp.mean()),
        "detector_accuracy": accuracy.get("detector_accuracy"),
        "precision": accuracy.get("precision"),
        "recall": accuracy.get("recall"),
        "governance_mode": MODE,
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    tl = build_ticker_timelines()
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    catalog = _read_csv(RESULTS / "late_exit_catalog.csv")
    framework = _read_json(RESULTS / "exit_timing_framework.json")
    governance = _read_json(RESULTS / "exit_timing_governance.json")

    positions = _load_open_positions()
    candidates = detect_open_positions(tl, lifecycle)
    _log(f"open positions: {len(positions)}, scored: {len(candidates)}")

    registry_cols = [
        "ticker",
        "score",
        "classification",
        "days_since_peak",
        "drawdown_from_peak",
        "confidence_change",
        "risk_change",
        "position_age",
        "estimated_opportunity_cost",
        "generated_at",
    ]
    if not candidates.empty:
        candidates[registry_cols].to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=registry_cols).to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")

    history = append_tracking_history(candidates)
    accuracy = build_accuracy(catalog)
    cert = build_certification(candidates, history, accuracy)
    executive = build_executive(candidates, accuracy, len(positions))

    report = {
        "generated_at": _now_iso(),
        "governance": GOVERNANCE_BANNER,
        "governance_reference": {
            "phase_171_approval": governance.get("executive_report", {}).get(
                "approved_proposals", []
            ),
            "framework_certification": (framework.get("certification") or {}).get("status"),
        },
        "executive_report": executive,
        "detector_certification": cert,
        "accuracy": accuracy,
        "current_candidates": candidates.to_dict(orient="records"),
        "tracking_history_rows": int(len(history)),
        "score_bands": {
            "0-24": "NORMAL",
            "25-49": "WATCH",
            "50-74": "REVIEW",
            "75-89": "ESCALATE",
            "90-100": "EXIT_CANDIDATE",
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CANDIDATES.name}, {OUT_HISTORY.name}")
    _log(f"executive: {json.dumps(executive, default=str)}")
    _log(f"certification: {cert['detector_certification_status']}")
    _log(f"mode: {MODE} — no actions generated")
    return report


if __name__ == "__main__":
    run()
