"""
Opportunity Promotion Engine (READ-ONLY analytics).

Reads:
    data/results/portfolio_allocation_recommendations.csv
    data/results/signals_with_rationale.csv
    data/results/signal_lifecycle_effective.csv

Writes:
    data/results/opportunity_promotion_recommendations.csv
    data/results/opportunity_promotion_summary.json

Purpose
-------
The portfolio_allocation_engine emits a `recommended_action` per symbol.
"WATCH" tags mean: a positive opportunity, but at least one promotion
gate (edge / confidence / delta_pct) was not yet strong enough. This
module re-scores WATCH symbols on a tighter, dedicated rubric and
classifies each one as:

    PROMOTE_OPEN_NEW  — ready to become an OPEN_NEW candidate
    KEEP_WATCH        — still positive but one or more gates miss
    REJECT            — momentum / confidence / signal has rolled over

The output is intended to feed dashboards and a human/operator decision
loop. It NEVER places trades, modifies execution, or mutates lifecycle
state; the trading layer (execute_trades) remains the sole owner of
order placement.

Decision precedence (spec):
    REJECT first  if delta_pct <= 0
                  OR confidence < 0.45
                  OR neither signal nor lifecycle is positive
    PROMOTE       if confidence  >= 0.50
                  AND delta_pct  > 0
                  AND (edge_score >= 0.50 OR allocation_score >= 0.50)
                  AND (lifecycle_action in {BUY,ADD} OR signal in {BUY,ADD})
    KEEP_WATCH    otherwise (positive signal, sub-threshold gate)

Safety
------
* Read-only. Missing inputs warn and continue.
* All numeric coercion is defensive (NaN / inf / blank -> None).
* main() returns 0 on success, 2 only on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_ALLOCATION_CSV = RESULTS_DIR / "portfolio_allocation_recommendations.csv"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "opportunity_promotion_recommendations.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "opportunity_promotion_summary.json"

# -----------------------------------------------------------
# Thresholds & labels (analytics-only)
# -----------------------------------------------------------
# Spec gates for promotion (rule A).
MIN_PROMOTE_CONFIDENCE = 0.50
MIN_PROMOTE_EDGE = 0.50
MIN_PROMOTE_ALLOCATION = 0.50

# Spec gate for rejection (rule C).
MIN_KEEP_CONFIDENCE = 0.45

DECISION_PROMOTE = "PROMOTE_OPEN_NEW"
DECISION_KEEP = "KEEP_WATCH"
DECISION_REJECT = "REJECT"

WATCH_FILTER_VALUE = "WATCH"

POSITIVE_SIGNAL_LABELS: frozenset = frozenset({"BUY", "ADD"})
LIFECYCLE_ACTION_POSITIVE: frozenset = frozenset({"BUY", "ADD"})

OUTPUT_COLUMNS = [
    "ticker",
    "allocation_score",
    "edge_score",
    "confidence",
    "delta_pct",
    "lifecycle_action",
    "effective_stance",
    "signal",
    "promotion_score",
    "promotion_decision",
    "reason",
]


# -----------------------------------------------------------
# Safe IO helpers
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[OPPORTUNITY_PROMOTION_WARN] {msg}", flush=True)


def _safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    """Return a DataFrame; never raises. Missing/empty/unreadable -> empty df."""
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); continuing without it")
            return pd.DataFrame()
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip", keep_default_na=False)
        except Exception as e:
            _warn(f"failed to read {label} ({path}): {type(e).__name__}: {e}")
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_safe)
    os.replace(tmp, path)


def _json_safe(o: Any) -> Any:
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, (datetime,)):
        return o.replace(microsecond=0).isoformat()
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


# -----------------------------------------------------------
# Coercion helpers
# -----------------------------------------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_float_or_zero(x: Any) -> float:
    v = _to_float(x)
    return 0.0 if v is None else v


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_symbol(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s == "BRK-B":
        s = "BRK.B"
    return s


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------
# Loaders
# -----------------------------------------------------------
def _load_allocation_watch_rows(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {allocation_score, edge_score, confidence, delta_pct,
              lifecycle_action, effective_stance, signal}}

    Only rows with `recommended_action == WATCH` are kept. The
    allocation engine has already aligned these fields across the
    underlying inputs, so this is the canonical seed for promotion.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col or "recommended_action" not in df.columns:
        _warn(
            "portfolio_allocation_recommendations.csv missing ticker/symbol or "
            "recommended_action column; nothing to promote"
        )
        return out

    for _, r in df.iterrows():
        action = _norm_upper(r.get("recommended_action"))
        if action != WATCH_FILTER_VALUE:
            continue
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "allocation_score": _to_float(r.get("allocation_score")),
            "edge_score": _to_float(r.get("edge_score")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get("delta_pct")),
            "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
            "effective_stance": _norm_upper(r.get("effective_stance")),
            "signal": _norm_upper(r.get("signal")),
        }
    return out


def _load_signals_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {signal, confidence, delta_pct}}

    Used to refresh allocation-row values when the allocation CSV is
    stale (e.g. allocation built before the latest signals run).
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("signals_with_rationale.csv missing ticker/symbol column")
        return out

    delta_col = _pick_first_present(df, ("delta_pct", "delta_pct_snapshot"))

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "signal": _norm_upper(r.get("signal")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get(delta_col)) if delta_col else None,
        }
    return out


def _load_lifecycle_effective_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {lifecycle_action, effective_stance, signal, confidence, delta_pct}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("signal_lifecycle_effective.csv missing ticker/symbol column")
        return out

    delta_col = _pick_first_present(df, ("delta_pct", "delta_pct_snapshot"))

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
            "effective_stance": _norm_upper(r.get("effective_stance"))
            or _norm_upper(r.get("stance")),
            "signal": _norm_upper(r.get("signal")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get(delta_col)) if delta_col else None,
        }
    return out


# -----------------------------------------------------------
# Scoring & decision
# -----------------------------------------------------------
def _is_positive_signal(signal: str) -> bool:
    return _norm_upper(signal) in POSITIVE_SIGNAL_LABELS


def _is_positive_lifecycle(lifecycle_action: str) -> bool:
    return _norm_upper(lifecycle_action) in LIFECYCLE_ACTION_POSITIVE


def _promotion_score(
    *,
    allocation_score: Optional[float],
    confidence: Optional[float],
    delta_pct: Optional[float],
    edge_score: Optional[float],
    signal: str,
    lifecycle_action: str,
) -> float:
    """
    Convex combination of the spec's six promotion factors.

    Term scaling:
      * allocation_score is already roughly in [-1, 1] (clamped).
      * confidence is [0, 1].
      * delta_pct is small (~[-0.05, 0.05]); 20x scaling gives reasonable
        spread before clamp.
      * edge_score is [-1, 1] (clamped).
      * signal_confirm = 1.0 if positive raw signal, 0.0 otherwise.
      * lifecycle_confirm = 1.0 if positive lifecycle, 0.0 otherwise.

    Higher = closer to ready. The score is used only for ranking
    (top_promotions); it never overrides the decision precedence.
    """
    alloc_norm = _clamp(_to_float_or_zero(allocation_score), -1.0, 1.0)
    conf_norm = _clamp(_to_float_or_zero(confidence), 0.0, 1.0)
    delta_norm = _clamp(_to_float_or_zero(delta_pct) * 20.0, -1.0, 1.0)
    edge_norm = _clamp(_to_float_or_zero(edge_score), -1.0, 1.0)
    signal_confirm = 1.0 if _is_positive_signal(signal) else 0.0
    lifecycle_confirm = 1.0 if _is_positive_lifecycle(lifecycle_action) else 0.0

    score = (
        0.30 * alloc_norm
        + 0.25 * conf_norm
        + 0.20 * delta_norm
        + 0.15 * edge_norm
        + 0.05 * signal_confirm
        + 0.05 * lifecycle_confirm
    )
    return round(score, 6)


def _decide_promotion(
    *,
    allocation_score: Optional[float],
    edge_score: Optional[float],
    confidence: Optional[float],
    delta_pct: Optional[float],
    signal: str,
    lifecycle_action: str,
) -> Tuple[str, str]:
    """
    Apply the spec's C → A → B precedence and return (decision, reason).

    Reasons are short, machine-greppable strings naming the specific gate
    or gates that drove the decision so the dashboard / audit logs can be
    filtered without re-deriving them.
    """
    pos_sig = _is_positive_signal(signal)
    pos_lc = _is_positive_lifecycle(lifecycle_action)
    pos_either = pos_sig or pos_lc

    # ── C: REJECT (highest precedence) ──────────────────────────────
    if delta_pct is None or delta_pct <= 0:
        return DECISION_REJECT, "delta_pct<=0_or_missing"
    if confidence is None or confidence < MIN_KEEP_CONFIDENCE:
        return DECISION_REJECT, f"confidence<{MIN_KEEP_CONFIDENCE}"
    if not pos_either:
        return DECISION_REJECT, "no_positive_signal_or_lifecycle"

    # ── A: PROMOTE_OPEN_NEW ─────────────────────────────────────────
    conf_ok = confidence >= MIN_PROMOTE_CONFIDENCE
    edge_ok = (edge_score is not None) and (edge_score >= MIN_PROMOTE_EDGE)
    alloc_ok = (allocation_score is not None) and (allocation_score >= MIN_PROMOTE_ALLOCATION)
    edge_or_alloc_ok = edge_ok or alloc_ok

    if conf_ok and edge_or_alloc_ok and pos_either:
        return DECISION_PROMOTE, "all_promotion_gates_met"

    # ── B: KEEP_WATCH (positive but a gate misses) ─────────────────
    why: List[str] = []
    if not conf_ok:
        why.append(f"conf<{MIN_PROMOTE_CONFIDENCE}")
    if not edge_or_alloc_ok:
        why.append(f"edge<{MIN_PROMOTE_EDGE}_and_alloc<{MIN_PROMOTE_ALLOCATION}")
    return DECISION_KEEP, "positive_signal_but_" + (",".join(why) or "below_threshold")


# -----------------------------------------------------------
# Build recommendations
# -----------------------------------------------------------
def _enrich_with_latest(
    seed: Dict[str, Any],
    sig_row: Optional[Dict[str, Any]],
    lc_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Allocation row wins by default (it's already cross-source aligned).
    Fall back to lifecycle, then signals, when allocation has blanks.
    """
    out = dict(seed)

    def _fill(field: str, *sources: Optional[Dict[str, Any]]) -> None:
        if out.get(field) not in (None, "", "NAN"):
            return
        for s in sources:
            if not s:
                continue
            v = s.get(field)
            if v not in (None, "", "NAN"):
                out[field] = v
                return

    _fill("confidence", lc_row, sig_row)
    _fill("delta_pct", lc_row, sig_row)
    _fill("lifecycle_action", lc_row)
    _fill("effective_stance", lc_row)
    _fill("signal", lc_row, sig_row)
    return out


def build_promotions(
    *,
    watch_seed: Dict[str, Dict[str, Any]],
    signals: Dict[str, Dict[str, Any]],
    lifecycle_eff: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For each WATCH symbol from the allocation report, compute promotion
    score + decision and return (df, summary).
    """
    rows: List[Dict[str, Any]] = []
    for sym in sorted(watch_seed.keys()):
        seed = watch_seed[sym]
        enriched = _enrich_with_latest(
            seed,
            sig_row=signals.get(sym),
            lc_row=lifecycle_eff.get(sym),
        )

        allocation_score = enriched.get("allocation_score")
        edge_score = enriched.get("edge_score")
        confidence = enriched.get("confidence")
        delta_pct = enriched.get("delta_pct")
        lifecycle_action = enriched.get("lifecycle_action") or ""
        effective_stance = enriched.get("effective_stance") or ""
        signal = enriched.get("signal") or ""

        promo_score = _promotion_score(
            allocation_score=allocation_score,
            confidence=confidence,
            delta_pct=delta_pct,
            edge_score=edge_score,
            signal=signal,
            lifecycle_action=lifecycle_action,
        )

        decision, reason = _decide_promotion(
            allocation_score=allocation_score,
            edge_score=edge_score,
            confidence=confidence,
            delta_pct=delta_pct,
            signal=signal,
            lifecycle_action=lifecycle_action,
        )

        rows.append(
            {
                "ticker": sym,
                "allocation_score": (
                    round(float(allocation_score), 6) if allocation_score is not None else None
                ),
                "edge_score": (round(float(edge_score), 6) if edge_score is not None else None),
                "confidence": (round(float(confidence), 6) if confidence is not None else None),
                "delta_pct": (round(float(delta_pct), 6) if delta_pct is not None else None),
                "lifecycle_action": lifecycle_action,
                "effective_stance": effective_stance,
                "signal": signal,
                "promotion_score": promo_score,
                "promotion_decision": decision,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    def _count(decision: str) -> int:
        if df.empty:
            return 0
        return int((df["promotion_decision"] == decision).sum())

    promoted_n = _count(DECISION_PROMOTE)
    kept_n = _count(DECISION_KEEP)
    rejected_n = _count(DECISION_REJECT)

    # top_promotions: rank PROMOTE first, then KEEP_WATCH; sort by
    # promotion_score desc then confidence desc then ticker asc.
    top_list: List[Dict[str, Any]] = []
    if not df.empty:
        cand = df[df["promotion_decision"].isin([DECISION_PROMOTE, DECISION_KEEP])].copy()
        if not cand.empty:
            cand["__rank_class"] = (cand["promotion_decision"] == DECISION_PROMOTE).astype(int)
            cand["__pscore"] = pd.to_numeric(cand["promotion_score"], errors="coerce").fillna(-1e9)
            cand["__conf"] = pd.to_numeric(cand["confidence"], errors="coerce").fillna(-1e9)
            cand = cand.sort_values(
                by=["__rank_class", "__pscore", "__conf", "ticker"],
                ascending=[False, False, False, True],
            )
            for _, r in cand.head(5).iterrows():
                top_list.append(
                    {
                        "ticker": str(r["ticker"]),
                        "promotion_decision": str(r["promotion_decision"]),
                        "promotion_score": float(r["__pscore"]),
                        "allocation_score": (
                            float(r["allocation_score"])
                            if pd.notna(r["allocation_score"])
                            else None
                        ),
                        "edge_score": (
                            float(r["edge_score"]) if pd.notna(r["edge_score"]) else None
                        ),
                        "confidence": (
                            float(r["confidence"]) if pd.notna(r["confidence"]) else None
                        ),
                        "delta_pct": (float(r["delta_pct"]) if pd.notna(r["delta_pct"]) else None),
                        "reason": str(r["reason"]),
                    }
                )

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "watch_candidates": int(len(df)),
        "promoted_open_new": promoted_n,
        "kept_watch": kept_n,
        "rejected": rejected_n,
        "thresholds": {
            "min_promote_confidence": MIN_PROMOTE_CONFIDENCE,
            "min_promote_edge": MIN_PROMOTE_EDGE,
            "min_promote_allocation": MIN_PROMOTE_ALLOCATION,
            "min_keep_confidence": MIN_KEEP_CONFIDENCE,
        },
        "top_promotions": top_list,
    }

    return df, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only opportunity promotion engine (no trading effect).",
    )
    p.add_argument("--allocation", default=str(DEFAULT_ALLOCATION_CSV))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--lifecycle-effective", default=str(DEFAULT_LIFECYCLE_EFFECTIVE_CSV))
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    print("[OPPORTUNITY_PROMOTION] starting (read-only intelligence layer)", flush=True)

    allocation_df = _safe_read_csv(
        Path(args.allocation), label="portfolio_allocation_recommendations.csv"
    )
    signals_df = _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")
    lifecycle_df = _safe_read_csv(
        Path(args.lifecycle_effective), label="signal_lifecycle_effective.csv"
    )

    watch_seed = _load_allocation_watch_rows(allocation_df)
    signals = _load_signals_map(signals_df)
    lifecycle_eff = _load_lifecycle_effective_map(lifecycle_df)

    df, summary = build_promotions(
        watch_seed=watch_seed,
        signals=signals,
        lifecycle_eff=lifecycle_eff,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    try:
        _atomic_write_csv(df, out_csv)
    except Exception as e:
        _warn(f"failed to write {out_csv}: {type(e).__name__}: {e}")
        return 2

    try:
        _atomic_write_json(summary, out_json)
    except Exception as e:
        _warn(f"failed to write {out_json}: {type(e).__name__}: {e}")
        return 2

    print(
        "[OPPORTUNITY_PROMOTION] "
        f"watch={summary['watch_candidates']} "
        f"promoted={summary['promoted_open_new']} "
        f"kept_watch={summary['kept_watch']} "
        f"rejected={summary['rejected']}",
        flush=True,
    )
    top_syms = [o["ticker"] for o in summary.get("top_promotions", [])]
    print(
        f"[OPPORTUNITY_TOP_PROMOTIONS] symbols={top_syms}",
        flush=True,
    )
    print(
        f"[OPPORTUNITY_PROMOTION_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
