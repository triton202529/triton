"""
Portfolio Allocation Engine (READ-ONLY analytics).

Reads:
    data/results/positions_snapshot.csv
    data/results/performance_risk_overlay.csv
    data/results/performance_intelligence_by_symbol.csv
    data/results/edge_sizing_recommendations.csv
    data/results/signals_with_rationale.csv
    data/results/signal_lifecycle_effective.csv

Writes:
    data/results/portfolio_allocation_recommendations.csv
    data/results/portfolio_allocation_summary.json

For each symbol in the union of these inputs, the engine produces a single
self-contained row capturing current position state, signal/edge quality,
risk posture, performance health, an `allocation_score`, and a
`recommended_action` derived from the precedence rules:

    A. EXIT       if risk_flag contains FORCE_EXIT
    B. TRIM       if risk_flag contains TRIM_PRIORITY
    C. BLOCK      if risk_flag contains BLOCK_NEW_BUY
    D. ADD        if currently held
                  AND effective_stance in {ADD, HOLD}
                  AND edge_score is strong
                  AND risk_flag is OK
    E. OPEN_NEW   if not currently held
                  AND lifecycle_action in {BUY, ADD}
                  AND edge_score is strong
                  AND confidence >= 0.50
                  AND delta_pct > 0
    F. WATCH      if not held and signal is positive but not strong enough
    G. HOLD       otherwise

Allocation-score combines: edge_score, confidence, delta_pct, performance
health, risk penalty, and a current-weight overweight penalty. It is used
to rank entry/add candidates in the JSON summary (`top_5_opportunities`).

Backwards compatibility:
    `recommended_action` values `EXIT` and `TRIM` are unchanged — these are
    the only values `manage_positions._load_portfolio_allocation_overlay_map`
    acts on for EXIT/TRIM overlay. The other action strings have evolved
    (`BLOCK_NEW_BUY` → `BLOCK`, `INCREASE` → `ADD`, plus new `OPEN_NEW` /
    `WATCH`) but those values were already loaded-but-unused downstream,
    so the rename is safe and contained to the reporting layer.

Safety:
    * Read-only. No trades, no broker calls, no mutation of execution,
      lifecycle, sizing, or manage_positions logic.
    * Missing inputs warn and continue — each loader returns an empty map.
    * Malformed rows are coerced defensively; main() returns rc=2 only on
      output write errors.
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
CONFIG_DIR = ROOT / "config"

DEFAULT_POSITIONS_CSV = RESULTS_DIR / "positions_snapshot.csv"
DEFAULT_EDGE_SIZING_CSV = RESULTS_DIR / "edge_sizing_recommendations.csv"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_PERF_BY_SYMBOL_CSV = RESULTS_DIR / "performance_intelligence_by_symbol.csv"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "portfolio_allocation_recommendations.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "portfolio_allocation_summary.json"

DEFAULT_EXECUTE_TRADES_CONFIG = CONFIG_DIR / "execute_trades.json"

# -----------------------------------------------------------
# Tunables (analytics-only thresholds)
# -----------------------------------------------------------
OVERWEIGHT_PCT = 15.0
UNDERWEIGHT_PCT = 5.0

# "Strong edge" gate used by ADD / OPEN_NEW rules. Either an explicit
# sizing_tier == STRONG_EDGE from the sizing engine, or a numeric edge_score
# at/above this threshold qualifies.
STRONG_EDGE_SCORE = 0.50

# Confidence floor for OPEN_NEW (spec rule E).
MIN_OPEN_NEW_CONFIDENCE = 0.50

# Default position cap fallback when execute_trades.json is missing.
# Mirrors services.execution_guard's hard-coded fallback.
MAX_POSITIONS_FALLBACK = 25

# Action precedence — first match wins (mirrors spec rules A..G).
ACTION_EXIT = "EXIT"
ACTION_TRIM = "TRIM"
ACTION_BLOCK = "BLOCK"
ACTION_ADD = "ADD"
ACTION_OPEN_NEW = "OPEN_NEW"
ACTION_WATCH = "WATCH"
ACTION_HOLD = "HOLD"

# Spec: risk_flag is a pipe-joined union (e.g. "FORCE_EXIT|TRIM_PRIORITY").
RISK_FLAG_FORCE_EXIT = "FORCE_EXIT"
RISK_FLAG_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_FLAG_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"

SIZING_TIER_STRONG_EDGE = "STRONG_EDGE"

# Stances that signal "add more to a held position".
EFFECTIVE_STANCE_ADD_OK: frozenset = frozenset({"ADD", "HOLD"})

# Lifecycle actions that authorize opening a brand-new position.
LIFECYCLE_ACTION_OPEN_OK: frozenset = frozenset({"BUY", "ADD"})

# Raw model signals that count as "positive but not yet strong" for WATCH.
POSITIVE_SIGNAL_LABELS: frozenset = frozenset({"BUY", "ADD"})

OUTPUT_COLUMNS = [
    "ticker",
    "is_currently_held",
    "current_position_value",
    "current_weight_pct",
    "allocation_band",
    "total_pl",
    "unrealized_pl",
    "risk_flag",
    "edge_score",
    "sizing_tier",
    "confidence",
    "delta_pct",
    "lifecycle_action",
    "effective_stance",
    "signal",
    "performance_bucket",
    "allocation_score",
    "recommended_action",
    "reason",
]


# -----------------------------------------------------------
# Logging / safe IO helpers
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[PORTFOLIO_ALLOCATION_WARN] {msg}", flush=True)


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
    """Last-resort JSON encoder: NaN/inf become None, datetime becomes iso."""
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
    """Return finite float or None for blanks / non-numeric / NaN / inf."""
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


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_max_positions(config_path: Path) -> Tuple[int, str]:
    """
    Returns (max_positions, source). Best-effort: reads only the JSON key we
    need so we don't depend on broker / ExecutionGuard imports.
    """
    try:
        if not config_path.is_file():
            return MAX_POSITIONS_FALLBACK, "fallback_default"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        v = cfg.get("max_positions")
        if v is None:
            return MAX_POSITIONS_FALLBACK, "fallback_default"
        n = max(1, int(v))
        return n, "execute_trades.json"
    except Exception as e:
        _warn(
            f"could not read max_positions from {config_path}: "
            f"{type(e).__name__}: {e}; using fallback {MAX_POSITIONS_FALLBACK}"
        )
        return MAX_POSITIONS_FALLBACK, "fallback_default"


# -----------------------------------------------------------
# Per-source loaders -> normalized {ticker -> dict}
# -----------------------------------------------------------
def _load_positions_map(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    {ticker: {"market_value", "qty", "unrealized_pl", "avg_entry_price",
              "current_price"}}

    Positions with non-positive market_value or qty are filtered: a long
    book with mv<=0 indicates either a closed position or a short; neither
    is a long-equity allocation candidate.
    """
    out: Dict[str, Dict[str, float]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("positions_snapshot.csv missing ticker/symbol column")
        return out

    mv_col = _pick_first_present(df, ("market_value", "value"))
    qty_col = _pick_first_present(df, ("qty", "qty_available"))
    upl_col = _pick_first_present(df, ("unrealized_pl",))
    avg_col = _pick_first_present(df, ("avg_entry_price",))
    px_col = _pick_first_present(df, ("current_price", "lastday_price"))

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        mv = _to_float_or_zero(r.get(mv_col)) if mv_col else 0.0
        qty = _to_float_or_zero(r.get(qty_col)) if qty_col else 0.0
        upl = _to_float_or_zero(r.get(upl_col)) if upl_col else 0.0
        avg = _to_float(r.get(avg_col)) if avg_col else None
        cp = _to_float(r.get(px_col)) if px_col else None

        if mv <= 0.0 or qty <= 0.0:
            continue

        existing = out.get(sym)
        if existing is None:
            out[sym] = {
                "market_value": mv,
                "qty": qty,
                "unrealized_pl": upl,
                "avg_entry_price": avg if avg is not None else 0.0,
                "current_price": cp if cp is not None else 0.0,
            }
        else:
            existing["market_value"] += mv
            existing["qty"] += qty
            existing["unrealized_pl"] += upl
    return out


def _load_edge_sizing_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {"edge_score", "sizing_tier", "risk_flag",
              "confidence", "delta_pct"}}

    confidence / delta_pct here mirror the sizing engine's per-symbol view;
    they are used as a fallback when the lifecycle effective file lacks a
    row for an entry candidate.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("edge_sizing_recommendations.csv missing ticker/symbol column")
        return out

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "edge_score": _to_float(r.get("edge_score")),
            "sizing_tier": str(r.get("sizing_tier") or "").strip().upper(),
            "risk_flag": str(r.get("risk_flag") or "").strip().upper(),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get("delta_pct")),
        }
    return out


def _load_risk_overlay_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {"risk_flag", "total_pl", "unrealized_pl", "drag_flag"}}

    Canonical source of `risk_flag` for action precedence.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("performance_risk_overlay.csv missing ticker/symbol column")
        return out

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "risk_flag": str(r.get("risk_flag") or "").strip().upper(),
            "total_pl": _to_float(r.get("total_pl")),
            "unrealized_pl": _to_float(r.get("unrealized_pl")),
            "drag_flag": str(r.get("drag_flag") or "").strip().lower() in {"true", "1", "yes"},
        }
    return out


def _load_perf_by_symbol_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {"total_pl", "unrealized_pl", "performance_bucket",
              "severity_bucket"}}

    `performance_bucket` feeds the perf-health term of allocation_score.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("symbol", "ticker"))
    if not sym_col:
        _warn("performance_intelligence_by_symbol.csv missing symbol/ticker column")
        return out

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "total_pl": _to_float(r.get("total_pl")),
            "unrealized_pl": _to_float(r.get("unrealized_pl")),
            "performance_bucket": str(r.get("performance_bucket") or "").strip().upper(),
            "severity_bucket": str(r.get("severity_bucket") or "").strip().upper(),
        }
    return out


def _load_signals_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {"signal", "confidence", "score", "delta_pct"}}

    `delta_pct` is taken from delta_pct_snapshot when delta_pct is absent.
    Used as a fallback when the lifecycle effective file lacks a row.
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
            "signal": str(r.get("signal") or "").strip().upper(),
            "confidence": _to_float(r.get("confidence")),
            "score": _to_float(r.get("score")),
            "delta_pct": _to_float(r.get(delta_col)) if delta_col else None,
        }
    return out


def _load_lifecycle_effective_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    {ticker: {"lifecycle_action", "stance", "effective_stance",
              "position_state", "confidence", "delta_pct", "signal"}}

    Canonical source for `lifecycle_action`, `effective_stance`, and the
    confidence/delta_pct used by the OPEN_NEW / ADD gates.
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
            "lifecycle_action": str(r.get("lifecycle_action") or "").strip().upper(),
            "stance": str(r.get("stance") or "").strip().upper(),
            "effective_stance": str(r.get("effective_stance") or "").strip().upper(),
            "position_state": str(r.get("position_state") or "").strip().upper(),
            "effective_position_state": str(r.get("effective_position_state") or "")
            .strip()
            .upper(),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get(delta_col)) if delta_col else None,
            "signal": str(r.get("signal") or "").strip().upper(),
        }
    return out


# -----------------------------------------------------------
# Per-row reasoning
# -----------------------------------------------------------
def _risk_components(risk_flag: str) -> List[str]:
    """Split a pipe-joined risk_flag string into normalized components."""
    if not risk_flag:
        return []
    parts = [p.strip().upper() for p in str(risk_flag).split("|")]
    return [p for p in parts if p and p != "OK"]


def _allocation_band(weight_pct: float, *, has_position: bool) -> str:
    """Classify a current weight into the spec's allocation bands."""
    if not has_position:
        return "NONE"
    if weight_pct > OVERWEIGHT_PCT:
        return "OVERWEIGHT"
    if weight_pct >= UNDERWEIGHT_PCT:
        return "NORMAL"
    return "UNDERWEIGHT"


def _is_strong_edge(edge_score: Optional[float], sizing_tier: str) -> bool:
    """STRONG_EDGE sizing tier or numeric edge_score >= STRONG_EDGE_SCORE."""
    if (sizing_tier or "").strip().upper() == SIZING_TIER_STRONG_EDGE:
        return True
    if edge_score is None:
        return False
    try:
        return float(edge_score) >= STRONG_EDGE_SCORE
    except Exception:
        return False


def _perf_health_term(perf_bucket: str) -> float:
    """Maps performance_bucket → numeric contribution to allocation_score."""
    b = (perf_bucket or "").strip().upper()
    if b == "STRONG_WINNER":
        return 0.20
    if b == "WINNER":
        return 0.10
    if b in {"NEUTRAL", ""}:
        return 0.0
    if b == "WEAK":
        return -0.10
    if b in {"LOSER", "STRONG_LOSER", "WEAK_LOSER"}:
        return -0.20
    return 0.0


def _risk_penalty(risk_components: List[str]) -> float:
    """Negative contribution proportional to risk-flag severity."""
    p = 0.0
    if RISK_FLAG_FORCE_EXIT in risk_components:
        p -= 0.50
    if RISK_FLAG_TRIM_PRIORITY in risk_components:
        p -= 0.30
    if RISK_FLAG_BLOCK_NEW_BUY in risk_components:
        p -= 0.20
    return p


def _weight_penalty(weight_pct: float) -> float:
    """Discourage further loading of an already-overweight book leg."""
    if weight_pct > OVERWEIGHT_PCT:
        return -0.10
    return 0.0


def _allocation_score(
    *,
    edge_score: Optional[float],
    confidence: Optional[float],
    delta_pct: Optional[float],
    perf_bucket: str,
    risk_components: List[str],
    weight_pct: float,
) -> float:
    """
    Convex combination of signal/health/risk terms in roughly [-1.5, +1.6].

    Each term is normalized to a comparable scale before weighting:
      * edge_score is treated as already in [-1, 1] (clamped).
      * confidence is [0, 1].
      * delta_pct (~[-0.05, 0.05] typical) is scaled by 10x then clamped.
      * perf_health: -0.20..+0.20 mapped from performance_bucket.
      * risk_penalty: 0..-1.0 from risk_flag composition.
      * weight_penalty: 0 or -0.10 for OVERWEIGHT.

    Higher = better opportunity. The score is used purely for ranking
    (top_5_opportunities) and never replaces the action precedence rules.
    """
    edge_norm = _clamp(_to_float_or_zero(edge_score), -1.0, 1.0)
    conf_norm = _clamp(_to_float_or_zero(confidence), 0.0, 1.0)
    delta_norm = _clamp(_to_float_or_zero(delta_pct) * 10.0, -1.0, 1.0)
    perf_term = _perf_health_term(perf_bucket)
    risk_p = _risk_penalty(risk_components)
    weight_p = _weight_penalty(weight_pct)

    score = (
        0.30 * edge_norm
        + 0.25 * conf_norm
        + 0.20 * delta_norm
        + 0.15 * perf_term
        + risk_p
        + weight_p
    )
    return round(score, 6)


def _decide_action(
    *,
    is_held: bool,
    risk_components: List[str],
    edge_score: Optional[float],
    sizing_tier: str,
    effective_stance: str,
    lifecycle_action: str,
    signal: str,
    confidence: Optional[float],
    delta_pct: Optional[float],
) -> Tuple[str, str]:
    """
    Apply spec precedence A..G and return (action, reason).

    The reason is a short machine-greppable string surfacing which rule
    fired and the inputs that drove it.
    """
    # A
    if RISK_FLAG_FORCE_EXIT in risk_components:
        return ACTION_EXIT, f"risk_flag_contains={RISK_FLAG_FORCE_EXIT}"
    # B
    if RISK_FLAG_TRIM_PRIORITY in risk_components:
        return ACTION_TRIM, f"risk_flag_contains={RISK_FLAG_TRIM_PRIORITY}"
    # C
    if RISK_FLAG_BLOCK_NEW_BUY in risk_components:
        return ACTION_BLOCK, f"risk_flag_contains={RISK_FLAG_BLOCK_NEW_BUY}"

    risk_ok = len(risk_components) == 0
    strong_edge = _is_strong_edge(edge_score, sizing_tier)
    eff_stance_u = (effective_stance or "").strip().upper()
    lc_action_u = (lifecycle_action or "").strip().upper()
    signal_u = (signal or "").strip().upper()

    # D
    if is_held and eff_stance_u in EFFECTIVE_STANCE_ADD_OK and strong_edge and risk_ok:
        return (
            ACTION_ADD,
            f"held_and_effective_stance={eff_stance_u}_and_strong_edge",
        )

    # E
    conf_ok = (confidence is not None) and (confidence >= MIN_OPEN_NEW_CONFIDENCE)
    delta_ok = (delta_pct is not None) and (delta_pct > 0.0)
    if (
        (not is_held)
        and lc_action_u in LIFECYCLE_ACTION_OPEN_OK
        and strong_edge
        and conf_ok
        and delta_ok
    ):
        return (
            ACTION_OPEN_NEW,
            (
                f"new_and_lifecycle_action={lc_action_u}_and_strong_edge_"
                f"conf>={MIN_OPEN_NEW_CONFIDENCE}_delta>0"
            ),
        )

    # F — non-held with a positive but sub-threshold opportunity.
    if not is_held:
        positive_intent = (
            lc_action_u in LIFECYCLE_ACTION_OPEN_OK or signal_u in POSITIVE_SIGNAL_LABELS
        )
        if positive_intent:
            why = []
            if not strong_edge:
                why.append("edge_weak")
            if confidence is None or confidence < MIN_OPEN_NEW_CONFIDENCE:
                why.append(f"conf<{MIN_OPEN_NEW_CONFIDENCE}")
            if delta_pct is None or delta_pct <= 0:
                why.append("delta<=0")
            return ACTION_WATCH, "positive_signal_but_" + (",".join(why) or "below_threshold")

    # G
    return ACTION_HOLD, "no_actionable_rule_matched"


# -----------------------------------------------------------
# Aggregation
# -----------------------------------------------------------
def build_recommendations(
    *,
    positions: Dict[str, Dict[str, float]],
    edge_sizing: Dict[str, Dict[str, Any]],
    risk_overlay: Dict[str, Dict[str, Any]],
    perf_by_symbol: Dict[str, Dict[str, Any]],
    signals: Dict[str, Dict[str, Any]],
    lifecycle_eff: Dict[str, Dict[str, Any]],
    max_positions: int,
    max_positions_source: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregate all six inputs into one row per symbol and apply action rules.

    Symbol universe is the union of all loaders so every actionable insight
    (held position, risk flag, edge candidate, fresh signal) appears.
    """
    universe = (
        set(positions.keys())
        | set(edge_sizing.keys())
        | set(risk_overlay.keys())
        | set(perf_by_symbol.keys())
        | set(signals.keys())
        | set(lifecycle_eff.keys())
    )

    total_position_value = sum(p["market_value"] for p in positions.values())

    rows: List[Dict[str, Any]] = []
    for sym in sorted(universe):
        pos = positions.get(sym)
        edge = edge_sizing.get(sym, {})
        risk = risk_overlay.get(sym, {})
        perf = perf_by_symbol.get(sym, {})
        sig = signals.get(sym, {})
        lc = lifecycle_eff.get(sym, {})

        is_held = pos is not None
        cur_value = float(pos["market_value"]) if pos else 0.0
        weight_pct = (cur_value / total_position_value) * 100.0 if total_position_value > 0 else 0.0

        # risk_flag precedence: overlay (canonical) > edge (mirrored) > OK.
        risk_flag_str = (
            str(risk.get("risk_flag") or "").strip().upper()
            or str(edge.get("risk_flag") or "").strip().upper()
            or ""
        )
        if not risk_flag_str:
            risk_flag_str = "OK"
        risk_comps = _risk_components(risk_flag_str)

        # total_pl precedence: overlay > perf > position.unrealized.
        total_pl: Optional[float] = risk.get("total_pl")
        if total_pl is None:
            total_pl = perf.get("total_pl")
        if total_pl is None and pos:
            total_pl = float(pos.get("unrealized_pl") or 0.0)

        # unrealized_pl precedence: position (live broker) > overlay > perf.
        unreal_pl: Optional[float]
        if pos:
            unreal_pl = float(pos.get("unrealized_pl") or 0.0)
        else:
            unreal_pl = risk.get("unrealized_pl")
            if unreal_pl is None:
                unreal_pl = perf.get("unrealized_pl")

        edge_score = edge.get("edge_score")
        sizing_tier = str(edge.get("sizing_tier") or "").strip().upper()

        # Lifecycle is authoritative for stance/action; fall back to signals.
        lifecycle_action = lc.get("lifecycle_action") or ""
        effective_stance = lc.get("effective_stance") or lc.get("stance") or ""
        signal_label = lc.get("signal") or sig.get("signal") or ""

        # Confidence / delta: prefer lifecycle, then signals, then edge.
        confidence = lc.get("confidence")
        if confidence is None:
            confidence = sig.get("confidence")
        if confidence is None:
            confidence = edge.get("confidence")

        delta_pct = lc.get("delta_pct")
        if delta_pct is None:
            delta_pct = sig.get("delta_pct")
        if delta_pct is None:
            delta_pct = edge.get("delta_pct")

        perf_bucket = perf.get("performance_bucket") or ""

        action, reason = _decide_action(
            is_held=is_held,
            risk_components=risk_comps,
            edge_score=edge_score,
            sizing_tier=sizing_tier,
            effective_stance=effective_stance,
            lifecycle_action=lifecycle_action,
            signal=signal_label,
            confidence=confidence,
            delta_pct=delta_pct,
        )

        alloc_score = _allocation_score(
            edge_score=edge_score,
            confidence=confidence,
            delta_pct=delta_pct,
            perf_bucket=perf_bucket,
            risk_components=risk_comps,
            weight_pct=weight_pct,
        )

        band = _allocation_band(weight_pct, has_position=is_held)

        rows.append(
            {
                "ticker": sym,
                "is_currently_held": bool(is_held),
                "current_position_value": round(cur_value, 4),
                "current_weight_pct": round(weight_pct, 4),
                "allocation_band": band,
                "total_pl": (round(total_pl, 4) if total_pl is not None else None),
                "unrealized_pl": (round(unreal_pl, 4) if unreal_pl is not None else None),
                "risk_flag": risk_flag_str,
                "edge_score": (round(float(edge_score), 6) if edge_score is not None else None),
                "sizing_tier": sizing_tier or "",
                "confidence": (round(float(confidence), 6) if confidence is not None else None),
                "delta_pct": (round(float(delta_pct), 6) if delta_pct is not None else None),
                "lifecycle_action": lifecycle_action,
                "effective_stance": effective_stance,
                "signal": signal_label,
                "performance_bucket": perf_bucket,
                "allocation_score": alloc_score,
                "recommended_action": action,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    held_df = df[df["is_currently_held"] == True] if not df.empty else df  # noqa: E712

    # Portfolio shape diagnostics (held-only).
    largest_symbol: Optional[str] = None
    largest_weight: float = 0.0
    if not held_df.empty:
        idx = held_df["current_weight_pct"].astype(float).idxmax()
        largest_symbol = str(held_df.at[idx, "ticker"])
        largest_weight = round(float(held_df.at[idx, "current_weight_pct"]), 4)

    # Counters per recommended_action.
    def _count_action(act: str) -> int:
        if df.empty:
            return 0
        return int((df["recommended_action"] == act).sum())

    open_new_n = _count_action(ACTION_OPEN_NEW)
    add_n = _count_action(ACTION_ADD)
    exit_n = _count_action(ACTION_EXIT)
    trim_n = _count_action(ACTION_TRIM)
    block_n = _count_action(ACTION_BLOCK)
    watch_n = _count_action(ACTION_WATCH)
    hold_n = _count_action(ACTION_HOLD)

    current_positions = int(len(held_df))
    available_slots = max(0, int(max_positions) - current_positions)

    # Allocation diagnostics: explain why candidates are not becoming OPEN_NEW / ADD.
    # These are read-only counters used for debugging the candidate pipeline.
    if df.empty:
        diag_buy_add_lifecycle = 0
        diag_positive_signal = 0
        diag_strong_edge = 0
        diag_conf_pass = 0
        diag_delta_pass = 0
        diag_open_gate_candidates = 0
        rejection_reasons: Dict[str, int] = {}
    else:
        edge_numeric = pd.to_numeric(df["edge_score"], errors="coerce")
        confidence_numeric = pd.to_numeric(df["confidence"], errors="coerce")
        delta_numeric = pd.to_numeric(df["delta_pct"], errors="coerce")
        lifecycle_upper = df["lifecycle_action"].astype(str).str.upper()
        signal_upper = df["signal"].astype(str).str.upper()
        sizing_upper = df["sizing_tier"].astype(str).str.upper()

        strong_edge_mask = (edge_numeric >= STRONG_EDGE_SCORE) | (
            sizing_upper == SIZING_TIER_STRONG_EDGE
        )
        conf_pass_mask = confidence_numeric >= MIN_OPEN_NEW_CONFIDENCE
        delta_pass_mask = delta_numeric > 0
        buy_add_lifecycle_mask = lifecycle_upper.isin(["BUY", ACTION_ADD])
        positive_signal_mask = signal_upper.isin(list(POSITIVE_SIGNAL_LABELS))

        diag_buy_add_lifecycle = int(buy_add_lifecycle_mask.sum())
        diag_positive_signal = int(positive_signal_mask.sum())
        diag_strong_edge = int(strong_edge_mask.sum())
        diag_conf_pass = int(conf_pass_mask.sum())
        diag_delta_pass = int(delta_pass_mask.sum())

        open_gate_mask = (
            (df["is_currently_held"] == False)  # noqa: E712
            & buy_add_lifecycle_mask
            & strong_edge_mask
            & conf_pass_mask
            & delta_pass_mask
        )
        diag_open_gate_candidates = int(open_gate_mask.sum())

        rejection_reasons = {}
        for reason in df["reason"].astype(str).fillna("UNKNOWN"):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    # top_5_opportunities: actionable adds/opens ranked by allocation_score
    # then edge_score then confidence (deterministic tie-break).
    top_opps: List[Dict[str, Any]] = []
    if not df.empty:
        candidates = df[df["recommended_action"].isin([ACTION_OPEN_NEW, ACTION_ADD])].copy()
        if not candidates.empty:
            candidates["__alloc"] = pd.to_numeric(
                candidates["allocation_score"], errors="coerce"
            ).fillna(-1e9)
            candidates["__edge"] = pd.to_numeric(candidates["edge_score"], errors="coerce").fillna(
                -1e9
            )
            candidates["__conf"] = pd.to_numeric(candidates["confidence"], errors="coerce").fillna(
                -1e9
            )
            candidates = candidates.sort_values(
                by=["__alloc", "__edge", "__conf", "ticker"],
                ascending=[False, False, False, True],
            )
            for _, r in candidates.head(5).iterrows():
                top_opps.append(
                    {
                        "ticker": str(r["ticker"]),
                        "recommended_action": str(r["recommended_action"]),
                        "allocation_score": float(r["__alloc"]),
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

    def _flag_count(component: str) -> int:
        if df.empty:
            return 0
        return int(df["risk_flag"].astype(str).str.contains(component, regex=False, na=False).sum())

    def _flag_symbols(component: str) -> List[str]:
        if df.empty:
            return []
        return [
            str(t)
            for t in df.loc[
                df["risk_flag"].astype(str).str.contains(component, regex=False, na=False),
                "ticker",
            ].tolist()
        ]

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        # Portfolio shape
        "total_position_value": round(float(total_position_value), 4),
        "current_positions": current_positions,
        "max_positions": int(max_positions),
        "max_positions_source": max_positions_source,
        "available_slots": int(available_slots),
        "largest_position_symbol": largest_symbol,
        "largest_position_weight_pct": largest_weight,
        "overweight_count": (
            int((held_df["allocation_band"] == "OVERWEIGHT").sum()) if not held_df.empty else 0
        ),
        "underweight_count": (
            int((held_df["allocation_band"] == "UNDERWEIGHT").sum()) if not held_df.empty else 0
        ),
        # Action counters
        "open_new_candidates": open_new_n,
        "add_candidates": add_n,
        "exit_candidates": exit_n,
        "trim_candidates": trim_n,
        "blocked_candidates": block_n,
        "watch_candidates": watch_n,
        "hold_candidates": hold_n,
        # Risk-flag detail
        "force_exit_count": _flag_count(RISK_FLAG_FORCE_EXIT),
        "trim_priority_count": _flag_count(RISK_FLAG_TRIM_PRIORITY),
        "block_new_buy_count": _flag_count(RISK_FLAG_BLOCK_NEW_BUY),
        "force_exit_symbols": _flag_symbols(RISK_FLAG_FORCE_EXIT),
        "trim_priority_symbols": _flag_symbols(RISK_FLAG_TRIM_PRIORITY),
        "blocked_new_buy_symbols": _flag_symbols(RISK_FLAG_BLOCK_NEW_BUY),
        # high_risk_positions kept for backward-compat with prior consumers.
        "high_risk_positions": sorted(
            set(_flag_symbols(RISK_FLAG_FORCE_EXIT) + _flag_symbols(RISK_FLAG_TRIM_PRIORITY))
        ),
        # Diagnostics
        "allocation_diagnostics": {
            "universe_count": int(len(df)),
            "currently_held_count": current_positions,
            "available_slots": int(available_slots),
            "buy_add_lifecycle_count": diag_buy_add_lifecycle,
            "positive_signal_count": diag_positive_signal,
            "strong_edge_count": diag_strong_edge,
            "confidence_pass_count": diag_conf_pass,
            "delta_pass_count": diag_delta_pass,
            "open_gate_candidates": diag_open_gate_candidates,
            "open_new_candidates": open_new_n,
            "add_candidates": add_n,
            "watch_candidates": watch_n,
            "top_rejection_reasons": dict(
                sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        },
        # Opportunity ranking
        "top_5_opportunities": top_opps,
    }

    return df, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only portfolio allocation engine (no trading effect).",
    )
    p.add_argument("--positions", default=str(DEFAULT_POSITIONS_CSV))
    p.add_argument("--edge-sizing", default=str(DEFAULT_EDGE_SIZING_CSV))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--perf-by-symbol", default=str(DEFAULT_PERF_BY_SYMBOL_CSV))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--lifecycle-effective", default=str(DEFAULT_LIFECYCLE_EFFECTIVE_CSV))
    p.add_argument("--config", default=str(DEFAULT_EXECUTE_TRADES_CONFIG))
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    print("[PORTFOLIO_ALLOCATION] starting (read-only intelligence layer)", flush=True)

    positions_df = _safe_read_csv(Path(args.positions), label="positions_snapshot.csv")
    edge_df = _safe_read_csv(Path(args.edge_sizing), label="edge_sizing_recommendations.csv")
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")
    perf_df = _safe_read_csv(
        Path(args.perf_by_symbol), label="performance_intelligence_by_symbol.csv"
    )
    signals_df = _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")
    lifecycle_df = _safe_read_csv(
        Path(args.lifecycle_effective), label="signal_lifecycle_effective.csv"
    )

    positions = _load_positions_map(positions_df)
    edge_sizing = _load_edge_sizing_map(edge_df)
    risk_overlay = _load_risk_overlay_map(risk_df)
    perf_by_symbol = _load_perf_by_symbol_map(perf_df)
    signals = _load_signals_map(signals_df)
    lifecycle_eff = _load_lifecycle_effective_map(lifecycle_df)

    max_positions, max_positions_source = _load_max_positions(Path(args.config))

    df, summary = build_recommendations(
        positions=positions,
        edge_sizing=edge_sizing,
        risk_overlay=risk_overlay,
        perf_by_symbol=perf_by_symbol,
        signals=signals,
        lifecycle_eff=lifecycle_eff,
        max_positions=max_positions,
        max_positions_source=max_positions_source,
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
        "[PORTFOLIO_ALLOCATION] "
        f"positions={summary['current_positions']} "
        f"max_positions={summary['max_positions']} "
        f"available_slots={summary['available_slots']} "
        f"open_new={summary['open_new_candidates']} "
        f"add={summary['add_candidates']} "
        f"exit={summary['exit_candidates']} "
        f"trim={summary['trim_candidates']} "
        f"block={summary['blocked_candidates']}",
        flush=True,
    )
    diag = summary.get("allocation_diagnostics", {})
    print(
        "[ALLOC_DIAGNOSTICS] "
        f"universe={diag.get('universe_count', 0)} "
        f"held={diag.get('currently_held_count', 0)} "
        f"slots={diag.get('available_slots', 0)} "
        f"buy_add_lifecycle={diag.get('buy_add_lifecycle_count', 0)} "
        f"positive_signal={diag.get('positive_signal_count', 0)} "
        f"strong_edge={diag.get('strong_edge_count', 0)} "
        f"conf_pass={diag.get('confidence_pass_count', 0)} "
        f"delta_pass={diag.get('delta_pass_count', 0)} "
        f"open_gate={diag.get('open_gate_candidates', 0)} "
        f"open_new={diag.get('open_new_candidates', 0)} "
        f"add={diag.get('add_candidates', 0)} "
        f"watch={diag.get('watch_candidates', 0)}",
        flush=True,
    )
    print(
        f"[ALLOC_REJECTION_TOP] {diag.get('top_rejection_reasons', {})}",
        flush=True,
    )
    top_syms = [o["ticker"] for o in summary.get("top_5_opportunities", [])]
    print(
        f"[PORTFOLIO_TOP_OPPORTUNITIES] symbols={top_syms}",
        flush=True,
    )
    print(
        f"[PORTFOLIO_ALLOCATION_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
