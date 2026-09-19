"""
Edge-based sizing intelligence layer (READ-ONLY).

Combines three existing artifacts to rank trade opportunities by quality
and emit a per-symbol size-multiplier recommendation that downstream code
*may* (later) consult — this module never places, modifies, cancels, or
filters orders, and never mutates execution / manage_positions / broker
state. Output is a single CSV at:

    data/results/edge_sizing_recommendations.csv

Inputs (all optional; missing files degrade gracefully):
  - data/results/trade_opportunities.csv               (driver: per-row opportunities)
  - data/results/performance_risk_overlay.csv          (risk_flag join)
  - data/results/performance_intelligence_by_symbol.csv (total_pl join)

Run:
    python -m services.edge_sizing_engine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_OPPORTUNITIES_CSV = RESULTS_DIR / "trade_opportunities.csv"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_PERF_BY_SYMBOL_CSV = RESULTS_DIR / "performance_intelligence_by_symbol.csv"
DEFAULT_OUTPUT_CSV = RESULTS_DIR / "edge_sizing_recommendations.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"

EDGE_PROVENANCE_SCHEMA = "198T-SF23-EDGE-PROVENANCE-1"
OPPS_PROVENANCE_SCHEMA = "198T-SF23-OPPORTUNITIES-PROVENANCE-1"
STATUS_VALID = "VALID"
STATUS_FAIL_CLOSED = "FAIL_CLOSED"
PROVENANCE_COLUMNS = (
    "session_date",
    "lifecycle_sha256",
    "opportunities_sha256",
    "provenance_status",
)
OPPS_PROVENANCE_COLUMNS = (
    "session_date",
    "lifecycle_sha256",
    "provenance_status",
)


# ------------------------------------------------------------------
# Tunable scoring constants
# ------------------------------------------------------------------
# Delta is small in absolute terms (~±2.5% typical), so scale up to land
# in a meaningful range against confidence (0..1) and bound it so a
# single component cannot dominate the final score.
DELTA_SCALE = 20.0  # delta_pct=0.025 -> 0.5
DELTA_CAP = 0.5  # absolute cap for |delta_score|

PERF_BONUS_LARGE = 0.10
PERF_BONUS_SMALL = 0.05
PERF_BONUS_LARGE_THRESHOLD = 100.0  # |total_pl| >= 100 -> large bonus/penalty

RISK_PENALTY_HEAVY = 0.50  # FORCE_EXIT / BLOCK_NEW_BUY
RISK_PENALTY_MEDIUM = 0.20  # TRIM_PRIORITY

# Tier thresholds applied to the FINAL edge_score (after blocked override).
TIER_STRONG_MIN = 0.75
TIER_NORMAL_MIN = 0.40

TIER_MULTIPLIER: Dict[str, float] = {
    "STRONG_EDGE": 1.25,
    "NORMAL_EDGE": 1.00,
    "WEAK_EDGE": 0.50,
    "BLOCKED": 0.00,
}

# Risk-flag components that are hard-blocking (override score-based tier).
HARD_BLOCK_FLAGS = frozenset({"FORCE_EXIT", "BLOCK_NEW_BUY"})


# ------------------------------------------------------------------
# Safe IO helpers
# ------------------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[edge_sizing_engine] WARN {msg}", file=sys.stderr, flush=True)


def _safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    """Defensive CSV loader. Missing/empty/malformed -> empty DataFrame."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return pd.DataFrame()
    except OSError as e:
        _warn(f"{label}: stat failed ({e})")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        _warn(f"{label}: read failed ({e})")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _norm_sym(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().upper()


def _norm_risk_flag(x: Any) -> str:
    """Pipe-joined risk_flag normalized; empty/None -> empty string."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().upper()


def _risk_components(risk_flag: str) -> set[str]:
    if not risk_flag:
        return set()
    return {c.strip() for c in risk_flag.split("|") if c.strip()}


# ------------------------------------------------------------------
# Current-session provenance (mtime is never evidence)
# ------------------------------------------------------------------
def sha256_file(path: Path) -> Optional[str]:
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_sidecar_path(csv_path: Path) -> Path:
    return csv_path.parent / f"{csv_path.stem}_provenance.json"


def normalize_session_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    try:
        ts = pd.to_datetime(text, errors="coerce")
    except Exception:
        return ""
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def unique_session_date_from_df(df: Optional[pd.DataFrame]) -> str:
    """Single session_as_of_date/session_date, else empty (unproven)."""
    if df is None or df.empty:
        return ""
    for col in ("session_as_of_date", "session_date", "as_of_date"):
        if col not in df.columns:
            continue
        found = {normalize_session_date(v) for v in df[col].tolist()}
        found.discard("")
        if len(found) == 1:
            return found.pop()
        if len(found) > 1:
            return ""
    return ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _empty_edge_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "opportunity_type",
            "confidence",
            "delta_pct",
            "total_pl",
            "risk_flag",
            "edge_score",
            "sizing_tier",
            "size_multiplier",
            "reason",
            *PROVENANCE_COLUMNS,
        ]
    )


def attach_provenance_columns(
    df: pd.DataFrame,
    *,
    session_date: str,
    lifecycle_sha256: str,
    opportunities_sha256: str,
    status: str,
) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    if out.empty:
        out = _empty_edge_frame()
    out["session_date"] = session_date or ""
    out["lifecycle_sha256"] = lifecycle_sha256 or ""
    out["opportunities_sha256"] = opportunities_sha256 or ""
    out["provenance_status"] = status
    return out


def write_edge_provenance_sidecar(
    csv_path: Path,
    *,
    status: str,
    session_date: str = "",
    lifecycle_sha256: str = "",
    opportunities_sha256: str = "",
    row_count: int = 0,
    tickers: Optional[List[str]] = None,
    invalid_reason: Optional[str] = None,
) -> None:
    payload = {
        "schema": EDGE_PROVENANCE_SCHEMA,
        "status": status,
        "session_date": session_date or None,
        "generated_at_utc": _utc_now_iso(),
        "lifecycle_sha256": lifecycle_sha256 or None,
        "opportunities_sha256": opportunities_sha256 or None,
        "row_count": int(row_count),
        "tickers": list(tickers or []),
        "invalid_reason": invalid_reason,
    }
    write_json_atomic(provenance_sidecar_path(csv_path), payload)


def write_fail_closed_edge(
    out_path: Path,
    reason: str,
    *,
    session_date: str = "",
    lifecycle_sha256: str = "",
    opportunities_sha256: str = "",
) -> None:
    """Replace any prior edge CSV so it cannot be treated as current evidence."""
    empty = _empty_edge_frame()
    _atomic_write_csv(empty, out_path)
    write_edge_provenance_sidecar(
        out_path,
        status=STATUS_FAIL_CLOSED,
        session_date=session_date,
        lifecycle_sha256=lifecycle_sha256,
        opportunities_sha256=opportunities_sha256,
        row_count=0,
        tickers=[],
        invalid_reason=reason,
    )
    print(
        f"[EDGE_SIZING_FAIL_CLOSED] reason={reason} path={out_path}",
        flush=True,
    )


def load_json_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def evaluate_opportunities_provenance(
    opps_path: Path, lifecycle_path: Path
) -> Tuple[bool, str, Dict[str, Any]]:
    """Opportunities are current only with a VALID sidecar bound to this lifecycle."""
    info: Dict[str, Any] = {
        "session_date": "",
        "lifecycle_sha256": sha256_file(lifecycle_path) or "",
        "opportunities_sha256": sha256_file(opps_path) or "",
    }
    sidecar = load_json_sidecar(provenance_sidecar_path(opps_path))
    if sidecar is None:
        return False, "opportunities_provenance_sidecar_missing", info
    if sidecar.get("schema") != OPPS_PROVENANCE_SCHEMA:
        return False, "opportunities_provenance_schema_mismatch", info
    if str(sidecar.get("status") or "").upper() != STATUS_VALID:
        return False, "opportunities_provenance_not_valid", info
    lc_sha = info["lifecycle_sha256"]
    if not lc_sha or lc_sha != sidecar.get("lifecycle_sha256"):
        return False, "opportunities_lifecycle_hash_mismatch", info
    lifecycle_df = _safe_read_csv(lifecycle_path, label="signal_lifecycle_effective")
    session = unique_session_date_from_df(lifecycle_df)
    if not session or session != sidecar.get("session_date"):
        return False, "opportunities_session_unproven", info
    info["session_date"] = session
    if not info["opportunities_sha256"]:
        return False, "opportunities_file_missing", info
    return True, "", info


def load_proven_edge_sizing_map(
    edge_path: Path,
    *,
    lifecycle_path: Path,
    opportunities_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Allocation join: only current-session, hash-bound, unique, finite edge rows.
    File mtime is ignored. Missing/stale/forged evidence returns an empty map.
    """
    diag: Dict[str, Any] = {
        "status": STATUS_FAIL_CLOSED,
        "reason": None,
        "proven_row_count": 0,
        "session_date": None,
    }
    sidecar = load_json_sidecar(provenance_sidecar_path(edge_path))
    if sidecar is None:
        diag["reason"] = "edge_provenance_sidecar_missing"
        return {}, diag
    if sidecar.get("schema") != EDGE_PROVENANCE_SCHEMA:
        diag["reason"] = "edge_provenance_schema_mismatch"
        return {}, diag
    if str(sidecar.get("status") or "").upper() != STATUS_VALID:
        diag["reason"] = "edge_provenance_not_valid"
        return {}, diag

    lc_sha = sha256_file(lifecycle_path) or ""
    opps_sha = sha256_file(opportunities_path) or ""
    if not lc_sha or lc_sha != sidecar.get("lifecycle_sha256"):
        diag["reason"] = "edge_lifecycle_hash_mismatch"
        return {}, diag
    if not opps_sha or opps_sha != sidecar.get("opportunities_sha256"):
        diag["reason"] = "edge_opportunities_hash_mismatch"
        return {}, diag

    opps_ok, opps_reason, opps_info = evaluate_opportunities_provenance(
        opportunities_path, lifecycle_path
    )
    if not opps_ok:
        diag["reason"] = opps_reason
        return {}, diag

    session = opps_info.get("session_date") or ""
    if not session or session != sidecar.get("session_date"):
        diag["reason"] = "edge_session_unproven"
        return {}, diag

    df = _safe_read_csv(edge_path, label="edge_sizing_recommendations")
    if df.empty:
        diag["status"] = STATUS_VALID
        diag["session_date"] = session
        diag["proven_row_count"] = 0
        return {}, diag

    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        sym = _norm_sym(row.get("ticker") if "ticker" in df.columns else row.get("symbol"))
        if not sym:
            diag["reason"] = "edge_row_missing_ticker"
            return {}, diag
        counts[sym] = counts.get(sym, 0) + 1
    if any(n > 1 for n in counts.values()):
        diag["reason"] = "edge_duplicate_tickers"
        return {}, diag

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        sym = _norm_sym(row.get("ticker") if "ticker" in df.columns else row.get("symbol"))
        if str(row.get("provenance_status") or "").upper() != STATUS_VALID:
            continue
        if normalize_session_date(row.get("session_date")) != session:
            continue
        if str(row.get("lifecycle_sha256") or "") != lc_sha:
            continue
        if str(row.get("opportunities_sha256") or "") != opps_sha:
            continue
        score = _to_float(row.get("edge_score"))
        if score is None:
            continue
        out[sym] = {
            "edge_score": score,
            "sizing_tier": str(row.get("sizing_tier") or "").strip().upper(),
            "risk_flag": str(row.get("risk_flag") or "").strip().upper(),
            "confidence": _to_float(row.get("confidence")),
            "delta_pct": _to_float(row.get("delta_pct")),
        }
    diag["status"] = STATUS_VALID
    diag["session_date"] = session
    diag["proven_row_count"] = len(out)
    return out, diag


# ------------------------------------------------------------------
# Per-row scoring
# ------------------------------------------------------------------
def _confidence_score(confidence: Optional[float]) -> float:
    """Clamp to [0, 1]. Missing -> 0 (neutral, neither bonus nor penalty)."""
    if confidence is None:
        return 0.0
    return max(0.0, min(1.0, float(confidence)))


def _delta_score(delta_pct: Optional[float]) -> float:
    """Bounded linear scaling. Stronger positive delta -> higher score."""
    if delta_pct is None:
        return 0.0
    raw = float(delta_pct) * DELTA_SCALE
    if raw > DELTA_CAP:
        return DELTA_CAP
    if raw < -DELTA_CAP:
        return -DELTA_CAP
    return raw


def _performance_bonus(total_pl: Optional[float]) -> float:
    """
    Small additive bonus/penalty proportional to realized+unrealized PL
    sign and magnitude. Missing PL -> 0.
    """
    if total_pl is None:
        return 0.0
    pl = float(total_pl)
    if pl >= PERF_BONUS_LARGE_THRESHOLD:
        return PERF_BONUS_LARGE
    if pl > 0:
        return PERF_BONUS_SMALL
    if pl <= -PERF_BONUS_LARGE_THRESHOLD:
        return -PERF_BONUS_LARGE
    if pl < 0:
        return -PERF_BONUS_SMALL
    return 0.0


def _risk_penalty(risk_components_set: set[str]) -> float:
    """Sum of per-component penalties. Empty set -> 0."""
    if not risk_components_set:
        return 0.0
    p = 0.0
    for c in risk_components_set:
        if c in HARD_BLOCK_FLAGS:
            p += RISK_PENALTY_HEAVY
        elif c == "TRIM_PRIORITY":
            p += RISK_PENALTY_MEDIUM
    return p


def _classify_tier(edge_score: float, risk_components_set: set[str]) -> str:
    """
    Hard block if any FORCE_EXIT or BLOCK_NEW_BUY component is present,
    regardless of edge_score (those are veto signals, not just penalties).
    Otherwise threshold by edge_score.
    """
    if risk_components_set & HARD_BLOCK_FLAGS:
        return "BLOCKED"
    if edge_score >= TIER_STRONG_MIN:
        return "STRONG_EDGE"
    if edge_score >= TIER_NORMAL_MIN:
        return "NORMAL_EDGE"
    return "WEAK_EDGE"


def _build_reason(
    *,
    confidence_score: float,
    delta_score: float,
    performance_bonus: float,
    risk_penalty: float,
    risk_components_set: set[str],
    tier: str,
) -> str:
    """Compact, auditable reason string showing the major contributions."""
    parts = [
        f"conf={confidence_score:.2f}",
        f"delta={delta_score:+.2f}",
        f"perf={performance_bonus:+.2f}",
        f"risk={'|'.join(sorted(risk_components_set)) if risk_components_set else 'NONE'}",
        f"penalty={-risk_penalty:.2f}",
    ]
    if tier == "BLOCKED":
        parts.append("override=hard_block")
    return ";".join(parts)


# ------------------------------------------------------------------
# Build pipeline
# ------------------------------------------------------------------
def _build_lookup_map(
    df: pd.DataFrame,
    *,
    sym_candidates: tuple[str, ...],
    keep_columns: tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    """Reduce a side DataFrame to a {symbol -> {col: value}} dict."""
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out
    sym_col: Optional[str] = None
    for cand in sym_candidates:
        if cand in df.columns:
            sym_col = cand
            break
    if sym_col is None:
        return out
    cols_present = [c for c in keep_columns if c in df.columns]
    for _, row in df.iterrows():
        sym = _norm_sym(row.get(sym_col))
        if not sym or sym in out:
            continue
        out[sym] = {c: row.get(c) for c in cols_present}
    return out


def build_recommendations(
    opps: pd.DataFrame,
    risk_overlay: pd.DataFrame,
    perf_by_symbol: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score every row in `opps` against the side overlays. Returns a
    DataFrame with the spec'd columns. Rows without a usable ticker are
    skipped (silently — defensive).
    """
    if opps is None or opps.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "opportunity_type",
                "confidence",
                "delta_pct",
                "total_pl",
                "risk_flag",
                "edge_score",
                "sizing_tier",
                "size_multiplier",
                "reason",
            ]
        )

    risk_map = _build_lookup_map(
        risk_overlay,
        sym_candidates=("ticker", "symbol"),
        keep_columns=("risk_flag",),
    )
    perf_map = _build_lookup_map(
        perf_by_symbol,
        sym_candidates=("symbol", "ticker"),
        keep_columns=("total_pl",),
    )

    rows: list[Dict[str, Any]] = []
    for _, r in opps.iterrows():
        sym = _norm_sym(r.get("ticker") if "ticker" in opps.columns else r.get("symbol"))
        if not sym:
            continue

        opp_type = (
            str(r.get("opportunity_type") or "").strip().upper()
            if "opportunity_type" in opps.columns
            else ""
        )
        confidence = _to_float(r.get("confidence")) if "confidence" in opps.columns else None
        delta_pct = _to_float(r.get("delta_pct")) if "delta_pct" in opps.columns else None

        total_pl = _to_float(perf_map.get(sym, {}).get("total_pl"))
        risk_flag_raw = _norm_risk_flag(risk_map.get(sym, {}).get("risk_flag"))
        risk_components_set = _risk_components(risk_flag_raw)

        c_score = _confidence_score(confidence)
        d_score = _delta_score(delta_pct)
        p_bonus = _performance_bonus(total_pl)
        r_pen = _risk_penalty(risk_components_set)
        edge = c_score + d_score + p_bonus - r_pen

        tier = _classify_tier(edge, risk_components_set)
        mult = TIER_MULTIPLIER[tier]
        reason = _build_reason(
            confidence_score=c_score,
            delta_score=d_score,
            performance_bonus=p_bonus,
            risk_penalty=r_pen,
            risk_components_set=risk_components_set,
            tier=tier,
        )

        rows.append(
            {
                "ticker": sym,
                "opportunity_type": opp_type,
                "confidence": confidence if confidence is not None else "",
                "delta_pct": delta_pct if delta_pct is not None else "",
                "total_pl": total_pl if total_pl is not None else "",
                "risk_flag": risk_flag_raw or "",
                "edge_score": round(float(edge), 6),
                "sizing_tier": tier,
                "size_multiplier": mult,
                "reason": reason,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "opportunity_type",
            "confidence",
            "delta_pct",
            "total_pl",
            "risk_flag",
            "edge_score",
            "sizing_tier",
            "size_multiplier",
            "reason",
        ],
    )


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write to .tmp then os.replace so concurrent readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(
    *,
    opps_path: Path = DEFAULT_OPPORTUNITIES_CSV,
    risk_path: Path = DEFAULT_RISK_OVERLAY_CSV,
    perf_path: Path = DEFAULT_PERF_BY_SYMBOL_CSV,
    out_path: Path = DEFAULT_OUTPUT_CSV,
    lifecycle_path: Path = DEFAULT_LIFECYCLE_EFFECTIVE_CSV,
) -> int:
    """
    Public entry point. Returns 0 when current-session evidence is written
    (including a proven empty file). Returns 2 after fail-closed replacement
    when opportunities provenance cannot be proven.
    """
    opps_ok, opps_reason, opps_info = evaluate_opportunities_provenance(opps_path, lifecycle_path)
    session_date = str(opps_info.get("session_date") or "")
    lifecycle_sha = str(opps_info.get("lifecycle_sha256") or "")
    opps_sha = str(opps_info.get("opportunities_sha256") or "")
    if not opps_ok:
        _warn(f"opportunities provenance failed ({opps_reason}); replacing prior edge file")
        write_fail_closed_edge(
            out_path,
            opps_reason,
            session_date=session_date,
            lifecycle_sha256=lifecycle_sha,
            opportunities_sha256=opps_sha,
        )
        return 2

    opps = _safe_read_csv(opps_path, label="trade_opportunities")
    risk_overlay = _safe_read_csv(risk_path, label="performance_risk_overlay")
    perf_by_symbol = _safe_read_csv(perf_path, label="performance_intelligence_by_symbol")

    if opps.empty:
        recs = _empty_edge_frame()
        print("[EDGE_DIAGNOSTICS] no recommendations generated", flush=True)
        print(
            "[EDGE_SIZING] total=0 strong=0 normal=0 weak=0 blocked=0",
            flush=True,
        )
        recs = attach_provenance_columns(
            recs,
            session_date=session_date,
            lifecycle_sha256=lifecycle_sha,
            opportunities_sha256=opps_sha,
            status=STATUS_VALID,
        )
        _atomic_write_csv(recs, out_path)
        write_edge_provenance_sidecar(
            out_path,
            status=STATUS_VALID,
            session_date=session_date,
            lifecycle_sha256=lifecycle_sha,
            opportunities_sha256=opps_sha,
            row_count=0,
            tickers=[],
        )
        print(f"[EDGE_SIZING_OUT] path={out_path}", flush=True)
        return 0

    recs = build_recommendations(opps, risk_overlay, perf_by_symbol)
    tickers = []
    if not recs.empty and "ticker" in recs.columns:
        tickers = [_norm_sym(v) for v in recs["ticker"].tolist() if _norm_sym(v)]
        if len(tickers) != len(set(tickers)):
            write_fail_closed_edge(
                out_path,
                "edge_duplicate_tickers",
                session_date=session_date,
                lifecycle_sha256=lifecycle_sha,
                opportunities_sha256=opps_sha,
            )
            return 2

    counts = {"STRONG_EDGE": 0, "NORMAL_EDGE": 0, "WEAK_EDGE": 0, "BLOCKED": 0}
    if not recs.empty:
        for tier, n in recs["sizing_tier"].value_counts().items():
            counts[str(tier)] = int(n)

    if not recs.empty:
        edge_scores = pd.to_numeric(recs["edge_score"], errors="coerce")
        print(
            "[EDGE_DIAGNOSTICS] "
            f"avg_edge={edge_scores.mean():.3f} "
            f"max_edge={edge_scores.max():.3f} "
            f"min_edge={edge_scores.min():.3f} "
            f"strong_threshold={TIER_STRONG_MIN:.2f} "
            f"normal_threshold={TIER_NORMAL_MIN:.2f}",
            flush=True,
        )

        top = recs.copy()
        top["__edge_score_num"] = pd.to_numeric(top["edge_score"], errors="coerce")
        top = top.sort_values("__edge_score_num", ascending=False).head(10)

        print(
            "[EDGE_TOP] "
            + ", ".join(
                [
                    f"{str(r.get('ticker', ''))}:{float(r.get('__edge_score_num', 0.0)):.2f}"
                    for _, r in top.iterrows()
                    if pd.notna(r.get("__edge_score_num"))
                ]
            ),
            flush=True,
        )
    else:
        print(
            "[EDGE_DIAGNOSTICS] no recommendations generated",
            flush=True,
        )

    print(
        f"[EDGE_SIZING] total={len(recs)} "
        f"strong={counts['STRONG_EDGE']} "
        f"normal={counts['NORMAL_EDGE']} "
        f"weak={counts['WEAK_EDGE']} "
        f"blocked={counts['BLOCKED']}",
        flush=True,
    )

    recs = attach_provenance_columns(
        recs,
        session_date=session_date,
        lifecycle_sha256=lifecycle_sha,
        opportunities_sha256=opps_sha,
        status=STATUS_VALID,
    )
    _atomic_write_csv(recs, out_path)
    write_edge_provenance_sidecar(
        out_path,
        status=STATUS_VALID,
        session_date=session_date,
        lifecycle_sha256=lifecycle_sha,
        opportunities_sha256=opps_sha,
        row_count=len(recs),
        tickers=tickers,
    )
    print(f"[EDGE_SIZING_OUT] path={out_path}", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Read-only edge-based sizing intelligence layer. Combines trade "
            "opportunities with the performance risk overlay and per-symbol "
            "performance intelligence to emit edge_sizing_recommendations.csv."
        )
    )
    ap.add_argument(
        "--opportunities",
        type=Path,
        default=DEFAULT_OPPORTUNITIES_CSV,
        help="Source trade_opportunities.csv",
    )
    ap.add_argument(
        "--risk-overlay",
        type=Path,
        default=DEFAULT_RISK_OVERLAY_CSV,
        help="performance_risk_overlay.csv (optional join)",
    )
    ap.add_argument(
        "--perf-by-symbol",
        type=Path,
        default=DEFAULT_PERF_BY_SYMBOL_CSV,
        help="performance_intelligence_by_symbol.csv (optional join)",
    )
    ap.add_argument(
        "--lifecycle-effective",
        type=Path,
        default=DEFAULT_LIFECYCLE_EFFECTIVE_CSV,
        help="signal_lifecycle_effective.csv used for session provenance",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path",
    )
    ap.add_argument(
        "--invalidate-current-evidence",
        action="store_true",
        help="Replace any prior edge file with fail-closed empty evidence and exit 0.",
    )
    args = ap.parse_args(argv)
    if args.invalidate_current_evidence:
        write_fail_closed_edge(
            args.out,
            "producer_failed_or_incomplete",
            lifecycle_sha256=sha256_file(args.lifecycle_effective) or "",
            opportunities_sha256=sha256_file(args.opportunities) or "",
        )
        return 0
    return run(
        opps_path=args.opportunities,
        risk_path=args.risk_overlay,
        perf_path=args.perf_by_symbol,
        out_path=args.out,
        lifecycle_path=args.lifecycle_effective,
    )


if __name__ == "__main__":
    sys.exit(main())
