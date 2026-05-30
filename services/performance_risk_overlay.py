"""
Performance Risk Overlay (READ-ONLY analytics).

Reads:
    data/results/performance_intelligence_by_symbol.csv

Writes:
    data/results/performance_risk_overlay.csv

For each symbol, derives a `risk_flag` from observed P/L and (when present)
win-rate/drag attributes. The output is informational only and is not consumed
by execution, sizing, lifecycle, or broker logic.

Classification rules (a symbol may match more than one — the resulting
risk_flag is a pipe-joined union, e.g. ``FORCE_EXIT|TRIM_PRIORITY``; symbols
matching no rule are tagged ``OK``):

    A. FORCE_EXIT
       unrealized_pl < -200  OR  total_pl < -300

    B. TRIM_PRIORITY
       unrealized_pl < -100  OR  drag_flag is True

    C. BLOCK_NEW_BUY
       total_pl < -150       OR  win_rate < 0.4   (only if win_rate column exists)

Safety:
    * Missing/empty input -> warn and exit 0 without writing anything.
    * Malformed rows are coerced defensively; the module never raises.
    * No trading logic is executed here.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_INPUT_CSV = RESULTS_DIR / "performance_intelligence_by_symbol.csv"
DEFAULT_OUTPUT_CSV = RESULTS_DIR / "performance_risk_overlay.csv"

# -----------------------------------------------------------
# Tunables (analytics-only thresholds; do not change behaviour anywhere)
# -----------------------------------------------------------
FORCE_EXIT_UNREALIZED_PL_LT = -200.0
FORCE_EXIT_TOTAL_PL_LT = -300.0

TRIM_PRIORITY_UNREALIZED_PL_LT = -100.0

BLOCK_NEW_BUY_TOTAL_PL_LT = -150.0
BLOCK_NEW_BUY_WIN_RATE_LT = 0.4

# Order in which flags are concatenated when more than one rule matches.
RISK_FLAG_ORDER = ("FORCE_EXIT", "TRIM_PRIORITY", "BLOCK_NEW_BUY")


# -----------------------------------------------------------
# Logging / safe IO helpers
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[PERFORMANCE_RISK_WARN] {msg}", flush=True)


def _safe_read_csv(path: Path, *, label: str) -> Optional[pd.DataFrame]:
    """Return a DataFrame, an empty DataFrame, or None when the file is missing.

    Returns:
        None              -- file does not exist (caller should treat as no-op)
        empty DataFrame   -- file exists but contains no usable rows
        DataFrame         -- successfully parsed
    Never raises.
    """
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); nothing to do")
            return None
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return None

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


# -----------------------------------------------------------
# Coercion helpers
# -----------------------------------------------------------
def _to_float(x: Any) -> Optional[float]:
    """Return a finite float or None for blanks / non-numeric values."""
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


def _to_bool(x: Any) -> bool:
    """Best-effort boolean coercion for drag_flag-like columns."""
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            return bool(int(x))
        except Exception:
            return False
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "y", "yes"}


def _norm_symbol(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s == "BRK-B":
        s = "BRK.B"
    return s


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _derive_drag_flag_series(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series indicating drag for each row.

    Order of preference:
        1. explicit ``drag_flag`` column when present
        2. ``performance_bucket == HIGH_DRAG``
        3. ``severity_bucket == HIGH`` AND ``total_pl < 0``
        4. all False (no signal)
    """
    n = len(df)
    if "drag_flag" in df.columns:
        try:
            return df["drag_flag"].apply(_to_bool).astype(bool)
        except Exception:
            return pd.Series([False] * n, index=df.index)

    if "performance_bucket" in df.columns:
        try:
            return df["performance_bucket"].astype(str).str.strip().str.upper() == "HIGH_DRAG"
        except Exception:
            pass

    if "severity_bucket" in df.columns and "total_pl" in df.columns:
        try:
            sev = df["severity_bucket"].astype(str).str.strip().str.upper() == "HIGH"
            pl = df["total_pl"].apply(_to_float)
            return sev & pl.apply(lambda v: v is not None and v < 0.0)
        except Exception:
            pass

    return pd.Series([False] * n, index=df.index)


# -----------------------------------------------------------
# Core classification
# -----------------------------------------------------------
def _classify_row(
    *,
    total_pl: Optional[float],
    unrealized_pl: Optional[float],
    drag_flag: bool,
    win_rate: Optional[float],
    has_win_rate: bool,
) -> str:
    """Return a pipe-joined union of triggered rules, or 'OK'."""
    triggers: List[str] = []

    # A. FORCE_EXIT
    cond_force_unreal = unrealized_pl is not None and unrealized_pl < FORCE_EXIT_UNREALIZED_PL_LT
    cond_force_total = total_pl is not None and total_pl < FORCE_EXIT_TOTAL_PL_LT
    if cond_force_unreal or cond_force_total:
        triggers.append("FORCE_EXIT")

    # B. TRIM_PRIORITY
    cond_trim_unreal = unrealized_pl is not None and unrealized_pl < TRIM_PRIORITY_UNREALIZED_PL_LT
    if cond_trim_unreal or drag_flag:
        triggers.append("TRIM_PRIORITY")

    # C. BLOCK_NEW_BUY
    cond_block_total = total_pl is not None and total_pl < BLOCK_NEW_BUY_TOTAL_PL_LT
    cond_block_winrate = (
        has_win_rate and win_rate is not None and win_rate < BLOCK_NEW_BUY_WIN_RATE_LT
    )
    if cond_block_total or cond_block_winrate:
        triggers.append("BLOCK_NEW_BUY")

    if not triggers:
        return "OK"
    # Preserve the canonical order regardless of evaluation order above.
    return "|".join([flag for flag in RISK_FLAG_ORDER if flag in triggers])


def build_overlay(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Build the overlay DataFrame and an aggregate counts dict."""
    counts = {
        "total_symbols": 0,
        "force_exit": 0,
        "trim_priority": 0,
        "block_new_buy": 0,
        "ok": 0,
    }

    if df is None or df.empty:
        empty = pd.DataFrame(
            columns=["ticker", "total_pl", "unrealized_pl", "drag_flag", "risk_flag"]
        )
        return empty, counts

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    has_total = "total_pl" in df.columns
    has_unreal = "unrealized_pl" in df.columns
    has_winrate = "win_rate" in df.columns

    drag_series = _derive_drag_flag_series(df)

    rows: List[dict] = []
    for i, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col)) if sym_col else ""
        if not sym:
            continue
        total_pl = _to_float(r.get("total_pl")) if has_total else None
        unreal_pl = _to_float(r.get("unrealized_pl")) if has_unreal else None
        win_rate = _to_float(r.get("win_rate")) if has_winrate else None
        drag = bool(drag_series.iloc[i]) if i in drag_series.index else False

        flag = _classify_row(
            total_pl=total_pl,
            unrealized_pl=unreal_pl,
            drag_flag=drag,
            win_rate=win_rate,
            has_win_rate=has_winrate,
        )

        rows.append(
            {
                "ticker": sym,
                "total_pl": total_pl,
                "unrealized_pl": unreal_pl,
                "drag_flag": bool(drag),
                "risk_flag": flag,
            }
        )

    out = pd.DataFrame(
        rows, columns=["ticker", "total_pl", "unrealized_pl", "drag_flag", "risk_flag"]
    )
    counts["total_symbols"] = int(len(out))
    if not out.empty:
        # A symbol can match more than one rule; count each rule independently.
        flags = out["risk_flag"].fillna("OK").astype(str)
        counts["force_exit"] = int(flags.str.contains("FORCE_EXIT", regex=False).sum())
        counts["trim_priority"] = int(flags.str.contains("TRIM_PRIORITY", regex=False).sum())
        counts["block_new_buy"] = int(flags.str.contains("BLOCK_NEW_BUY", regex=False).sum())
        counts["ok"] = int((flags == "OK").sum())

    return out, counts


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only performance-aware risk overlay (no trading effect).",
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to performance_intelligence_by_symbol.csv",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to write performance_risk_overlay.csv",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    in_path = Path(args.input)
    out_path = Path(args.output)

    print("[PERFORMANCE_RISK] starting (read-only intelligence layer)", flush=True)

    df = _safe_read_csv(in_path, label="performance_intelligence_by_symbol.csv")
    if df is None:
        # Spec: file missing -> warn, do nothing, do not crash.
        print(
            "[PERFORMANCE_RISK] total_symbols=0 force_exit=0 trim_priority=0 block_new_buy=0",
            flush=True,
        )
        return 0

    if df.empty:
        _warn(f"input is empty: {in_path}; writing empty overlay")

    overlay, counts = build_overlay(df)

    try:
        _atomic_write_csv(overlay, out_path)
    except Exception as e:
        _warn(f"failed to write {out_path}: {type(e).__name__}: {e}")
        return 2

    print(
        "[PERFORMANCE_RISK] "
        f"total_symbols={counts['total_symbols']} "
        f"force_exit={counts['force_exit']} "
        f"trim_priority={counts['trim_priority']} "
        f"block_new_buy={counts['block_new_buy']}",
        flush=True,
    )
    print(f"[PERFORMANCE_RISK_OUT] overlay={out_path.as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
