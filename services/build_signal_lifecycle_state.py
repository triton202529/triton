# services/build_signal_lifecycle_state.py
# ---------------------------------------------------------
# TRITON — Signal Lifecycle STATE Reducer (Phase 1.5)
#
# Purpose:
#   Convert lifecycle HISTORY (many rows per ticker over time)
#   into lifecycle STATE (exactly one row per ticker, NOW).
#
# Reads:
#   data/results/signals_lifecycle.csv   (preferred)
#   data/results/signal_lifecycle.csv    (fallback if history missing)
#
# Writes:
#   data/results/signal_lifecycle.csv    (STATE table: unique ticker)
#
# Adds:
#   freshness (FRESH/OK/STALE/UNKNOWN) derived from heartbeat.json / pipeline_status.json
#
# Run:
#   python services/build_signal_lifecycle_state.py --results-dir data/results
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    s = str(x).strip()
    if not s:
        return None

    # ISO
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # pandas
    try:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def heartbeat_timestamp(results_dir: Path) -> datetime | None:
    # Prefer heartbeat.json
    hb = read_json(results_dir / "heartbeat.json")
    if isinstance(hb, dict):
        for k in [
            "last_success_utc",
            "last_success",
            "last_run_utc",
            "last_run",
            "timestamp_utc",
            "timestamp",
        ]:
            if k in hb:
                dt = parse_dt(hb.get(k))
                if dt:
                    return dt

    # Fallback pipeline_status.json
    ps = read_json(results_dir / "pipeline_status.json")
    if isinstance(ps, dict):
        for k in [
            "last_success_utc",
            "last_success",
            "last_run_utc",
            "last_run",
            "timestamp_utc",
            "timestamp",
        ]:
            if k in ps:
                dt = parse_dt(ps.get(k))
                if dt:
                    return dt

    return None


def compute_freshness(results_dir: Path) -> tuple[str, float | None, str]:
    """
    Returns: (freshness_label, age_minutes, source_file)
      - FRESH: <= 30m
      - OK:    <= 180m
      - STALE: > 180m
      - UNKNOWN: no timestamp found
    """
    src = "heartbeat.json"
    dt = None
    hb = read_json(results_dir / "heartbeat.json")
    if isinstance(hb, dict):
        dt = heartbeat_timestamp(results_dir)
    if not dt:
        src = "pipeline_status.json"
        dt = heartbeat_timestamp(results_dir)

    if not dt:
        return "UNKNOWN", None, "none"

    age_min = max((utcnow() - dt).total_seconds() / 60.0, 0.0)

    if age_min <= 30:
        return "FRESH", age_min, src
    if age_min <= 180:
        return "OK", age_min, src
    return "STALE", age_min, src


def pick_time_col(df: pd.DataFrame) -> str | None:
    # Choose the best column to determine "latest"
    candidates = [
        "generated_at_utc",
        "updated_utc",
        "as_of_date",
        "asof_date",
        "date",
        "timestamp",
        "datetime",
        "time",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ticker normalization
    if "ticker" not in df.columns:
        if "symbol" in df.columns:
            df["ticker"] = df["symbol"]
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # stance normalization: if stance missing, map lifecycle_action -> stance
    if "stance" not in df.columns:
        if "lifecycle_action" in df.columns:
            df["stance"] = df["lifecycle_action"]
        elif "action" in df.columns:
            df["stance"] = df["action"]

    # position_state normalization
    if "position_state" not in df.columns:
        if "position" in df.columns:
            df["position_state"] = df["position"]

    return df


def build_state(results_dir: Path) -> pd.DataFrame:
    # Prefer your pipeline history file
    history_path = results_dir / "signals_lifecycle.csv"

    # Fallback: if user copied/renamed things, try signal_lifecycle.csv as source
    fallback_path = results_dir / "signal_lifecycle.csv"

    src_path = history_path if history_path.exists() else fallback_path
    if not src_path.exists():
        raise FileNotFoundError(
            f"Missing lifecycle source. Expected {history_path.name} (preferred) "
            f"or {fallback_path.name} (fallback) in {results_dir}"
        )

    df = pd.read_csv(src_path)
    if df is None or len(df) == 0:
        raise ValueError(f"{src_path.name} is empty.")

    df = sanitize_cols(df)
    df = normalize_required_fields(df)

    if "ticker" not in df.columns:
        raise ValueError(f"{src_path.name} has no ticker/symbol column.")

    # Determine "latest" row per ticker
    time_col = pick_time_col(df)
    if time_col:
        df["_dt"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    else:
        df["_dt"] = pd.NaT

    # Sort so tail(1) is latest per ticker
    df = df.sort_values(["ticker", "_dt"], ascending=[True, True])

    state = df.groupby("ticker", as_index=False).tail(1).copy()

    # Add freshness
    freshness, age_min, hb_src = compute_freshness(results_dir)
    state["freshness"] = freshness
    state["freshness_age_min"] = age_min if age_min is not None else ""
    state["freshness_source"] = hb_src

    # Add updated_utc
    state["updated_utc"] = utcnow().isoformat()

    # Add source trace
    state["state_source_file"] = src_path.name
    state["state_time_col"] = time_col if time_col else ""

    # Cleanup
    state = state.drop(columns=["_dt"], errors="ignore")

    # Minimal recommended column order (keep the rest too)
    preferred = [
        "ticker",
        "stance",
        "lifecycle_action",
        "last_action",
        "position_state",
        "signal",
        "confidence",
        "edge_pct",
        "delta_pct",
        "as_of_date",
        "date",
        "generated_at_utc",
        "freshness",
        "freshness_age_min",
        "freshness_source",
        "updated_utc",
        "state_source_file",
        "state_time_col",
        "rationale",
        "notes",
    ]
    cols = [c for c in preferred if c in state.columns] + [
        c for c in state.columns if c not in preferred
    ]
    state = state[cols]

    # Guarantee unique ticker (hard safety)
    # If any duplicates remain (shouldn't), keep the last after sort
    state = state.sort_values(["ticker"])
    state = state.drop_duplicates(subset=["ticker"], keep="last")

    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results", help="Path to data/results")
    ap.add_argument("--out", default="signal_lifecycle.csv", help="Output filename (STATE table)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    state = build_state(results_dir)

    out_path = results_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(out_path, index=False)

    print(f"Wrote STATE: {out_path}")
    print(f"Rows: {len(state):,} (unique tickers)")
    print(
        f"Freshness: {state['freshness'].iloc[0] if 'freshness' in state.columns and len(state) else 'n/a'}"
    )


if __name__ == "__main__":
    main()
