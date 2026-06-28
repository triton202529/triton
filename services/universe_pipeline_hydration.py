"""
universe_pipeline_hydration.py
Phase 197B — Sync governed tickers.txt membership into active processed/signal/lifecycle chain.

Uses existing pipeline modules only; never forces execution or bypasses gates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
TICKERS_FILE = ROOT / "data" / "config" / "tickers.txt"
RESULTS = ROOT / "data" / "results"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PREDICTIONS = ROOT / "data" / "predictions"

INDEX_PREFIX = "^"

HYDRATION_CHAIN = (
    ("fetch_fundamentals", [sys.executable, "-m", "services.fetch_fundamentals"]),
    ("score_stocks", [sys.executable, "-m", "services.score_stocks"]),
    ("train_model", [sys.executable, "-m", "services.train_model", "--min-rows", "30"]),
    ("generate_signals", [sys.executable, "-m", "services.generate_signals"]),
    ("apply_signal_lifecycle", [sys.executable, "-m", "services.apply_signal_lifecycle"]),
    (
        "build_effective_lifecycle",
        [sys.executable, "-m", "services.build_effective_lifecycle"],
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_governed_universe_tickers(*, include_indices: bool = True) -> List[str]:
    from services.fetch_raw_data import _load_tickers_from_file

    tickers = _load_tickers_from_file(TICKERS_FILE)
    if include_indices:
        return tickers
    return [t for t in tickers if not str(t).startswith(INDEX_PREFIX)]


def load_tradable_universe_tickers() -> List[str]:
    """Equity/ETF symbols from tickers.txt (excludes ^ index reference tickers)."""
    return load_governed_universe_tickers(include_indices=False)


def _csv_has_ticker(path: Path, ticker: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import pandas as pd

        df = pd.read_csv(path, nrows=5000)
        col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if not col:
            return False
        return bool((df[col].astype(str).str.upper() == ticker.upper()).any())
    except Exception:
        return False


def assess_symbol_pipeline_state(
    symbol: str,
    *,
    tradable_universe: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    tradable = set(tradable_universe or load_tradable_universe_tickers())
    raw_path = RAW / f"{sym}.csv"
    processed_path = PROCESSED / f"{sym}.parquet"
    prediction_path = PREDICTIONS / f"{sym}_predictions.parquet"

    state: Dict[str, Any] = {
        "symbol": sym,
        "in_config_universe": sym in tradable,
        "is_index_reference": sym.startswith(INDEX_PREFIX),
        "raw_exists": raw_path.is_file(),
        "processed_exists": processed_path.is_file(),
        "prediction_exists": prediction_path.is_file(),
        "in_signals": _csv_has_ticker(RESULTS / "signals_with_rationale.csv", sym)
        or _csv_has_ticker(RESULTS / "signals.csv", sym),
        "in_lifecycle": _csv_has_ticker(RESULTS / "signal_lifecycle.csv", sym),
        "in_effective_lifecycle": _csv_has_ticker(RESULTS / "signal_lifecycle_effective.csv", sym),
        "in_allocation": _csv_has_ticker(RESULTS / "portfolio_allocation_recommendations.csv", sym),
        "in_risk_overlay": _csv_has_ticker(RESULTS / "performance_risk_overlay.csv", sym),
    }
    state["hydrated"] = bool(
        state["in_config_universe"]
        and state["processed_exists"]
        and state["in_signals"]
        and state["in_effective_lifecycle"]
    )
    gaps: List[str] = []
    if state["in_config_universe"] and not state["raw_exists"]:
        gaps.append("raw_missing")
    if state["in_config_universe"] and not state["processed_exists"]:
        gaps.append("processed_missing")
    if state["in_config_universe"] and not state["in_signals"]:
        gaps.append("signals_missing")
    if state["in_config_universe"] and not state["in_effective_lifecycle"]:
        gaps.append("effective_lifecycle_missing")
    if state["in_config_universe"] and not state["in_allocation"]:
        gaps.append("allocation_missing")
    state["gaps"] = gaps
    return state


def assess_universe_hydration(
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    tradable = load_tradable_universe_tickers()
    target = [str(s).strip().upper() for s in (symbols or tradable) if str(s).strip()]
    per_symbol = [assess_symbol_pipeline_state(s, tradable_universe=tradable) for s in target]
    processed_count = len(list(PROCESSED.glob("*.parquet")))
    lifecycle_count = 0
    allocation_count = 0
    try:
        import pandas as pd

        lc_path = RESULTS / "signal_lifecycle_effective.csv"
        if lc_path.is_file():
            lc = pd.read_csv(lc_path, low_memory=False)
            col = "ticker" if "ticker" in lc.columns else "symbol"
            if col in lc.columns:
                lifecycle_count = int(lc[col].astype(str).str.upper().nunique())
        alloc_path = RESULTS / "portfolio_allocation_recommendations.csv"
        if alloc_path.is_file():
            al = pd.read_csv(alloc_path, low_memory=False)
            col = "ticker" if "ticker" in al.columns else "symbol"
            if col in al.columns:
                allocation_count = int(al[col].astype(str).str.upper().nunique())
    except Exception:
        pass

    stranded = [s["symbol"] for s in per_symbol if s.get("in_config_universe") and s.get("gaps")]
    return {
        "generated_at": _utc_now_iso(),
        "governed_tradable_count": len(tradable),
        "processed_parquet_count": processed_count,
        "lifecycle_universe_count": lifecycle_count,
        "allocation_universe_count": allocation_count,
        "stranded_symbols": stranded,
        "symbols": per_symbol,
    }


def _run_step(name: str, cmd: List[str], *, verbose: bool) -> Tuple[bool, Dict[str, Any]]:
    if verbose:
        print(f"[UNIVERSE_HYDRATE] step={name} cmd={' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        env=env,
    )
    detail = {
        "step": name,
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1200:],
        "stderr_tail": (proc.stderr or "")[-600:],
    }
    ok = proc.returncode == 0
    if verbose:
        status = "OK" if ok else "FAIL"
        print(f"[UNIVERSE_HYDRATE] step={name} status={status} rc={proc.returncode}", flush=True)
    return ok, detail


def ensure_raw_data(
    symbols: Sequence[str],
    *,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    missing = [s.upper() for s in symbols if s and not (RAW / f"{s.upper()}.csv").is_file()]
    if not missing:
        return True, {"skipped": True, "reason": "raw_present"}
    cmd = [
        sys.executable,
        "-m",
        "services.fetch_raw_data",
        "--tickers",
        ",".join(missing),
        "--min-ok",
        "0",
        "--min-asof-ok",
        "0",
        "--retries",
        "3",
        "--timeout",
        "45",
    ]
    return _run_step("fetch_raw", cmd, verbose=verbose)


def preprocess_symbols(
    symbols: Sequence[str],
    *,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Build data/processed/{SYMBOL}.parquet from existing raw CSVs."""
    import pandas as pd

    from run_preprocessing import get_as_of_date, normalize_date_series
    from services.preprocess_data import preprocess_stock_csv

    as_of_str = get_as_of_date()
    as_of_dt = pd.to_datetime(as_of_str, errors="coerce").normalize()
    written: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for sym in sorted({str(s).strip().upper() for s in symbols if s}):
        raw_path = RAW / f"{sym}.csv"
        if not raw_path.is_file():
            skipped.append(f"{sym}:raw_missing")
            continue
        try:
            df = preprocess_stock_csv(str(raw_path))
            if df.empty or "date" not in df.columns:
                skipped.append(f"{sym}:empty_after_preprocess")
                continue
            df = df.copy()
            df["date"] = normalize_date_series(df["date"])
            df = df.dropna(subset=["date"])
            last_dt = pd.to_datetime(df["date"].max(), errors="coerce")
            if pd.isna(last_dt):
                skipped.append(f"{sym}:invalid_dates")
                continue
            last_dt = last_dt.normalize()
            if last_dt < as_of_dt:
                skipped.append(f"{sym}:stale_last_date={last_dt.date()}")
                continue
            out = PROCESSED / f"{sym}.parquet"
            df.to_parquet(out, index=False)
            written.append(sym)
            if verbose:
                print(f"[UNIVERSE_HYDRATE] processed {sym} rows={len(df)}", flush=True)
        except Exception as exc:
            errors.append(f"{sym}:{exc}")

    ok = bool(written) and not errors
    return ok, {
        "step": "preprocess_symbols",
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "as_of_date": as_of_str,
    }


def hydrate_universe_symbols(
    symbols: Optional[Sequence[str]] = None,
    *,
    verbose: bool = False,
    skip_raw_fetch: bool = False,
) -> Dict[str, Any]:
    """
    Run the standard signal pipeline chain so configured symbols reach
    processed → fundamentals/scores → train → signals → lifecycle.
    """
    target = [
        str(s).strip().upper()
        for s in (symbols or load_tradable_universe_tickers())
        if str(s).strip()
    ]
    before = assess_universe_hydration(target)
    steps_run: List[Dict[str, Any]] = []

    if not skip_raw_fetch:
        ok, detail = ensure_raw_data(target, verbose=verbose)
        steps_run.append(detail)
        if not ok:
            return {
                "ok": False,
                "reason": "fetch_raw_failed",
                "before": before,
                "steps": steps_run,
            }

    ok, detail = preprocess_symbols(target, verbose=verbose)
    steps_run.append(detail)
    if not ok:
        return {
            "ok": False,
            "reason": "preprocess_failed",
            "before": before,
            "steps": steps_run,
        }

    for step_name, cmd in HYDRATION_CHAIN:
        ok, detail = _run_step(step_name, cmd, verbose=verbose)
        steps_run.append(detail)
        if not ok and step_name in ("preprocess", "generate_signals", "apply_signal_lifecycle"):
            return {
                "ok": False,
                "reason": f"{step_name}_failed",
                "before": before,
                "steps": steps_run,
            }

    after = assess_universe_hydration(target)
    target_states = {s["symbol"]: s for s in after.get("symbols") or []}
    all_hydrated = all(target_states.get(s, {}).get("hydrated") for s in target)
    return {
        "ok": all_hydrated,
        "reason": "OK" if all_hydrated else "partial_hydration",
        "before": before,
        "after": after,
        "steps": steps_run,
        "symbols": target,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Hydrate governed universe symbols into signal pipeline"
    )
    ap.add_argument("--symbols", default="", help="Comma-separated symbols (default: all tradable)")
    ap.add_argument("--assess-only", action="store_true")
    ap.add_argument("--skip-raw-fetch", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    )
    if args.assess_only:
        report = assess_universe_hydration(symbols)
        print(json.dumps(report, indent=2, default=str))
        return 0

    result = hydrate_universe_symbols(
        symbols,
        verbose=bool(args.verbose),
        skip_raw_fetch=bool(args.skip_raw_fetch),
    )
    print(json.dumps({"ok": result.get("ok"), "reason": result.get("reason")}, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
