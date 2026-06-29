"""
runtime_universe_hydration_preflight.py
Phase 197C — Fast runtime check for stranded governed-universe symbols.

Runs before allocation/paper-entry in the market-open sequence.
Never submits broker orders or changes execution authorization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PROCESSED = ROOT / "data" / "processed"
AUDIT_JSON = RESULTS / "runtime_universe_hydration_preflight.json"

PREVIOUS_PHASE = "197B"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_csv_ticker_set(path: Path) -> Set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    try:
        import pandas as pd

        df = pd.read_csv(path, low_memory=False)
        col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if not col:
            return set()
        return {
            str(x).strip().upper()
            for x in df[col].dropna().astype(str)
            if str(x).strip() and str(x).strip().upper() not in ("NAN", "NONE")
        }
    except Exception:
        return set()


def _processed_ticker_set() -> Set[str]:
    if not PROCESSED.is_dir():
        return set()
    return {
        p.stem.upper()
        for p in PROCESSED.glob("*.parquet")
        if p.stem.upper() not in ("STOCK_DATA", "STOCK_DATA_MERGED")
    }


def detect_hydration_gaps(
    *,
    tradable: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fast set-based gap scan (no per-symbol subprocess)."""
    from services.universe_pipeline_hydration import load_tradable_universe_tickers

    configured = sorted(
        {str(s).strip().upper() for s in (tradable or load_tradable_universe_tickers()) if s}
    )
    configured_set = set(configured)
    processed_set = _processed_ticker_set()
    lifecycle_set = _load_csv_ticker_set(RESULTS / "signal_lifecycle_effective.csv")
    allocation_set = _load_csv_ticker_set(RESULTS / "portfolio_allocation_recommendations.csv")
    signals_set = _load_csv_ticker_set(
        RESULTS / "signals_with_rationale.csv"
    ) | _load_csv_ticker_set(RESULTS / "signals.csv")

    missing_processed = sorted(configured_set - processed_set)
    missing_lifecycle = sorted(configured_set - lifecycle_set)
    missing_allocation = sorted(configured_set - allocation_set)
    missing_signals = sorted(configured_set - signals_set)

    critical: List[str] = []
    for sym in configured:
        if sym not in processed_set or sym not in lifecycle_set or sym not in signals_set:
            critical.append(sym)
    critical = sorted(set(critical))

    allocation_only = sorted(
        s for s in configured if s in lifecycle_set and s in signals_set and s not in allocation_set
    )

    return {
        "configured_symbols": configured,
        "configured_symbol_count": len(configured),
        "processed_symbol_count": len(processed_set & configured_set),
        "lifecycle_symbol_count": len(lifecycle_set & configured_set),
        "allocation_symbol_count": len(allocation_set & configured_set),
        "missing_processed_symbols": missing_processed,
        "missing_lifecycle_symbols": missing_lifecycle,
        "missing_allocation_symbols": missing_allocation,
        "missing_signals_symbols": missing_signals,
        "stranded_symbols": critical,
        "allocation_refresh_only_symbols": allocation_only,
        "hydration_needed": bool(critical),
    }


def _market_open() -> bool:
    from services.guard_auto import is_market_open_et

    return bool(is_market_open_et(datetime.now(timezone.utc)))


def _live_execution_permitted() -> bool:
    try:
        doc = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
        return bool(doc.get("live_execution_permitted", True))
    except Exception:
        return True


def _allow_expensive_during_market_open() -> bool:
    return os.environ.get("TRITON_RUNTIME_HYDRATION_ALLOW_MARKET_OPEN", "").strip() in (
        "1",
        "true",
        "yes",
    )


def run_runtime_hydration_preflight(
    *,
    verbose: bool = False,
    force: bool = False,
    check_only: bool = False,
) -> Dict[str, Any]:
    from services.universe_pipeline_hydration import (
        ensure_raw_data,
        hydrate_universe_symbols,
        preprocess_symbols,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    gaps = detect_hydration_gaps()
    market_open = _market_open()

    audit: Dict[str, Any] = {
        "phase": "197C",
        "timestamp": _utc_now_iso(),
        "framework": "runtime_universe_hydration_preflight",
        **gaps,
        "market_open": market_open,
        "hydration_attempted": False,
        "hydrated_symbols": [],
        "deferred_symbols": [],
        "failed_symbols": [],
        "broker_orders_submitted": False,
        "live_execution_permitted": _live_execution_permitted() or load_live_execution_flag(),
        "runtime_authority_unchanged": True,
        "continuous_loop_interval_minutes": 45,
    }

    if not gaps["hydration_needed"]:
        audit["status"] = "SKIPPED_NO_GAPS"
        audit["reason"] = "all configured tradable symbols hydrated in processed/signals/lifecycle"
        _atomic_write_json(AUDIT_JSON, audit)
        if verbose:
            print(
                f"[RUNTIME_HYDRATION_PREFLIGHT] status=SKIPPED_NO_GAPS "
                f"configured={gaps['configured_symbol_count']} "
                f"processed={gaps['processed_symbol_count']} "
                f"lifecycle={gaps['lifecycle_symbol_count']}",
                flush=True,
            )
        return audit

    stranded = list(gaps["stranded_symbols"])
    audit["hydration_needed"] = True

    if check_only:
        audit["status"] = "GAPS_DETECTED"
        audit["reason"] = f"stranded={','.join(stranded)}"
        _atomic_write_json(AUDIT_JSON, audit)
        if verbose:
            print(f"[RUNTIME_HYDRATION_PREFLIGHT] check_only stranded={stranded}", flush=True)
        return audit

    needs_expensive = any(
        sym in gaps["missing_lifecycle_symbols"] or sym in gaps["missing_signals_symbols"]
        for sym in stranded
    )
    defer_expensive = (
        needs_expensive and market_open and not force and not _allow_expensive_during_market_open()
    )

    audit["hydration_attempted"] = True
    hydrated: List[str] = []
    deferred: List[str] = []
    failed: List[str] = []

    cheap_targets = [
        s
        for s in stranded
        if s in gaps["missing_processed_symbols"] or not (PROCESSED / f"{s}.parquet").is_file()
    ]
    if cheap_targets:
        ok_raw, _ = ensure_raw_data(cheap_targets, verbose=verbose)
        ok_pre, pre_detail = preprocess_symbols(cheap_targets, verbose=verbose)
        written = list(pre_detail.get("written") or [])
        hydrated.extend(written)
        for sym in cheap_targets:
            if sym not in written and (sym in (pre_detail.get("errors") or []) or not ok_pre):
                failed.append(sym)
        if verbose:
            print(
                f"[RUNTIME_HYDRATION_PREFLIGHT] cheap_hydration written={written}",
                flush=True,
            )

    signal_chain_targets = [
        s
        for s in stranded
        if s in gaps["missing_lifecycle_symbols"] or s in gaps["missing_signals_symbols"]
    ]

    if signal_chain_targets and defer_expensive:
        deferred.extend(signal_chain_targets)
        audit["status"] = "DEFERRED_MARKET_OPEN"
        audit["reason"] = (
            "signal-chain hydration deferred while market open; "
            f"symbols={','.join(signal_chain_targets)}"
        )
    elif signal_chain_targets:
        result = hydrate_universe_symbols(
            signal_chain_targets,
            verbose=verbose,
            skip_raw_fetch=True,
        )
        after = result.get("after") or {}
        for sym in signal_chain_targets:
            states = {s["symbol"]: s for s in (after.get("symbols") or [])}
            if states.get(sym, {}).get("hydrated"):
                if sym not in hydrated:
                    hydrated.append(sym)
            else:
                failed.append(sym)
        if result.get("ok"):
            audit["status"] = "HYDRATED"
            audit["reason"] = f"hydrated={','.join(sorted(set(hydrated)))}"
        elif hydrated:
            audit["status"] = "PARTIAL"
            audit["reason"] = (
                f"partial_hydration hydrated={','.join(sorted(set(hydrated)))} "
                f"failed={','.join(sorted(set(failed)))}"
            )
        else:
            audit["status"] = "HYDRATION_INCOMPLETE"
            audit["reason"] = str(result.get("reason") or "hydration_incomplete")
    elif cheap_targets and not signal_chain_targets:
        audit["status"] = "PREPROCESSED_ONLY"
        audit["reason"] = f"processed={','.join(hydrated)}; signal chain not required"
    elif not signal_chain_targets and not cheap_targets:
        audit["status"] = "SKIPPED_NO_GAPS"
        audit["reason"] = "no actionable hydration targets"
    else:
        audit["status"] = "HYDRATION_INCOMPLETE"
        audit["reason"] = "hydration produced no symbols"

    audit["hydrated_symbols"] = sorted(set(hydrated))
    audit["deferred_symbols"] = sorted(set(deferred))
    audit["failed_symbols"] = sorted(set(failed))

    post = detect_hydration_gaps()
    audit["postflight"] = {
        "processed_symbol_count": post["processed_symbol_count"],
        "lifecycle_symbol_count": post["lifecycle_symbol_count"],
        "allocation_symbol_count": post["allocation_symbol_count"],
        "stranded_symbols": post["stranded_symbols"],
    }

    _atomic_write_json(AUDIT_JSON, audit)
    if verbose:
        print(
            f"[RUNTIME_HYDRATION_PREFLIGHT] status={audit['status']} reason={audit['reason']}",
            flush=True,
        )
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Runtime universe hydration preflight")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="Run expensive hydration even if market open"
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    audit = run_runtime_hydration_preflight(
        verbose=bool(args.verbose),
        force=bool(args.force),
        check_only=bool(args.check_only),
    )
    status = str(audit.get("status") or "")
    non_fatal = {
        "SKIPPED_NO_GAPS",
        "HYDRATED",
        "PREPROCESSED_ONLY",
        "DEFERRED_MARKET_OPEN",
        "GAPS_DETECTED",
        "PARTIAL",
        "HYDRATION_INCOMPLETE",
    }
    if status in non_fatal:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
