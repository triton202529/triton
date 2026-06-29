"""
symbol_hydration_quarantine.py
Phase 197D — Classify configured symbols that cannot yet enter the active signal chain.

Quarantined symbols are excluded from hydration failure counts and execution paths.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PREDICTIONS = ROOT / "data" / "predictions"
UNIVERSE_ASSETS = ROOT / "data" / "config" / "universe_assets.json"
QUARANTINE_REGISTRY = ROOT / "data" / "governance" / "symbol_hydration_quarantine.json"

TRAIN_MIN_ROWS = int(os.environ.get("TRITON_HYDRATION_TRAIN_MIN_ROWS", "30"))
IPO_HISTORY_MAX_ROWS = int(os.environ.get("TRITON_HYDRATION_IPO_MAX_ROWS", "60"))

QUARANTINE_CLASSIFICATIONS = (
    "QUARANTINED_UNSUPPORTED",
    "QUARANTINED_INSUFFICIENT_HISTORY",
    "QUARANTINED_IPO_EDGE",
    "QUARANTINED_VENDOR_DATA_UNAVAILABLE",
    "QUARANTINED_NOT_TRADABLE",
    "QUARANTINED_PENDING_DATA",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_universe_meta(symbol: str) -> Dict[str, Any]:
    if not UNIVERSE_ASSETS.is_file():
        return {}
    try:
        doc = json.loads(UNIVERSE_ASSETS.read_text(encoding="utf-8"))
        return dict((doc.get("assets") or {}).get(symbol.upper()) or {})
    except Exception:
        return {}


def _raw_row_count(symbol: str) -> int:
    path = RAW / f"{symbol.upper()}.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _processed_row_count(symbol: str) -> int:
    path = PROCESSED / f"{symbol.upper()}.parquet"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_parquet(path))
    except Exception:
        return 0


def _skipped_ticker_reason(symbol: str) -> str:
    path = RESULTS / "skipped_tickers.csv"
    if not path.is_file():
        return ""
    try:
        import pandas as pd

        df = pd.read_csv(path, low_memory=False)
        if "ticker" not in df.columns:
            return ""
        sub = df[df["ticker"].astype(str).str.upper() == symbol.upper()]
        if sub.empty:
            return ""
        return str(sub.iloc[-1].get("reason") or "")
    except Exception:
        return ""


def classify_symbol_quarantine(symbol: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Return (classification, reason, details).
    classification is None when symbol is actionable for hydration.
    """
    sym = str(symbol or "").strip().upper()
    details: Dict[str, Any] = {"symbol": sym}
    meta = _load_universe_meta(sym)
    details["universe_metadata"] = meta

    raw_rows = _raw_row_count(sym)
    proc_rows = _processed_row_count(sym)
    details["raw_rows"] = raw_rows
    details["processed_rows"] = proc_rows
    details["train_min_rows"] = TRAIN_MIN_ROWS

    if not (RAW / f"{sym}.csv").is_file():
        return (
            "QUARANTINED_VENDOR_DATA_UNAVAILABLE",
            "raw_data_missing",
            details,
        )

    if raw_rows == 0:
        return (
            "QUARANTINED_VENDOR_DATA_UNAVAILABLE",
            "raw_data_empty",
            details,
        )

    if raw_rows < TRAIN_MIN_ROWS or proc_rows < TRAIN_MIN_ROWS:
        if raw_rows <= IPO_HISTORY_MAX_ROWS:
            return (
                "QUARANTINED_IPO_EDGE",
                f"insufficient_history raw_rows={raw_rows} processed_rows={proc_rows} "
                f"train_min_rows={TRAIN_MIN_ROWS}",
                details,
            )
        return (
            "QUARANTINED_INSUFFICIENT_HISTORY",
            f"insufficient_history raw_rows={raw_rows} processed_rows={proc_rows} "
            f"train_min_rows={TRAIN_MIN_ROWS}",
            details,
        )

    skip_reason = _skipped_ticker_reason(sym)
    if skip_reason:
        details["skipped_ticker_reason"] = skip_reason
        if "Missing fundamentals or score" in skip_reason:
            return (
                "QUARANTINED_UNSUPPORTED",
                skip_reason,
                details,
            )
        if "Too few rows after lagging" in skip_reason or "Empty dataset" in skip_reason:
            return (
                "QUARANTINED_INSUFFICIENT_HISTORY",
                skip_reason,
                details,
            )

    pred_path = PREDICTIONS / f"{sym}_predictions.parquet"
    if proc_rows >= TRAIN_MIN_ROWS and not pred_path.is_file():
        # Processed history exists but signal chain never produced predictions.
        return (
            "QUARANTINED_PENDING_DATA",
            "processed_history_present_predictions_missing",
            details,
        )

    return None, "", details


def evaluate_quarantine_for_symbols(symbols: Sequence[str]) -> Dict[str, Any]:
    quarantined: Dict[str, Dict[str, Any]] = {}
    actionable: List[str] = []
    for sym in sorted({str(s).strip().upper() for s in symbols if s}):
        classification, reason, details = classify_symbol_quarantine(sym)
        if classification:
            quarantined[sym] = {
                "classification": classification,
                "reason": reason,
                "details": details,
            }
        else:
            actionable.append(sym)
    return {
        "quarantined_symbols": sorted(quarantined.keys()),
        "quarantine_reasons_by_symbol": {
            sym: f"{info['classification']}:{info['reason']}" for sym, info in quarantined.items()
        },
        "quarantine_details_by_symbol": quarantined,
        "actionable_stranded_symbols": sorted(actionable),
    }


def apply_quarantine_to_gaps(gaps: Dict[str, Any]) -> Dict[str, Any]:
    """Split stranded symbols into quarantined vs actionable hydration targets."""
    stranded = list(gaps.get("stranded_symbols") or [])
    q = evaluate_quarantine_for_symbols(stranded)
    actionable = list(q["actionable_stranded_symbols"])
    out = dict(gaps)
    out["quarantined_symbols"] = list(q["quarantined_symbols"])
    out["quarantine_reasons_by_symbol"] = dict(q["quarantine_reasons_by_symbol"])
    out["quarantine_details_by_symbol"] = dict(q["quarantine_details_by_symbol"])
    out["actionable_stranded_symbols"] = actionable
    out["stranded_symbols"] = actionable
    out["no_actionable_hydration_needed"] = not actionable
    out["hydration_needed"] = bool(actionable)
    return out


def persist_quarantine_registry(quarantine_details: Dict[str, Dict[str, Any]]) -> None:
    doc = {
        "generated_at": _utc_now_iso(),
        "framework": "Phase-197D",
        "symbols": quarantine_details,
    }
    QUARANTINE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUARANTINE_REGISTRY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, QUARANTINE_REGISTRY)
