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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


def _pipeline_symbol_sets() -> Tuple[Set[str], Set[str]]:
    lifecycle: Set[str] = set()
    signals: Set[str] = set()
    for path, target in (
        (RESULTS / "signal_lifecycle_effective.csv", lifecycle),
        (RESULTS / "signals_with_rationale.csv", signals),
        (RESULTS / "signals.csv", signals),
    ):
        if not path.is_file():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(path, low_memory=False)
            col = (
                "ticker"
                if "ticker" in df.columns
                else ("symbol" if "symbol" in df.columns else None)
            )
            if not col:
                continue
            target.update(
                str(x).strip().upper()
                for x in df[col].dropna().astype(str)
                if str(x).strip() and str(x).strip().upper() not in ("NAN", "NONE")
            )
        except Exception:
            continue
    return lifecycle, signals


def classify_symbol_quarantine(symbol: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Return (classification, reason, details).
    classification is None when symbol is actionable for hydration/repair.
    """
    sym = str(symbol or "").strip().upper()
    details: Dict[str, Any] = {"symbol": sym}
    meta = _load_universe_meta(sym)
    details["universe_metadata"] = meta
    lifecycle_set, signals_set = _pipeline_symbol_sets()
    details["in_lifecycle"] = sym in lifecycle_set
    details["in_signals"] = sym in signals_set

    if sym in lifecycle_set or sym in signals_set:
        details["artifact_gap"] = "PIPELINE_ARTIFACT_MISSING"
        return None, "", details

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

    if raw_rows >= TRAIN_MIN_ROWS:
        details["artifact_gap"] = "PIPELINE_ARTIFACT_MISSING"
        return None, "", details

    if raw_rows <= IPO_HISTORY_MAX_ROWS:
        return (
            "QUARANTINED_IPO_EDGE",
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

    return (
        "QUARANTINED_INSUFFICIENT_HISTORY",
        f"insufficient_history raw_rows={raw_rows} processed_rows={proc_rows} "
        f"train_min_rows={TRAIN_MIN_ROWS}",
        details,
    )


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


REGISTRY_FRAMEWORK = "Phase-197D"


def load_quarantine_registry() -> Dict[str, Any]:
    if not QUARANTINE_REGISTRY.is_file():
        return {}
    try:
        return json.loads(QUARANTINE_REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_symbols_for_compare(symbols: Dict[str, Any]) -> str:
    return json.dumps(symbols or {}, sort_keys=True, default=str)


def registry_substance_unchanged(
    existing: Dict[str, Any],
    quarantine_details: Dict[str, Dict[str, Any]],
) -> bool:
    """True when symbol quarantine content matches existing registry (ignores generated_at)."""
    if not existing:
        return False
    if str(existing.get("framework") or "") != REGISTRY_FRAMEWORK:
        return False
    return _normalize_symbols_for_compare(
        existing.get("symbols")
    ) == _normalize_symbols_for_compare(quarantine_details)


def persist_quarantine_registry(quarantine_details: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Write governance registry only when substantive quarantine content changes.
    Runtime timestamps belong in runtime_universe_hydration_preflight.json, not here.
    """
    rel_path = str(QUARANTINE_REGISTRY.relative_to(ROOT)).replace("\\", "/")
    existing = load_quarantine_registry()
    if registry_substance_unchanged(existing, quarantine_details):
        return {
            "written": False,
            "skipped": True,
            "skip_reason": "unchanged_substance",
            "path": rel_path,
        }

    doc = {
        "generated_at": _utc_now_iso(),
        "framework": REGISTRY_FRAMEWORK,
        "symbols": quarantine_details,
    }
    QUARANTINE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2, default=str)
    if QUARANTINE_REGISTRY.is_file():
        prior = QUARANTINE_REGISTRY.read_bytes()
        ends_with_newline = prior.endswith(b"\n") or prior.endswith(b"\r\n")
    else:
        ends_with_newline = True
    if ends_with_newline and not payload.endswith("\n"):
        payload += "\n"
    tmp = QUARANTINE_REGISTRY.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, QUARANTINE_REGISTRY)
    return {
        "written": True,
        "skipped": False,
        "skip_reason": None,
        "reason": "substantive_quarantine_change",
        "path": rel_path,
    }
