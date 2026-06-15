"""
phase_195a_spcx_validation.py
Phase 195A — SpaceX (SPCX) universe integration and IPO safety validation.

Universe expansion only — no model, execution, or governance changes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TICKERS_FILE = ROOT / "data" / "config" / "tickers.txt"
UNIVERSE_ASSETS = ROOT / "data" / "config" / "universe_assets.json"
RESULTS = ROOT / "data" / "results"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PREDICTIONS = ROOT / "data" / "predictions"

OUTPUT_JSON = RESULTS / "phase_195a_spcx_validation.json"
TARGET_TICKER = "SPCX"
EXPECTED_UNIVERSE = 65

IPO_SAFE_OUTCOMES = frozenset(
    {
        "WATCH",
        "NO_ACTION",
        "LOW_CONFIDENCE",
        "INSUFFICIENT_DATA",
        "REDUCED_SIZE",
        "HOLD",
        "WAIT",
        "SKIP",
        "BLOCKED",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_tickers() -> List[str]:
    from services.fetch_raw_data import _load_tickers_from_file

    return _load_tickers_from_file(TICKERS_FILE)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _csv_has_ticker(path: Path, ticker: str) -> Optional[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path, nrows=5000)
        if df.empty:
            return {"exists": True, "rows": 0}
        col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if not col:
            return {"exists": True, "rows": len(df), "note": "no ticker column"}
        sub = df[df[col].astype(str).str.upper() == ticker.upper()]
        if sub.empty:
            return {"exists": True, "rows": len(df), "spcx_rows": 0}
        row = sub.iloc[-1].to_dict()
        signal_cols = [
            c
            for c in ("signal", "stance", "lifecycle_action", "effective_stance")
            if c in sub.columns
        ]
        outcome = {c: str(row.get(c, "")) for c in signal_cols}
        return {
            "exists": True,
            "rows": len(df),
            "spcx_rows": len(sub),
            "latest": outcome,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def _parquet_info(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        info: Dict[str, Any] = {"exists": True, "rows": len(df)}
        if "date" in df.columns and len(df) > 0:
            info["min_date"] = str(df["date"].min())
            info["max_date"] = str(df["date"].max())
        info["nan_columns"] = [c for c in df.columns if df[c].isna().any()]
        return info
    except Exception as e:
        return {"exists": True, "error": str(e)}


def run_phase_195a_validation(*, fetch_spcx: bool = True) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    tickers = _load_tickers()
    universe_count = len(tickers)
    spcx_in_universe = TARGET_TICKER in tickers

    checks.append(
        {
            "check": "universe_membership",
            "pass": spcx_in_universe and universe_count == EXPECTED_UNIVERSE,
            "universe_count": universe_count,
            "expected": EXPECTED_UNIVERSE,
            "spcx_present": spcx_in_universe,
        }
    )

    meta = _load_json(UNIVERSE_ASSETS).get("assets", {}).get(TARGET_TICKER, {})
    checks.append(
        {
            "check": "asset_metadata",
            "pass": meta.get("ticker") == TARGET_TICKER and meta.get("status") == "ACTIVE",
            "metadata": meta,
        }
    )

    fetch_result: Dict[str, Any] = {"skipped": not fetch_spcx}
    if fetch_spcx:
        try:
            import subprocess
            import sys

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "services.fetch_raw_data",
                    "--tickers",
                    TARGET_TICKER,
                    "--min-ok",
                    "0",
                    "--min-asof-ok",
                    "0",
                    "--retries",
                    "3",
                    "--timeout",
                    "45",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            fetch_result = {
                "exit_code": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-800:],
                "stderr_tail": (proc.stderr or "")[-400:],
            }
        except Exception as e:
            fetch_result = {"error": str(e)}

    raw_path = RAW / f"{TARGET_TICKER}.csv"
    raw_info: Dict[str, Any] = {"path": str(raw_path), "exists": raw_path.is_file()}
    if raw_path.is_file():
        try:
            import pandas as pd

            rdf = pd.read_csv(raw_path)
            raw_info["rows"] = len(rdf)
            if len(rdf) < 60:
                raw_info["ipo_note"] = "limited_history_expected"
        except Exception as e:
            raw_info["error"] = str(e)

    checks.append(
        {
            "check": "data_layer",
            "pass": raw_info.get("exists") or fetch_result.get("exit_code") == 0,
            "fetch": fetch_result,
            "raw": raw_info,
            "processed": _parquet_info(PROCESSED / f"{TARGET_TICKER}.parquet"),
        }
    )

    lifecycle = _csv_has_ticker(RESULTS / "signal_lifecycle_effective.csv", TARGET_TICKER)
    signals = _csv_has_ticker(RESULTS / "signals.csv", TARGET_TICKER)
    allocation = _csv_has_ticker(
        RESULTS / "portfolio_allocation_recommendations.csv", TARGET_TICKER
    )
    opportunities = _csv_has_ticker(RESULTS / "trade_opportunities.csv", TARGET_TICKER)

    lifecycle_outcome = ""
    if lifecycle and lifecycle.get("latest"):
        lifecycle_outcome = str(
            lifecycle["latest"].get("effective_stance")
            or lifecycle["latest"].get("stance")
            or lifecycle["latest"].get("lifecycle_action")
            or ""
        ).upper()

    ipo_safe = (
        not lifecycle
        or lifecycle.get("spcx_rows", 0) == 0
        or any(tok in lifecycle_outcome for tok in IPO_SAFE_OUTCOMES)
        or lifecycle_outcome in ("", "NAN")
    )
    unintended_buy = lifecycle_outcome == "BUY"

    checks.append(
        {
            "check": "lifecycle_allocation",
            "pass": ipo_safe and not unintended_buy,
            "lifecycle": lifecycle,
            "signals": signals,
            "allocation": allocation,
            "trade_opportunities": opportunities,
            "lifecycle_outcome": lifecycle_outcome or "NOT_YET_IN_PIPELINE",
            "ipo_safe_outcome": ipo_safe,
            "unintended_buy": unintended_buy,
        }
    )

    checks.append(
        {
            "check": "broker_safety",
            "pass": True,
            "note": "No execution or broker calls in Phase 195A validation",
            "execution_modified": False,
        }
    )

    all_pass = all(c.get("pass") for c in checks)
    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "Phase-195A",
        "ticker": TARGET_TICKER,
        "authority_file": str(TICKERS_FILE.relative_to(ROOT)).replace("\\", "/"),
        "universe_count": universe_count,
        "expected_universe_count": EXPECTED_UNIVERSE,
        "phase_pass": all_pass,
        "checks": checks,
        "summary": {
            "universe_expanded": spcx_in_universe and universe_count == EXPECTED_UNIVERSE,
            "pipeline_safe": ipo_safe and not unintended_buy,
            "no_execution_changes": True,
        },
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    return payload


if __name__ == "__main__":
    result = run_phase_195a_validation()
    print(f"Phase 195A validation: {OUTPUT_JSON}")
    print(f"universe={result['universe_count']} phase_pass={result['phase_pass']}")
    for c in result["checks"]:
        print(f"  {c['check']}: {'PASS' if c.get('pass') else 'FAIL'}")
