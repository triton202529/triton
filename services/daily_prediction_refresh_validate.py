"""
daily_prediction_refresh_validate.py

Validate that a daily prediction-refresh run advanced raw / processed /
prediction artifacts to the required last completed NYSE session.

No broker, network, paper-arm, or execution side effects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tickers(tickers_file: Path) -> List[str]:
    from services.fetch_raw_data import _load_tickers_from_file

    return list(_load_tickers_from_file(tickers_file))


def _max_date_from_file(path: Path) -> Optional[date]:
    import pandas as pd

    if not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception:
        return None
    col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if not col:
        return None
    d = pd.to_datetime(df[col], errors="coerce").max()
    if pd.isna(d):
        return None
    return d.date()


def _skipped_insufficient_history(skipped_path: Path) -> set[str]:
    out: set[str] = set()
    if not skipped_path.is_file():
        return out
    try:
        import pandas as pd

        df = pd.read_csv(skipped_path)
        if "ticker" not in df.columns:
            return out
        reason_col = "reason" if "reason" in df.columns else None
        for _, row in df.iterrows():
            t = str(row.get("ticker") or "").strip().upper()
            if not t:
                continue
            reason = str(row.get(reason_col) or "") if reason_col else ""
            if "too few rows" in reason.lower() or "empty dataset" in reason.lower():
                # Only equity-like skipped symbols (exclude ARM_* memory junk noise).
                if t.startswith("ARM_") or t.endswith("_MEMORY") or t.endswith("_HISTORY"):
                    continue
                out.add(t)
    except Exception:
        return out
    return out


def assess_layer(
    *,
    tickers: Sequence[str],
    path_for,
    target: date,
    allowed_missing: Optional[set[str]] = None,
) -> Dict[str, Any]:
    allowed_missing = allowed_missing or set()
    dist: Counter[str] = Counter()
    missing: List[str] = []
    stale: List[str] = []
    schema_invalid: List[str] = []
    ok = 0
    for t in tickers:
        p = path_for(t)
        if not p.is_file():
            if t in allowed_missing:
                continue
            missing.append(t)
            continue
        d = _max_date_from_file(p)
        if d is None:
            schema_invalid.append(t)
            continue
        dist[d.isoformat()] += 1
        if d == target:
            ok += 1
        else:
            stale.append(t)
    expected = [t for t in tickers if t not in allowed_missing]
    return {
        "expected_count": len(expected),
        "ok_count": ok,
        "missing": missing,
        "stale": stale,
        "schema_invalid": schema_invalid,
        "max_date_distribution": dict(dist),
        "passed": ok == len(expected) and not missing and not stale and not schema_invalid,
    }


def validate_refresh(
    *,
    repo_root: Path,
    target_session: Optional[date] = None,
    run_id: str = "",
    tickers_file: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    from services.market_calendar import last_completed_nyse_session

    target = target_session or last_completed_nyse_session()
    tickers_path = tickers_file or (repo_root / "data" / "config" / "tickers.txt")
    tickers = _load_tickers(tickers_path)
    skipped = _skipped_insufficient_history(repo_root / "data" / "results" / "skipped_tickers.csv")
    # Predictions may omit index refs and policy-skipped symbols like SPCX.
    pred_allowed_missing = set(skipped) | {t for t in tickers if t.startswith("^")}

    raw = assess_layer(
        tickers=tickers,
        path_for=lambda t: repo_root / "data" / "raw" / f"{t}.csv",
        target=target,
    )
    processed = assess_layer(
        tickers=tickers,
        path_for=lambda t: repo_root / "data" / "processed" / f"{t}.parquet",
        target=target,
    )
    predictions = assess_layer(
        tickers=tickers,
        path_for=lambda t: repo_root / "data" / "predictions" / f"{t}_predictions.parquet",
        target=target,
        allowed_missing=pred_allowed_missing,
    )

    # Freshness of skipped_tickers / model_comparison for this run (mtime today UTC optional).
    skipped_path = repo_root / "data" / "results" / "skipped_tickers.csv"
    model_cmp = repo_root / "data" / "results" / "model_comparison.csv"
    train_artifacts_present = skipped_path.is_file() and model_cmp.is_file()

    passed = bool(
        raw["passed"] and processed["passed"] and predictions["passed"] and train_artifacts_present
    )
    result: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id or "",
        "framework": "daily_prediction_refresh_validate",
        "target_session": target.isoformat(),
        "ticker_count": len(tickers),
        "raw": raw,
        "processed": processed,
        "predictions": predictions,
        "policy_skipped_predictions": sorted(skipped),
        "train_artifacts_present": train_artifacts_present,
        "passed": passed,
        "broker_called": False,
        "network_called": False,
    }
    if not passed:
        reasons = []
        if not raw["passed"]:
            reasons.append("raw_not_advanced")
        if not processed["passed"]:
            reasons.append("processed_not_advanced")
        if not predictions["passed"]:
            reasons.append("predictions_not_advanced")
        if not train_artifacts_present:
            reasons.append("train_artifacts_missing")
        result["failure_reasons"] = reasons
    return result


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate daily prediction refresh advancement")
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--target-session", default=None, help="YYYY-MM-DD override")
    ap.add_argument("--run-id", default="")
    ap.add_argument(
        "--summary-out",
        default="",
        help="Optional path for fresh run summary JSON (default: data/results/daily_prediction_refresh_summary.json)",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo = Path(args.repo_root)
    target = None
    if args.target_session:
        target = date.fromisoformat(str(args.target_session))
    result = validate_refresh(repo_root=repo, target_session=target, run_id=str(args.run_id or ""))

    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else (repo / "data" / "results" / "daily_prediction_refresh_summary.json")
    )
    _atomic_write_json(summary_path, result)
    result["summary_path"] = str(summary_path)

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        print(
            f"[daily_prediction_refresh_validate] passed={result['passed']} "
            f"target={result['target_session']} run_id={result.get('run_id')}"
        )
        if not result["passed"]:
            print(f"[daily_prediction_refresh_validate] reasons={result.get('failure_reasons')}")

    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
