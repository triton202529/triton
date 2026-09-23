"""
daily_prediction_refresh_validate.py

Validate daily prediction-refresh coverage and current-run output advancement.

Contract (accepted):
  raw required        = full canonical universe (66)
  processed required  = full canonical universe (66)
  prediction required = universe minus SPCX only (65), including ^ index refs
  SPCX                = sole established insufficient-history exclusion

No broker, network, paper-arm, or execution side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_EXCLUSION_SPCX = "SPCX"
INDEX_REFERENCE_SYMBOLS = ("^DJI", "^GSPC", "^IXIC", "^VIX")
MIN_TRADABLE_COVERAGE = 0.90
MISSING_EXPECTED_SESSION_BAR = "MISSING_EXPECTED_SESSION_BAR"
INPUT_SKIPS_REL = "data/results/daily_refresh_input_skips.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso_dt(value: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_tickers(tickers_file: Path) -> List[str]:
    from services.fetch_raw_data import _load_tickers_from_file

    return list(_load_tickers_from_file(tickers_file))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_owner_evidence_json(path: Path) -> Any:
    """Load owner-generated current-run JSON evidence.

    Accepts UTF-8 with or without a single leading BOM (Windows PowerShell 5.1
    ``Set-Content -Encoding UTF8``). Uses strict ``json.loads``; does not
    swallow decode or JSON errors; does not apply lossy decoding.
    """
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")  # UTF-8; strips one leading BOM if present
    return json.loads(text)


def _file_identity(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        "sha256": h.hexdigest(),
    }


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


def _prediction_quality(path: Path) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    out: Dict[str, Any] = {
        "schema_valid": False,
        "empty": True,
        "nonfinite": False,
        "max_date": None,
    }
    if not path.is_file():
        return out
    try:
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception:
        return out
    if df.empty:
        out["empty"] = True
        return out
    out["empty"] = False
    need = {"date", "close", "predicted_close", "signal"}
    if not need.issubset(set(df.columns)):
        return out
    out["schema_valid"] = True
    d = pd.to_datetime(df["date"], errors="coerce").max()
    out["max_date"] = None if pd.isna(d) else d.date().isoformat()
    preds = pd.to_numeric(df["predicted_close"], errors="coerce")
    arr = preds.to_numpy(dtype=float)
    out["nonfinite"] = bool(preds.isna().any() or not np.isfinite(arr).all())
    return out


def load_dated_input_skips(repo_root: Path, target: date) -> List[Dict[str, Any]]:
    """Explicit target-session skips. A prediction file cannot clear one."""
    path = repo_root / INPUT_SKIPS_REL
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if str(doc.get("target_session") or "") != target.isoformat():
        return []
    rows: List[Dict[str, Any]] = []
    for row in doc.get("skips") or []:
        if not isinstance(row, dict):
            continue
        if row.get("reason") != MISSING_EXPECTED_SESSION_BAR:
            continue
        if str(row.get("session") or "") != target.isoformat():
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        rows.append(
            {
                "symbol": symbol,
                "session": target.isoformat(),
                "reason": MISSING_EXPECTED_SESSION_BAR,
                "operationally_eligible": False,
            }
        )
    return rows


def required_symbol_sets(tickers: Sequence[str]) -> Dict[str, Any]:
    universe = [str(t).strip().upper() for t in tickers if str(t).strip()]
    excluded = [EXPECTED_EXCLUSION_SPCX] if EXPECTED_EXCLUSION_SPCX in universe else []
    index_refs = [t for t in universe if t.startswith("^")]
    pred_required = [t for t in universe if t not in set(excluded)]
    return {
        "canonical_symbols": universe,
        "canonical_symbol_count": len(universe),
        "raw_required": list(universe),
        "processed_required": list(universe),
        "prediction_required": pred_required,
        "expected_excluded_symbols": excluded,
        "index_reference_symbols": index_refs,
        "index_reference_predictions_required": True,
        "raw_required_count": len(universe),
        "processed_required_count": len(universe),
        "prediction_required_count": len(pred_required),
    }


def _layer_paths(repo_root: Path, symbol: str, layer: str) -> Path:
    if layer == "raw":
        return repo_root / "data" / "raw" / f"{symbol}.csv"
    if layer == "processed":
        return repo_root / "data" / "processed" / f"{symbol}.parquet"
    if layer == "predictions":
        return repo_root / "data" / "predictions" / f"{symbol}_predictions.parquet"
    raise ValueError(f"unknown layer: {layer}")


def capture_baseline(
    *,
    repo_root: Path,
    run_id: str,
    run_started_at: str,
    tickers_file: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    tickers_path = tickers_file or (repo_root / "data" / "config" / "tickers.txt")
    sets = required_symbol_sets(_load_tickers(tickers_path))
    artifacts: Dict[str, Any] = {}
    for layer, symbols in (
        ("raw", sets["raw_required"]),
        ("processed", sets["processed_required"]),
        ("predictions", sets["prediction_required"]),
    ):
        for sym in symbols:
            rel = _layer_paths(repo_root, sym, layer).relative_to(repo_root).as_posix()
            ident = _file_identity(repo_root / rel)
            artifacts[rel] = ident or {"exists": False}
    # Train diagnostic artifacts used as ancillary current-run evidence.
    for rel in (
        "data/results/skipped_tickers.csv",
        "data/results/model_comparison.csv",
        "data/results/feature_importance.csv",
    ):
        ident = _file_identity(repo_root / rel)
        artifacts[rel] = ident or {"exists": False}

    payload = {
        "framework": "daily_prediction_refresh_baseline",
        "run_id": run_id,
        "run_started_at_utc": run_started_at,
        "captured_at_utc": _utc_now_iso(),
        "symbol_contract": sets,
        "artifacts": artifacts,
    }
    return payload


def _advanced_vs_baseline(
    *,
    repo_root: Path,
    rel: str,
    baseline_artifacts: Dict[str, Any],
    run_started_at: datetime,
) -> Dict[str, Any]:
    path = repo_root / rel
    pre = baseline_artifacts.get(rel) or {"exists": False}
    post = _file_identity(path)
    if not post or not post.get("exists"):
        return {"advanced": False, "reason": "missing_after_run", "pre": pre, "post": None}
    if not pre.get("exists"):
        return {"advanced": True, "reason": "created_this_run", "pre": pre, "post": post}
    changed = (
        post.get("sha256") != pre.get("sha256")
        or int(post.get("size_bytes") or -1) != int(pre.get("size_bytes") or -2)
        or int(post.get("mtime_ns") or 0) > int(pre.get("mtime_ns") or 0)
    )
    # Also require the write happened at/after run start (mtime seconds).
    mtime_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    after_start = mtime_utc >= (run_started_at - __import__("datetime").timedelta(seconds=2))
    advanced = bool(changed and after_start)
    return {
        "advanced": advanced,
        "reason": "rewritten_this_run" if advanced else "unchanged_from_baseline",
        "pre": pre,
        "post": post,
        "mtime_utc": mtime_utc.isoformat(),
        "after_run_start": after_start,
    }


def assess_required_layer(
    *,
    repo_root: Path,
    layer: str,
    required_symbols: Sequence[str],
    excluded_symbols: Sequence[str],
    target: date,
    baseline_artifacts: Dict[str, Any],
    run_started_at: datetime,
    require_advancement: bool,
) -> Dict[str, Any]:
    required = [str(s).upper() for s in required_symbols]
    excluded = {str(s).upper() for s in excluded_symbols}
    ok_symbols: List[str] = []
    missing: List[str] = []
    stale: List[str] = []
    schema_invalid: List[str] = []
    empty: List[str] = []
    nonfinite: List[str] = []
    not_advanced: List[str] = []
    dist: Counter[str] = Counter()
    advancement: Dict[str, Any] = {}

    for sym in required:
        path = _layer_paths(repo_root, sym, layer)
        rel = path.relative_to(repo_root).as_posix()
        adv = _advanced_vs_baseline(
            repo_root=repo_root,
            rel=rel,
            baseline_artifacts=baseline_artifacts,
            run_started_at=run_started_at,
        )
        advancement[sym] = adv
        if not path.is_file():
            missing.append(sym)
            continue
        if layer == "predictions":
            q = _prediction_quality(path)
            if q["empty"]:
                empty.append(sym)
                continue
            if not q["schema_valid"]:
                schema_invalid.append(sym)
                continue
            if q["nonfinite"]:
                nonfinite.append(sym)
                continue
            max_date = q["max_date"]
        else:
            d = _max_date_from_file(path)
            max_date = None if d is None else d.isoformat()
            if d is None:
                schema_invalid.append(sym)
                continue
        if max_date is None:
            schema_invalid.append(sym)
            continue
        dist[max_date] += 1
        if max_date != target.isoformat():
            stale.append(sym)
            continue
        if require_advancement and not adv.get("advanced"):
            not_advanced.append(sym)
            continue
        ok_symbols.append(sym)

    # Unexpected prediction files outside canonical required+excluded universe.
    unexpected: List[str] = []
    if layer == "predictions":
        allowed = set(required) | excluded
        for p in sorted((repo_root / "data" / "predictions").glob("*_predictions.parquet")):
            name = p.name[: -len("_predictions.parquet")].upper()
            if name not in allowed:
                unexpected.append(name)

    present_ok = set(ok_symbols)
    required_set = set(required)
    # Exact membership: every required symbol ok; extras reported separately.
    passed = (
        present_ok == required_set
        and not missing
        and not stale
        and not schema_invalid
        and not empty
        and not nonfinite
        and not not_advanced
        and not unexpected
    )
    return {
        "layer": layer,
        "required_symbols": required,
        "required_count": len(required),
        "ok_symbols": ok_symbols,
        "ok_count": len(ok_symbols),
        "missing_required_symbols": missing,
        "stale_symbols": stale,
        "schema_invalid_symbols": schema_invalid,
        "empty_symbols": empty,
        "nonfinite_symbols": nonfinite,
        "not_advanced_symbols": not_advanced,
        "unexpected_symbols": unexpected,
        "expected_excluded_symbols": sorted(excluded),
        "max_date_distribution": dict(dist),
        "advancement": advancement,
        "passed": passed,
    }


def validate_refresh(
    *,
    repo_root: Path,
    target_session: Optional[date] = None,
    run_id: str = "",
    run_started_at: Optional[str] = None,
    baseline: Optional[Dict[str, Any]] = None,
    baseline_path: Optional[Path] = None,
    stage_receipts: Optional[Sequence[Dict[str, Any]]] = None,
    tickers_file: Optional[Path] = None,
    require_advancement: bool = True,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    from services.market_calendar import last_completed_nyse_session

    target = target_session or last_completed_nyse_session()
    tickers_path = tickers_file or (repo_root / "data" / "config" / "tickers.txt")
    full_sets = required_symbol_sets(_load_tickers(tickers_path))
    dated_skips = load_dated_input_skips(repo_root, target)
    skip_symbols = {row["symbol"] for row in dated_skips}
    sets = dict(full_sets)
    for key in ("raw_required", "processed_required", "prediction_required"):
        sets[key] = [symbol for symbol in full_sets[key] if symbol not in skip_symbols]
    sets["dated_input_skips"] = dated_skips

    if baseline is None and baseline_path is not None and Path(baseline_path).is_file():
        baseline = load_owner_evidence_json(Path(baseline_path))
        if not isinstance(baseline, dict):
            raise ValueError("baseline evidence must be a JSON object")
    baseline = baseline or {}
    baseline_artifacts = dict(baseline.get("artifacts") or {})

    started = run_started_at or baseline.get("run_started_at_utc") or _utc_now_iso()
    started_dt = _parse_iso_dt(str(started))
    rid = str(run_id or baseline.get("run_id") or "")

    # Reject old summary reuse / mismatched baseline binding.
    binding_errors: List[str] = []
    if require_advancement:
        if not rid:
            binding_errors.append("missing_run_id")
        if baseline.get("run_id") and str(baseline.get("run_id")) != rid:
            binding_errors.append("baseline_run_id_mismatch")
        if not baseline_artifacts:
            binding_errors.append("missing_baseline_artifacts")
        if (
            baseline.get("run_started_at_utc")
            and run_started_at
            and str(baseline.get("run_started_at_utc")) != str(run_started_at)
        ):
            binding_errors.append("baseline_started_at_mismatch")

    raw = assess_required_layer(
        repo_root=repo_root,
        layer="raw",
        required_symbols=sets["raw_required"],
        excluded_symbols=[],
        target=target,
        baseline_artifacts=baseline_artifacts,
        run_started_at=started_dt,
        require_advancement=require_advancement,
    )
    processed = assess_required_layer(
        repo_root=repo_root,
        layer="processed",
        required_symbols=sets["processed_required"],
        excluded_symbols=[],
        target=target,
        baseline_artifacts=baseline_artifacts,
        run_started_at=started_dt,
        require_advancement=require_advancement,
    )
    predictions = assess_required_layer(
        repo_root=repo_root,
        layer="predictions",
        required_symbols=sets["prediction_required"],
        excluded_symbols=sets["expected_excluded_symbols"],
        target=target,
        baseline_artifacts=baseline_artifacts,
        run_started_at=started_dt,
        require_advancement=require_advancement,
    )

    # Ancillary train diagnostics must also advance when training succeeds.
    train_diag_advanced = True
    train_diag_details: Dict[str, Any] = {}
    if require_advancement:
        for rel in ("data/results/skipped_tickers.csv", "data/results/model_comparison.csv"):
            train_diag_details[rel] = _advanced_vs_baseline(
                repo_root=repo_root,
                rel=rel,
                baseline_artifacts=baseline_artifacts,
                run_started_at=started_dt,
            )
            if not train_diag_details[rel].get("advanced"):
                train_diag_advanced = False

    receipts = list(stage_receipts or [])
    receipt_errors: List[str] = []
    if require_advancement:
        expected_order = [
            "fetch_raw_data",
            "stale_data_gate",
            "preprocess",
            "train_model",
        ]
        names = [str(r.get("name") or "") for r in receipts]
        if names[: len(expected_order)] != expected_order:
            receipt_errors.append("stage_order_invalid")
        if len(names) != len(set(names)):
            receipt_errors.append("duplicate_stage_receipt")
        for r in receipts:
            if int(r.get("exit_code") or 0) != 0:
                receipt_errors.append(f"stage_nonzero:{r.get('name')}")

    tradables = [s for s in full_sets["canonical_symbols"] if not str(s).startswith("^")]
    raw_ok = set(raw["ok_symbols"])
    covered = [s for s in tradables if s in raw_ok]
    coverage_ratio = (len(covered) / len(tradables)) if tradables else 0.0
    coverage_ok = coverage_ratio + 1e-12 >= MIN_TRADABLE_COVERAGE
    ineligible = []
    for row in dated_skips:
        pred_path = _layer_paths(repo_root, row["symbol"], "predictions")
        ineligible.append(
            {
                "symbol": row["symbol"],
                "session": row["session"],
                "reason": row["reason"],
                "prediction_present": pred_path.is_file(),
                "operationally_eligible": False,
            }
        )

    passed = bool(
        not binding_errors
        and not receipt_errors
        and coverage_ok
        and raw["passed"]
        and processed["passed"]
        and predictions["passed"]
        and train_diag_advanced
    )

    result: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": rid,
        "run_started_at_utc": started,
        "run_ended_at_utc": _utc_now_iso(),
        "framework": "daily_prediction_refresh_validate",
        "target_session": target.isoformat(),
        "symbol_contract": sets,
        "raw": raw,
        "processed": processed,
        "predictions": predictions,
        "expected_excluded_symbols": sets["expected_excluded_symbols"],
        "index_reference_predictions_required": True,
        "exact_symbol_set_validation": True,
        "train_diagnostics_advancement": train_diag_details,
        "train_diagnostics_advanced": train_diag_advanced,
        "stage_receipts": receipts,
        "baseline_run_id": baseline.get("run_id"),
        "baseline_captured_at_utc": baseline.get("captured_at_utc"),
        "require_advancement": require_advancement,
        "binding_errors": binding_errors,
        "receipt_errors": receipt_errors,
        "dated_skips": dated_skips,
        "tradable_coverage": {
            "covered": len(covered),
            "denominator": len(tradables),
            "ratio": coverage_ratio,
            "minimum": MIN_TRADABLE_COVERAGE,
        },
        "skipped_symbols_remain_ineligible": ineligible,
        "passed": passed,
        "broker_called": False,
        "network_called": False,
        "first_failure": None,
    }
    if not passed:
        reasons: List[str] = []
        reasons.extend(binding_errors)
        reasons.extend(receipt_errors)
        if not coverage_ok:
            reasons.append("coverage_below_minimum")
        if not raw["passed"]:
            reasons.append("raw_validation_failed")
        if not processed["passed"]:
            reasons.append("processed_validation_failed")
        if not predictions["passed"]:
            reasons.append("predictions_validation_failed")
        if not train_diag_advanced:
            reasons.append("train_diagnostics_not_advanced")
        result["failure_reasons"] = reasons
        # First failure priority.
        if binding_errors:
            result["first_failure"] = binding_errors[0]
        elif receipt_errors:
            result["first_failure"] = receipt_errors[0]
        elif not coverage_ok:
            result["first_failure"] = "coverage_below_minimum"
        elif raw.get("not_advanced_symbols"):
            result["first_failure"] = "raw_not_advanced"
        elif processed.get("not_advanced_symbols"):
            result["first_failure"] = "processed_not_advanced"
        elif predictions.get("not_advanced_symbols"):
            result["first_failure"] = "predictions_not_advanced"
        elif predictions.get("missing_required_symbols"):
            result["first_failure"] = "predictions_missing_required"
        else:
            result["first_failure"] = reasons[0] if reasons else "validation_failed"
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Daily prediction refresh validate / baseline capture")
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--target-session", default=None)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--run-started-at", default="")
    ap.add_argument("--capture-baseline", action="store_true")
    ap.add_argument("--baseline-out", default="")
    ap.add_argument("--baseline-in", default="")
    ap.add_argument("--stage-receipts", default="", help="JSON file of ordered stage receipts")
    ap.add_argument("--summary-out", default="")
    ap.add_argument(
        "--allow-stale-without-advancement",
        action="store_true",
        help="Test/debug only: skip advancement binding",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo = Path(args.repo_root)
    rid = str(args.run_id or "")
    started = str(args.run_started_at or _utc_now_iso())

    if args.capture_baseline:
        if not rid:
            raise SystemExit("--run-id is required for --capture-baseline")
        payload = capture_baseline(repo_root=repo, run_id=rid, run_started_at=started)
        out = (
            Path(args.baseline_out)
            if args.baseline_out
            else (repo / "data" / "results" / f"daily_prediction_refresh_baseline_{rid}.json")
        )
        _atomic_write_json(out, payload)
        if args.json:
            sys.stdout.write(
                json.dumps({"ok": True, "baseline_path": str(out), **payload}, indent=2) + "\n"
            )
        else:
            print(f"[daily_prediction_refresh_validate] baseline_written={out}")
        return 0

    target = date.fromisoformat(args.target_session) if args.target_session else None
    receipts = []
    if args.stage_receipts:
        receipts = load_owner_evidence_json(Path(args.stage_receipts))
        if not isinstance(receipts, list):
            raise SystemExit("stage receipts must be a JSON list")

    result = validate_refresh(
        repo_root=repo,
        target_session=target,
        run_id=rid,
        run_started_at=started,
        baseline_path=Path(args.baseline_in) if args.baseline_in else None,
        stage_receipts=receipts,
        require_advancement=not bool(args.allow_stale_without_advancement),
    )
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
            print(
                f"[daily_prediction_refresh_validate] first_failure={result.get('first_failure')} "
                f"reasons={result.get('failure_reasons')}"
            )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
