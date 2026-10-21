"""Isolated offline D1a as-of check for one completed DailyRefresh run.

Reads existing refresh artifacts and market files under ``--repo-root``.
Writes one new ``signals_asof_diagnostics_<validation_id>.json`` and nothing else.
Does not import or call an operational signal publisher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.daily_refresh_stage_receipts import resolve_stage_receipts  # noqa: E402
from services.generate_signals import compute_asof_policy  # noqa: E402
from services.lifecycle_artifact_guard import parse_universe_ticker_identities  # noqa: E402

REQUIRED_SESSION = "2026-09-28"
SHARED_DIAGNOSTICS = "data/results/signals_asof_diagnostics.json"
_UNTOUCHED = (
    SHARED_DIAGNOSTICS,
    "data/results/signals.csv",
    "data/results/signals_with_rationale.csv",
    "data/results/heartbeat.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> Dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_utc(text: str) -> datetime:
    raw = str(text).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _max_date(path: Path) -> date:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    columns = {str(column).lower(): column for column in frame.columns}
    key = columns.get("date")
    if key is None:
        raise ValueError(f"missing date column: {path}")
    parsed = pd.to_datetime(frame[key], errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"no dates: {path}")
    return parsed.max().date()


def _output_path(root: Path, validation_id: str) -> Path:
    name = f"signals_asof_diagnostics_{validation_id}.json"
    return root / "data" / "results" / name


def _fail(message: str) -> Tuple[int, str, None]:
    return 1, message, None


def _layer_dates(
    root: Path, symbols: Sequence[str], folder: str, suffix: str
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Optional[str]]:
    dates: Dict[str, str] = {}
    records: List[Dict[str, Any]] = []
    for symbol in symbols:
        path = root / "data" / folder / f"{symbol}{suffix}"
        if not path.is_file():
            return {}, [], f"missing {folder} file for {symbol}"
        try:
            found = _max_date(path)
        except (OSError, ValueError) as exc:
            return {}, [], str(exc)
        dates[symbol] = found.isoformat()
        item = _record(path, root)
        item["max_date"] = found.isoformat()
        records.append(item)
    return dates, records, None


def validate_d1a(
    root: Path,
    *,
    source_run_id: str,
    session: str,
    validation_id: str,
) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """Return ``(exit_code, message, payload)``. Payload is set only on success."""
    root = root.resolve()
    if session != REQUIRED_SESSION:
        return _fail(f"session must be {REQUIRED_SESSION}")
    if not validation_id or any(char in validation_id for char in "\\/\0"):
        return _fail("validation_id is not a single file token")
    destination = _output_path(root, validation_id)
    if destination.exists():
        return 2, f"refusing existing output: {destination}", None
    for relative in _UNTOUCHED:
        # Presence is irrelevant. These paths are never opened as evidence.
        del relative

    session_date = date.fromisoformat(session)
    stamp = source_run_id.removeprefix("daily_prediction_refresh_")
    required = {
        "log": root / "logs" / f"daily_refresh_{stamp}.log",
        "baseline": root
        / "data"
        / "results"
        / f"daily_prediction_refresh_baseline_{source_run_id}.json",
        "fetch": root / "data" / "results" / "daily_refresh_fetch_receipt.json",
        "summary": root / "data" / "results" / "daily_prediction_refresh_summary.json",
        "tickers": root / "data" / "config" / "tickers.txt",
    }
    resolved = resolve_stage_receipts(root / "data" / "results", source_run_id)
    missing = [name for name, path in required.items() if not path.is_file()]
    if resolved.status == "missing":
        missing.append("stage")
    if missing:
        return _fail("missing source artifacts: " + ",".join(missing))
    if resolved.status not in {"generation", "legacy"} or not isinstance(resolved.document, list):
        return _fail(f"stage receipts {resolved.status}")
    if resolved.path is None:
        return _fail("stage receipts missing")
    required["stage"] = resolved.path
    log_text = required["log"].read_text(encoding="utf-8", errors="replace")
    if source_run_id not in log_text:
        return _fail("log is not bound to the source run")

    try:
        baseline = _load_json(required["baseline"])
        stage_rows = resolved.document
        summary = _load_json(required["summary"])
        fetch = _load_json(required["fetch"])
        symbols = parse_universe_ticker_identities(required["tickers"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(f"unreadable source artifact: {exc}")
    if not isinstance(stage_rows, list) or not symbols:
        return _fail("stage receipts or universe are empty")
    if baseline.get("run_id") != source_run_id or summary.get("run_id") != source_run_id:
        return _fail("source-run binding mismatch")
    if summary.get("target_session") != session or summary.get("passed") is not True:
        return _fail("summary is not a passed run for the required session")
    if summary.get("broker_called") is not False or summary.get("network_called") is not False:
        return _fail("summary reports broker or network contact")

    fetch_rows = [
        row
        for row in stage_rows
        if isinstance(row, dict)
        and row.get("name") == "fetch_raw_data"
        and row.get("run_id") == source_run_id
    ]
    if len(fetch_rows) != 1:
        return _fail("missing or mismatched source-run fetch binding")
    bound = fetch_rows[0]
    if bound.get("status") != "COMPLETE" or bound.get("outcome") != "ok":
        return _fail("fetch binding is not complete")
    if bound.get("target_session") != session or fetch.get("target_session") != session:
        return _fail("fetch target session mismatch")
    if fetch.get("passed") is not True or bound.get("artifact_after_baseline") is not True:
        return _fail("fetch receipt is not a passing post-baseline artifact")
    fetch_record = _record(required["fetch"], root)
    if (
        bound.get("artifact_sha256") != fetch_record["sha256"]
        or int(bound.get("artifact_size_bytes") or -1) != fetch_record["size_bytes"]
    ):
        return _fail("fetch receipt hash or size does not match the binding")
    try:
        written = _parse_utc(str(bound.get("artifact_written_at_utc")))
        captured = _parse_utc(str(baseline.get("captured_at_utc")))
        bound_captured = _parse_utc(str(bound.get("baseline_captured_at_utc")))
    except (TypeError, ValueError) as exc:
        return _fail(f"unreadable baseline timing: {exc}")
    if captured != bound_captured or written <= captured:
        return _fail("fetch artifact is not after the baseline capture")

    layers: Dict[str, Dict[str, str]] = {}
    layer_records: List[Dict[str, Any]] = []
    for folder, suffix, label in (
        ("raw", ".csv", "raw"),
        ("processed", ".parquet", "processed"),
        ("predictions", "_predictions.parquet", "predictions"),
    ):
        found, records, error = _layer_dates(root, symbols, folder, suffix)
        if error:
            return _fail(error)
        layers[label] = found
        for item in records:
            item["layer"] = label
        layer_records.extend(records)

    required_count = len(symbols)
    checks = {}
    for label, found in layers.items():
        ok = sum(1 for value in found.values() if value == session)
        checks[label] = {"ok": ok, "required": required_count, "passed": ok == required_count}
        if ok != required_count:
            return _fail(f"{label} is not {required_count}/{required_count} on {session}")

    tradable = [symbol for symbol in symbols if not symbol.startswith("^")]
    indexes = [symbol for symbol in symbols if symbol.startswith("^")]
    latest = {symbol: date.fromisoformat(layers["predictions"][symbol]) for symbol in symbols}
    _effective, policy = compute_asof_policy(
        latest,
        session_date,
        tradable_tickers=tradable,
        index_tickers=indexes,
    )
    have_session = int(policy["have_session"])
    denominator = int(policy["tradable_tickers"])
    coverage = float(policy["coverage"])
    lag = int(policy["lag_sessions_vs_session"])
    max_lag = int(policy["max_lag_sessions"])
    failed = bool(policy["fail"])
    if (
        failed
        or have_session != denominator
        or denominator != len(tradable)
        or coverage < 1.0
        or lag != 0
        or max_lag != 1
        or policy.get("max_prediction_date") != session
    ):
        return _fail("coverage policy rejected the required session")

    if destination.exists():
        return 2, f"refusing existing output: {destination}", None
    payload: Dict[str, Any] = {
        "validation_id": validation_id,
        "source_run_id": source_run_id,
        "session_asof_date": session,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_artifacts": [
            {**_record(required[name], root), "role": name}
            for name in ("log", "baseline", "fetch", "stage", "summary")
        ],
        "layer_artifacts": layer_records,
        "bound_fetch_receipt": {
            "run_id": bound.get("run_id"),
            "target_session": bound.get("target_session"),
            "sha256": fetch_record["sha256"],
            "size_bytes": fetch_record["size_bytes"],
            "artifact_after_baseline": True,
            "artifact_written_at_utc": bound.get("artifact_written_at_utc"),
            "baseline_captured_at_utc": baseline.get("captured_at_utc"),
            "post_baseline": written > captured,
        },
        "raw_max_dates": layers["raw"],
        "processed_max_dates": layers["processed"],
        "prediction_max_dates": layers["predictions"],
        "raw_check": checks["raw"],
        "processed_check": checks["processed"],
        "prediction_check": checks["predictions"],
        "prediction_max_date": policy.get("max_prediction_date"),
        "tradable": {
            "have_session": have_session,
            "denominator": denominator,
            "coverage": coverage,
            "lag_sessions_vs_session": lag,
            "max_lag_sessions": max_lag,
            "fail": failed,
        },
        "shared_diagnostics_used_as_evidence": False,
        "shared_diagnostics_path": SHARED_DIAGNOSTICS,
        "evidence_excludes_shared_diagnostics": True,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(str(destination), flags)
    except FileExistsError:
        return 2, f"refusing existing output: {destination}", None
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return 0, str(destination), payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated D1a as-of validator")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--validation-id", required=True)
    args = parser.parse_args(argv)
    code, message, _payload = validate_d1a(
        args.repo_root,
        source_run_id=args.source_run_id,
        session=args.session,
        validation_id=args.validation_id,
    )
    stream = sys.stdout if code == 0 else sys.stderr
    print(message, file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
