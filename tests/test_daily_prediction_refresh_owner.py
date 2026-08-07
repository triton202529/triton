"""
Focused offline tests for daily prediction-refresh owner/validator repair.

Temporary roots + mocks only. No broker, network, or production artifact mutation.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DAILY_REFRESH = ROOT / "run_daily_refresh.ps1"
MARKET_OPEN = ROOT / "scripts" / "run_market_open.ps1"


def _write_ohlcv(path: Path, last: date, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [last - timedelta(days=i) for i in range(rows - 1, -1, -1)]
    pd.DataFrame(
        {
            "Date": [d.isoformat() for d in dates],
            "Open": 1.0,
            "High": 1.1,
            "Low": 0.9,
            "Close": 1.0,
            "Volume": 1000,
        }
    ).to_csv(path, index=False)


def _write_processed(path: Path, last: date, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [last - timedelta(days=i) for i in range(rows - 1, -1, -1)]
    pd.DataFrame({"date": dates, "close": [1.0] * rows}).to_parquet(path, index=False)


def _write_pred(
    path: Path,
    last: date,
    rows: int = 5,
    *,
    predicted: float = 1.01,
    empty: bool = False,
    drop_col: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if empty:
        pd.DataFrame(
            {"date": [], "close": [], "predicted_close": [], "signal": [], "ticker": []}
        ).to_parquet(path, index=False)
        return
    dates = [last - timedelta(days=i) for i in range(rows - 1, -1, -1)]
    ticker = path.name.split("_")[0]
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [1.0] * rows,
            "predicted_close": [predicted] * rows,
            "signal": ["HOLD"] * rows,
            "ticker": [ticker] * rows,
        }
    )
    if drop_col:
        df = df.drop(columns=[drop_col])
    df.to_parquet(path, index=False)


def _universe_66() -> list[str]:
    base = [f"S{i:02d}" for i in range(1, 62)]  # 61 synthetics
    # Force exact accepted shape: 61 + SPCX + 4 indices = 66
    return base + ["SPCX", "^DJI", "^GSPC", "^IXIC", "^VIX"]


def _seed_repo(tmp: Path, *, target: date, include_index_preds: bool = True) -> Path:
    tickers = _universe_66()
    assert len(tickers) == 66
    (tmp / "data" / "config").mkdir(parents=True)
    (tmp / "data" / "config" / "tickers.txt").write_text(
        "\n".join(tickers) + "\n", encoding="utf-8"
    )
    (tmp / "data" / "results").mkdir(parents=True)
    for t in tickers:
        _write_ohlcv(tmp / "data" / "raw" / f"{t}.csv", target)
        _write_processed(tmp / "data" / "processed" / f"{t}.parquet", target)
        if t == "SPCX":
            continue
        if t.startswith("^") and not include_index_preds:
            continue
        _write_pred(tmp / "data" / "predictions" / f"{t}_predictions.parquet", target)
    pd.DataFrame([{"ticker": "SPCX", "reason": "Too few rows after lagging (have 14)"}]).to_csv(
        tmp / "data" / "results" / "skipped_tickers.csv", index=False
    )
    pd.DataFrame([{"ticker": "S01", "model": "rf"}]).to_csv(
        tmp / "data" / "results" / "model_comparison.csv", index=False
    )
    pd.DataFrame([{"feature": "x", "importance": 1.0}]).to_csv(
        tmp / "data" / "results" / "feature_importance.csv", index=False
    )
    return tmp


def _touch_rewrite_pred(path: Path, target: date) -> None:
    time.sleep(0.02)
    _write_pred(path, target, predicted=1.02)


def _advance_all(tmp: Path, target: date) -> None:
    tickers = _universe_66()
    time.sleep(0.05)
    for t in tickers:
        _write_ohlcv(tmp / "data" / "raw" / f"{t}.csv", target)
        _write_processed(tmp / "data" / "processed" / f"{t}.parquet", target)
        if t == "SPCX":
            continue
        _write_pred(
            tmp / "data" / "predictions" / f"{t}_predictions.parquet", target, predicted=1.03
        )
    pd.DataFrame([{"ticker": "SPCX", "reason": "Too few rows after lagging (have 14)"}]).to_csv(
        tmp / "data" / "results" / "skipped_tickers.csv", index=False
    )
    pd.DataFrame([{"ticker": "S01", "model": "rf", "v": 2}]).to_csv(
        tmp / "data" / "results" / "model_comparison.csv", index=False
    )


def _receipts_ok() -> list[dict]:
    return [
        {"name": "fetch_raw_data", "exit_code": 0, "outcome": "ok"},
        {"name": "stale_data_gate", "exit_code": 0, "outcome": "ok"},
        {"name": "preprocess", "exit_code": 0, "outcome": "ok"},
        {"name": "train_model", "exit_code": 0, "outcome": "ok"},
    ]


# --- Pre-repair defect reproductions (document old broken comparison) ---


def test_pre_repair_count_defect_65_vs_61():
    """Historical defect: excluding indices from expected while counting their ok files."""
    tickers = _universe_66()
    excluded = {"SPCX"}
    indices = {t for t in tickers if t.startswith("^")}
    pred_allowed_missing = excluded | indices
    expected = [t for t in tickers if t not in pred_allowed_missing]
    # Production-like: 65 prediction files present and current.
    ok_count = 65
    assert len(expected) == 61
    assert ok_count != len(expected)
    pre_repair_passed = ok_count == len(expected) and True
    assert pre_repair_passed is False


def test_pre_repair_stale_same_session_accepted_without_advancement():
    """Historical defect: date-only validation accepts unchanged same-session artifacts."""
    from services.daily_prediction_refresh_validate import validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        # No baseline / no advancement requirement mimics old behavior.
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="old_style",
            require_advancement=False,
        )
        assert result["passed"] is True
        assert result["predictions"]["required_count"] == 65


# --- Contract / repair tests ---


def test_owner_script_contract_no_broker_and_baseline_before_fetch():
    src = DAILY_REFRESH.read_text(encoding="utf-8")
    assert "capture-baseline" in src or "capture_baseline" in src
    assert "baseline-in" in src or "--baseline-in" in src
    assert "services.fetch_raw_data" in src
    assert "services.preprocess" in src
    assert "services.train_model" in src
    for banned in (
        "run_full_pipeline",
        "snapshot_equity",
        "snapshot_positions",
        "manage_open_orders",
        "manage_positions",
        "run_market_open",
        "generate_signals",
    ):
        assert banned not in src


def test_symbol_contract_66_66_65_with_indices_required():
    from services.daily_prediction_refresh_validate import required_symbol_sets

    sets = required_symbol_sets(_universe_66())
    assert sets["raw_required_count"] == 66
    assert sets["processed_required_count"] == 66
    assert sets["prediction_required_count"] == 65
    assert sets["expected_excluded_symbols"] == ["SPCX"]
    for idx in ("^DJI", "^GSPC", "^IXIC", "^VIX"):
        assert idx in sets["prediction_required"]
    assert "SPCX" not in sets["prediction_required"]


def test_count_defect_fixed_65_predictions_pass_after_advancement():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target, include_index_preds=True)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r1", run_started_at=started)
        _advance_all(root, target)
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r1",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
            require_advancement=True,
        )
        assert result["passed"] is True
        assert result["predictions"]["required_count"] == 65
        assert result["predictions"]["ok_count"] == 65
        assert result["expected_excluded_symbols"] == ["SPCX"]


def test_spcx_is_exclusion_not_missing_required():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r2", run_started_at=started)
        _advance_all(root, target)
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r2",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert "SPCX" not in result["predictions"]["missing_required_symbols"]
        assert result["expected_excluded_symbols"] == ["SPCX"]


@pytest.mark.parametrize("idx", ["^DJI", "^GSPC", "^IXIC", "^VIX"])
def test_missing_index_prediction_fails(idx: str):
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r3", run_started_at=started)
        _advance_all(root, target)
        (root / "data" / "predictions" / f"{idx}_predictions.parquet").unlink()
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r3",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert result["passed"] is False
        assert idx in result["predictions"]["missing_required_symbols"]


def test_missing_non_index_required_prediction_fails():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r4", run_started_at=started)
        _advance_all(root, target)
        (root / "data" / "predictions" / "S01_predictions.parquet").unlink()
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r4",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert result["passed"] is False
        assert "S01" in result["predictions"]["missing_required_symbols"]


def test_unexpected_extra_cannot_mask_missing_required():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r5", run_started_at=started)
        _advance_all(root, target)
        (root / "data" / "predictions" / "S01_predictions.parquet").unlink()
        _write_pred(root / "data" / "predictions" / "ZZZZ_predictions.parquet", target)
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r5",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert result["passed"] is False
        assert "S01" in result["predictions"]["missing_required_symbols"]
        assert "ZZZZ" in result["predictions"]["unexpected_symbols"]


def test_schema_empty_nonfinite_wrong_date_fail():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r6", run_started_at=started)
        _advance_all(root, target)
        _write_pred(
            root / "data" / "predictions" / "S01_predictions.parquet",
            target,
            drop_col="signal",
        )
        r1 = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r6",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert r1["passed"] is False
        assert "S01" in r1["predictions"]["schema_invalid_symbols"]

        baseline2 = capture_baseline(repo_root=root, run_id="r6b", run_started_at=started)
        _advance_all(root, target)
        _write_pred(root / "data" / "predictions" / "S02_predictions.parquet", target, empty=True)
        r2 = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r6b",
            run_started_at=started,
            baseline=baseline2,
            stage_receipts=_receipts_ok(),
        )
        assert "S02" in r2["predictions"]["empty_symbols"]

        baseline3 = capture_baseline(repo_root=root, run_id="r6c", run_started_at=started)
        _advance_all(root, target)
        _write_pred(
            root / "data" / "predictions" / "S03_predictions.parquet",
            target,
            predicted=float("nan"),
        )
        r3 = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r6c",
            run_started_at=started,
            baseline=baseline3,
            stage_receipts=_receipts_ok(),
        )
        assert "S03" in r3["predictions"]["nonfinite_symbols"]

        baseline4 = capture_baseline(repo_root=root, run_id="r6d", run_started_at=started)
        _advance_all(root, target)
        _write_pred(root / "data" / "predictions" / "S04_predictions.parquet", date(2026, 7, 31))
        r4 = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r6d",
            run_started_at=started,
            baseline=baseline4,
            stage_receipts=_receipts_ok(),
        )
        assert "S04" in r4["predictions"]["stale_symbols"]


def test_unchanged_same_session_artifacts_fail_advancement():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r7", run_started_at=started)
        # Children "exit 0" but no rewrite.
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r7",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert result["passed"] is False
        assert result["predictions"]["not_advanced_symbols"]
        assert "predictions_not_advanced" in result.get("failure_reasons", []) or result.get(
            "first_failure"
        )


def test_partial_layer_advancement_fails():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r8", run_started_at=started)
        time.sleep(0.05)
        for t in _universe_66():
            _write_ohlcv(root / "data" / "raw" / f"{t}.csv", target)
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r8",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert result["passed"] is False
        assert result["raw"]["passed"] is True or result["raw"]["ok_count"] == 66
        assert (
            result["processed"]["not_advanced_symbols"]
            or result["predictions"]["not_advanced_symbols"]
        )

        baseline2 = capture_baseline(repo_root=root, run_id="r8b", run_started_at=started)
        time.sleep(0.05)
        for t in _universe_66():
            _write_ohlcv(root / "data" / "raw" / f"{t}.csv", target)
            _write_processed(root / "data" / "processed" / f"{t}.parquet", target)
        result2 = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r8b",
            run_started_at=started,
            baseline=baseline2,
            stage_receipts=_receipts_ok(),
        )
        assert result2["passed"] is False
        assert result2["predictions"]["not_advanced_symbols"]


def test_rewritten_same_session_distinguishable_from_baseline():
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="r9", run_started_at=started)
        _advance_all(root, target)
        result = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="r9",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert result["passed"] is True
        # Same business date, but advancement flags true.
        sample = result["predictions"]["advancement"]["S01"]
        assert sample["advanced"] is True


def test_old_summary_and_mismatched_run_id_rejected():
    from services.daily_prediction_refresh_validate import capture_baseline, main, validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = capture_baseline(repo_root=root, run_id="runA", run_started_at=started)
        _advance_all(root, target)
        bad = validate_refresh(
            repo_root=root,
            target_session=target,
            run_id="runB",
            run_started_at=started,
            baseline=baseline,
            stage_receipts=_receipts_ok(),
        )
        assert bad["passed"] is False
        assert "baseline_run_id_mismatch" in bad["binding_errors"]

        summary = root / "data" / "results" / "daily_prediction_refresh_summary.json"
        summary.write_text(
            json.dumps({"run_id": "ancient", "passed": True, "target_session": "2026-08-07"}),
            encoding="utf-8",
        )
        started2 = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline2 = capture_baseline(repo_root=root, run_id="runC", run_started_at=started2)
        (root / "baseline.json").write_text(json.dumps(baseline2), encoding="utf-8")
        (root / "receipts.json").write_text(json.dumps(_receipts_ok()), encoding="utf-8")
        # No advancement vs baseline2 — historical summary must not force pass.
        rc = main(
            [
                "--repo-root",
                str(root),
                "--target-session",
                target.isoformat(),
                "--run-id",
                "runC",
                "--run-started-at",
                started2,
                "--baseline-in",
                str(root / "baseline.json"),
                "--stage-receipts",
                str(root / "receipts.json"),
                "--summary-out",
                str(summary),
            ]
        )
        assert rc == 1
        doc = json.loads(summary.read_text(encoding="utf-8"))
        assert doc["run_id"] == "runC"
        assert doc["passed"] is False


def test_stage_order_and_failure_propagation_harness():
    script = textwrap.dedent(
        r"""
        $ErrorActionPreference = 'Stop'
        $ran = New-Object System.Collections.Generic.List[string]
        function Run-PyStep([string]$Name) {
            $ran.Add($Name) | Out-Null
            if ($Name -eq 'preprocess') { throw 'STEP_FAILED:preprocess:3' }
        }
        try {
            Run-PyStep 'fetch_raw_data'
            Run-PyStep 'stale_data_gate'
            Run-PyStep 'preprocess'
            Run-PyStep 'train_model'
            Write-Host 'UNEXPECTED_FULL_SUCCESS'
        } catch {
            Write-Host ('CAUGHT=' + $_.Exception.Message)
            Write-Host ('RAN=' + ($ran -join ','))
        }
        """
    )
    with tempfile.TemporaryDirectory() as td:
        ps1 = Path(td) / "h.ps1"
        ps1.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            capture_output=True,
            text=True,
            cwd=td,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        assert "STEP_FAILED:preprocess:3" in out
        assert "train_model" not in out.split("RAN=")[-1]


def test_first_nonzero_exit_preserved_in_owner_script():
    src = DAILY_REFRESH.read_text(encoding="utf-8")
    assert 'throw ("STEP_FAILED:" + $Name + ":" + $code)' in src
    assert "exit $exitCode" in src


def test_d1a_wrapper_unchanged():
    src = MARKET_OPEN.read_text(encoding="utf-8")
    assert "function Run-RefreshStep" in src
    assert '$ErrorActionPreference = "Continue"' in src


def test_validator_module_has_no_network_or_broker_side_effects():
    src = (ROOT / "services" / "daily_prediction_refresh_validate.py").read_text(encoding="utf-8")
    assert "yfinance" not in src
    assert "alpaca" not in src.lower()
    assert "place_order" not in src
    assert "broker_called" in src
    assert "network_called" in src
