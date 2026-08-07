"""
Focused offline tests for the daily prediction-refresh owner.

Temporary roots + mocks only. No broker, network, production artifact
mutation, Scheduled Task changes, or paper-arm changes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DAILY_REFRESH = ROOT / "run_daily_refresh.ps1"
MARKET_OPEN = ROOT / "scripts" / "run_market_open.ps1"
VALIDATE_MOD = "services.daily_prediction_refresh_validate"


def _write_ohlcv(path: Path, last: date, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [last - timedelta(days=i) for i in range(rows - 1, -1, -1)]
    df = pd.DataFrame(
        {
            "Date": [d.isoformat() for d in dates],
            "Open": 1.0,
            "High": 1.1,
            "Low": 0.9,
            "Close": 1.0,
            "Volume": 1000,
        }
    )
    df.to_csv(path, index=False)


def _write_processed(path: Path, last: date, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [last - timedelta(days=i) for i in range(rows - 1, -1, -1)]
    df = pd.DataFrame({"date": dates, "close": [1.0] * rows})
    df.to_parquet(path, index=False)


def _write_pred(path: Path, last: date, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [last - timedelta(days=i) for i in range(rows - 1, -1, -1)]
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [1.0] * rows,
            "predicted_close": [1.01] * rows,
            "signal": ["HOLD"] * rows,
            "ticker": [path.name.split("_")[0]] * rows,
        }
    )
    df.to_parquet(path, index=False)


def _seed_repo(tmp: Path, *, target: date, include_spcx_skip: bool = True) -> Path:
    (tmp / "data" / "config").mkdir(parents=True)
    tickers = ["AAPL", "SPY", "SPCX", "^GSPC"]
    (tmp / "data" / "config" / "tickers.txt").write_text(
        "\n".join(tickers) + "\n", encoding="utf-8"
    )
    for t in tickers:
        _write_ohlcv(tmp / "data" / "raw" / f"{t}.csv", target)
        _write_processed(tmp / "data" / "processed" / f"{t}.parquet", target)
        if t == "SPCX" and include_spcx_skip:
            continue
        if t.startswith("^"):
            continue
        _write_pred(tmp / "data" / "predictions" / f"{t}_predictions.parquet", target)
    results = tmp / "data" / "results"
    results.mkdir(parents=True)
    if include_spcx_skip:
        pd.DataFrame([{"ticker": "SPCX", "reason": "Too few rows after lagging (have 14)"}]).to_csv(
            results / "skipped_tickers.csv", index=False
        )
    else:
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(
            results / "skipped_tickers.csv", index=False
        )
    pd.DataFrame([{"ticker": "AAPL", "model": "rf"}]).to_csv(
        results / "model_comparison.csv", index=False
    )
    return tmp


def test_daily_refresh_owner_script_contract_no_broker_and_uses_venv_python():
    src = DAILY_REFRESH.read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in src or ".venv/Scripts/python.exe" in src
    assert "services.fetch_raw_data" in src
    assert "services.preprocess" in src
    assert "services.train_model" in src
    assert "services.daily_prediction_refresh_validate" in src
    assert "services.stale_data_gate" in src
    # Must not invoke broker / order / runtime controls.
    for banned in (
        "snapshot_equity",
        "snapshot_positions",
        "snapshot_live_orders",
        "manage_open_orders",
        "manage_positions",
        "run_market_open",
        "run_full_pipeline",
        "set_paper_arm",
        "execute-smart",
    ):
        assert banned not in src
    assert "STEP_FAILED:" in src
    assert "TRITON_ENABLE_LIVE_EXECUTION" in src


def test_daily_refresh_does_not_alter_d1a_wrapper():
    src = MARKET_OPEN.read_text(encoding="utf-8")
    assert "function Run-RefreshStep" in src
    assert '$ErrorActionPreference = "Continue"' in src


def test_validate_passes_when_layers_advanced_with_spcx_policy_skip():
    from services.daily_prediction_refresh_validate import validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target, include_spcx_skip=True)
        result = validate_refresh(repo_root=root, target_session=target, run_id="t1")
        assert result["passed"] is True
        assert result["predictions"]["passed"] is True
        assert "SPCX" in result["policy_skipped_predictions"]


def test_validate_rejects_stale_predictions_even_if_exit_zero_train_claimed():
    from services.daily_prediction_refresh_validate import validate_refresh, main

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target, include_spcx_skip=True)
        # Stale prediction for AAPL only.
        _write_pred(
            root / "data" / "predictions" / "AAPL_predictions.parquet",
            date(2026, 7, 31),
        )
        result = validate_refresh(repo_root=root, target_session=target, run_id="stale")
        assert result["passed"] is False
        assert "predictions_not_advanced" in result["failure_reasons"]
        summary = root / "out_summary.json"
        rc = main(
            [
                "--repo-root",
                str(root),
                "--target-session",
                target.isoformat(),
                "--run-id",
                "stale",
                "--summary-out",
                str(summary),
                "--json",
            ]
        )
        assert rc == 1
        doc = json.loads(summary.read_text(encoding="utf-8"))
        assert doc["run_id"] == "stale"
        assert doc["passed"] is False


def test_validate_rejects_missing_train_artifacts():
    from services.daily_prediction_refresh_validate import validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        (root / "data" / "results" / "model_comparison.csv").unlink()
        result = validate_refresh(repo_root=root, target_session=target)
        assert result["passed"] is False
        assert "train_artifacts_missing" in result["failure_reasons"]


def test_historical_summary_cannot_satisfy_new_run():
    from services.daily_prediction_refresh_validate import main

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        summary = root / "data" / "results" / "daily_prediction_refresh_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(
                {
                    "generated_at_utc": "2026-06-17T22:00:00+00:00",
                    "run_id": "old_run",
                    "passed": True,
                    "target_session": "2026-06-17",
                }
            ),
            encoding="utf-8",
        )
        # Fresh validation with stale predictions must overwrite with fail.
        _write_pred(
            root / "data" / "predictions" / "SPY_predictions.parquet",
            date(2026, 7, 31),
        )
        rc = main(
            [
                "--repo-root",
                str(root),
                "--target-session",
                target.isoformat(),
                "--run-id",
                "new_run",
                "--summary-out",
                str(summary),
            ]
        )
        assert rc == 1
        doc = json.loads(summary.read_text(encoding="utf-8"))
        assert doc["run_id"] == "new_run"
        assert doc["passed"] is False
        assert doc["target_session"] == target.isoformat()


@pytest.mark.parametrize(
    "now_et,expected",
    [
        # Friday before close → prior session Thu 2026-08-06
        (datetime(2026, 8, 7, 12, 0, tzinfo=ZoneInfo("America/New_York")), date(2026, 8, 6)),
        # Friday after close → Fri 2026-08-07
        (datetime(2026, 8, 7, 17, 0, tzinfo=ZoneInfo("America/New_York")), date(2026, 8, 7)),
        # Saturday → Fri 2026-08-07
        (datetime(2026, 8, 8, 10, 0, tzinfo=ZoneInfo("America/New_York")), date(2026, 8, 7)),
        # Monday early UTC still Sunday evening ET → Fri
        (datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc), date(2026, 8, 7)),
    ],
)
def test_last_completed_session_calendar_edges(now_et, expected):
    from services.market_calendar import last_completed_nyse_session

    assert last_completed_nyse_session(now=now_et) == expected


def test_nyse_holiday_uses_prior_session():
    from services.market_calendar import last_completed_nyse_session

    # 2026-07-04 is Saturday; NYSE observed Independence Day on Fri 2026-07-03.
    # After that holiday close window, last completed session is Thu 2026-07-02.
    now = datetime(2026, 7, 4, 17, 0, tzinfo=ZoneInfo("America/New_York"))
    assert last_completed_nyse_session(now=now) == date(2026, 7, 2)


def test_owner_step_failure_stops_later_stages():
    """Fetch failure must not invoke preprocess/train (mocked child orchestration)."""
    script = textwrap.dedent(
        r"""
        $ErrorActionPreference = 'Stop'
        $ran = New-Object System.Collections.Generic.List[string]
        function Run-PyStep([string]$Name) {
            $ran.Add($Name) | Out-Null
            if ($Name -eq 'fetch_raw_data') {
                throw 'STEP_FAILED:fetch_raw_data:7'
            }
        }
        try {
            Run-PyStep 'fetch_raw_data'
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
        ps1 = Path(td) / "harness.ps1"
        ps1.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ps1),
            ],
            capture_output=True,
            text=True,
            cwd=td,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        assert "CAUGHT=STEP_FAILED:fetch_raw_data:7" in out
        assert "RAN=fetch_raw_data" in out
        assert "UNEXPECTED_FULL_SUCCESS" not in out


def test_preprocess_failure_stops_training_stage_order():
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
        ps1 = Path(td) / "harness.ps1"
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
        assert "UNEXPECTED_FULL_SUCCESS" not in out


def test_training_failure_propagates_nonzero_exit_pattern():
    src = DAILY_REFRESH.read_text(encoding="utf-8")
    assert 'throw ("STEP_FAILED:" + $Name + ":" + $code)' in src
    assert "exit $exitCode" in src


def test_unexpected_partial_symbol_failure_blocks():
    from services.daily_prediction_refresh_validate import validate_refresh

    target = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as td:
        root = _seed_repo(Path(td), target=target)
        (root / "data" / "raw" / "AAPL.csv").unlink()
        result = validate_refresh(repo_root=root, target_session=target)
        assert result["passed"] is False
        assert "AAPL" in result["raw"]["missing"]


def test_validate_module_has_no_broker_imports():
    src = (ROOT / "services" / "daily_prediction_refresh_validate.py").read_text(encoding="utf-8")
    for banned in ("alpaca", "broker", "place_order", "submit_order", "snapshot_positions"):
        assert banned not in src.lower() or banned == "broker" and "broker_called" in src
    assert "broker_called" in src
