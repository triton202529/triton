"""
Regression: market-open D1 refresh failure must safe-block, and IPO-band
symbols with stale vendor history must quarantine instead of aborting E0.

Isolated tmp fixtures / mocks only. No broker, network, paper-arm, live,
Scheduled Tasks, or production artifact writes.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKET_OPEN = ROOT / "scripts" / "run_market_open.ps1"
CONTINUOUS = ROOT / "scripts" / "run_continuous_loop.ps1"


def test_run_market_open_refresh_step_captures_python_output_before_bool_return():
    src = MARKET_OPEN.read_text(encoding="utf-8")
    fn = src[
        src.index("function Run-RefreshStep") : src.index("function Write-MarketOpenSequenceStatus")
    ]
    assert "$output = & python @PyArgs 2>&1" in fn
    assert "$code = $LASTEXITCODE" in fn
    assert "$script:LastRefreshStepExitCode = $code" in fn
    assert "return $false" in fn
    assert "return $true" in fn
    # Must not invoke python on the bare success stream without capture.
    assert "& python @PyArgs\n" not in fn.replace("$output = & python @PyArgs 2>&1", "")


def test_run_market_open_refresh_failure_uses_explicit_true_compare():
    src = MARKET_OPEN.read_text(encoding="utf-8")
    # Strip comments so the historical anti-pattern mention in the docstring
    # does not false-trigger the executable-code assertion.
    code_lines = [ln for ln in src.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert '$stepOk = Run-RefreshStep -Label "D1a. GENERATE SIGNALS"' in code
    assert "$stepOk -ne $true" in code
    assert "if (-not (Run-RefreshStep" not in code
    assert "[REFRESH_SAFE_BLOCK]" in code
    assert "Write-MarketOpenSequenceStatus" in code


def test_continuous_loop_logs_failed_step_from_status_artifact():
    src = CONTINUOUS.read_text(encoding="utf-8")
    assert "market_open_sequence_status.json" in src
    assert "failed_step=" in src
    assert "PAPER_ENTRY_EXECUTION_BLOCKED" in src


def test_powershell_refresh_bool_pollution_trap_is_closed():
    """
    Prove the historical failure mode (python stdout pollutes function return,
    making `if (-not (...))` skip the failure branch) and that capture+eq-$true
    closes it.
    """
    script = textwrap.dedent(
        r"""
        $ErrorActionPreference = 'Stop'
        function Bad-Refresh {
            # Simulate python writing lines to the success stream, then failing.
            Write-Output 'signal row 1'
            Write-Output 'signal row 2'
            $code = 1
            if ($code -ne 0) { return $false }
            return $true
        }
        function Good-Refresh {
            $output = @('signal row 1', 'signal row 2')
            $code = 1
            foreach ($line in @($output)) { Write-Host $line }
            if ($code -ne 0) { return $false }
            return $true
        }
        $bad = Bad-Refresh
        $badWouldIgnore = -not ($bad)
        $good = Good-Refresh
        $goodCaught = ($good -ne $true)
        Write-Output ("bad_count={0};bad_ignore={1};good_caught={2}" -f @($bad).Count, [bool]$badWouldIgnore, [bool]$goodCaught)
        """
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "refresh_pollution.ps1"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("bad_count=")][-1]
    assert "bad_ignore=False" in line  # polluted return looks truthy → failure ignored
    assert "good_caught=True" in line


def test_stale_ipo_band_symbol_is_quarantined(tmp_path, monkeypatch):
    from services import symbol_hydration_quarantine as q

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    results = tmp_path / "results"
    raw_dir.mkdir()
    proc_dir.mkdir()
    results.mkdir()
    monkeypatch.setattr(q, "RAW", raw_dir)
    monkeypatch.setattr(q, "PROCESSED", proc_dir)
    monkeypatch.setattr(q, "RESULTS", results)
    monkeypatch.setattr(q, "UNIVERSE_ASSETS", tmp_path / "missing_universe.json")
    monkeypatch.setattr(q, "QUARANTINE_REGISTRY", tmp_path / "quarantine.json")

    as_of = datetime(2026, 8, 6)
    dates = [as_of - timedelta(days=i) for i in range(40, 6, -1)]  # last date < as_of
    assert len(dates) >= 30
    assert dates[-1].date() < as_of.date()
    df = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1000,
        }
    )
    df.to_csv(raw_dir / "SPCX.csv", index=False)
    df.to_parquet(proc_dir / "SPCX.parquet", index=False)

    with (
        patch.object(q, "_pipeline_symbol_sets", return_value=(set(), set())),
        patch.object(q, "_as_of_date_normalized", return_value=pd.Timestamp(as_of).normalize()),
    ):
        classification, reason, details = q.classify_symbol_quarantine("SPCX")

    assert classification == "QUARANTINED_IPO_EDGE"
    assert "stale_or_insufficient_history" in reason
    assert details["raw_rows"] >= q.TRAIN_MIN_ROWS
    assert details["raw_rows"] <= q.IPO_HISTORY_MAX_ROWS


def test_fresh_ipo_band_symbol_remains_actionable_when_not_stale(tmp_path, monkeypatch):
    from services import symbol_hydration_quarantine as q

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    results = tmp_path / "results"
    raw_dir.mkdir()
    proc_dir.mkdir()
    results.mkdir()
    monkeypatch.setattr(q, "RAW", raw_dir)
    monkeypatch.setattr(q, "PROCESSED", proc_dir)
    monkeypatch.setattr(q, "RESULTS", results)
    monkeypatch.setattr(q, "UNIVERSE_ASSETS", tmp_path / "missing_universe.json")

    as_of = datetime(2026, 8, 6)
    dates = [as_of - timedelta(days=i) for i in range(33, -1, -1)]
    dates = dates[:34]
    assert dates[-1].date() == as_of.date()
    df = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1000,
        }
    )
    df.to_csv(raw_dir / "SPCX.csv", index=False)

    with (
        patch.object(q, "_pipeline_symbol_sets", return_value=(set(), set())),
        patch.object(q, "_as_of_date_normalized", return_value=pd.Timestamp(as_of).normalize()),
    ):
        classification, reason, details = q.classify_symbol_quarantine("SPCX")

    assert classification is None
    assert details.get("artifact_gap") == "PIPELINE_ARTIFACT_MISSING"


def test_apply_quarantine_removes_stale_ipo_from_actionable_stranded(tmp_path, monkeypatch):
    from services import symbol_hydration_quarantine as q

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    results = tmp_path / "results"
    raw_dir.mkdir()
    proc_dir.mkdir()
    results.mkdir()
    monkeypatch.setattr(q, "RAW", raw_dir)
    monkeypatch.setattr(q, "PROCESSED", proc_dir)
    monkeypatch.setattr(q, "RESULTS", results)
    monkeypatch.setattr(q, "UNIVERSE_ASSETS", tmp_path / "missing_universe.json")
    monkeypatch.setattr(q, "QUARANTINE_REGISTRY", tmp_path / "quarantine.json")

    as_of = datetime(2026, 8, 6)
    dates = [as_of - timedelta(days=i) for i in range(40, 6, -1)]
    df = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "close": 1.0,
        }
    )
    df.to_csv(raw_dir / "SPCX.csv", index=False)

    gaps = {
        "stranded_symbols": ["SPCX", "AAPL"],
        "actionable_stranded_symbols": ["SPCX", "AAPL"],
        "hydration_needed": True,
        "no_actionable_hydration_needed": False,
        "quarantined_symbols": [],
        "quarantine_reasons_by_symbol": {},
    }

    with (
        patch.object(q, "_pipeline_symbol_sets", return_value=(set(), set())),
        patch.object(q, "_as_of_date_normalized", return_value=pd.Timestamp(as_of).normalize()),
        patch.object(q, "_raw_row_count", side_effect=lambda s: 34 if s == "SPCX" else 500),
        patch.object(q, "_processed_row_count", return_value=0),
        patch.object(
            q,
            "_raw_last_date",
            side_effect=lambda s: pd.Timestamp("2026-07-31").normalize() if s == "SPCX" else None,
        ),
    ):
        # AAPL: large history, missing last-date helper → stays actionable via
        # raw_rows >= TRAIN and > IPO band (500 > 60).
        out = q.apply_quarantine_to_gaps(gaps)

    assert "SPCX" in out["quarantined_symbols"]
    assert "SPCX" not in out["actionable_stranded_symbols"]
    assert "SPCX" not in out["stranded_symbols"]


def test_preflight_quarantined_gaps_is_non_fatal_for_stale_ipo(tmp_path, monkeypatch):
    """E0 must return exit 0 when only stale IPO-band gaps remain."""
    from services import runtime_universe_hydration_preflight as pre
    from services import symbol_hydration_quarantine as q

    audit_path = tmp_path / "runtime_universe_hydration_preflight.json"
    monkeypatch.setattr(pre, "AUDIT_JSON", audit_path)
    monkeypatch.setattr(pre, "RESULTS", tmp_path)
    monkeypatch.setattr(pre, "PROCESSED", tmp_path / "processed")
    (tmp_path / "processed").mkdir()

    gaps = {
        "configured_symbols": ["SPCX"],
        "configured_symbol_count": 1,
        "processed_symbol_count": 1,
        "lifecycle_symbol_count": 0,
        "allocation_symbol_count": 0,
        "missing_processed_symbols": [],
        "missing_lifecycle_symbols": ["SPCX"],
        "missing_allocation_symbols": ["SPCX"],
        "missing_signals_symbols": ["SPCX"],
        "raw_stranded_symbols": ["SPCX"],
        "stranded_symbols": [],
        "artifact_regression_symbols": [],
        "pipeline_artifact_missing_symbols": [],
        "allocation_refresh_only_symbols": [],
        "quarantined_symbols": ["SPCX"],
        "quarantine_reasons_by_symbol": {
            "SPCX": "QUARANTINED_IPO_EDGE:stale_or_insufficient_history"
        },
        "quarantine_details_by_symbol": {
            "SPCX": {
                "classification": "QUARANTINED_IPO_EDGE",
                "reason": "stale_or_insufficient_history",
                "details": {"symbol": "SPCX"},
            }
        },
        "actionable_stranded_symbols": [],
        "no_actionable_hydration_needed": True,
        "hydration_needed": False,
    }

    with (
        patch.object(pre, "detect_hydration_gaps", return_value=gaps),
        patch.object(pre, "_market_open", return_value=False),
        patch.object(pre, "_live_execution_permitted", return_value=False),
        patch(
            "services.portfolio_history_sync.load_live_execution_flag",
            return_value=False,
        ),
        patch.object(
            q,
            "persist_quarantine_registry",
            return_value={"written": False, "skipped": True, "skip_reason": "test"},
        ),
    ):
        audit = pre.run_runtime_hydration_preflight(verbose=False)
        rc = pre.main([])

    assert audit["status"] == "QUARANTINED_GAPS"
    assert rc == 0
    assert audit_path.is_file()
    # Ensure test wrote only under tmp_path
    assert str(audit_path).startswith(str(tmp_path))
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    assert saved["status"] == "QUARANTINED_GAPS"
