"""SF61 provenance precision and writer exclusion. TEMP roots only."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_daily_refresh.ps1"
FRESH = ROOT / "scripts" / "run_data_freshness_refresh.ps1"
LOCK_UTILS = ROOT / "scripts" / "loop_lock_utils.ps1"
PY = Path(r"C:\Users\akimw\triton\.venv\Scripts\python.exe")
PROD = Path(r"C:\Users\akimw\triton")
WATCH = [
    PROD / "run_daily_refresh.ps1",
    PROD / "scripts" / "run_data_freshness_refresh.ps1",
    PROD / "data" / "results" / "daily_refresh_fetch_receipt.json",
    PROD / "data" / "results" / "daily_prediction_refresh_summary.json",
    PROD / "data" / "results" / "training_receipt.json",
    PROD / "data" / "results" / "heartbeat.json",
    PROD / "data" / "results" / "positions_snapshot.csv",
    PROD / "data" / "results" / "recent_orders.csv",
    PROD
    / "data"
    / "results"
    / ("daily_prediction_refresh_stage_receipts_" "daily_prediction_refresh_20260926_090907.json"),
    PROD / "logs" / "daily_refresh_20260926_090907.log",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session() -> str:
    proc = subprocess.run(
        [
            str(PY),
            "-c",
            "from services.market_calendar import last_completed_nyse_session; "
            "print(last_completed_nyse_session().isoformat())",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout
    return proc.stdout.strip()


def _ps(tmp: Path, body: str) -> subprocess.CompletedProcess[str]:
    driver = tmp / "driver.ps1"
    driver.parent.mkdir(parents=True, exist_ok=True)
    driver.write_text(body, encoding="utf-8")
    return subprocess.run(
        ["powershell", "-NoProfile", "-File", str(driver)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _holder(name: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Start-Sleep -Seconds 90; Write-Output '{name}'",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_visible(tmp: Path, name: str) -> None:
    body = f"""
. '{LOCK_UTILS}'
if (Test-LoopProcessRunning -ScriptName '{name}') {{ 'VISIBLE' }} else {{ 'HIDDEN' }}
"""
    for _ in range(20):
        proc = _ps(tmp, body)
        if "VISIBLE" in proc.stdout:
            return
        time.sleep(0.25)
    raise AssertionError(proc.stdout)


def _receipt(session: str) -> dict:
    return {
        "target_session": session,
        "tradable_coverage": {
            "covered": 63,
            "denominator": 63,
            "ratio": 1.0,
            "minimum": 0.9,
        },
        "decisions": [
            {
                "symbol": "SPY",
                "action": "promote",
                "reason": "monotonic_append",
                "index": False,
            }
        ],
        "dated_skips": [],
        "promoted": ["SPY"],
        "production_raw_unchanged": False,
        "passed": True,
        "promotion_error": None,
    }


def _bind(
    tmp: Path,
    *,
    mtime: datetime,
    captured: datetime,
    started: str,
    ended: str,
) -> subprocess.CompletedProcess[str]:
    results = tmp / "data" / "results"
    results.mkdir(parents=True, exist_ok=True)
    artifact = results / "daily_refresh_fetch_receipt.json"
    artifact.write_text(json.dumps(_receipt(_session())), encoding="utf-8")
    os.utime(artifact, (mtime.timestamp(), mtime.timestamp()))
    baseline = results / "baseline.json"
    baseline.write_text(
        json.dumps({"captured_at_utc": _stamp(captured), "artifacts": {}}),
        encoding="utf-8",
    )
    receipt_path = results / "stage.json"
    body = f"""
$ErrorActionPreference = 'Stop'
$resultsDir = '{results}'
$baselinePath = '{baseline}'
$receiptsPath = '{receipt_path}'
$runId = 'daily_prediction_refresh_synthetic_sf61'
$py = '{PY}'
. '{SCRIPT}'
$script:StageReceipts = @([ordered]@{{
    name = 'fetch_raw_data'
    started_at = '{started}'
    ended_at = '{ended}'
    exit_code = 0
    outcome = 'ok'
}})
try {{
    Bind-FetchArtifact
    Write-Output 'BIND_OK'
}} catch {{
    Write-Output ('BIND_FAIL ' + $_.Exception.Message)
}}
"""
    return _ps(tmp, body)


def test_round_trip_keeps_fractional_seconds(tmp_path: Path) -> None:
    proc = _ps(
        tmp_path,
        f"""
. '{SCRIPT}'
$parsed = ConvertFrom-UtcRoundTripText '2026-09-26T13:12:10.1266632Z'
$text = ConvertTo-UtcRoundTripText $parsed
Write-Output $text
Write-Output (ConvertTo-UtcRoundTripText (ConvertFrom-UtcRoundTripText '2026-09-26T13:12:10Z'))
""",
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines[-2] == "2026-09-26T13:12:10.1266632Z", proc.stdout
    assert lines[-1] == "2026-09-26T13:12:10Z", proc.stdout


def test_same_second_fractional_write_binds(tmp_path: Path) -> None:
    moment = datetime(2026, 9, 26, 13, 12, 10, 126663, tzinfo=timezone.utc)
    proc = _bind(
        tmp_path,
        mtime=moment,
        captured=moment - timedelta(minutes=5),
        started="2026-09-26T13:11:39.0000000Z",
        ended="2026-09-26T13:12:10.5000000Z",
    )
    assert "BIND_OK" in proc.stdout, proc.stdout
    row = json.loads((tmp_path / "data" / "results" / "stage.json").read_text(encoding="utf-8"))[0]
    assert row["artifact_written_at_utc"].startswith("2026-09-26T13:12:10.")
    assert not row["artifact_written_at_utc"].endswith(":10Z")


def test_stale_and_late_artifacts_fail(tmp_path: Path) -> None:
    moment = datetime(2026, 9, 26, 13, 12, 10, 126663, tzinfo=timezone.utc)
    stale = _bind(
        tmp_path / "stale",
        mtime=moment - timedelta(minutes=10),
        captured=moment - timedelta(minutes=5),
        started="2026-09-26T13:11:39.0000000Z",
        ended="2026-09-26T13:12:10.5000000Z",
    )
    assert "FETCH_ARTIFACT_PREDATED_BASELINE" in stale.stdout, stale.stdout
    late = _bind(
        tmp_path / "late",
        mtime=moment,
        captured=moment - timedelta(minutes=5),
        started="2026-09-26T13:11:39.0000000Z",
        ended="2026-09-26T13:12:10.1000000Z",
    )
    assert "FETCH_ARTIFACT_PROVENANCE_WINDOW" in late.stdout, late.stdout


def test_lock_order_release_and_refusal(tmp_path: Path) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    window = source.split("Invoke-LockedDailyRefreshWindow -LiveDir $liveDir -Body {", 1)[1]
    assert window.index('Run-PyStep -Name "capture_baseline"') < window.index(
        'Run-PyStep -Name "daily_prediction_refresh_validate"'
    )
    assert (
        "Exit-DailyRefreshMutationLock"
        in source.split("function Invoke-LockedDailyRefreshWindow", 1)[1].split(
            "function Bind-FetchArtifact", 1
        )[0]
    )
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    marker = tmp_path / "held.txt"
    success = _ps(
        tmp_path / "ok",
        f"""
. '{SCRIPT}'
Invoke-LockedDailyRefreshWindow -LiveDir '{live}' -Body {{
    if (-not (Test-Path -LiteralPath '{live / "daily_refresh.lock"}')) {{ throw 'LOCK_MISSING' }}
    Set-Content -LiteralPath '{marker}' -Value 'held'
}}
if (Test-Path -LiteralPath '{live / "daily_refresh.lock"}') {{ 'STILL_HELD' }} else {{ 'RELEASED' }}
""",
    )
    assert "RELEASED" in success.stdout, success.stdout
    assert marker.read_text(encoding="utf-8").strip() == "held"
    failed = _ps(
        tmp_path / "bad",
        f"""
. '{SCRIPT}'
try {{
    Invoke-LockedDailyRefreshWindow -LiveDir '{live}' -Body {{ throw 'STEP_FAILED:fetch_raw_data:2' }}
}} catch {{
    Write-Output 'CAUGHT'
}}
if (Test-Path -LiteralPath '{live / "daily_refresh.lock"}') {{ 'STILL_HELD' }} else {{ 'RELEASED' }}
""",
    )
    assert "CAUGHT" in failed.stdout and "RELEASED" in failed.stdout, failed.stdout


def test_live_foreign_lock_is_not_removed(tmp_path: Path) -> None:
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    lock = live / "data_freshness_refresh.lock"
    lock.write_text("pid=424242\nsource=data_freshness_refresh\n", encoding="utf-8")
    before = lock.read_bytes()
    holder = _holder("run_data_freshness_refresh.ps1")
    try:
        _wait_visible(tmp_path / "see", "run_data_freshness_refresh.ps1")
        proc = _ps(
            tmp_path / "refuse",
            f"""
. '{SCRIPT}'
if (Enter-DailyRefreshMutationLock -LiveDir '{live}') {{ 'ACQUIRED' }} else {{ 'REFUSED' }}
""",
        )
        assert "REFUSED" in proc.stdout, proc.stdout
        assert lock.read_bytes() == before
        assert not (live / "daily_refresh.lock").exists()
    finally:
        holder.kill()


def test_freshness_refuses_without_snapshot_writes(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(LOCK_UTILS, scripts / "loop_lock_utils.ps1")
    for rel in (
        "data/results/heartbeat.json",
        "data/results/positions_snapshot.csv",
        "data/results/recent_orders.csv",
        "data/raw/SPY.csv",
        "data/processed/SPY.parquet",
        "data/predictions/SPY_predictions.parquet",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen-" + rel.encode("ascii"))
    before = {
        rel: (tmp_path / rel).read_bytes()
        for rel in (
            "data/results/heartbeat.json",
            "data/results/positions_snapshot.csv",
            "data/results/recent_orders.csv",
            "data/raw/SPY.csv",
            "data/processed/SPY.parquet",
            "data/predictions/SPY_predictions.parquet",
        )
    }
    lock = tmp_path / "data" / "live" / "daily_refresh.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("pid=424242\nsource=daily_refresh\n", encoding="utf-8")
    original = lock.read_bytes()
    holder = _holder("run_daily_refresh.ps1")
    try:
        _wait_visible(tmp_path / "see", "run_daily_refresh.ps1")
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-File",
                str(FRESH),
                "-RepoRoot",
                str(tmp_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout
        assert "blocked: daily refresh owns the writer lock" in proc.stdout
        assert lock.read_bytes() == original
        for rel, payload in before.items():
            assert (tmp_path / rel).read_bytes() == payload
    finally:
        holder.kill()


def test_overlapping_launches_allow_one_writer(tmp_path: Path) -> None:
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    first = tmp_path / "first.ps1"
    second = tmp_path / "second.ps1"
    first.write_text(
        f"""
. '{SCRIPT}'
if (Enter-DailyRefreshMutationLock -LiveDir '{live}') {{
    Set-Content -LiteralPath '{tmp_path / "first.txt"}' -Value 'ACQUIRED'
    Start-Sleep -Seconds 20
    Exit-DailyRefreshMutationLock -LiveDir '{live}'
}} else {{
    Set-Content -LiteralPath '{tmp_path / "first.txt"}' -Value 'REFUSED'
}}
""",
        encoding="utf-8",
    )
    second.write_text(
        f"""
. '{SCRIPT}'
if (Enter-DailyRefreshMutationLock -LiveDir '{live}') {{ 'ACQUIRED' }} else {{ 'REFUSED' }}
if (Test-Path -LiteralPath '{live / "daily_refresh.lock"}') {{ 'LOCK_PRESENT' }} else {{ 'LOCK_ABSENT' }}
""",
        encoding="utf-8",
    )
    holder = subprocess.Popen(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f". '{SCRIPT}'; "
                f"if (Enter-DailyRefreshMutationLock -LiveDir '{live}') {{ "
                f"Set-Content -LiteralPath '{tmp_path / 'first.txt'}' -Value 'ACQUIRED'; "
                f"Start-Sleep -Seconds 20; "
                f"Exit-DailyRefreshMutationLock -LiveDir '{live}' "
                f"}} else {{ Set-Content -LiteralPath '{tmp_path / 'first.txt'}' -Value 'REFUSED' }}"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            if (tmp_path / "first.txt").exists():
                break
            time.sleep(0.25)
        assert (tmp_path / "first.txt").read_text(encoding="utf-8").strip() == "ACQUIRED"
        _wait_visible(tmp_path / "see", "run_daily_refresh.ps1")
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(second)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert "REFUSED" in proc.stdout and "LOCK_PRESENT" in proc.stdout, proc.stdout
    finally:
        holder.kill()


def test_dot_source_does_not_write_production(tmp_path: Path) -> None:
    before = {path: _sha(path) for path in WATCH if path.is_file()}
    proc = _ps(tmp_path, f". '{SCRIPT}'\nWrite-Output 'DEFINED'")
    assert "DEFINED" in proc.stdout, proc.stdout
    after = {path: _sha(path) for path in before}
    assert after == before
