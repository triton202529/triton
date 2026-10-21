"""SF55 atomic stage-receipt publication. No production writes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_daily_refresh.ps1"
RUN_ID = "daily_prediction_refresh_20260925_183007"
HISTORICAL = Path(
    r"C:\Users\akimw\triton\data\results"
    r"\daily_prediction_refresh_stage_receipts_daily_prediction_refresh_20260925_183007.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp: Path, body: str) -> subprocess.CompletedProcess[str]:
    driver = tmp / "driver.ps1"
    driver.write_text(body, encoding="utf-8")
    return subprocess.run(
        ["powershell", "-NoProfile", "-File", str(driver)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _generation(results: Path, generation: int) -> Path:
    return results / ("daily_prediction_refresh_stage_receipts_" f"{RUN_ID}.gen.{generation}.json")


def _publish(tmp: Path, results: Path, stages: list[dict]) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    spec = tmp / "stages.json"
    spec.write_text(json.dumps(stages), encoding="utf-8")
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}'
$script:StageReceipts = @(Get-Content -LiteralPath '{spec}' -Raw | ConvertFrom-Json)
Write-StageReceipts -ResultsDirectory '{results}' -RunId '{RUN_ID}'
"""
    proc = _run(tmp, body)
    assert proc.returncode == 0, proc.stdout


def test_first_publish_and_replace_and_spaces(tmp_path: Path) -> None:
    results = tmp_path / "dir with spaces"
    legacy = results / f"daily_prediction_refresh_stage_receipts_{RUN_ID}.json"
    _publish(tmp_path, results, [{"name": "capture_baseline", "exit_code": 0, "outcome": "ok"}])
    assert not legacy.exists()
    first_path = _generation(results, 1)
    raw = first_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    first = json.loads(raw.decode("utf-8"))
    assert first[0]["name"] == "capture_baseline"
    before = first_path.read_bytes()
    _publish(
        tmp_path,
        results,
        [
            {"name": "capture_baseline", "exit_code": 0, "outcome": "ok", "status": "COMPLETE"},
            {"name": "fetch_raw_data", "exit_code": 0, "outcome": "ok", "status": "COMPLETE"},
        ],
    )
    assert first_path.read_bytes() == before
    second = json.loads(_generation(results, 2).read_text(encoding="utf-8"))
    assert [row["name"] for row in second] == ["capture_baseline", "fetch_raw_data"]
    assert list(results.glob("*.tmp")) == []


def test_replace_failure_preserves_bytes_and_cleans_temp(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    legacy = results / f"daily_prediction_refresh_stage_receipts_{RUN_ID}.json"
    legacy.write_text('[{"name":"legacy"}]\n', encoding="utf-8")
    before = legacy.read_bytes()
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}'
$script:StageReceipts = @([pscustomobject]@{{ name = 'fetch_raw_data'; exit_code = 0 }})
$fs = [System.IO.File]::Open('{legacy}', 'Open', 'Read', 'Read')
try {{
    Write-StageReceipts -ResultsDirectory '{results}' -RunId '{RUN_ID}'
}} finally {{
    $fs.Dispose()
}}
$left = @(Get-ChildItem -LiteralPath '{results}' -Force -Filter '*.tmp')
if ($left.Count -ne 0) {{ throw 'temporary file remains' }}
"""
    proc = _run(tmp_path, body)
    assert proc.returncode == 0, proc.stdout
    assert legacy.read_bytes() == before
    published = json.loads(_generation(results, 1).read_text(encoding="utf-8"))
    assert published[0]["name"] == "fetch_raw_data"


def test_two_runs_are_deterministic(tmp_path: Path) -> None:
    one = tmp_path / "a"
    two = tmp_path / "b"
    stages = [
        {"name": "capture_baseline", "exit_code": 0, "outcome": "ok", "status": "COMPLETE"},
        {
            "name": "fetch_raw_data",
            "run_id": "daily_prediction_refresh_synthetic_sf55",
            "exit_code": 0,
            "outcome": "ok",
            "status": "COMPLETE",
            "artifact_path": "data/results/daily_refresh_fetch_receipt.json",
        },
    ]
    _publish(tmp_path / "run1", one, stages)
    _publish(tmp_path / "run2", two, stages)
    assert json.loads(_generation(one, 1).read_text(encoding="utf-8")) == json.loads(
        _generation(two, 1).read_text(encoding="utf-8")
    )


def test_historical_receipt_and_import_are_untouched(tmp_path: Path) -> None:
    before = _sha(HISTORICAL) if HISTORICAL.is_file() else None
    proc = _run(
        tmp_path,
        f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}'
if (Get-Command Write-StageReceipts -ErrorAction SilentlyContinue) {{ 'defined' }}
""",
    )
    assert proc.returncode == 0, proc.stdout
    assert "defined" in proc.stdout
    if before is not None:
        assert _sha(HISTORICAL) == before


def test_parser_has_zero_errors() -> None:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$e=$null; $t=$null; "
            f"[void][System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}', [ref]$t, [ref]$e); "
            "if ($e) { $e.Count } else { 0 }",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip().endswith("0")
