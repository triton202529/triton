"""SF58 fetch-artifact binding. No production writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_daily_refresh.ps1"
PY = Path(r"C:\Users\akimw\triton\.venv\Scripts\python.exe")
PROD = Path(r"C:\Users\akimw\triton")
FROZEN = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF58_20260925_224200\frozen"
    r"\data\results\daily_refresh_fetch_receipt.json"
)
FETCH_SHA = "afe0ebdb937d055b27eea27431fd89014a65c2b31c2325636c785661572e7bc7"
SF57_RECEIPT = PROD / (
    "data/results/daily_prediction_refresh_stage_receipts_"
    "daily_prediction_refresh_20260925_221253.json"
)
SF57_RECEIPT_SHA = "43a7296f0f3fab1b2ae87e926891af375abcde856783246e13d422465ee49d32"
HIST_RECEIPT = PROD / (
    "data/results/daily_prediction_refresh_stage_receipts_"
    "daily_prediction_refresh_20260925_183007.json"
)
HIST_RECEIPT_SHA = "790e61ef2758c345eb80fc4326e4b6084df1e9bb4f2099eb41d930ce7a1df947"
HIST_TMP = PROD / (
    "data/results/daily_prediction_refresh_stage_receipts_"
    "daily_prediction_refresh_20260925_183007.json.tmp"
)
SUMMARY = PROD / "data/results/daily_prediction_refresh_summary.json"
SUMMARY_SHA = "eec5099c5c386d0d0a94c840e92e3837dabe6acb8d1071bd93f6dfb0a10d8b76"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _receipt(session: str, **overrides) -> dict:
    payload = {
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
    payload.update(overrides)
    return payload


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _bind(
    tmp: Path,
    payload: dict,
    *,
    mtime: datetime,
    captured: datetime,
    started: datetime,
    ended: datetime,
    prior_sha: str = "0" * 64,
    hook: str = "",
) -> subprocess.CompletedProcess[str]:
    results = tmp / "data" / "results"
    results.mkdir(parents=True, exist_ok=True)
    artifact = results / "daily_refresh_fetch_receipt.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(artifact, (mtime.timestamp(), mtime.timestamp()))
    baseline = results / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "captured_at_utc": _stamp(captured),
                "artifacts": {
                    "data/results/daily_refresh_fetch_receipt.json": {
                        "exists": True,
                        "sha256": prior_sha,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = results / "stage.json"
    body = f"""
$ErrorActionPreference = 'Stop'
$resultsDir = '{results}'
$baselinePath = '{baseline}'
$receiptsPath = '{receipt}'
$runId = 'daily_prediction_refresh_synthetic_sf58'
$py = '{PY}'
. '{SCRIPT}'
$script:StageReceipts = @([ordered]@{{
    name = 'fetch_raw_data'
    started_at = '{_stamp(started)}'
    ended_at = '{_stamp(ended)}'
    exit_code = 0
    outcome = 'ok'
}})
{hook}
try {{
    Bind-FetchArtifact
    Write-Output 'BIND_OK'
}} catch {{
    Write-Output ('BIND_FAIL ' + $_.Exception.Message)
}}
Write-Output ('ATTEMPTS=' + $script:FetchBindingPublishAttempts)
"""
    return _run(tmp, body)


def test_production_schema_binds_and_large_document_parses(tmp_path: Path) -> None:
    session = _session()
    moment = datetime(2026, 9, 25, 22, 14, 2, tzinfo=timezone.utc)
    proc = _bind(
        tmp_path,
        _receipt(session),
        mtime=moment,
        captured=moment - timedelta(minutes=5),
        started=moment - timedelta(minutes=1),
        ended=moment + timedelta(minutes=1),
    )
    assert "BIND_OK" in proc.stdout, proc.stdout
    assert "ATTEMPTS=1" in proc.stdout
    receipt = json.loads((tmp_path / "data" / "results" / "stage.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in receipt] == ["fetch_raw_data"]
    row = receipt[0]
    assert row["run_id"] == "daily_prediction_refresh_synthetic_sf58"
    assert row["status"] == "COMPLETE"
    assert row["target_session"] == session
    assert row["artifact_path"] == "data/results/daily_refresh_fetch_receipt.json"
    assert row["artifact_after_baseline"] is True
    assert row["baseline_captured_at_utc"] == _stamp(moment - timedelta(minutes=5))
    artifact = tmp_path / "data" / "results" / "daily_refresh_fetch_receipt.json"
    assert row["artifact_sha256"] == _sha(artifact)
    assert row["artifact_size_bytes"] == artifact.stat().st_size
    assert row["artifact_written_at_utc"] == _stamp(moment)

    large = _receipt(session, padding="x" * 1_100_000)
    proc = _bind(
        tmp_path / "large",
        large,
        mtime=moment,
        captured=moment - timedelta(minutes=5),
        started=moment - timedelta(minutes=1),
        ended=moment + timedelta(minutes=1),
    )
    assert "BIND_OK" in proc.stdout, proc.stdout
    stored = (
        (tmp_path / "large" / "data" / "results" / "daily_refresh_fetch_receipt.json")
        .stat()
        .st_size
    )
    assert stored > 1_000_000


def test_specific_rejections(tmp_path: Path) -> None:
    session = _session()
    moment = datetime(2026, 9, 25, 22, 14, 2, tzinfo=timezone.utc)
    cases = [
        (
            "passed-string",
            _receipt(session, passed="true"),
            moment,
            "FETCH_ARTIFACT_PASSED_NOT_BOOLEAN",
        ),
        ("passed-false", _receipt(session, passed=False), moment, "FETCH_ARTIFACT_FAILED"),
        (
            "wrong-session",
            _receipt(session, target_session="1999-01-01"),
            moment,
            "FETCH_ARTIFACT_WRONG_SESSION",
        ),
        (
            "low-coverage",
            _receipt(
                session,
                tradable_coverage={
                    "covered": 30,
                    "denominator": 63,
                    "ratio": 30 / 63,
                    "minimum": 0.9,
                },
            ),
            moment,
            "FETCH_ARTIFACT_COVERAGE",
        ),
        (
            "missing-session",
            _receipt(session),
            moment,
            "FETCH_ARTIFACT_SESSION_MISSING",
        ),
        (
            "predates",
            _receipt(session),
            moment - timedelta(minutes=10),
            "FETCH_ARTIFACT_PREDATED_BASELINE",
        ),
        (
            "provenance-missing",
            _receipt(session),
            datetime(1999, 1, 1, tzinfo=timezone.utc),
            "FETCH_ARTIFACT_PROVENANCE_MISSING",
        ),
        (
            "provenance-window",
            _receipt(session),
            moment + timedelta(minutes=10),
            "FETCH_ARTIFACT_PROVENANCE_WINDOW",
        ),
    ]
    missing = cases[4][1]
    missing.pop("target_session")
    for name, payload, stamp, reason in cases:
        proc = _bind(
            tmp_path / name,
            payload,
            mtime=stamp,
            captured=moment - timedelta(minutes=5),
            started=moment - timedelta(minutes=1),
            ended=moment + timedelta(minutes=1),
        )
        assert reason in proc.stdout, (name, proc.stdout)
        assert "ATTEMPTS=0" in proc.stdout


def test_malformed_json_and_byte_change_do_not_publish(tmp_path: Path) -> None:
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    artifact = results / "daily_refresh_fetch_receipt.json"
    artifact.write_text("{", encoding="utf-8")
    moment = datetime(2026, 9, 25, 22, 14, 2, tzinfo=timezone.utc)
    baseline = results / "baseline.json"
    baseline.write_text(
        json.dumps({"captured_at_utc": _stamp(moment - timedelta(minutes=5)), "artifacts": {}}),
        encoding="utf-8",
    )
    receipt = results / "stage.json"
    body = f"""
$ErrorActionPreference = 'Stop'
$resultsDir = '{results}'
$baselinePath = '{baseline}'
$receiptsPath = '{receipt}'
$runId = 'daily_prediction_refresh_synthetic_sf58'
$py = '{PY}'
. '{SCRIPT}'
$script:StageReceipts = @([ordered]@{{
    name = 'fetch_raw_data'
    started_at = '{_stamp(moment - timedelta(minutes=1))}'
    ended_at = '{_stamp(moment + timedelta(minutes=1))}'
    exit_code = 0
    outcome = 'ok'
}})
try {{ Bind-FetchArtifact; Write-Output 'BIND_OK' }} catch {{ Write-Output ('BIND_FAIL ' + $_.Exception.Message) }}
Write-Output ('ATTEMPTS=' + $script:FetchBindingPublishAttempts)
"""
    proc = _run(tmp_path, body)
    assert "FETCH_ARTIFACT_JSON_PARSE" in proc.stdout, proc.stdout
    assert "ATTEMPTS=0" in proc.stdout
    assert not receipt.exists()

    session = _session()
    hook = r"""
$script:FetchBindingAfterRead = {
    param($path)
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $bytes[5] = 90
    [System.IO.File]::WriteAllBytes($path, $bytes)
}
"""
    proc = _bind(
        tmp_path / "changed",
        _receipt(session),
        mtime=moment,
        captured=moment - timedelta(minutes=5),
        started=moment - timedelta(minutes=1),
        ended=moment + timedelta(minutes=1),
        hook=hook,
    )
    assert "FETCH_ARTIFACT_CHANGED" in proc.stdout, proc.stdout
    assert "ATTEMPTS=0" in proc.stdout


def test_failed_publication_preserves_receipt_once(tmp_path: Path) -> None:
    session = _session()
    moment = datetime(2026, 9, 25, 22, 14, 2, tzinfo=timezone.utc)
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    artifact = results / "daily_refresh_fetch_receipt.json"
    artifact.write_text(json.dumps(_receipt(session)), encoding="utf-8")
    os.utime(artifact, (moment.timestamp(), moment.timestamp()))
    baseline = results / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "captured_at_utc": _stamp(moment - timedelta(minutes=5)),
                "artifacts": {
                    "data/results/daily_refresh_fetch_receipt.json": {
                        "exists": True,
                        "sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = results / "stage.json"
    body = f"""
$ErrorActionPreference = 'Stop'
$resultsDir = '{results}'
$baselinePath = '{baseline}'
$receiptsPath = '{receipt}'
$runId = 'daily_prediction_refresh_synthetic_sf58'
$py = '{PY}'
. '{SCRIPT}'
$script:StageReceipts = @([ordered]@{{
    name = 'fetch_raw_data'
    started_at = '{_stamp(moment - timedelta(minutes=1))}'
    ended_at = '{_stamp(moment + timedelta(minutes=1))}'
    exit_code = 0
    outcome = 'ok'
}})
Write-StageReceipts
$before = [System.IO.File]::ReadAllBytes('{receipt}')
$fs = [System.IO.File]::Open('{receipt}', 'Open', 'Read', 'None')
try {{
    Bind-FetchArtifact
    Write-Output 'BIND_OK'
}} catch {{
    Write-Output ('BIND_FAIL ' + $_.Exception.Message)
}} finally {{
    $fs.Dispose()
}}
Write-Output ('ATTEMPTS=' + $script:FetchBindingPublishAttempts)
$after = [System.IO.File]::ReadAllBytes('{receipt}')
$same = $true
if ($before.Length -ne $after.Length) {{ $same = $false }}
else {{
    for ($i = 0; $i -lt $before.Length; $i++) {{
        if ($before[$i] -ne $after[$i]) {{ $same = $false; break }}
    }}
}}
Write-Output ('UNCHANGED=' + $same)
"""
    proc = _run(tmp_path, body)
    assert "BIND_FAIL" in proc.stdout, proc.stdout
    assert "ATTEMPTS=1" in proc.stdout
    assert "UNCHANGED=True" in proc.stdout
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert "artifact_sha256" not in saved[0]


def test_frozen_artifact_validator_and_failures(tmp_path: Path) -> None:
    from services.daily_prediction_refresh_validate import _bound_fetch_receipt

    session = _session()
    direct = tmp_path / "direct"
    direct_results = direct / "data" / "results"
    direct_results.mkdir(parents=True)
    direct_artifact = direct_results / "daily_refresh_fetch_receipt.json"
    shutil.copy2(FROZEN, direct_artifact)
    assert direct_artifact.read_bytes() == FROZEN.read_bytes()
    mtime = datetime.fromtimestamp(direct_artifact.stat().st_mtime, timezone.utc)
    captured = mtime - timedelta(minutes=5)
    started = mtime - timedelta(minutes=2)
    ended = mtime + timedelta(minutes=2)
    baseline = direct_results / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "captured_at_utc": _stamp(captured),
                "artifacts": {
                    "data/results/daily_refresh_fetch_receipt.json": {
                        "exists": True,
                        "sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    receipt_path = direct_results / "stage.json"
    body = f"""
$ErrorActionPreference = 'Stop'
$resultsDir = '{direct_results}'
$baselinePath = '{baseline}'
$receiptsPath = '{receipt_path}'
$runId = 'daily_prediction_refresh_synthetic_sf58'
$py = '{PY}'
. '{SCRIPT}'
$script:StageReceipts = @([ordered]@{{
    name = 'fetch_raw_data'
    started_at = '{_stamp(started)}'
    ended_at = '{_stamp(ended)}'
    exit_code = 0
    outcome = 'ok'
}})
Bind-FetchArtifact
Write-Output 'BIND_OK'
"""
    proc = _run(direct, body)
    assert "BIND_OK" in proc.stdout, proc.stdout
    assert direct_artifact.read_bytes() == FROZEN.read_bytes()
    stage = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stage[0]["artifact_sha256"] == FETCH_SHA
    assert stage[0]["artifact_size_bytes"] == 1222054
    assert stage[0]["target_session"] == session
    assert stage[0]["run_id"] == "daily_prediction_refresh_synthetic_sf58"
    rejected, parsed = _bound_fetch_receipt(
        repo_root=direct,
        target=datetime.fromisoformat(session).date(),
        run_id="daily_prediction_refresh_synthetic_sf58",
        stage_receipts=stage,
        baseline=json.loads(baseline.read_text(encoding="utf-8")),
        run_started_at=captured - timedelta(minutes=1),
    )
    assert rejected == {}, rejected
    assert parsed["passed"] is True
    assert parsed["target_session"] == session

    raw = bytearray(direct_artifact.read_bytes())
    raw[20] ^= 0x01
    direct_artifact.write_bytes(raw)
    rejected, _parsed = _bound_fetch_receipt(
        repo_root=direct,
        target=datetime.fromisoformat(session).date(),
        run_id="daily_prediction_refresh_synthetic_sf58",
        stage_receipts=stage,
        baseline=json.loads(baseline.read_text(encoding="utf-8")),
        run_started_at=captured - timedelta(minutes=1),
    )
    assert rejected.get("receipt") == "forged_fetch_artifact"
    direct_artifact.write_bytes(FROZEN.read_bytes())
    unbound = [dict(stage[0])]
    unbound[0].pop("artifact_sha256")
    rejected, _parsed = _bound_fetch_receipt(
        repo_root=direct,
        target=datetime.fromisoformat(session).date(),
        run_id="daily_prediction_refresh_synthetic_sf58",
        stage_receipts=unbound,
        baseline=json.loads(baseline.read_text(encoding="utf-8")),
        run_started_at=captured - timedelta(minutes=1),
    )
    assert rejected.get("receipt") == "missing_binding_field"


def test_import_parse_and_historical_evidence_are_untouched(tmp_path: Path) -> None:
    before = {
        PROD / "data/results/daily_refresh_fetch_receipt.json": FETCH_SHA,
        SF57_RECEIPT: SF57_RECEIPT_SHA,
        HIST_RECEIPT: HIST_RECEIPT_SHA,
        HIST_TMP: _sha(HIST_TMP),
        SUMMARY: SUMMARY_SHA,
    }
    proc = _run(
        tmp_path,
        f"""
. '{SCRIPT}'
Write-Output 'DEFINED'
""",
    )
    assert proc.returncode == 0, proc.stdout
    assert "DEFINED" in proc.stdout
    for path, digest in before.items():
        assert _sha(path) == digest
    historical = json.loads(SF57_RECEIPT.read_text(encoding="utf-8"))
    assert [row["name"] for row in historical] == ["fetch_raw_data"]
    assert "artifact_sha256" not in historical[0]
    parse = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$e=$null; $t=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('"
            + str(SCRIPT)
            + "', [ref]$t, [ref]$e); if ($e) { $e.Count } else { 0 }",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert parse.stdout.strip().splitlines()[-1] == "0"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "$text | ConvertFrom-Json" not in source
    assert "ConvertFrom-Json -InputObject" in source
    assert "yfinance" not in source
    assert "alpaca" not in source.lower()
