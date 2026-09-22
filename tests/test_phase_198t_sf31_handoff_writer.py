"""Isolated import and call contract for the market-open handoff writer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_market_open.ps1"
CONTROL_SCRIPT = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF31_control_isolated\scripts\run_market_open.ps1"
)
SCHEMA_KEYS = [
    "schema_version",
    "status",
    "market_open_invocation_id",
    "session_date",
    "issued_at",
    "source",
    "creates_no_authority",
    "live_execution_permitted",
    "broker_orders_submitted",
    "ready_at",
    "generated_at",
    "interlock_sha256",
    "f_binding_sha256",
]
INV = "mo-20260921_171800-sf310000000000000000000000000001"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_ps(body: str, cwd: Path, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", body],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _writer_prefix(root: Path) -> str:
    return f". '{SCRIPT}' -FunctionsOnly\n"


def test_functions_only_precedes_every_side_effect():
    src = SCRIPT.read_text(encoding="utf-8")
    ret = src.index("if ($FunctionsOnly)")
    assert ret < src.index("Set-Location $repoRoot")
    assert ret < src.index("Start-Transcript")
    assert ret < src.index("$script:MarketOpenInvocationId = (")
    assert ret < src.index('Write-InvocationHandoff -Status "PENDING"')
    assert ret < src.index("Run-Step -Label")
    assert 'Write-InvocationHandoff -Status "READY"' in src
    assert "function Write-Utf8NoBomAtomicFile" in src
    assert "UTF8Encoding($false)" in src


def test_functions_only_load_has_no_side_effects(tmp_path: Path):
    scheduled = ROOT / "data" / "results" / "scheduled_runs"
    before = {p.name for p in scheduled.glob("market_open_*.log")} if scheduled.is_dir() else set()
    body = f"""
$ErrorActionPreference = 'Stop'
$before = (Get-Location).Path
$beforeId = $env:TRITON_MARKET_OPEN_INVOCATION_ID
$beforeKids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$PID" | ForEach-Object {{ $_.Name }})
{_writer_prefix(ROOT)}
$after = (Get-Location).Path
if ($before -ne $after) {{ throw 'CWD_CHANGED' }}
if ($env:TRITON_MARKET_OPEN_INVOCATION_ID -ne $beforeId) {{ throw 'ENV_ID_CHANGED' }}
if ($script:MarketOpenInvocationId) {{ throw 'ID_GENERATED' }}
$afterKids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$PID" | ForEach-Object {{ $_.Name }})
$newKids = @($afterKids | Where-Object {{ $beforeKids -notcontains $_ }})
if ($newKids -contains 'python.exe') {{ throw 'PYTHON_STARTED' }}
if (-not (Get-Command Write-InvocationHandoff -ErrorAction SilentlyContinue)) {{ throw 'WRITER_MISSING' }}
'AFTER'
"""
    completed = _run_ps(body, tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, captured[-2000:]
    assert "AFTER" in completed.stdout
    assert "status=PENDING" not in captured
    assert "status=READY" not in captured
    assert list(tmp_path.rglob("*")) == []
    after = {p.name for p in scheduled.glob("market_open_*.log")} if scheduled.is_dir() else set()
    assert after == before


def _isolated_call(root: Path, *, extra: str, status: str = "READY") -> str:
    return f"""
$ErrorActionPreference = 'Stop'
{_writer_prefix(root)}
try {{
  Write-InvocationHandoff -Isolated -Status '{status}' {extra}
  'WROTE'
}} catch {{
  'ERR:' + $_.Exception.Message
  exit 0
}}
"""


def test_explicit_isolated_call_writes_one_bound_handoff(tmp_path: Path):
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    interlock = results / "paper_sell_exit_before_entry.json"
    readiness = results / "execution_readiness.json"
    binding = results / "manage_positions_cycle_binding.json"
    interlock.write_bytes(b'{"schema":"interlock"}\n')
    readiness.write_bytes(b'{"readiness_status":"READY"}\n')
    binding.write_bytes(b'{"f_stage_status":"COMPLETE"}\n')
    issued = "2026-09-21T17:17:22.0000000Z"
    ready = "2026-09-21T17:18:00.0000000Z"
    extra = " ".join(
        [
            f"-RepoRoot '{tmp_path}'",
            f"-InvocationId '{INV}'",
            "-SessionDate '2026-09-21'",
            f"-IssuedAt '{issued}'",
            f"-ReadyAt '{ready}'",
            f"-GeneratedAt '{ready}'",
            f"-InterlockPath '{interlock}'",
            f"-InterlockSha256 '{_sha(interlock)}'",
            f"-FBindingSha256 '{_sha(binding)}'",
            f"-ReadinessPath '{readiness}'",
            f"-ReadinessSha256 '{_sha(readiness)}'",
            "-Mode 'PAPER_ONLY'",
            "-LiveExecutionPermitted $false",
            f"-SourceArtifactHashes @{{ '{binding}' = '{_sha(binding)}' }}",
        ]
    )
    completed = _run_ps(_isolated_call(tmp_path, extra=extra), tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert "WROTE" in captured, captured[-2000:]
    dest = results / "market_open_invocation_handoff.json"
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert list(doc.keys()) == SCHEMA_KEYS
    assert doc["status"] == "READY"
    assert doc["market_open_invocation_id"] == INV
    assert doc["session_date"] == "2026-09-21"
    assert doc["issued_at"] == issued
    assert doc["ready_at"] == ready
    assert doc["generated_at"] == ready
    assert doc["interlock_sha256"] == _sha(interlock)
    assert doc["f_binding_sha256"] == _sha(binding)
    assert doc["source"] == "scripts/run_market_open.ps1"
    assert doc["creates_no_authority"] is True
    assert doc["live_execution_permitted"] is False
    assert doc["broker_orders_submitted"] is False
    assert doc["schema_version"] == 2
    assert dest.read_bytes()[:3] != b"\xef\xbb\xbf"


@pytest.mark.parametrize(
    "extra,token",
    [
        (
            "-RepoRoot '{root}' -SessionDate '2026-09-21' -IssuedAt '2026-09-21T17:17:22Z' "
            "-Mode 'PAPER' -LiveExecutionPermitted $false",
            "HANDOFF_ISOLATED_INVOCATION_ID_REQUIRED",
        ),
        (
            "-RepoRoot 'C:\\SF31_refused_root' -InvocationId '{inv}' "
            "-SessionDate '2026-09-21' -IssuedAt '2026-09-21T17:17:22Z' "
            "-Mode 'PAPER' -LiveExecutionPermitted $false",
            "HANDOFF_ISOLATED_ROOT_REFUSED",
        ),
        (
            "-RepoRoot '{root}' -HandoffPath 'C:\\SF31_refused_root\\handoff.json' "
            "-InvocationId '{inv}' -SessionDate '2026-09-21' -IssuedAt '2026-09-21T17:17:22Z' "
            "-Mode 'PAPER' -LiveExecutionPermitted $false",
            "HANDOFF_OUTPUT_OUTSIDE_ROOT",
        ),
        (
            "-RepoRoot '{root}' -InvocationId '{inv}' -SessionDate '2026-09-21' "
            "-IssuedAt '2026-09-21T17:17:22Z' -Mode 'PAPER'",
            "HANDOFF_LIVE_EVIDENCE_MISSING",
        ),
        (
            "-RepoRoot '{root}' -InvocationId '{inv}' -SessionDate '2026-09-21' "
            "-IssuedAt '2026-09-21T17:17:22Z' -Mode 'PAPER' -LiveExecutionPermitted $true",
            "HANDOFF_LIVE_EVIDENCE_AMBIGUOUS",
        ),
        (
            "-RepoRoot '{root}' -InvocationId '{inv}' -SessionDate '2026-09-21' "
            "-IssuedAt '2026-09-21T17:17:22Z' -Mode 'LIVE' -LiveExecutionPermitted $false",
            "HANDOFF_MODE_NOT_PAPER",
        ),
    ],
)
def test_isolated_writer_refuses_without_writing(tmp_path: Path, extra: str, token: str):
    rendered = extra.format(root=str(tmp_path), inv=INV)
    completed = _run_ps(_isolated_call(tmp_path, extra=rendered, status="PENDING"), tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert token in captured, captured[-2000:]
    assert "WROTE" not in captured
    assert not (tmp_path / "data" / "results" / "market_open_invocation_handoff.json").exists()
    assert not Path(r"C:\SF31_refused_root").exists()


def test_isolated_ready_rejects_interlock_hash_mismatch(tmp_path: Path):
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    interlock = results / "paper_sell_exit_before_entry.json"
    interlock.write_bytes(b"canonical\n")
    extra = " ".join(
        [
            f"-RepoRoot '{tmp_path}'",
            f"-InvocationId '{INV}'",
            "-SessionDate '2026-09-21'",
            "-IssuedAt '2026-09-21T17:17:22Z'",
            "-ReadyAt '2026-09-21T17:18:00Z'",
            f"-InterlockPath '{interlock}'",
            "-InterlockSha256 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
            "-FBindingSha256 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'",
            "-Mode 'PAPER'",
            "-LiveExecutionPermitted $false",
        ]
    )
    completed = _run_ps(_isolated_call(tmp_path, extra=extra), tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert "HANDOFF_INTERLOCK_HASH_MISMATCH" in captured, captured[-2000:]
    assert not (results / "market_open_invocation_handoff.json").exists()


def _install_fake_python(iso: Path, *, fail_f: bool) -> None:
    bin_dir = iso / "bin"
    scripts = iso / "scripts"
    results = iso / "data" / "results"
    bin_dir.mkdir(parents=True)
    scripts.mkdir(parents=True)
    results.mkdir(parents=True)
    (scripts / "triton_env_bootstrap.ps1").write_text(
        "# isolated no-op bootstrap\n", encoding="utf-8"
    )
    flag = "1" if fail_f else "0"
    fake = bin_dir / "fake_python.py"
    fake.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['FAKE_PY_LOG'])\n"
        "joined = ' '.join(sys.argv[1:])\n"
        "prior = log.read_text(encoding='utf-8') if log.is_file() else ''\n"
        "log.write_text(prior + joined + '\\n', encoding='utf-8')\n"
        "if 'services.defensive_exit_approval_readiness' in joined:\n"
        "    root = Path(os.environ['FAKE_RESULTS'])\n"
        "    (root / 'paper_sell_exit_before_entry.json').write_text('{}\\n', encoding='utf-8')\n"
        "    (root / 'manage_positions_cycle_binding.json').write_text('{}\\n', encoding='utf-8')\n"
        f"if os.environ.get('FAKE_FAIL_F') == '1' and 'services.manage_positions' in joined:\n"
        "    raise SystemExit(2)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    wrapper = f'@echo off\r\n"{sys.executable}" "{fake}" %*\r\nexit /b %ERRORLEVEL%\r\n'
    (bin_dir / "python.cmd").write_text(wrapper, encoding="ascii")
    (bin_dir / "python.bat").write_text(wrapper, encoding="ascii")
    (iso / "fail_f.txt").write_text(flag, encoding="ascii")


def _run_sequence(script: Path, tmp_path: Path, *, fail_f: bool) -> dict:
    iso = tmp_path / ("fail" if fail_f else "ok")
    _install_fake_python(iso, fail_f=fail_f)
    (iso / "scripts" / "run_market_open.ps1").write_bytes(script.read_bytes())
    log_path = iso / "fake_python.log"
    env = dict(os.environ)
    env["PATH"] = str(iso / "bin") + os.pathsep + env.get("PATH", "")
    env["FAKE_PY_LOG"] = str(log_path)
    env["FAKE_RESULTS"] = str(iso / "data" / "results")
    env["FAKE_FAIL_F"] = "1" if fail_f else "0"
    env["TRITON_ENABLE_LIVE_EXECUTION"] = "0"
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(iso / "scripts" / "run_market_open.ps1"),
        ],
        cwd=str(iso),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    results = iso / "data" / "results"
    handoff_path = results / "market_open_invocation_handoff.json"
    status_path = results / "market_open_sequence_status.json"
    transcripts = list((results / "scheduled_runs").glob("market_open_*.log"))
    transcript = transcripts[0].read_text(encoding="utf-8", errors="replace") if transcripts else ""
    handoff = {}
    if handoff_path.is_file():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
    status = {}
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    modules = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "-m " in line:
                modules.append(line.split("-m ", 1)[1].split()[0])
    return {
        "rc": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "handoff": handoff,
        "status": status,
        "modules": modules,
        "transcript": transcript,
        "interlock": results / "paper_sell_exit_before_entry.json",
    }


def _constant_fields(doc: dict) -> dict:
    return {
        "schema_version": doc.get("schema_version"),
        "status": doc.get("status"),
        "source": doc.get("source"),
        "creates_no_authority": doc.get("creates_no_authority"),
        "live_execution_permitted": doc.get("live_execution_permitted"),
        "broker_orders_submitted": doc.get("broker_orders_submitted"),
        "keys": list(doc.keys()),
    }


def test_file_invocation_matches_control_on_success_and_failure(tmp_path: Path):
    assert CONTROL_SCRIPT.is_file()
    cand_ok = _run_sequence(SCRIPT, tmp_path / "cand_ok", fail_f=False)
    ctrl_ok = _run_sequence(CONTROL_SCRIPT, tmp_path / "ctrl_ok", fail_f=False)
    cand_fail = _run_sequence(SCRIPT, tmp_path / "cand_fail", fail_f=True)
    ctrl_fail = _run_sequence(CONTROL_SCRIPT, tmp_path / "ctrl_fail", fail_f=True)
    for label, row in (("cand_ok", cand_ok), ("ctrl_ok", ctrl_ok)):
        assert row["rc"] == 0, label + row["stdout"][-1500:] + row["stderr"][-1500:]
        assert row["modules"]
        assert row["handoff"]["status"] == "READY"
        assert row["handoff"]["interlock_sha256"] == _sha(row["interlock"])
        assert row["status"].get("status") == "COMPLETE"
        assert "transcript end" in row["transcript"].lower()
    assert cand_ok["modules"] == ctrl_ok["modules"]
    assert _constant_fields(cand_ok["handoff"]) == _constant_fields(ctrl_ok["handoff"])
    assert cand_ok["handoff"]["market_open_invocation_id"].startswith("mo-")
    assert ctrl_ok["handoff"]["market_open_invocation_id"].startswith("mo-")
    for label, row in (("cand_fail", cand_fail), ("ctrl_fail", ctrl_fail)):
        assert row["rc"] == 2, label + row["stdout"][-1500:] + row["stderr"][-1500:]
        assert "defensive_exit_approval_readiness" not in row["modules"]
        assert row["handoff"]["status"] == "PENDING"
        assert row["handoff"]["interlock_sha256"] is None
        assert int(row["status"].get("exit_code") or 0) == 2
        assert "transcript end" in row["transcript"].lower()
    assert cand_fail["modules"] == ctrl_fail["modules"]
    assert _constant_fields(cand_fail["handoff"]) == _constant_fields(ctrl_fail["handoff"])


def test_results_root_ignores_script_invocation_id(tmp_path: Path):
    derived = tmp_path / "data" / "results"
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}' -FunctionsOnly
$script:MarketOpenInvocationId = 'mo-script-global-must-not-win'
Write-InvocationHandoff -Status 'PENDING' -RepoRoot '{tmp_path}' -ResultsRoot '{derived}' -InvocationId '{INV}' `
  -SessionDate '2026-09-21' -IssuedAt '2026-09-21T17:17:22.0000000Z' `
  -Mode 'PAPER' -LiveExecutionPermitted $false
'WROTE'
"""
    completed = _run_ps(body, tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert "WROTE" in captured, captured[-1500:]
    dest = tmp_path / "data" / "results" / "market_open_invocation_handoff.json"
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["status"] == "PENDING"
    assert doc["market_open_invocation_id"] == INV
    assert "script-global" not in json.dumps(doc)
    assert not (tmp_path / "data" / "results" / "data" / "results").exists()


def test_results_root_is_not_extended_again(tmp_path: Path):
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}' -FunctionsOnly
Write-InvocationHandoff -Status 'PENDING' -ResultsRoot '{results}' -InvocationId '{INV}' `
  -SessionDate '2026-09-21' -IssuedAt '2026-09-21T17:17:22.0000000Z' `
  -Mode 'PAPER' -LiveExecutionPermitted $false
'WROTE'
"""
    completed = _run_ps(body, tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert "WROTE" in captured, captured[-1500:]
    assert (results / "market_open_invocation_handoff.json").is_file()
    assert not (results / "data" / "results").exists()


def test_conflicting_roots_write_nothing(tmp_path: Path):
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}' -FunctionsOnly
try {{
  Write-InvocationHandoff -Status 'PENDING' -RepoRoot '{tmp_path}' -ResultsRoot '{tmp_path}' `
    -InvocationId '{INV}' -SessionDate '2026-09-21' -IssuedAt '2026-09-21T17:17:22.0000000Z' `
    -Mode 'PAPER' -LiveExecutionPermitted $false
  'WROTE'
}} catch {{
  'ERR:' + $_.Exception.Message
}}
"""
    completed = _run_ps(body, tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert "HANDOFF_ROOT_CONFLICT" in captured, captured[-1500:]
    assert "WROTE" not in captured
    assert not (tmp_path / "data" / "results" / "market_open_invocation_handoff.json").exists()


def test_failure_status_stays_under_results_root(tmp_path: Path):
    body = f"""
$ErrorActionPreference = 'Stop'
. '{SCRIPT}' -FunctionsOnly
try {{
  Write-MarketOpenSequenceStatus -Status 'ABORTED' -FailedStep 'F. POSITION MANAGEMENT' `
    -Reason 'step_failed' -ExitCode 2 -ResultsRoot 'C:\\SF31_refused_root'
  'WROTE_BAD'
}} catch {{
  'ERR:' + $_.Exception.Message
}}
Write-MarketOpenSequenceStatus -Status 'ABORTED' -FailedStep 'F. POSITION MANAGEMENT' `
  -Reason 'step_failed' -ExitCode 2 -RepoRoot '{tmp_path}'
'WROTE'
"""
    completed = _run_ps(body, tmp_path)
    captured = (completed.stdout or "") + (completed.stderr or "")
    assert "HANDOFF_ISOLATED_ROOT_REFUSED" in captured, captured[-1500:]
    assert "WROTE_BAD" not in captured
    assert "WROTE" in captured
    status = json.loads(
        (tmp_path / "data" / "results" / "market_open_sequence_status.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert status["status"] == "ABORTED"
    assert int(status["exit_code"]) == 2
    assert status["live_execution_permitted"] is False
    assert not Path(r"C:\SF31_refused_root").exists()
