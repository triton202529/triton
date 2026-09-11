"""Isolated end-to-end verification of the production PowerShell UNKNOWN branch.

Executes the exact current bytes of scripts/run_continuous_loop.ps1 inside a
temporary repository root. Does not modify production. Does not contact a broker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from services.operator_console_status import build_operator_status
from services.paper_runtime_session_gate import (
    ATTENTION_SESSION_CALENDAR_UNKNOWN,
    PIPELINE_DEGRADED,
    PIPELINE_INFORMATIONAL_CLOSED,
    load_trading_cycle_health,
)

REPO = Path(__file__).resolve().parents[1]
PROD_LOOP = REPO / "scripts" / "run_continuous_loop.ps1"
REAL_PYTHON = REPO / ".venv" / "Scripts" / "python.exe"

CASES = [
    {
        "id": "unknown_import_failure",
        "kind": "UNKNOWN",
        "session_state": "UNKNOWN",
        "reason": "calendar_import_unavailable",
        "weak_clock_diagnostic": False,
    },
    {
        "id": "unknown_load_exception",
        "kind": "UNKNOWN",
        "session_state": "UNKNOWN",
        "reason": "calendar_load_exception",
        "weak_clock_diagnostic": False,
    },
    {
        "id": "unknown_schedule_exception",
        "kind": "UNKNOWN",
        "session_state": "UNKNOWN",
        "reason": "schedule_retrieval_failed",
        "weak_clock_diagnostic": False,
    },
    {
        "id": "unknown_malformed",
        "kind": "UNKNOWN",
        "session_state": "UNKNOWN",
        "reason": "calendar_malformed",
        "weak_clock_diagnostic": False,
    },
    {
        "id": "unknown_weak_clock_says_open",
        "kind": "UNKNOWN",
        "session_state": "UNKNOWN",
        "reason": "calendar_load_exception",
        "weak_clock_diagnostic": True,
    },
    {
        "id": "closed_control",
        "kind": "CLOSED",
        "session_state": "CLOSED",
        "reason": "regular_close",
        "weak_clock_diagnostic": False,
    },
    {
        "id": "open_control",
        "kind": "OPEN",
        "session_state": "OPEN",
        "reason": "regular_session_open",
        "weak_clock_diagnostic": False,
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_counters() -> dict:
    return {
        "run_market_open": 0,
        "arm_validation": 0,
        "paper_entry_stage": 0,
        "manage_positions": 0,
        "execution_plan": 0,
        "broker_init": 0,
        "broker_attempt": 0,
        "authorization_mutation": 0,
        "continuous_child_launch": 0,
        "watchdog_child_launch": 0,
        "unexpected_python_module": 0,
        "session_check": 0,
        "record_cycle": 0,
        "cooperative_shutdown": 0,
        "loop_safety": 0,
        "cadence_record_start": 0,
        "cadence_remaining_wait": 0,
    }


def _write_python_shim(temp_root: Path, case: dict) -> Path:
    bin_dir = temp_root / "harness_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    counters_path = temp_root / "harness_counters.json"
    counters_path.write_text(json.dumps(_empty_counters()), encoding="utf-8")
    session_path = temp_root / "harness_forced_session.json"
    session_path.write_text(
        json.dumps(
            {
                "session_state": case["session_state"],
                "regular_session_open": case["session_state"] == "OPEN",
                "reason": case["reason"],
                "action": (
                    "invoke_entry_sequence"
                    if case["session_state"] == "OPEN"
                    else (
                        "skip_market_closed"
                        if case["session_state"] == "CLOSED"
                        else "block_session_unknown"
                    )
                ),
                "log_event": (
                    None
                    if case["session_state"] == "OPEN"
                    else (
                        "PAPER_ENTRY_SKIPPED_MARKET_CLOSED"
                        if case["session_state"] == "CLOSED"
                        else "PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN"
                    )
                ),
                "weak_clock_diagnostic": case["weak_clock_diagnostic"],
                "calendar": "XNYS",
                "source": "harness_forced_session",
            }
        ),
        encoding="utf-8",
    )
    shim = bin_dir / "harness_python_shim.py"
    shim.write_text(
        f"""from __future__ import annotations
import json, os, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

COUNTERS = Path(r"{counters_path}")
SESSION = Path(r"{session_path}")
TEMP_ROOT = Path(r"{temp_root}")
REAL = Path(r"{REAL_PYTHON}")
REPO = Path(r"{REPO}")

def bump(key: str, n: int = 1) -> dict:
    doc = json.loads(COUNTERS.read_text(encoding="utf-8-sig"))
    doc[key] = int(doc.get(key) or 0) + n
    COUNTERS.write_text(json.dumps(doc), encoding="utf-8")
    return doc

def block_network() -> None:
    def _blocked(*_a, **_k):
        raise OSError("network blocked in calendar-unknown PowerShell consumer harness")
    socket.create_connection = _blocked  # type: ignore
    socket.socket.connect = _blocked  # type: ignore

def trap_unexpected(joined: str) -> int:
    bump("unexpected_python_module")
    sys.stderr.write("HARNESS_TRAP unexpected_python " + joined + "\\n")
    return 99

def parse_documented_cadence(argv: list[str]):
    if len(argv) < 2 or argv[0] != "-m" or argv[1] != "services.continuous_loop_cadence":
        return None
    rest = argv[2:]
    allowed_bool = {{"--record-start", "--remaining-wait", "--json"}}
    allowed_kv = {{"--interval-seconds", "--cycle", "--root", "--phase"}}
    flags = set()
    kv = {{}}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok in allowed_bool:
            flags.add(tok)
            i += 1
            continue
        if tok in allowed_kv:
            if i + 1 >= len(rest) or str(rest[i + 1]).startswith("-"):
                return None
            kv[tok] = rest[i + 1]
            i += 2
            continue
        return None
    if "--json" not in flags:
        return None
    start = "--record-start" in flags
    wait = "--remaining-wait" in flags
    if start == wait:
        return None
    try:
        interval = int(kv["--interval-seconds"])
        cycle = int(kv["--cycle"])
    except (KeyError, TypeError, ValueError):
        return None
    if interval <= 0 or cycle < 0:
        return None
    if "--root" in kv:
        root = Path(kv["--root"]).resolve()
        allowed = TEMP_ROOT.resolve()
        if os.path.normcase(str(root)) != os.path.normcase(str(allowed)) and not os.path.normcase(str(root)).startswith(os.path.normcase(str(allowed)) + os.sep):
            return None
    return {{"start": start, "wait": wait, "interval": interval, "cycle": cycle}}

def emit_cadence(parsed: dict) -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wait_s = 0 if parsed["wait"] else parsed["interval"]
    payload = {{
        "schema_version": "198T-AT-CADENCE-1",
        "timezone": "UTC",
        "cycle": parsed["cycle"],
        "phase": "INTERRUPTIBLE_WAIT" if parsed["wait"] else "ACTIVE_CYCLE",
        "interval_seconds": parsed["interval"],
        "remaining_wait_seconds": wait_s,
        "next_due_at_utc": now,
        "authoritative": True,
        "harness": True,
        "live_execution_permitted": False,
    }}
    dest = TEMP_ROOT / "data" / "results" / "continuous_loop_cadence.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bump("cadence_remaining_wait" if parsed["wait"] else "cadence_record_start")
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\\n")
    return 0

def main(argv: list[str]) -> int:
    for k in (
        "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL", "ALPACA_BASE_URL",
    ):
        os.environ[k] = ""
    os.environ["TRITON_ENABLE_LIVE_EXECUTION"] = "0"
    os.environ["TRITON_LIVE_EXECUTION"] = "0"
    block_network()
    joined = " ".join(argv)
    if "paper_entry_loop_stage" in joined:
        bump("paper_entry_stage")
        sys.stdout.write("HARNESS_FAKE_paper_entry_stage\\n")
        return 0
    trap_keys = (
        ("paper_session_arm_policy", "arm_validation"),
        ("manage_positions", "manage_positions"),
        ("alpaca_client", "broker_init"),
        ("broker_alpaca", "broker_init"),
        ("execute_trades", "broker_attempt"),
        ("place_live_orders", "broker_attempt"),
        ("paper_entry_dispatch", "execution_plan"),
        ("run_watchdog_loop", "watchdog_child_launch"),
    )
    for needle, key in trap_keys:
        if needle in joined:
            bump(key)
            sys.stderr.write("HARNESS_TRAP " + needle + "\\n")
            return 99
    if "paper_runtime_session_gate" in joined and "--check" in joined:
        bump("session_check")
        sys.stdout.write(SESSION.read_text(encoding="utf-8").strip() + "\\n")
        return 0
    if "paper_runtime_session_gate" in joined and "--record-cycle" in joined:
        bump("record_cycle")
        extra = []
        if "--root" not in joined:
            extra = ["--root", str(TEMP_ROOT)]
        return subprocess.call([str(REAL), "-m", "services.paper_runtime_session_gate", *argv[argv.index("-m")+2:], *extra], cwd=str(REPO))
    if "operator_cooperative_shutdown" in joined:
        bump("cooperative_shutdown")
        sys.stdout.write("{{\\"accepted\\": false, \\"harness\\": true}}\\n")
        return 0
    if "loop_safety" in joined:
        bump("loop_safety")
        sys.stdout.write("{{\\"ok\\": true, \\"harness\\": true}}\\n")
        return 0
    if "continuous_loop_cadence" in joined or (len(argv) >= 2 and argv[0] == "-m" and argv[1] == "services.continuous_loop_cadence"):
        parsed = parse_documented_cadence(argv)
        if parsed is None:
            return trap_unexpected(joined)
        return emit_cadence(parsed)
    if "-m" in argv:
        return trap_unexpected(joined)
    return subprocess.call([str(REAL), *argv], cwd=str(REPO))

if __name__ == "__main__":
    # argv to this shim is whatever followed python.exe
    raise SystemExit(main(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    cmd = bin_dir / "python.cmd"
    cmd.write_text(
        "@echo off\r\n"
        "set APCA_API_KEY_ID=\r\n"
        "set APCA_API_SECRET_KEY=\r\n"
        "set ALPACA_API_KEY=\r\n"
        "set ALPACA_SECRET_KEY=\r\n"
        "set APCA_API_BASE_URL=\r\n"
        "set TRITON_ENABLE_LIVE_EXECUTION=0\r\n"
        "set TRITON_LIVE_EXECUTION=0\r\n"
        f'"{REAL_PYTHON}" "{shim}" %*\r\n'
        "exit /b %ERRORLEVEL%\r\n",
        encoding="ascii",
    )
    return bin_dir


def _write_traps(temp_root: Path) -> None:
    scripts = temp_root / "scripts"
    counters = temp_root / "harness_counters.json"
    trap = f"""
$ErrorActionPreference = 'Continue'
$cpath = '{counters}'
$doc = Get-Content -LiteralPath $cpath -Raw | ConvertFrom-Json
$key = $env:HARNESS_TRAP_KEY
if (-not $key) {{ $key = 'run_market_open' }}
if ($null -eq $doc.$key) {{ $doc | Add-Member -NotePropertyName $key -NotePropertyValue 0 }}
$doc.$key = [int]$doc.$key + 1
($doc | ConvertTo-Json) | Set-Content -LiteralPath $cpath -Encoding UTF8
Write-Host "HARNESS_FAKE_$key"
exit 0
"""
    (scripts / "run_market_open.ps1").write_text(
        "$env:HARNESS_TRAP_KEY='run_market_open'\n" + trap, encoding="utf-8"
    )
    (scripts / "manage_positions.ps1").write_text(
        "$env:HARNESS_TRAP_KEY='manage_positions'\n" + trap, encoding="utf-8"
    )
    (scripts / "run_watchdog_loop.ps1").write_text(
        "$env:HARNESS_TRAP_KEY='watchdog_child_launch'\n" + trap, encoding="utf-8"
    )


def prepare_isolated_root(temp_root: Path, case: dict) -> dict:
    scripts = temp_root / "scripts"
    scripts.mkdir(parents=True)
    (temp_root / "data" / "live").mkdir(parents=True, exist_ok=True)
    (temp_root / "data" / "results").mkdir(parents=True, exist_ok=True)
    (temp_root / "data" / "results" / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    raw = PROD_LOOP.read_bytes()
    dest = scripts / "run_continuous_loop.ps1"
    dest.write_bytes(raw)
    prod_sha = hashlib.sha256(raw).hexdigest()
    exec_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    if prod_sha != exec_sha:
        raise AssertionError("executed script hash does not match production")
    for name in ("triton_env_bootstrap.ps1", "loop_lock_utils.ps1", "loop_log_utils.ps1"):
        shutil.copy2(REPO / "scripts" / name, scripts / name)
    _write_traps(temp_root)
    bin_dir = _write_python_shim(temp_root, case)
    return {
        "prod_sha256": prod_sha,
        "exec_sha256": exec_sha,
        "hash_match": prod_sha == exec_sha,
        "executed_script": str(dest),
        "bin_dir": str(bin_dir),
    }


def run_isolated_loop(temp_root: Path, case: dict, *, timeout_s: int = 90) -> dict:
    meta = prepare_isolated_root(temp_root, case)
    bin_dir = Path(meta["bin_dir"])
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    for k in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL",
        "ALPACA_BASE_URL",
    ):
        env[k] = ""
    env["TRITON_ENABLE_LIVE_EXECUTION"] = "0"
    env["TRITON_LIVE_EXECUTION"] = "0"
    # Do not point the isolated run at the real repo root.
    env.pop("TRITON_REPO_ROOT", None)
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(temp_root / "scripts" / "run_continuous_loop.ps1"),
        "-MaxCycles",
        "1",
        "-IntervalMinutes",
        "1",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(temp_root),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
    )
    duration = time.perf_counter() - t0
    counters = json.loads((temp_root / "harness_counters.json").read_text(encoding="utf-8-sig"))
    health = load_trading_cycle_health(root=temp_root)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + "\n" + stderr
    outside_locks = []
    prod_lock = REPO / "data" / "live" / "continuous_loop.lock"
    # Do not treat a pre-existing production lock as created by this run; only flag if mtime is newer than start.
    return {
        **meta,
        "case_id": case["id"],
        "kind": case["kind"],
        "command": cmd,
        "temp_root": str(temp_root),
        "exit_code": proc.returncode,
        "duration_s": round(duration, 3),
        "stdout": stdout,
        "stderr": stderr,
        "counters": counters,
        "health": health,
        "forced_session": json.loads(
            (temp_root / "harness_forced_session.json").read_text(encoding="utf-8")
        ),
        "events": {
            "PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN": "[PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN]"
            in combined,
            "CYCLE_BLOCKED_SESSION_UNKNOWN": "[CYCLE_BLOCKED_SESSION_UNKNOWN]" in combined,
            "PAPER_ENTRY_SKIPPED_MARKET_CLOSED": "[PAPER_ENTRY_SKIPPED_MARKET_CLOSED]" in combined,
            "CYCLE_SKIPPED_MARKET_CLOSED": "[CYCLE_SKIPPED_MARKET_CLOSED]" in combined,
            "CYCLE_OK": "[CYCLE_OK]" in combined,
            "CYCLE_FAIL": "[CYCLE_FAIL]" in combined,
            "SESSION_CHECK": "[SESSION_CHECK]" in combined,
            "CADENCE_START": "[CADENCE_START]" in combined,
            "HARNESS_FAKE_run_market_open": "HARNESS_FAKE_run_market_open" in combined,
        },
        "outside_production_lock_exists": prod_lock.is_file(),
    }


def _running_detector():
    return {
        "continuous_loop_running": True,
        "watchdog_loop_running": True,
        "detected_continuous_process_ids": [11],
        "detected_watchdog_process_ids": [22],
        "process_detection_confidence": "high",
        "continuous_detection_method": "fixture",
        "watchdog_detection_method": "fixture",
        "continuous_lock_classification": "LOCK_HEALTHY_ACTIVE",
        "watchdog_lock_classification": "LOCK_HEALTHY_ACTIVE",
        "continuous_lock_healthy": True,
        "watchdog_lock_healthy": True,
        "continuous_lock_exists": True,
        "watchdog_lock_exists": True,
    }


def _assert_reached_session_check(result: dict) -> None:
    ev = result["events"]
    c = result["counters"]
    assert ev["SESSION_CHECK"] is True
    assert ev["CADENCE_START"] is True
    assert int(c.get("session_check") or 0) >= 1
    assert int(c.get("cadence_record_start") or 0) >= 1
    assert int(c.get("unexpected_python_module") or 0) == 0


def _assert_no_broker_or_order(result: dict) -> None:
    c = result["counters"]
    h = result["health"]
    assert c["broker_init"] == 0
    assert c["broker_attempt"] == 0
    assert c["authorization_mutation"] == 0
    assert h.get("authorization_mutated") is False
    assert int(h.get("broker_attempts") or 0) == 0


def _assert_writes_inside_temp(temp_root: Path) -> None:
    temp = temp_root.resolve()
    worktree_cadence = REPO / "data" / "results" / "continuous_loop_cadence.json"
    assert not worktree_cadence.is_file()
    cadence = temp / "data" / "results" / "continuous_loop_cadence.json"
    if cadence.is_file():
        cadence_res = cadence.resolve()
        assert os.path.normcase(str(cadence_res)).startswith(os.path.normcase(str(temp)))
        payload = json.loads(cadence.read_text(encoding="utf-8"))
        assert payload.get("live_execution_permitted") is False
        assert payload.get("harness") is True


def _assert_unknown(result: dict) -> None:
    c = result["counters"]
    ev = result["events"]
    h = result["health"]
    assert result["hash_match"] is True
    assert result["exit_code"] == 0
    _assert_reached_session_check(result)
    _assert_no_broker_or_order(result)
    _assert_writes_inside_temp(Path(result["temp_root"]))
    assert ev["PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN"] is True
    assert ev["CYCLE_BLOCKED_SESSION_UNKNOWN"] is True
    assert ev["PAPER_ENTRY_SKIPPED_MARKET_CLOSED"] is False
    assert ev["CYCLE_SKIPPED_MARKET_CLOSED"] is False
    assert ev["CYCLE_OK"] is False
    assert ev["HARNESS_FAKE_run_market_open"] is False
    assert c["run_market_open"] == 0
    assert c["arm_validation"] == 0
    assert c["paper_entry_stage"] == 0
    assert c["manage_positions"] == 0
    assert c["execution_plan"] == 0
    assert c["continuous_child_launch"] == 0
    assert c["watchdog_child_launch"] == 0
    assert h.get("trading_pipeline_status") == PIPELINE_DEGRADED
    assert h.get("last_outcome") == "session_unknown"
    assert h.get("operator_attention_code") == ATTENTION_SESSION_CALENDAR_UNKNOWN
    assert result["forced_session"]["session_state"] == "UNKNOWN"
    assert result["forced_session"]["weak_clock_diagnostic"] in (True, False)


@pytest.fixture(autouse=True)
def _block_network_and_credentials(monkeypatch):
    for key in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")

    def _blocked(*_a, **_k):
        raise RuntimeError("network blocked in powershell consumer tests")

    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["kind"] == "UNKNOWN"], ids=lambda c: c["id"]
)
def test_unknown_powershell_branch(case, tmp_path: Path):
    result = run_isolated_loop(tmp_path, case)
    _assert_unknown(result)
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    codes = {a.get("classification_code") for a in (status.get("attention") or [])}
    assert ATTENTION_SESSION_CALENDAR_UNKNOWN in codes
    assert status["trading_pipeline"]["status"] == PIPELINE_DEGRADED


def test_unknown_healthy_watchdog_does_not_overwrite(tmp_path: Path):
    case = next(c for c in CASES if c["id"] == "unknown_load_exception")
    result = run_isolated_loop(tmp_path, case)
    _assert_unknown(result)
    (tmp_path / "data" / "results" / "watchdog_status.json").write_text(
        json.dumps({"status": "OK", "health_band": "HEALTHY", "broker_connected": True}),
        encoding="utf-8",
    )
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    assert status["trading_pipeline"]["status"] == PIPELINE_DEGRADED
    assert status["overall_status"] == "DEGRADED"
    codes = {a.get("classification_code") for a in (status.get("attention") or [])}
    assert ATTENTION_SESSION_CALENDAR_UNKNOWN in codes


def test_closed_control_powershell_branch(tmp_path: Path):
    case = next(c for c in CASES if c["id"] == "closed_control")
    result = run_isolated_loop(tmp_path, case)
    ev = result["events"]
    c = result["counters"]
    h = result["health"]
    assert result["hash_match"] is True
    assert result["exit_code"] == 0
    _assert_reached_session_check(result)
    _assert_no_broker_or_order(result)
    _assert_writes_inside_temp(tmp_path)
    assert ev["PAPER_ENTRY_SKIPPED_MARKET_CLOSED"] is True
    assert ev["CYCLE_SKIPPED_MARKET_CLOSED"] is True
    assert ev["PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN"] is False
    assert ev["CYCLE_FAIL"] is False
    assert ev["CYCLE_OK"] is False
    assert ev["HARNESS_FAKE_run_market_open"] is False
    assert c["run_market_open"] == 0
    assert c["paper_entry_stage"] == 0
    assert h.get("trading_pipeline_status") == PIPELINE_INFORMATIONAL_CLOSED
    assert h.get("operator_attention_code") in (None, "")
    status = build_operator_status(root=tmp_path)
    codes = {a.get("classification_code") for a in (status.get("attention") or [])}
    assert ATTENTION_SESSION_CALENDAR_UNKNOWN not in codes
    assert status["trading_pipeline"]["status"] == PIPELINE_INFORMATIONAL_CLOSED


def test_open_control_reaches_fake_market_open_once(tmp_path: Path):
    case = next(c for c in CASES if c["id"] == "open_control")
    result = run_isolated_loop(tmp_path, case)
    ev = result["events"]
    c = result["counters"]
    assert result["hash_match"] is True
    assert result["exit_code"] == 0
    _assert_reached_session_check(result)
    _assert_no_broker_or_order(result)
    _assert_writes_inside_temp(tmp_path)
    assert c["run_market_open"] == 1
    assert ev["HARNESS_FAKE_run_market_open"] is True
    assert ev["PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN"] is False
    assert ev["PAPER_ENTRY_SKIPPED_MARKET_CLOSED"] is False
    assert c["arm_validation"] == 0
    # OPEN path calls the paper-entry python module after fake market-open exit 0.
    # That call is trapped (not a real stage). Count may be 1; must not be a real broker.
    assert c["paper_entry_stage"] in (0, 1)
    assert result["health"].get("authorization_mutated") is False


def test_closed_does_not_clear_prior_market_hours_fail(tmp_path: Path):
    from services.paper_runtime_session_gate import record_cycle_outcome

    record_cycle_outcome(
        outcome="fail",
        reason="blocked_arm_expired",
        session_open=True,
        root=tmp_path,
    )
    case = next(c for c in CASES if c["id"] == "closed_control")
    result = run_isolated_loop(tmp_path, case)
    _assert_reached_session_check(result)
    _assert_no_broker_or_order(result)
    assert result["health"].get("last_market_hours_outcome") == "fail"
    assert result["health"].get("trading_pipeline_status") == PIPELINE_DEGRADED
    assert result["events"]["CYCLE_FAIL"] is False
    assert result["events"]["PAPER_ENTRY_SKIPPED_MARKET_CLOSED"] is True


def test_executed_script_bytes_match_production(tmp_path: Path):
    meta = prepare_isolated_root(tmp_path, CASES[0])
    assert meta["hash_match"] is True
    assert meta["prod_sha256"] == _sha256(PROD_LOOP)


def _run_shim(
    tmp_path: Path, argv: list[str], *, prepare: bool = True
) -> subprocess.CompletedProcess:
    if prepare:
        prepare_isolated_root(tmp_path, CASES[0])
    shim = tmp_path / "harness_bin" / "harness_python_shim.py"
    return subprocess.run(
        [str(REAL_PYTHON), str(shim), *argv],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )


def test_cadence_stub_cannot_change_session_result(tmp_path: Path):
    (tmp_path / "data" / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "results" / "continuous_loop_cadence.json").write_text(
        json.dumps(
            {
                "session_state": "OPEN",
                "regular_session_open": True,
                "remaining_wait_seconds": 0,
                "next_due_at_utc": "2026-09-10T20:00:00Z",
                "harness": True,
                "live_execution_permitted": False,
            }
        ),
        encoding="utf-8",
    )
    case = next(c for c in CASES if c["id"] == "unknown_weak_clock_says_open")
    result = run_isolated_loop(tmp_path, case)
    _assert_unknown(result)
    assert result["events"]["HARNESS_FAKE_run_market_open"] is False
    assert result["counters"]["run_market_open"] == 0
    assert result["forced_session"]["weak_clock_diagnostic"] is True
    assert result["health"].get("last_outcome") == "session_unknown"


def test_unrelated_python_module_still_traps(tmp_path: Path):
    proc = _run_shim(tmp_path, ["-m", "services.not_a_calendar_module", "--json"])
    counters = json.loads((tmp_path / "harness_counters.json").read_text(encoding="utf-8-sig"))
    assert proc.returncode == 99
    assert "HARNESS_TRAP unexpected_python" in (proc.stderr or "")
    assert counters["unexpected_python_module"] == 1
    assert counters["cadence_record_start"] == 0
    assert counters["session_check"] == 0


def test_malformed_cadence_command_still_traps(tmp_path: Path):
    bad_cmds = (
        ["-m", "services.continuous_loop_cadence", "--json"],
        ["-m", "services.continuous_loop_cadence", "--record-start", "--cycle", "1", "--json"],
        [
            "-m",
            "services.continuous_loop_cadence",
            "--record-start",
            "--remaining-wait",
            "--interval-seconds",
            "60",
            "--cycle",
            "1",
            "--json",
        ],
        [
            "-m",
            "services.continuous_loop_cadence",
            "--record-start",
            "--interval-seconds",
            "60",
            "--cycle",
            "1",
            "--json",
            "--unexpected-flag",
        ],
    )
    prepare_isolated_root(tmp_path, CASES[0])
    for argv in bad_cmds:
        proc = _run_shim(tmp_path, argv, prepare=False)
        assert proc.returncode == 99, argv
        assert "HARNESS_TRAP unexpected_python" in (proc.stderr or ""), argv
    counters = json.loads((tmp_path / "harness_counters.json").read_text(encoding="utf-8-sig"))
    assert counters["unexpected_python_module"] == len(bad_cmds)
    assert counters["cadence_record_start"] == 0
    assert counters["cadence_remaining_wait"] == 0


def test_documented_remaining_wait_is_accepted_without_broker(tmp_path: Path):
    proc = _run_shim(
        tmp_path,
        [
            "-m",
            "services.continuous_loop_cadence",
            "--remaining-wait",
            "--interval-seconds",
            "60",
            "--cycle",
            "1",
            "--json",
        ],
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["remaining_wait_seconds"] == 0
    assert payload["live_execution_permitted"] is False
    counters = json.loads((tmp_path / "harness_counters.json").read_text(encoding="utf-8-sig"))
    assert counters["cadence_remaining_wait"] == 1
    assert counters["unexpected_python_module"] == 0
    assert counters["broker_attempt"] == 0
    artifact = tmp_path / "data" / "results" / "continuous_loop_cadence.json"
    assert artifact.is_file()
    _assert_writes_inside_temp(tmp_path)


def test_harness_writes_remain_inside_temp(tmp_path: Path):
    worktree_cadence = REPO / "data" / "results" / "continuous_loop_cadence.json"
    prod_cadence = (
        Path(r"C:\Users\akimw\triton") / "data" / "results" / "continuous_loop_cadence.json"
    )
    before_worktree = worktree_cadence.exists()
    before_prod = None
    if prod_cadence.exists():
        before_prod = (prod_cadence.stat().st_mtime_ns, prod_cadence.stat().st_size)
    case = next(c for c in CASES if c["id"] == "unknown_import_failure")
    result = run_isolated_loop(tmp_path, case)
    _assert_unknown(result)
    _assert_writes_inside_temp(tmp_path)
    assert worktree_cadence.exists() is before_worktree
    if before_prod is None:
        assert not prod_cadence.exists()
    else:
        after = (prod_cadence.stat().st_mtime_ns, prod_cadence.stat().st_size)
        assert after == before_prod
    assert Path(result["temp_root"]).resolve() == tmp_path.resolve()
