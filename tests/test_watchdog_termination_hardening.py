"""
Watchdog termination hardening — log I/O must not kill the loop; exit 11 stays
nonterminal; helper returns stay scalar; every exit path records TERMINAL_REASON.

Temp artifacts / subprocess harness only. No broker, network, paper-arm, live,
Scheduled Task, or production runtime writes.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "scripts" / "run_watchdog_loop.ps1"


def _wd_src() -> str:
    return WD.read_text(encoding="utf-8")


def test_log_line_swallows_add_content_failures():
    src = _wd_src()
    fn = src[src.index("function Log-Line") : src.index("function Write-TerminalReason")]
    assert "Add-Content" in fn
    assert "LOG_WRITE_FAILED" in fn
    assert "fallback" in fn
    # Must not rethrow after Add-Content failure
    assert "throw" not in fn


def test_helper_returns_capture_stdout_before_code():
    src = _wd_src()
    sleep_fn = src[
        src.index("function Invoke-InterruptibleSleep") : src.index("function Remove-OwnLoopLock")
    ]
    assert "$sleepOut = python" in sleep_fn
    assert "2>&1" in sleep_fn
    assert "$code = $LASTEXITCODE" in sleep_fn
    assert "return ,$code" in sleep_fn
    poll_fn = src[
        src.index("function Test-CooperativeShutdown") : src.index(
            "function Invoke-InterruptibleSleep"
        )
    ]
    assert "$out = python" in poll_fn
    assert "return ,$code" in poll_fn


def test_exit_code_11_is_nonterminal_continue_path():
    src = _wd_src()
    # Expired/refused exit 11 must continue (not break as accept).
    assert "($pollCode -ne 0) -and ($pollCode -ne 11)" in src
    assert "$pollCode -eq 10" in src
    body = src[
        src.index("while ($true)") : src.index("Write-TerminalReason -Reason $terminalReason")
    ]
    # break only after explicit accept / max cycles markers
    assert "SHUTDOWN_ACCEPTED" in body
    assert "LOOP_COMPLETE" in body


def test_every_marked_exit_records_terminal_reason():
    src = _wd_src()
    assert "Write-TerminalReason" in src
    assert "COOPERATIVE_SHUTDOWN_PRE_CYCLE" in src
    assert "COOPERATIVE_SHUTDOWN_POST_CYCLE" in src
    assert "COOPERATIVE_SHUTDOWN_INTERRUPTIBLE_WAIT" in src
    assert "MAX_CYCLES" in src
    assert "LOOP_ABORT" in src
    assert "[TERMINAL_REASON]" in src


def test_poll_code_normalized_before_eq_10_compare():
    """Guard against PowerShell array -eq containment treating polluted returns as accept."""
    src = _wd_src()
    assert '@((Test-CooperativeShutdown -CycleState "BETWEEN_CYCLES"))[-1]' in src
    assert "@((Invoke-InterruptibleSleep" in src


def test_log_lock_no_longer_aborts_loop_after_repair():
    """
    Deterministic repro of the 2026-08-06 failure mode: exclusive lock on the
    loop log under ErrorActionPreference=Stop. After repair, mirrored Log-Line
    must not terminate; a subsequent poll line must still be attempted.
    """
    script = textwrap.dedent(
        r"""
        $ErrorActionPreference = 'Stop'
        $td = $env:HARNESS_TD
        $loopLog = Join-Path $td 'watchdog.log'
        'seed' | Set-Content $loopLog
        function Log-Line($msg) {
            $line = "$(Get-Date -Format s) $msg"
            Write-Host $line
            try {
                Add-Content -Path $loopLog -Value $line -ErrorAction Stop
            } catch {
                try {
                    Add-Content -Path ($loopLog + '.fallback') -Value $line -ErrorAction SilentlyContinue
                } catch {}
                Write-Host ("[LOG_WRITE_FAILED] " + $_.Exception.Message)
            }
        }
        $fs = [System.IO.File]::Open($loopLog, 'Open', 'Read', 'None')
        try {
            Log-Line '[SHUTDOWN_POLL] between_cycles exit_code=11'
            Log-Line '[CYCLE_START] cycle=303'
            Write-Host 'CONTINUED_AFTER_LOG_LOCK=True'
        } finally {
            $fs.Dispose()
        }
        if (Test-Path ($loopLog + '.fallback')) {
            Write-Host 'FALLBACK_WRITTEN=True'
        }
        """
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "log_lock_repair.ps1"
        path.write_text(script, encoding="utf-8")
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["HARNESS_TD"] = td
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
            env=env,
        )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "CONTINUED_AFTER_LOG_LOCK=True" in proc.stdout
    assert "FALLBACK_WRITTEN=True" in proc.stdout
    assert "LOG_WRITE_FAILED" in proc.stdout


def test_expired_exit_11_does_not_eq_10_after_capture():
    script = textwrap.dedent(
        r"""
        $ErrorActionPreference = 'Stop'
        function Test-CooperativeShutdown {
            $out = '{"mode":"check","action":"CONTINUE_REFUSED","detail_state":"REFUSED_EXPIRED"}'
            $code = 11
            Write-Host "[SHUTDOWN_POLL] output=$out"
            return ,$code
        }
        $pollCode = @((Test-CooperativeShutdown))[-1]
        Write-Host ("type=" + $pollCode.GetType().FullName)
        Write-Host ("eq10=" + ($pollCode -eq 10))
        Write-Host ("continue=" + (($pollCode -eq 0) -or ($pollCode -eq 11)))
        """
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "exit11.ps1"
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
    assert "eq10=False" in proc.stdout
    assert "continue=True" in proc.stdout


def test_sleep_capture_prevents_false_termination_on_noise_ten():
    script = textwrap.dedent(
        r"""
        $ErrorActionPreference = 'Stop'
        function Invoke-InterruptibleSleep {
            $sleepOut = @('{"mode":"sleep","shutdown":false}', '10')
            $code = 0
            foreach ($line in @($sleepOut)) { Write-Host "[SLEEP_POLL] $line" }
            return ,$code
        }
        $sleepCode = @((Invoke-InterruptibleSleep))[-1]
        Write-Host ("eq10=" + ($sleepCode -eq 10))
        Write-Host ("continue=" + ($sleepCode -eq 0))
        """
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sleep_cap.ps1"
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
    assert "eq10=False" in proc.stdout
    assert "continue=True" in proc.stdout


def test_cooperative_exit_10_still_detected():
    script = textwrap.dedent(
        r"""
        function Test-CooperativeShutdown {
            $out = '{"mode":"check","action":"EXIT_NOW"}'
            $code = 10
            Write-Host "poll=$out"
            return ,$code
        }
        $pollCode = @((Test-CooperativeShutdown))[-1]
        Write-Host ("accept=" + ($pollCode -eq 10))
        """
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "exit10.ps1"
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
    assert "accept=True" in proc.stdout


def test_harness_scripts_do_not_target_production_live_locks():
    # Guardrail: embedded harness scripts must not write production live locks.
    here = Path(__file__).read_text(encoding="utf-8")
    harness_blocks = re.findall(r'r"""(.*?)"""', here, flags=re.S)
    joined = "\n".join(harness_blocks)
    assert "data\\live\\watchdog_loop.lock" not in joined
    assert "data/live/watchdog_loop.lock" not in joined
    assert "Set-Content" not in joined or "data\\live" not in joined
