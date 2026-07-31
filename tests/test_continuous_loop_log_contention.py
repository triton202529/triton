"""
Continuous-loop log contention hardening — synthetic file locks only.

Does not launch the continuous runtime, touch real data/live locks, or call brokers.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "scripts" / "loop_log_utils.ps1"
CONT = ROOT / "scripts" / "run_continuous_loop.ps1"
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")


def _run_ps(script: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ps_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def test_helper_normal_write_succeeds_once(tmp_path: Path):
    log = tmp_path / "loop.log"
    script = f"""
    . '{_ps_quote(UTILS)}'
    $r = Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'alpha' -MaxAttempts 5
    if (-not $r.ok) {{ throw 'expected ok' }}
    if ($r.attempts -ne 1) {{ throw "attempts=$($r.attempts)" }}
    Write-Output "OK attempts=$($r.attempts)"
    """
    proc = _run_ps(script)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert log.read_text(encoding="utf-8").strip() == "alpha"
    assert "attempts=1" in proc.stdout


def _hold_file_handle(path: Path):
    """
    Open a Win32 handle with share-mode 0 so Add-Content hits a sharing violation
    until the handle is closed.
    """
    import ctypes
    from ctypes import wintypes

    path.write_text("x", encoding="utf-8")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CreateFileW = kernel32.CreateFileW
    CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    CreateFileW.restype = wintypes.HANDLE
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
    handle = CreateFileW(
        str(path),
        GENERIC_READ | GENERIC_WRITE,
        0,  # no sharing
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE_VALUE or handle == -1:
        raise OSError(f"CreateFileW failed: {ctypes.get_last_error()}")
    return handle


def _release_file_handle(handle) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL
    CloseHandle(handle)


_ADD_CONTENT_SHARE_MOCK = r"""
function Add-Content {
    [CmdletBinding()]
    param(
        [Parameter()][string]$LiteralPath,
        [Parameter()][string]$Path,
        [Parameter()][AllowEmptyString()][object]$Value
    )
    $target = if ($LiteralPath) { $LiteralPath } else { $Path }
    if ($script:ShareFailLeft -gt 0) {
        $script:ShareFailLeft--
        throw [System.IO.IOException]::new(
            "The process cannot access the file '$target' because it is being used by another process."
        )
    }
    Microsoft.PowerShell.Management\Add-Content -LiteralPath $target -Value $Value -ErrorAction Stop
}
"""


def test_helper_one_transient_sharing_violation_then_success(tmp_path: Path):
    log = tmp_path / "loop.log"
    script = f"""
    . '{_ps_quote(UTILS)}'
    $script:ShareFailLeft = 1
    {_ADD_CONTENT_SHARE_MOCK}
    $r = Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'recovered' -MaxAttempts 5 -InitialBackoffMs 5
    Write-Output "OK attempts=$($r.attempts)"
    """
    proc = _run_ps(script, timeout=30)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    body = log.read_text(encoding="utf-8")
    assert "recovered" in body
    m = re.search(r"attempts=(\d+)", proc.stdout)
    assert m and int(m.group(1)) == 2


def test_helper_multiple_transient_then_success(tmp_path: Path):
    log = tmp_path / "loop.log"
    script = f"""
    . '{_ps_quote(UTILS)}'
    $script:ShareFailLeft = 3
    {_ADD_CONTENT_SHARE_MOCK}
    $r = Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'multi' -MaxAttempts 6 -InitialBackoffMs 5
    Write-Output "OK attempts=$($r.attempts)"
    """
    proc = _run_ps(script, timeout=30)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "multi" in log.read_text(encoding="utf-8")
    m = re.search(r"attempts=(\d+)", proc.stdout)
    assert m and int(m.group(1)) == 4


def test_helper_persistent_contention_fails_closed(tmp_path: Path):
    log = tmp_path / "loop.log"
    script = f"""
    . '{_ps_quote(UTILS)}'
    try {{
      Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'never' -MaxAttempts 3 -InitialBackoffMs 20
      Write-Output 'UNEXPECTED_SUCCESS'
      exit 0
    }} catch {{
      Write-Output ("FATAL:" + $_.Exception.Message)
      exit 7
    }}
    """

    handle = _hold_file_handle(log)
    try:
        proc = _run_ps(script, timeout=30)
    finally:
        _release_file_handle(handle)

    assert proc.returncode == 7, proc.stderr + proc.stdout
    assert "LOG_WRITE_FATAL" in proc.stdout or "LOG_WRITE_FATAL" in proc.stderr
    assert "never" not in log.read_text(encoding="utf-8")
    assert "UNEXPECTED_SUCCESS" not in proc.stdout


def test_helper_unexpected_error_not_suppressed(tmp_path: Path):
    missing = tmp_path / "missing_dir" / "nope.log"
    script = f"""
    . '{_ps_quote(UTILS)}'
    try {{
      Write-TritonLoopLogLine -Path '{_ps_quote(missing)}' -Line 'x' -MaxAttempts 3 -InitialBackoffMs 1
      Write-Output 'UNEXPECTED_SUCCESS'
      exit 0
    }} catch {{
      Write-Output ("ERR:" + $_.Exception.GetType().FullName + ":" + $_.Exception.Message)
      exit 9
    }}
    """
    proc = _run_ps(script)
    assert proc.returncode == 9, proc.stderr + proc.stdout
    assert "UNEXPECTED_SUCCESS" not in proc.stdout
    # Directory missing is not a sharing violation → should fail without LOG_WRITE_FATAL retry exhaustion messaging as primary
    assert "ERR:" in proc.stdout


def test_helper_preserves_message_order(tmp_path: Path):
    log = tmp_path / "loop.log"
    script = f"""
    . '{_ps_quote(UTILS)}'
    Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'one' | Out-Null
    Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'two' | Out-Null
    Write-TritonLoopLogLine -Path '{_ps_quote(log)}' -Line 'three' | Out-Null
    Write-Output 'OK'
    """
    proc = _run_ps(script)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == ["one", "two", "three"]


def test_original_unhandled_addcontent_sharing_failure_reproduced(tmp_path: Path):
    """Pre-hardening behavior: bare Add-Content + Stop fails on exclusive lock."""
    log = tmp_path / "loop.log"
    script = f"""
    $ErrorActionPreference = 'Stop'
    try {{
      Add-Content -LiteralPath '{_ps_quote(log)}' -Value 'raw' -ErrorAction Stop
      Write-Output 'UNEXPECTED_SUCCESS'
      exit 0
    }} catch {{
      Write-Output ("RAW_FAIL:" + $_.Exception.Message)
      exit 3
    }}
    """

    handle = _hold_file_handle(log)
    try:
        proc = _run_ps(script, timeout=20)
    finally:
        _release_file_handle(handle)

    assert proc.returncode == 3, proc.stderr + proc.stdout
    assert (
        "being used by another process" in proc.stdout.lower()
        or "cannot access the file" in proc.stdout.lower()
    )


def test_continuous_script_uses_hardened_writer_and_unique_log():
    src = CONT.read_text(encoding="utf-8")
    assert "loop_log_utils.ps1" in src
    assert "Write-TritonLoopLogLine" in src
    assert re.search(r"continuous_loop_\$startedAt\.log", src)
    assert 'Get-Date -Format "yyyyMMdd_HHmmss"' in src
    # Bare Add-Content must not remain in Log-Line body (helper owns append).
    log_fn = src[src.index("function Log-Line") : src.index("function Test-CooperativeShutdown")]
    assert "Write-TritonLoopLogLine" in log_fn
    assert "Add-Content" not in log_fn


def test_persistent_failure_exits_before_next_cycle_contract():
    src = CONT.read_text(encoding="utf-8")
    assert "[LOOP_ABORT]" in src
    assert "exit 1" in src
    # Outer catch wraps abort logging; finally still releases lock.
    assert "Remove-OwnLoopLock" in src
    assert "LOG_WRITE_FATAL" in src or "Write-TritonLoopLogLine" in src
    # While-loop work cannot continue after uncaught log throw (propagates to catch/exit).
    assert "while ($true)" in src


def test_no_broker_methods_in_log_utils():
    src = UTILS.read_text(encoding="utf-8").lower()
    assert "submit_order" not in src
    assert "create_order" not in src
    assert "place_order" not in src


def test_lock_cleanup_still_invoked_in_finally():
    src = CONT.read_text(encoding="utf-8")
    finally_block = src[src.index("finally {") :]
    assert "Remove-OwnLoopLock" in finally_block
    # Release happens before optional end log.
    rel = finally_block.index("Remove-OwnLoopLock")
    end = finally_block.index("TRITON CONTINUOUS LOOP END")
    assert rel < end


def test_powershell_parse_utils_and_continuous():
    for path in (UTILS, CONT):
        script = f"""
        $errs = $null
        $null = [System.Management.Automation.Language.Parser]::ParseFile('{_ps_quote(path)}', [ref]$null, [ref]$errs)
        if ($errs -and $errs.Count -gt 0) {{
          $errs | ForEach-Object {{ $_.ToString() }}
          exit 2
        }}
        Write-Output 'PARSE_OK'
        exit 0
        """
        proc = _run_ps(script)
        assert proc.returncode == 0, path.name + " " + proc.stderr + proc.stdout
        assert "PARSE_OK" in proc.stdout


def test_e0_hydration_unaffected_by_loop_log_helper():
    # Hydration module must not import or reference loop log utils.
    hyd = (ROOT / "services" / "runtime_universe_hydration_preflight.py").read_text(
        encoding="utf-8"
    )
    assert "loop_log_utils" not in hyd
    assert "Write-TritonLoopLogLine" not in hyd
