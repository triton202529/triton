"""SF64 entrypoint order. TEMP roots only. No production writes."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_daily_refresh.ps1"
LOCK_UTILS = ROOT / "scripts" / "loop_lock_utils.ps1"
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
    PROD / "logs" / "daily_refresh_20260926_090907.log",
]
HELPERS = {
    "ConvertTo-UtcRoundTripText",
    "Log",
    "Write-StageReceipts",
    "Invoke-LockedDailyRefreshWindow",
    "Run-PyStep",
    "Bind-FetchArtifact",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _ast(tmp: Path, script: Path) -> str:
    body = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$tokens, [ref]$errors)
Write-Output ('PARSE_ERRORS=' + @($errors).Count)
$funcs = $ast.FindAll({{ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }}, $true)
foreach ($fn in $funcs) {{
    Write-Output ('DEF ' + $fn.Name + ' ' + $fn.Extent.StartOffset)
}}
$cmds = $ast.FindAll({{ param($n) $n -is [System.Management.Automation.Language.CommandAst] }}, $true)
foreach ($cmd in $cmds) {{
    $name = $cmd.GetCommandName()
    if (-not $name) {{ continue }}
    $parent = $cmd.Parent
    $scope = 'script'
    while ($null -ne $parent) {{
        if ($parent -is [System.Management.Automation.Language.FunctionDefinitionAst]) {{
            $scope = 'function'
            break
        }}
        $parent = $parent.Parent
    }}
    Write-Output ('CALL ' + $scope + ' ' + $name + ' ' + $cmd.Extent.StartOffset)
}}
"""
    proc = _ps(tmp, body)
    assert proc.returncode == 0, proc.stdout
    return proc.stdout


def _defs_and_script_calls(listing: str) -> tuple[dict[str, int], list[tuple[str, int]]]:
    defs: dict[str, list[int]] = {}
    calls: list[tuple[str, int]] = []
    for line in listing.splitlines():
        parts = line.split()
        if line.startswith("DEF ") and len(parts) == 3:
            defs.setdefault(parts[1], []).append(int(parts[2]))
        elif line.startswith("CALL script ") and len(parts) == 4:
            calls.append((parts[2], int(parts[3])))
    collapsed = {name: offsets[0] for name, offsets in defs.items()}
    return collapsed, calls


def _compile_stub(root: Path) -> Path:
    exe = root / ".venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    source = root / "stub.cs"
    source.write_text(
        "\n".join(
            [
                "using System;",
                "using System.IO;",
                "public class Sf64Stub {",
                "  public static int Main(string[] args) {",
                '    string log = Environment.GetEnvironmentVariable("SF64_STUB_LOG");',
                '    string lockPath = Environment.GetEnvironmentVariable("SF64_LOCK");',
                "    bool held = !string.IsNullOrEmpty(lockPath) && File.Exists(lockPath);",
                '    File.AppendAllText(log, string.Join(" ", args) + " LOCK=" + held + "\\n");',
                '    if (args.Length > 0 && args[0] == "-c") Console.WriteLine("2026-09-25");',
                "    return 0;",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Add-Type -TypeDefinition (Get-Content -Raw -LiteralPath '"
                + str(source)
                + "') -OutputType ConsoleApplication -OutputAssembly '"
                + str(exe)
                + "'"
            ),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0 and exe.is_file(), proc.stdout
    return exe


def _stage(root: Path, script_text: str) -> tuple[Path, Path, Path]:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    script = root / "run_daily_refresh.ps1"
    script.write_text(script_text, encoding="utf-8", newline="\n")
    shutil.copyfile(LOCK_UTILS, root / "scripts" / "loop_lock_utils.ps1")
    _compile_stub(root)
    return script, root / "stub.log", root / "data" / "live" / "daily_refresh.lock"


def _run_entrypoint(
    root: Path, script: Path, stub_log: Path, lock: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SF64_STUB_LOG"] = str(stub_log)
    env["SF64_LOCK"] = str(lock)
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
        cwd=str(root),
    )


def _function_span(text: str) -> tuple[int, int]:
    start = text.find("function ConvertTo-UtcRoundTripText {")
    assert start >= 0
    opener = text.find("{", start)
    depth = 0
    for pos in range(opener, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                end = pos + 1
                if text.startswith("\r\n", end):
                    end += 2
                elif text.startswith("\n", end):
                    end += 1
                return start, end
    raise AssertionError("unclosed function")


def _break_order(text: str) -> str:
    start, end = _function_span(text)
    block = text[start:end]
    removed = text[:start] + text[end:]
    call = removed.find("$runStartedAt = ConvertTo-UtcRoundTripText")
    assert call >= 0 and "function ConvertTo-UtcRoundTripText" not in removed
    return removed.replace(
        "function ConvertFrom-UtcRoundTripText {",
        block + "function ConvertFrom-UtcRoundTripText {",
        1,
    )


def test_parser_has_zero_errors(tmp_path: Path) -> None:
    listing = _ast(tmp_path, SCRIPT)
    assert "PARSE_ERRORS=0" in listing


def test_round_trip_function_is_defined_once(tmp_path: Path) -> None:
    listing = _ast(tmp_path, SCRIPT)
    defs, _calls = _defs_and_script_calls(listing)
    assert listing.count("DEF ConvertTo-UtcRoundTripText ") == 1
    assert "ConvertTo-UtcRoundTripText" in defs


def test_definition_precedes_every_executable_call(tmp_path: Path) -> None:
    listing = _ast(tmp_path, SCRIPT)
    def_at = None
    call_at = []
    for line in listing.splitlines():
        parts = line.split()
        if line.startswith("DEF ConvertTo-UtcRoundTripText "):
            def_at = int(parts[2])
        elif line.startswith("CALL ") and parts[2] == "ConvertTo-UtcRoundTripText":
            call_at.append(int(parts[3]))
    assert def_at is not None and call_at
    assert all(def_at < call for call in call_at)


def test_entrypoint_helpers_are_defined_before_first_script_use(tmp_path: Path) -> None:
    listing = _ast(tmp_path, SCRIPT)
    defs, calls = _defs_and_script_calls(listing)
    seen = []
    for name, offset in calls:
        if name not in HELPERS:
            continue
        assert name in defs and defs[name] < offset
        seen.append(name)
    assert "ConvertTo-UtcRoundTripText" in seen
    assert "Run-PyStep" in seen
    assert "Invoke-LockedDailyRefreshWindow" in seen


def test_dot_source_creates_no_production_file(tmp_path: Path) -> None:
    before = {path: _sha(path) for path in WATCH if path.is_file()}
    proc = _ps(tmp_path, f". '{SCRIPT}'\nWrite-Output 'DEFINED'")
    assert proc.returncode == 0, proc.stdout
    assert "DEFINED" in proc.stdout
    after = {path: _sha(path) for path in before}
    assert after == before
    assert not (PROD / "data" / "live" / "daily_refresh.lock").exists()


def test_normal_entrypoint_passes_timestamp_and_releases_temp_lock(tmp_path: Path) -> None:
    before = {path: _sha(path) for path in WATCH if path.is_file()}
    script, stub_log, lock = _stage(tmp_path, SCRIPT.read_text(encoding="utf-8"))
    proc = _run_entrypoint(tmp_path, script, stub_log, lock)
    stub = stub_log.read_text(encoding="utf-8")
    assert "services.daily_prediction_refresh_validate" in stub
    assert "LOCK=True" in stub
    assert "not recognized" not in proc.stdout
    assert not lock.exists()
    assert (tmp_path / "logs").is_dir()
    assert any((tmp_path / "logs").glob("daily_refresh_*.log"))
    assert {path: _sha(path) for path in before} == before
    assert not (PROD / "data" / "live" / "daily_refresh.lock").exists()
    assert not list(PROD.glob("logs/daily_refresh_20260928_*.log"))


def test_definition_after_call_does_not_reach_stub(tmp_path: Path) -> None:
    script, stub_log, lock = _stage(tmp_path, _break_order(SCRIPT.read_text(encoding="utf-8")))
    proc = _run_entrypoint(tmp_path, script, stub_log, lock)
    assert proc.returncode != 0
    assert "ConvertTo-UtcRoundTripText" in proc.stdout
    assert not stub_log.exists() or stub_log.read_text(encoding="utf-8") == ""
    assert not lock.exists()
