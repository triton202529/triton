# scripts/run_continuous_loop.ps1
param(
    [int]$IntervalMinutes = 45,
    [int]$MaxCycles = 0
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

$loopLogDir = "data\results\continuous_loop"
New-Item -ItemType Directory -Force -Path $loopLogDir | Out-Null

$startedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$loopLog = Join-Path $loopLogDir "continuous_loop_$startedAt.log"
$lockFile = "data\live\continuous_loop.lock"

New-Item -ItemType Directory -Force -Path "data\live" | Out-Null

if (Test-Path $lockFile) {
    Write-Host "[BLOCKED] Continuous loop lock already exists: $lockFile" -ForegroundColor Red
    Write-Host "Delete it only if you are sure no loop is running."
    exit 1
}

Set-Content -Path $lockFile -Value "started_at=$(Get-Date -Format s)" -Encoding UTF8

function Log-Line($msg) {
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    Add-Content -Path $loopLog -Value $line
}

try {
    Log-Line "=== TRITON CONTINUOUS LOOP START ==="
    Log-Line "RepoRoot=$repoRoot"
    Log-Line "IntervalMinutes=$IntervalMinutes"
    Log-Line "MaxCycles=$MaxCycles"
    Log-Line "LoopLog=$loopLog"

    $cycle = 0

    while ($true) {
        $cycle++

        Log-Line ""
        Log-Line "[CYCLE_START] cycle=$cycle"

        powershell -ExecutionPolicy Bypass -File .\scripts\run_market_open.ps1
        $rc = $LASTEXITCODE

        if ($rc -ne 0) {
            Log-Line "[CYCLE_FAIL] cycle=$cycle exit_code=$rc"
        } else {
            Log-Line "[CYCLE_OK] cycle=$cycle"
        }

        if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
            Log-Line "[LOOP_COMPLETE] reached MaxCycles=$MaxCycles"
            break
        }

        Log-Line "[SLEEP] waiting $IntervalMinutes minutes before next cycle"
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
}
catch {
    Log-Line "[LOOP_ABORT] $($_.Exception.Message)"
    exit 1
}
finally {
    if (Test-Path $lockFile) {
        Remove-Item $lockFile -Force
    }
    Log-Line "=== TRITON CONTINUOUS LOOP END ==="
}
