[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
chcp 65001 | Out-Null

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Activate venv
. .\.venv\Scripts\Activate.ps1

# -----------------------------
# Logging
# -----------------------------
$logsDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir "daily_refresh_$ts.log"

function Log($msg) {
    $line = "$(Get-Date -Format s)  $msg"
    $line | Out-File -FilePath $logPath -Append -Encoding utf8
    Write-Host $line
}

function Run-Step([string]$name, [string]$cmd) {
    Log ">> $name"
    Log "CMD: $cmd"

    # IMPORTANT:
    # We DO NOT pipe native output through PowerShell (it causes NativeCommandError + truncation).
    # Instead, cmd.exe handles stdout/stderr redirection straight into the log.
    $cmdEsc = $cmd.Replace('"','\"')  # keep quoting safe
    $logEsc = $logPath.Replace('"','\"')

    cmd.exe /c "$cmdEsc 1>>""$logEsc"" 2>>&1"
    $code = $LASTEXITCODE

    Log "EXIT: $name code=$code"

    if ($code -ne 0) {
        Log "❌ $name FAILED. See full log: $logPath"
        throw "$name failed (exit=$code). Full log: $logPath"
    }
}

Log "=== TRITON Daily Refresh START ==="
Log "pwd=$pwd"
Log "log_path=$logPath"

# 1) Fetch raw data
Run-Step "fetch_raw_data" "python -m services.fetch_raw_data --verbose --min-ok 10"

# 2) Stale gate
Run-Step "stale_data_gate" "python -m services.stale_data_gate"

# 3) Full pipeline
Run-Step "run_full_pipeline" "python run_full_pipeline.py --verbose"

# 4) Snapshot equity (updates portfolio_history.csv)
Run-Step "snapshot_equity" "python -m services.snapshot_equity --mode paper"

# 5) Snapshot positions (updates positions_snapshot.csv)
Run-Step "snapshot_positions" "python -m services.snapshot_positions --mode paper"

Log "=== TRITON Daily Refresh OK ==="
