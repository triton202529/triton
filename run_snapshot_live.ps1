$ErrorActionPreference = "Stop"

$ROOT = "C:\Users\akimw\triton"
$PYTHON = "$ROOT\.venv\Scripts\python.exe"
$LOG = "$ROOT\data\results\scheduler_snapshot.log"

# Ensure results directory exists
if (-not (Test-Path "$ROOT\data\results")) {
    New-Item -ItemType Directory -Path "$ROOT\data\results" | Out-Null
}

# Log header
Add-Content -Path $LOG -Value "=== TRITON_SnapshotLive $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Add-Content -Path $LOG -Value "ROOT=$ROOT"
Add-Content -Path $LOG -Value "PYTHON=$PYTHON"

# Change directory
Set-Location $ROOT

# Run snapshot
& $PYTHON -m services.snapshot_live_orders --mode paper --verbose 2>&1 |
    Tee-Object -FilePath $LOG -Append

Add-Content -Path $LOG -Value "ExitCode=$LASTEXITCODE"
Add-Content -Path $LOG -Value ""
