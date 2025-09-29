# run_daily.ps1
$ErrorActionPreference = "Stop"

# Resolve repo root (folder of this script)
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv (PowerShell activation script)
$activate = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) { throw "Can't find venv at $activate" }
. $activate

# Paths
$py      = Join-Path $ROOT "run_daily.py"
$config  = Join-Path $ROOT "config\baseline.smart_weight.json"

# Optional: log file
$logDir  = Join-Path $ROOT "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log     = Join-Path $logDir "run_daily_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Run
Write-Host "Running: $py --config $config"
python $py --config $config *>&1 | Tee-Object -FilePath $log

Write-Host "Done. Log: $log"
