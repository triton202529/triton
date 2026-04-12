$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\.venv\Scripts\Activate.ps1

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PSScriptRoot "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "preflight_$ts.log"

python -m services.preflight_autorun --mode paper --auto-refresh --verbose --max-signal-lag-days 0 *>> $log

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
