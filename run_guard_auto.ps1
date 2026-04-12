# run_guard_auto.ps1
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

# Activate venv
. "$ROOT\.venv\Scripts\Activate.ps1"

# Ensure results dir exists
$resultsDir = Join-Path $ROOT "data\results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

# Log path
$log = Join-Path $resultsDir ("guard_auto_{0}.log" -f (Get-Date -Format "yyyyMMdd"))

# Header (UTF-8 NO BOM)
"--- $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8

# Capture stdout+stderr, then append with explicit encoding
$out = & python "$ROOT\services\guard_auto.py" --mode paper --verbose 2>&1
$out | Out-File -FilePath $log -Append -Encoding utf8

exit $LASTEXITCODE
