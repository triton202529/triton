$ErrorActionPreference = "Stop"

Set-Location "C:\Users\akimw\triton"
. .\.venv\Scripts\Activate.ps1

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "premarket_preflight_$ts.log"

"=== TRITON PreMarket Preflight START $ts ===" | Tee-Object -FilePath $log -Append

try {
    # Preflight autorun:
    # - first pass preflight
    # - if fail and safe to refresh (closed), it can refresh and re-check
    python -m services.preflight_autorun `
      --mode paper `
      --auto-refresh `
      --verbose `
      --max-signal-lag-days 0 `
      --max-open-orders 0 `
      --strict-open-orders `
      --raw-max-age-days 2 `
      --signals-max-age-days 2 `
      --max-signals-generated-age-minutes 240 `
      --max-snapshot-generated-age-minutes 240 `
      2>&1 | Tee-Object -FilePath $log -Append

    "=== TRITON PreMarket Preflight SUCCESS $ts ===" | Tee-Object -FilePath $log -Append
    exit 0
}
catch {
    "=== TRITON PreMarket Preflight FAIL $ts ===" | Tee-Object -FilePath $log -Append
    $_ | Tee-Object -FilePath $log -Append
    exit 1
}
