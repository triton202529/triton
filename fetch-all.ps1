param(
  [int]$Days = 365
)

$ErrorActionPreference = "Stop"

# Work from the repo root no matter how we're launched
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Paths
$python   = Join-Path $root ".venv\Scripts\python.exe"
$pipeline = Join-Path $root "pipeline_real_data.py"
$outCsv   = Join-Path $root "data\results\signals_with_rationale.csv"

# Logs (UTF-8)
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("fetch-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))

function Log { param([string]$m)
  $line = "{0} {1}" -f ([DateTime]::Now.ToString('s')), $m
  $line | Out-File -FilePath $log -Append -Encoding UTF8
  Write-Host $line
}

function Get-RowCount([string]$file) {
  if (Test-Path -LiteralPath $file) {
    try {
      $c = (Get-Content -LiteralPath $file -ErrorAction Stop).Count
      return [math]::Max($c - 1, 0)   # minus header
    } catch { return 0 }
  } else { return 0 }
}

Log "Starting run-fetch.ps1"
Log "Root: $root"
Log "Python: $python"
Log "Pipeline: $pipeline"
Log "NEWSAPI_KEY present: $([bool]$env:NEWSAPI_KEY)"

if (-not (Test-Path $python))   { Log "ERROR: Python not found at $python";   exit 1 }
if (-not (Test-Path $pipeline)) { Log "ERROR: pipeline_real_data.py not found at $pipeline"; exit 1 }

# Build universe from existing OHLC parquet files
$tickers = Get-ChildItem "$root\data\results\*.parquet" -ErrorAction SilentlyContinue |
  ForEach-Object { $_.BaseName } |
  Where-Object { $_ -notmatch "^\^" -or $_ -in "^DJI","^GSPC","^IXIC","^VIX" }

if (-not $tickers -or $tickers.Count -eq 0) {
  Log "No parquet files found; defaulting to core tickers."
  $tickers = @("AAPL","MSFT","GOOGL","NVDA","AMZN")
}
Log "Tickers count: $($tickers.Count)"

# Build argument list
$args = @($pipeline, "-t") + $tickers + @("-d", $Days)
Log ("Command: {0} {1}" -f $python, ($args -join ' '))

# Pre-run CSV row count
$before = Get-RowCount $outCsv
Log "Rows before: $before"

# Run Python via Start-Process and capture both streams
$stdOut = Join-Path $logDir ("stdout-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))
$stdErr = Join-Path $logDir ("stderr-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))

$p = Start-Process -FilePath $python `
  -ArgumentList $args `
  -WorkingDirectory $root `
  -NoNewWindow -Wait -PassThru `
  -RedirectStandardOutput $stdOut `
  -RedirectStandardError  $stdErr

$code = $p.ExitCode

if (Test-Path $stdOut) { Get-Content $stdOut -Raw | Out-File -FilePath $log -Append -Encoding UTF8 }
if (Test-Path $stdErr) { Get-Content $stdErr -Raw | Out-File -FilePath $log -Append -Encoding UTF8 }

# Post-run CSV row count
$after = Get-RowCount $outCsv
$delta = $after - $before
Log "Rows after:  $after (delta: $delta)"
Log "Pipeline exit code: $code"
exit $code
