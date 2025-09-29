param(
  [int]$Days = 365,
  [string]$Sentinel = "SPY"
)

$ErrorActionPreference = "Stop"

# Work from the repo root
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

function Log {
  param([string]$m)
  $line = "{0} {1}" -f ([DateTime]::Now.ToString('s')), $m
  $line | Out-File -FilePath $log -Append -Encoding UTF8
  Write-Host $line
}

function Get-LastDate([string]$file, [string]$ticker) {
  if (-not (Test-Path -LiteralPath $file)) { return $null }
  try {
    $d = Import-Csv -LiteralPath $file |
         Where-Object { $_.ticker -eq $ticker -and $_.date } |
         Sort-Object { [datetime]$_.date } |
         Select-Object -Last 1 -ExpandProperty date
    if ($d) { return [datetime]$d } else { return $null }
  } catch { return $null }
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

# Sentinel before-run
$beforeDate = Get-LastDate $outCsv $Sentinel
$beforeStr  = if ($beforeDate) { $beforeDate.ToString('yyyy-MM-dd') } else { '-' }
Log ("Sentinel ({0}) last date before: {1}" -f $Sentinel, $beforeStr)

# Build argument list
$args = @($pipeline, "-t") + $tickers + @("-d", "$Days")
$cmdLine = ($python + " " + ($args -join " "))
Log ("Command: {0}" -f $cmdLine)

# Run Python and capture output; get real exit code
$stdOut = Join-Path $logDir ("stdout-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))
$stdErr = Join-Path $logDir ("stderr-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))

$p = Start-Process -FilePath $python `
      -ArgumentList $args `
      -WorkingDirectory $root `
      -NoNewWindow -Wait -PassThru `
      -RedirectStandardOutput $stdOut `
      -RedirectStandardError  $stdErr

# Merge child logs into main log (UTF-8)
if (Test-Path $stdOut) { Get-Content $stdOut -Raw | Out-File -FilePath $log -Append -Encoding UTF8 }
if (Test-Path $stdErr) { Get-Content $stdErr -Raw | Out-File -FilePath $log -Append -Encoding UTF8 }

$code = $p.ExitCode

# Sentinel after-run
$afterDate = Get-LastDate $outCsv $Sentinel
$afterStr  = if ($afterDate) { $afterDate.ToString('yyyy-MM-dd') } else { '-' }
Log ("Sentinel ({0}) last date after:  {1}" -f $Sentinel, $afterStr)

if ($beforeDate -and $afterDate) {
  if ($afterDate -gt $beforeDate) { Log "Fresh data detected for $Sentinel ✅" }
  elseif ($afterDate -eq $beforeDate) { Log "No newer trading day available yet (window unchanged) ℹ️" }
} else {
  Log "Could not determine freshness (missing dates) ⚠️"
}

Log "Pipeline exit code: $code"
exit $code
