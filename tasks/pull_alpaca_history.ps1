<#  tasks\pull_alpaca_history.ps1
    Fetch Alpaca portfolio history → normalize → import into Triton → run STRICT analyzer.

    Examples:
      .\tasks\pull_alpaca_history.ps1 -PreferExisting -Backup
      .\tasks\pull_alpaca_history.ps1 -Live -Period 6M -Timeframe 1D -ExtendedHours
#>

[CmdletBinding()]
param(
  [ValidateSet("1D","1W","1M","3M","6M","12M","ALL")]
  [string] $Period        = "12M",         # Alpaca: months or ALL (not years)
  [ValidateSet("1Min","5Min","15Min","1H","1D")]
  [string] $Timeframe     = "1D",          # 1Min/5Min/15Min/1H/1D
  [switch] $ExtendedHours,                 # include extended hours on intraday
  [switch] $Live,                          # use live endpoint instead of paper
  [string] $OutCsv        = "$HOME\Downloads\alpaca_equity_history.csv",
  [string] $MinDate       = "2025-09-01",  # only import rows >= this date
  [switch] $PreferExisting,                # keep existing rows on timestamp collisions
  [switch] $Backup,                        # backup Triton history before writing
  [switch] $Quiet
)

$ErrorActionPreference = "Stop"

function Write-Info($msg){ if(-not $Quiet){ Write-Host $msg } }
function Fail($msg){ Write-Error $msg; exit 1 }

# --- Resolve endpoint & headers ---
$alpacaHost = if ($Live.IsPresent) { "api.alpaca.markets" } else { "paper-api.alpaca.markets" }
$path       = "/v2/account/portfolio/history"  # keep leading slash

$k = $env:APCA_API_KEY_ID
$s = $env:APCA_API_SECRET_KEY
if([string]::IsNullOrWhiteSpace($k) -or [string]::IsNullOrWhiteSpace($s)){
  Fail "APCA env vars not set. Run:
  `$env:APCA_API_KEY_ID = '...'
  `$env:APCA_API_SECRET_KEY = '...'"
}

# --- Build query string safely (and always prefix with '?') ---
$qh = @{
  period         = $Period
  timeframe      = $Timeframe
  extended_hours = $ExtendedHours.IsPresent.ToString().ToLower()
}

# Join non-empty key/values into k=v&k=v...
$qsPairs = @()
foreach ($kv in $qh.GetEnumerator()) {
  if ($null -ne $kv.Value -and "$($kv.Value)".Length -gt 0) {
    $qsPairs += ("{0}={1}" -f $kv.Key, $kv.Value)
  }
}
$qs  = ($qsPairs -join "&")
$URL = "https://$alpacaHost$path" + ($(if ($qs) { "?$qs" } else { "" }))

$hdr = @{ "APCA-API-KEY-ID" = $k; "APCA-API-SECRET-KEY" = $s }

Write-Info "[GET] $URL"
try {
  $resp = Invoke-RestMethod -Headers $hdr -Uri $URL -Method GET
} catch {
  Fail ("Alpaca request failed: " + $_.Exception.Message)
}

# --- Normalize response to date,equity ---
$propNames = $resp.PSObject.Properties.Name
$hasTimestamp  = $propNames -contains 'timestamp'
$hasTimestamps = $propNames -contains 'timestamps'
if(-not ($hasTimestamp -or $hasTimestamps)){
  Fail ("Response missing timestamp(s) field. Got: " + ($propNames -join ", "))
}

$ts = if($hasTimestamp){ $resp.timestamp } else { $resp.timestamps }
$eq = $resp.equity
if(-not $ts -or -not $eq -or $ts.Count -ne $eq.Count){
  Fail "Unexpected shape: timestamps=$($ts.Count) equity=$($eq.Count)"
}

$rows = for ($i=0; $i -lt $ts.Count; $i++) {
  $raw = $ts[$i]
  if ($raw -is [string]) {
    $dt = [datetime]::Parse($raw) # ISO8601
  } else {
    $dt = [DateTimeOffset]::FromUnixTimeSeconds([int64]$raw).UtcDateTime
  }
  [pscustomobject]@{
    date   = $dt.ToString("yyyy-MM-dd HH:mm:ss")
    equity = [double]$eq[$i]
  }
}
if(-not $rows -or $rows.Count -eq 0){ Fail "No rows parsed from Alpaca." }

# --- Write CSV then filter zero-equity rows (common in empty periods) ---
$dir = Split-Path -Parent $OutCsv
if(-not (Test-Path $dir)){ New-Item -ItemType Directory -Path $dir | Out-Null }

$rows | Export-Csv $OutCsv -NoTypeInformation -Encoding UTF8
Write-Info "[OK] Wrote $OutCsv"

$temp = [System.IO.Path]::ChangeExtension($OutCsv, ".filtered.csv")
$filtered = Import-Csv $OutCsv | Where-Object { [double]$_.equity -ne 0 }
if ($filtered -and $filtered.Count -gt 0) {
  $filtered | Export-Csv $temp -NoTypeInformation -Encoding UTF8
  Move-Item $temp $OutCsv -Force
  Write-Info "[OK] Filtered rows kept: $($filtered.Count)"
} else {
  if(Test-Path $temp){ Remove-Item $temp -Force }
  Write-Info "[WARN] Filter removed all rows; leaving original file."
}

# --- Import into TRITON ---
$importArgs = @(".\scripts\import_equity_history.py","--source",$OutCsv,"--min-date",$MinDate)
if($PreferExisting){ $importArgs += "--prefer-existing" }
if($Backup){ $importArgs += "--backup" }

Write-Info "[PY] python $($importArgs -join ' ')"
& python @importArgs
if($LASTEXITCODE -ne 0){ Fail "Importer returned exit code $LASTEXITCODE" }

# --- Run STRICT analyzer ---
Write-Info "[RUN] tasks\run_baseline_analyzer_strict.cmd"
& .\tasks\run_baseline_analyzer_strict.cmd
exit $LASTEXITCODE
