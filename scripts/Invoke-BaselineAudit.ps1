# scripts/Invoke-BaselineAudit.ps1
[CmdletBinding()]
param(
  [string]$MergeSha,                                  # optional: compare <sha>^ vs <sha>
  [Parameter(Mandatory = $true)][string]$Path,        # required: path to weights.csv
  [string]$BaseRef = "origin/main",                   # used when -MergeSha is omitted
  [int]$TopN = 20,
  [string]$OutDir = (Join-Path $PWD ("out\audit-" + (Get-Date -f yyyyMMdd_HHmmss)))
)

$ErrorActionPreference = 'Stop'

function Write-Note { param([string]$msg) Write-Host "[audit] $msg" }

# --- guards ---
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) { throw "git not found on PATH" }
if (-not $MergeSha) { Write-Note "Fetching $BaseRef ..."; git fetch origin main --depth=1 | Out-Null }

# Output dir
if (-not (Test-Path -LiteralPath $OutDir)) { [void](New-Item -ItemType Directory -Force -Path $OutDir) }

# --- helpers ---
function Parse-Weight {
  param([string]$s)
  if ($null -eq $s -or $s -eq '') { return [double]0 }
  $t = ($s.Trim() -replace ',', '')
  if ($t -match '%$') { return ([double]::Parse($t.TrimEnd('%'), [System.Globalization.CultureInfo]::InvariantCulture) / 100.0) }
  return [double]::Parse($t, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Load-WeightsFromCsv {
  param([string]$csvPath)
  if (-not (Test-Path -LiteralPath $csvPath)) { return @{} }
  $map = @{}
  $rows = Import-Csv -LiteralPath $csvPath
  foreach($r in $rows){
    if ($r.PSObject.Properties.Name -notcontains 'ticker' -or
        $r.PSObject.Properties.Name -notcontains 'target_weight') {
      throw "CSV '$csvPath' must contain columns: ticker,target_weight"
    }
    $map[$r.ticker] = Parse-Weight $r.target_weight
  }
  return $map
}

function SumX { param([double[]]$xs) return ( $xs | Measure-Object -Sum ).Sum }
function HHI  { param([double[]]$xs) return ( SumX ($xs | ForEach-Object { $_ * $_ }) ) }
function Ent  { param([double[]]$xs) $e=0.0; foreach($v in $xs){ if($v -gt 0){ $e += -1.0*$v*[math]::Log($v) } } return $e }
function EffN { param([double[]]$xs) return [math]::Exp( (Ent $xs) ) }
function Pct  { param([double]$x) return ("{0:P4}" -f $x) }

# Safely run `git show` and return string or $null (suppresses native command errors)
function GitShowSafe {
  param([string]$Spec)
  $old = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'SilentlyContinue'
    $out = (& git show --no-pager $Spec 2>$null | Out-String)
    $rc  = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $old
  }
  if ($rc -eq 0) { return $out } else { return $null }
}

# --- resolve CSV snapshots ---
$prevCsv = Join-Path $OutDir 'weights_prev.csv'
$newCsv  = Join-Path $OutDir 'weights_new.csv'

if ($MergeSha) {
  Write-Note "Auditing '$Path' at merge $MergeSha (prev=<sha>^, new=<sha>)"
  git cat-file -t $MergeSha 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) {
    git fetch --all --prune | Out-Null
    git cat-file -t $MergeSha 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Commit $MergeSha not found locally even after fetch." }
  }

  $x = GitShowSafe "$($MergeSha)^:$Path"
  if ($null -ne $x) { $x | Set-Content -Path $prevCsv -Encoding UTF8 } else { "" | Set-Content -Path $prevCsv -Encoding UTF8 }

  $y = GitShowSafe "$($MergeSha):$Path"
  if ($null -ne $y) { $y | Set-Content -Path $newCsv  -Encoding UTF8 } else { throw "File '$Path' not present at $MergeSha." }
}
else {
  Write-Note "Auditing '$Path' (prev=$BaseRef, new=working file)"
  $x = GitShowSafe "$($BaseRef):$Path"
  if ($null -ne $x) { $x | Set-Content -Path $prevCsv -Encoding UTF8 } else { "" | Set-Content -Path $prevCsv -Encoding UTF8 }

  Get-Content -Raw -LiteralPath $Path | Set-Content -Path $newCsv -Encoding UTF8
}

# --- load & metrics ---
$pm = Load-WeightsFromCsv -csvPath $prevCsv
$nm = Load-WeightsFromCsv -csvPath $newCsv
if ($nm.Count -eq 0) { throw "Failed to parse NEW CSV '$newCsv' or it has no rows." }

$pvals = [double[]]($pm.Values)
$nvals = [double[]]($nm.Values)

$aprev = [pscustomobject]@{
  Sum     = $( if ($pvals.Count) { SumX $pvals } else { 0.0 } )
  HHI     = $( if ($pvals.Count) { HHI  $pvals } else { 0.0 } )
  Entropy = $( if ($pvals.Count) { Ent  $pvals } else { 0.0 } )
  EffN    = $( if ($pvals.Count) { EffN $pvals } else { 0.0 } )
}
$anew  = [pscustomobject]@{
  Sum     = (SumX $nvals)
  HHI     = (HHI  $nvals)
  Entropy = (Ent  $nvals)
  EffN    = (EffN $nvals)
}

# Detect empty/absent previous snapshot (first run on a branch)
$prevEmpty = ( ($pvals | Measure-Object).Count -eq 0 ) -or ( [math]::Abs($aprev.Sum) -lt 1e-9 )

# --- diff ---
$all = ($pm.Keys + $nm.Keys) | Sort-Object -Unique
$diff = foreach($t in $all){
  $o = $( if ($pm.ContainsKey($t)) { $pm[$t] } else { 0.0 } )
  $n = $( if ($nm.ContainsKey($t)) { $nm[$t] } else { 0.0 } )
  [pscustomobject]@{ ticker=$t; old=[double]$o; new=[double]$n; delta=[double]($n-$o) }
}

# Turnover: treated as N/A on initial snapshot
$turnover = 0.5 * ( ( $diff | ForEach-Object { [math]::Abs($_.delta) } | Measure-Object -Sum ).Sum )
$turnoverDisplay = $( if ($prevEmpty) { "N/A (initial snapshot)" } else { (Pct $turnover) } )

# --- write CSV ---
$diff | Sort-Object @{Expression={ [math]::Abs($_.delta) }; Descending=$true} |
  Export-Csv (Join-Path $OutDir "baseline_diff.csv") -NoTypeInformation -Encoding UTF8

# --- write summary markdown ---
$top = $diff | Sort-Object @{Expression={ [math]::Abs($_.delta) }; Descending=$true} | Select-Object -First $TopN
$md = @()
$md += "# Baseline change summary"
$md += ""
$md += "## Metrics"
$md += ""
$md += "| metric | previous | new |"
$md += "|---|---:|---:|"
$md += "| sum      | {0} | {1} |" -f ("{0:N6}" -f $aprev.Sum), ("{0:N6}" -f $anew.Sum)
$md += "| HHI      | {0} | {1} |" -f ("{0:N4}" -f $aprev.HHI),  ("{0:N4}" -f $anew.HHI)
$md += "| Entropy  | {0} | {1} |" -f ("{0:N4}" -f $aprev.Entropy), ("{0:N4}" -f $anew.Entropy)
$md += "| EffN     | {0} | {1} |" -f ("{0:N2}" -f $aprev.EffN), ("{0:N2}" -f $anew.EffN)
$md += "| Turnover |  | **{0}** |" -f $turnoverDisplay
if ($prevEmpty) {
  $md += ""
  $md += "> Note: No previous baseline found on $BaseRef. Treating this as the initial snapshot; turnover is not applicable."
}
$md += ""
$md += "## Top $TopN weight changes"
$md += ""
$md += "| ticker | old | new | delta |"
$md += "|---|---:|---:|---:|"
foreach($r in $top){ $md += "| {0} | {1} | {2} | {3} |" -f $r.ticker, (Pct $r.old), (Pct $r.new), (Pct $r.delta) }
$summary = Join-Path $OutDir "SUMMARY.md"
$md -join "`n" | Out-File -FilePath $summary -Encoding UTF8

# --- console output ---
Write-Host ""
Write-Host "Metrics (prev -> new):"
Write-Host ("{0,-9} {1:N6} -> {2:N6}" -f "Sum:",     $aprev.Sum,     $anew.Sum)
Write-Host ("{0,-9} {1:N4} -> {2:N4}" -f "HHI:",     $aprev.HHI,     $anew.HHI)
Write-Host ("{0,-9} {1:N4} -> {2:N4}" -f "Entropy:", $aprev.Entropy, $anew.Entropy)
Write-Host ("{0,-9} {1:N2} -> {2:N2}" -f "EffN:",    $aprev.EffN,    $anew.EffN)
Write-Host ("{0,-9} {1}"    -f "Turnover:",          $turnoverDisplay)

Write-Host ""
Write-Host ("Top {0} |delta|:" -f $TopN)
$top | Format-Table ticker,
  @{n='old';e={'{0:P4}' -f $_.old}},
  @{n='new';e={'{0:P4}' -f $_.new}},
  @{n='delta';e={'{0:P4}' -f $_.delta}} | Out-Host

Write-Host ""
Write-Host "Wrote:"
Get-ChildItem -LiteralPath $OutDir | Format-Table Name,Length | Out-Host
