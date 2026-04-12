param(
  [string]$Ref = "main",
  [switch]$Download,   # download baseline-pr-assets artifact
  [switch]$Summary     # print out\audit\SUMMARY.md after downloading
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI (gh) not found on PATH."
}

function Get-Rid {
  param(
    [string]$Status,   # queued | in_progress | completed | "" (any)
    [string]$Event     # workflow_dispatch | schedule | "" (any)
  )
  $common = @("--limit","1","--json","databaseId","--jq",".[0].databaseId")
  $base1  = @("run","list","--workflow","baseline-pr")
  $base2  = @("run","list","--workflow","baseline-pr.yml")

  if ($Status) { $base1 += @("--status", $Status); $base2 += @("--status", $Status) }
  if ($Event)  { $base1 += @("--event",  $Event ); $base2 += @("--event",  $Event  ) }

  $rid = (& gh @base1 @common 2>$null)
  if (-not $rid -or $rid -eq "null") {
    $rid = (& gh @base2 @common 2>$null)
  }
  if ($rid -and $rid -ne "null") { return $rid.Trim() } else { return $null }
}

# 0) If a run is already queued or in-progress, just watch it.
$rid = Get-Rid -Status "in_progress" -Event ""
if (-not $rid) { $rid = Get-Rid -Status "queued" -Event "" }

if ($rid) {
  Write-Host "Found existing run $rid (queued/running) — watching instead of triggering…"
} else {
  # 1) Trigger a fresh run
  Write-Host "Triggering baseline-pr on `$Ref`..."
  & gh workflow run "baseline-pr.yml" --ref $Ref | Out-Null

  # 2) Poll for the new non-push run (prefer workflow_dispatch, then schedule)
  for ($i = 0; $i -lt 45 -and -not $rid; $i++) {
    Start-Sleep -Seconds 2
    $rid = Get-Rid -Status "" -Event "workflow_dispatch"
    if (-not $rid) { $rid = Get-Rid -Status "" -Event "schedule" }
  }

  if (-not $rid) {
    Write-Warning "Couldn't find a new run yet. Dumping recent runs for visibility..."
    (& gh run list --limit 8 --json databaseId,event,name,displayTitle,createdAt,status,conclusion |
      ConvertFrom-Json |
      Sort-Object {[datetime]$_.createdAt} -Descending |
      Format-Table createdAt,event,name,displayTitle,status,conclusion,databaseId)
    throw "No non-push baseline-pr run found after trigger."
  }
}

Write-Host "Watching $rid ..."
& gh run watch $rid

Write-Host "`n== Status =="
$runJson = & gh run view $rid --json name,event,status,conclusion,createdAt,updatedAt --jq "."
$run     = $null
try { $run = $runJson | ConvertFrom-Json } catch {}
$runJson

# Branch used by the workflow’s PR
$branch = "ci/baseline-$rid"

Write-Host "`n== PR =="
$prNum = $null
try {
  & gh pr view $branch --json number,url,state --jq "."
  $prNum = & gh pr view $branch --json number --jq .number 2>$null
  if ($prNum) {
    & gh pr view $branch --web | Out-Null
  }
} catch {
  Write-Warning ("PR not found for branch {0} (maybe cancelled or skipped)." -f $branch)
}

# Artifacts / Summary (only if run succeeded)
if ($Download) {
  $conc = if ($run) { $run.conclusion } else { "<unknown>" }
  if (-not $run -or $run.conclusion -ne "success") {
    Write-Warning ("Run conclusion is {0}; skipping artifact download." -f $conc)
    return
  }

  $dest = "out\gha\baseline-pr-assets-$rid"
  if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $dest | Out-Null

  try {
    & gh run download $rid --name baseline-pr-assets -D $dest
    Write-Host "Artifacts -> $dest"
  } catch {
    Write-Warning ("No baseline-pr-assets artifact found or download failed: {0}" -f ($_.Exception.Message))
    return
  }

  if ($Summary -and (Test-Path "$dest\out\audit\SUMMARY.md")) {
    Write-Host "`n== PR Summary (from artifact) =="
    Get-Content "$dest\out\audit\SUMMARY.md"
  }
}
