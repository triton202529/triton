param(
  [string]$Ref = 'main',
  [switch]$DownloadArtifacts,
  [switch]$ShowSummary,
  [switch]$OpenPR,
  [switch]$OnlyWatch,
  [int]$PollSeconds = 90,          # how long to look for the new run after triggering
  [int]$PollIntervalSeconds = 2    # polling interval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# ---- prerequisites ------------------------------------------------------------
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI ('gh') not found on PATH."
}

# prefer addressing the workflow by file; we’ll also try the name
$WorkflowKeys = @(
  '.github/workflows/baseline-pr.yml',
  'baseline-pr.yml',
  'baseline-pr'
)

# ---- helpers -----------------------------------------------------------------
function Run-Gh {
  param(
    [Parameter(Mandatory)][string[]]$Args,
    [switch]$Quiet
  )
  $out = & gh @Args 2>&1
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    $msg = $out | Out-String
    throw ("gh exited with code {0}`n{1}" -f $code, $msg)
  }
  if (-not $Quiet) { $out }
}

function Gh-Trim([string[]]$Args) {
  ($null + (Run-Gh -Args $Args -Quiet)).ToString().Trim()
}

function Get-RepoSlug {
  Gh-Trim @('repo','view','--json','nameWithOwner','--jq','.nameWithOwner')
}

function Try-Get-RunId-By {
  param(
    [string]$WorkflowKey,   # may be empty to search repo-wide
    [string]$Event,         # e.g., workflow_dispatch | schedule | (omit for repo-wide latest)
    [int]$Limit = 1
  )
  $args = @('run','list','--limit',"$Limit",'--json','databaseId')
  if ($WorkflowKey) { $args += @('--workflow', $WorkflowKey) }
  if ($Event)       { $args += @('--event', $Event) }
  $args += @('--jq','.[0].databaseId')
  $rid = Gh-Trim $args
  if ($rid -and $rid -ne 'null') { return $rid } else { return $null }
}

function Get-Latest-DispatchOrSchedule {
  # Try each workflow key; prefer workflow_dispatch, then schedule; finally repo-wide fallback
  foreach ($wf in $WorkflowKeys) {
    $rid = Try-Get-RunId-By -WorkflowKey $wf -Event 'workflow_dispatch'
    if ($rid) { return $rid }
  }
  foreach ($wf in $WorkflowKeys) {
    $rid = Try-Get-RunId-By -WorkflowKey $wf -Event 'schedule'
    if ($rid) { return $rid }
  }
  # repo-wide fallback (any dispatch)
  $rid = Try-Get-RunId-By -WorkflowKey '' -Event 'workflow_dispatch'
  if ($rid) { return $rid }
  return $null
}

function Get-InProgressRid {
  foreach ($wf in $WorkflowKeys) {
    $rid = Gh-Trim @('run','list','--workflow', $wf,'--status','in_progress','--limit','1','--json','databaseId','--jq','.[0].databaseId')
    if ($rid -and $rid -ne 'null') { return $rid }
  }
  # repo-wide fallback
  $rid = Gh-Trim @('run','list','--status','in_progress','--limit','1','--json','databaseId','--jq','.[0].databaseId')
  if ($rid -and $rid -ne 'null') { return $rid }
  return $null
}

function Poll-For-New-RunId {
  param(
    [datetime]$SinceUtc,
    [int]$Seconds,
    [int]$IntervalSeconds
  )
  $deadline = $SinceUtc.AddSeconds($Seconds)
  while ([datetime]::UtcNow -lt $deadline) {
    foreach ($wf in $WorkflowKeys) {
      $json = Run-Gh @('run','list','--workflow', $wf,'--limit','30','--json','databaseId,event,createdAt') -Quiet
      if ($json) {
        try {
          $runs = $json | ConvertFrom-Json
          $hit  = $runs |
            Where-Object { $_.event -in @('workflow_dispatch','schedule') -and ([datetime]$_.createdAt).ToUniversalTime() -ge $SinceUtc } |
            Sort-Object { [datetime]$_.createdAt } -Descending |
            Select-Object -First 1
          if ($hit) { return [string]$hit.databaseId }
        } catch { }
      }
    }
    # repo-wide last chance this loop
    $json = Run-Gh @('run','list','--limit','30','--json','databaseId,event,createdAt,name,displayTitle') -Quiet
    if ($json) {
      try {
        $runs = $json | ConvertFrom-Json
        $hit  = $runs |
          Where-Object {
            $_.event -in @('workflow_dispatch','schedule') -and
            ($_.name -eq 'baseline-pr' -or $_.displayTitle -eq 'baseline-pr') -and
            ([datetime]$_.createdAt).ToUniversalTime() -ge $SinceUtc
          } |
          Sort-Object { [datetime]$_.createdAt } -Descending |
          Select-Object -First 1
        if ($hit) { return [string]$hit.databaseId }
      } catch { }
    }
    Start-Sleep -Seconds $IntervalSeconds
  }
  return $null
}

function Show-Recent-Runs {
  param([int]$Count = 8)
  $json = Run-Gh @('run','list','--limit',"$Count",'--json','databaseId,event,name,displayTitle,createdAt,status,conclusion') -Quiet
  if (-not $json) { Write-Host "(no rows to display)"; return }
  try {
    $rows = $json | ConvertFrom-Json | Sort-Object { [datetime]$_.createdAt } -Descending
    $rows | Format-Table createdAt,event,name,displayTitle,status,conclusion,databaseId
  } catch {
    Write-Host "(no rows to display)"
  }
}

# ---- main --------------------------------------------------------------------
# If OnlyWatch: just find an in-progress run and watch it
if ($OnlyWatch) {
  $rid = Get-InProgressRid
  if (-not $rid) { throw "OnlyWatch was set but no in-progress baseline-pr run is active." }
  Write-Host "Watching run $rid ..."
  Run-Gh @('run','watch',"$rid") | Out-Null
  Write-Host "`n== Status =="
  Run-Gh @('run','view',"$rid",'--json','name,event,status,conclusion,createdAt,updatedAt','--jq','.') | Out-Null
  return
}

# Prefer an already in-progress run to avoid concurrency cancellation
$rid = Get-InProgressRid
if ($rid) {
  Write-Host "Found in-progress run $rid - watching instead of triggering a new one..."
  Run-Gh @('run','watch',"$rid") | Out-Null
} else {
  # Trigger a new run via the file key (most reliable)
  Write-Host "Triggering baseline-pr on '$Ref'..."
  # Pick the first existing workflow key
  $wfToUse = $WorkflowKeys[0]
  # If the first fails, try the next ones during resolve step below
  Run-Gh @('workflow','run',"$wfToUse",'--ref',"$Ref") -Quiet | Out-Null

  $since = (Get-Date).ToUniversalTime().AddSeconds(-3)  # small cushion
  Start-Sleep -Seconds 3

  # Poll for the new run to appear
  $rid = Poll-For-New-RunId -SinceUtc $since -Seconds $PollSeconds -IntervalSeconds $PollIntervalSeconds

  if (-not $rid) {
    Write-Warning "No run matched within time window; showing the $([math]::Min(8,$PollSeconds)) most recent runs repo-wide:"
    Show-Recent-Runs -Count 8
    throw "No non-push baseline-pr run found after trigger."
  }

  Write-Host "Watching run $rid ..."
  Run-Gh @('run','watch',"$rid") | Out-Null
}

# Status
Write-Host "`n== Status =="
Run-Gh @('run','view',"$rid",'--json','name,event,status,conclusion,createdAt,updatedAt','--jq','.') | Out-Null

# PR info (branch created by the workflow)
$branch = "ci/baseline-$rid"
try {
  Write-Host "`n== PR =="
  Run-Gh @('pr','view',"$branch",'--json','number,url,state','--jq','.') | Out-Null
  if ($OpenPR) { Run-Gh @('pr','view',"$branch",'--web') -Quiet | Out-Null }
} catch {
  Write-Warning "No PR found for branch $branch (maybe skipped due to data prerequisites?)."
}

# Artifacts
if ($DownloadArtifacts) {
  $repo = Get-RepoSlug
  $dest = "out\gha\baseline-pr-assets-$rid"
  if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $dest | Out-Null

  Write-Host "`n== Artifacts =="
  try {
    Run-Gh @('api',"repos/$repo/actions/runs/$rid/artifacts",'--jq','.artifacts[] | {name, size: .size_in_bytes, expired, url: .archive_download_url}') | Out-Null
  } catch {
    Write-Warning "Failed to list artifacts via API: $($_.Exception.Message)"
  }

  try {
    Run-Gh @('run','download',"$rid",'--name','baseline-pr-assets','-D',"$dest") -Quiet | Out-Null
    Write-Host "Artifacts -> $dest"
    if ($ShowSummary -and (Test-Path "$dest\out\audit\SUMMARY.md")) {
      Write-Host "`n== PR Summary (from artifact) =="
      Get-Content "$dest\out\audit\SUMMARY.md"
    }
  } catch {
    Write-Warning "No 'baseline-pr-assets' artifact found or download failed: $($_.Exception.Message)"
  }
}

<#
Examples:
  .\tools\Invoke-BaselinePr.ps1                      # trigger + watch + status + PR
  .\tools\Invoke-BaselinePr.ps1 -DownloadArtifacts   # also download artifacts
  .\tools\Invoke-BaselinePr.ps1 -ShowSummary -DownloadArtifacts
  .\tools\Invoke-BaselinePr.ps1 -OnlyWatch           # just attach to an in-flight run
  .\tools\Invoke-BaselinePr.ps1 -Ref main            # run against a specific ref
#>
