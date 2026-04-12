param(
  [string]$Ref = 'main',
  [switch]$DownloadArtifacts,
  [switch]$OpenPR,
  [switch]$ShowSummary,
  [switch]$OnlyWatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# -------------------- Preconditions --------------------
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI ('gh') not found on PATH."
}

# We'll try these identifiers when filtering run lists or triggering.
$WorkflowKeys = @('baseline-pr', 'baseline-pr.yml', '.github/workflows/baseline-pr.yml')

# -------------------- Helpers --------------------------
function Invoke-Gh {
  param(
    [Parameter(Mandatory)][string[]]$Args,
    [switch]$Quiet
  )
  $out  = (& gh @Args 2>&1)
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    $cmd = ($Args -join ' ')
    $msg = "gh exited with code {0}: {1}" -f $code, $cmd
    $detail = ($out | Out-String).Trim()
    if ($detail) { $msg = "$msg`n$detail" }
    throw $msg
  }
  if (-not $Quiet) { $out }
}

function Get-RepoSlug {
  # Prefer JSON to avoid parsing `gh repo view`'s human output
  $out = Invoke-Gh @('repo','view','--json','nameWithOwner','--jq','.nameWithOwner')
  $repo = ($out | Out-String).Trim()
  if (-not $repo) { throw "Could not determine repository slug." }
  return $repo
}

function List-Workflows {
  param([string]$Repo)
  # Use API + PowerShell filtering (no jq needed here)
  $json = Invoke-Gh @('api',"repos/$Repo/actions/workflows")
  if (-not $json) { return @() }
  try {
    $obj = $json | ConvertFrom-Json
    return $obj.workflows
  } catch {
    return @()
  }
}

function Resolve-Workflow {
  param([string]$Repo)
  # Try to find a workflow whose name == 'baseline-pr' OR whose path ends with 'baseline-pr.yml'
  $wfs = List-Workflows -Repo $Repo
  if ($wfs.Count -gt 0) {
    $hit = $wfs | Where-Object {
      $_.name -eq 'baseline-pr' -or ($_.path -match '(^|/)\Qbaseline-pr.yml\E$')
    } | Select-Object -First 1
    if ($hit) {
      return [pscustomobject]@{
        Id   = [string]$hit.id
        Name = [string]$hit.name
        Path = [string]$hit.path
      }
    }
  }
  # Fallback: we can still trigger by file key (WorkflowKeys[1]) and list runs by name filter.
  return $null
}

function Get-RunList {
  param(
    [string]$WorkflowFilter,     # name or path; can be $null for repo-wide
    [string]$ExtraArgs = '',     # e.g. '--status in_progress'
    [int]$Limit = 50
  )
  # Build args for: gh run list
  $args = @('run','list','--limit',"$Limit",'--json','databaseId,event,createdAt,status,name,displayTitle')
  if ($WorkflowFilter) { $args += @('--workflow', $WorkflowFilter) }
  if ($ExtraArgs) {
    # split on spaces while keeping tokens; user only passes simple flags here
    $args += ($ExtraArgs -split '\s+' | Where-Object { $_ -ne '' })
  }
  $json = Invoke-Gh -Args $args -Quiet
  if (-not $json) { return @() }
  try {
    return $json | ConvertFrom-Json
  } catch {
    return @()
  }
}

function Merge-Dedup {
  param([Object[]]$Items)
  if (-not $Items) { return @() }
  $seen = @{}
  $out  = New-Object System.Collections.Generic.List[object]
  foreach ($it in $Items) {
    $key = "$($it.databaseId)"
    if (-not $seen.ContainsKey($key)) {
      $seen[$key] = $true
      [void]$out.Add($it)
    }
  }
  return $out
}

function Find-NonPushRid {
  # newest workflow_dispatch or schedule across our workflow keys
  $all = @()
  foreach ($wf in $WorkflowKeys) { $all += Get-RunList -WorkflowFilter $wf }
  $all = Merge-Dedup $all
  $hit = $all |
    Where-Object { $_.event -in @('workflow_dispatch','schedule') } |
    Sort-Object { [datetime]$_.createdAt } -Descending |
    Select-Object -First 1
  if ($hit) { return [string]$hit.databaseId }
  return $null
}

function Find-InProgressRid {
  $all = @()
  foreach ($wf in $WorkflowKeys) { $all += Get-RunList -WorkflowFilter $wf -ExtraArgs '--status in_progress' -Limit 20 }
  $all = Merge-Dedup $all
  $hit = $all | Sort-Object { [datetime]$_.createdAt } -Descending | Select-Object -First 1
  if ($hit) { return [string]$hit.databaseId }
  return $null
}

function Find-TriggeredRid {
  param(
    [datetime]$SinceUtc,
    [int]$Attempts = 90,
    [int]$DelaySeconds = 2
  )
  $since = $SinceUtc.ToUniversalTime()
  for ($i = 0; $i -lt $Attempts; $i++) {
    $all = @()
    foreach ($wf in $WorkflowKeys) { $all += Get-RunList -WorkflowFilter $wf -Limit 30 }
    $all = Merge-Dedup $all
    $hit = $all |
      Where-Object { $_.event -in @('workflow_dispatch','schedule') } |
      Where-Object { ([datetime]$_.createdAt) -ge $since } |
      Sort-Object { [datetime]$_.createdAt } -Descending |
      Select-Object -First 1
    if ($hit) { return [string]$hit.databaseId }
    Start-Sleep -Seconds $DelaySeconds
  }
  return $null
}

function Show-RecentRuns {
  param([int]$Limit = 8)
  $list = Get-RunList -WorkflowFilter $null -Limit $Limit
  if (-not $list -or $list.Count -eq 0) {
    Write-Host "(no rows to display)"
    return
  }
  $list |
    Sort-Object { [datetime]$_.createdAt } -Descending |
    Select-Object createdAt, event, name, displayTitle, status, databaseId |
    Format-Table -AutoSize
}

# -------------------- Main -----------------------------
$repo = Get-RepoSlug
$wfResolved = Resolve-Workflow -Repo $repo  # may be $null; we can still operate

$rid = $null

if ($OnlyWatch) {
  $rid = Find-InProgressRid
  if (-not $rid) { throw "OnlyWatch was set but no in-progress baseline-pr run is active." }
  Write-Host "Watching run $rid ..."
  Invoke-Gh @('run','watch',$rid) | Out-Null
} else {
  # Prefer an in-progress run to avoid concurrency cancel
  $rid = Find-InProgressRid
  if ($rid) {
    Write-Host "Found in-progress run $rid - watching instead of triggering a new one..."
  } else {
    $triggerAt = (Get-Date).ToUniversalTime()
    Write-Host "Triggering baseline-pr on '$Ref'..."

    # Choose the most-specific key to trigger:
    $triggerKey = if ($wfResolved -and $wfResolved.Path) { $wfResolved.Path } else { $WorkflowKeys[1] }  # 'baseline-pr.yml'
    Invoke-Gh @('workflow','run', $triggerKey, '--ref', $Ref) -Quiet | Out-Null

    # Give Actions a moment to index the new run
    Start-Sleep -Seconds 3

    # Poll for the newly-triggered run
    $rid = Find-TriggeredRid -SinceUtc $triggerAt
    if (-not $rid) {
      Write-Warning "No run matched within time window; showing the $([int]8) most recent runs repo-wide:"
      Show-RecentRuns -Limit 8
      # last-resort fallback
      $rid = Find-NonPushRid
    }
    if (-not $rid) { throw "No non-push baseline-pr run found after trigger." }
  }

  Write-Host "Watching run $rid ..."
  Invoke-Gh @('run','watch',$rid) | Out-Null
}

Write-Host "`n== Status =="
Invoke-Gh @('run','view',$rid,'--json','name,event,status,conclusion,createdAt,updatedAt','--jq','.') | Out-Null
Invoke-Gh @('run','view',$rid,'--json','name,event,status,conclusion,createdAt,updatedAt','--jq','.')  # echo the JSON

# PR info: branch is created by the workflow as "ci/baseline-<rid>"
$branch = "ci/baseline-$rid"
try {
  $pr = Invoke-Gh @('pr','view',$branch,'--json','number,url,state','--jq','.')
  Write-Host "`n== PR =="
  $pr
  if ($OpenPR) { Invoke-Gh @('pr','view',$branch,'--web') -Quiet | Out-Null }
} catch {
  Write-Warning "No PR found for branch $branch (maybe skipped due to data prerequisites?)."
}

# Artifacts (optional)
if ($DownloadArtifacts) {
  $dest = "out\gha\baseline-pr-assets-$rid"
  if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $dest | Out-Null

  Write-Host "`n== Artifacts =="
  try {
    $arts = Invoke-Gh @('api',"repos/$repo/actions/runs/$rid/artifacts")
    if ($arts) {
      ($arts | ConvertFrom-Json).artifacts |
        Select-Object @{Name='name';Expression={$_.name}},
                      @{Name='size';Expression={$_.size_in_bytes}},
                      @{Name='expired';Expression={$_.expired}},
                      @{Name='url';Expression={$_.archive_download_url}} |
        Format-Table -AutoSize
    }
  } catch {
    Write-Warning "Failed to list artifacts: $($_.Exception.Message)"
  }

  try {
    Invoke-Gh @('run','download',$rid,'--name','baseline-pr-assets','-D', $dest) -Quiet | Out-Null
    Write-Host "Artifacts -> $dest"
    if ($ShowSummary -and (Test-Path "$dest\out\audit\SUMMARY.md")) {
      Write-Host "`n== PR Summary (from artifact) =="
      Get-Content "$dest\out\audit\SUMMARY.md"
    }
  } catch {
    Write-Warning "No 'baseline-pr-assets' artifact found or download failed: $($_.Exception.Message)"
  }
}
