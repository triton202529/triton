# scripts/poll_open.ps1
# ---------------------------------------
# TRITON — Poll Open Orders (Paper)
# Polls by LOG-SESSION (correct grouping)
#
# HARDENING:
#  - single-run lock per LogSession (prevents concurrent pollers)
#  - uses .venv python if present
#  - writes heartbeat JSON to data/results/poll_<LogSession>.json
#  - releases lock on exit (best-effort)

param(
  [string]$LogSession = "2026-02-02_OPEN",
  [int]$Loops = 30,
  [int]$SleepSec = 60
)

$ErrorActionPreference = "Stop"

# Repo root = parent of scripts/
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

# Prefer venv python if present
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Py = if (Test-Path $VenvPy) { $VenvPy } else { "python" }

# Lock + heartbeat paths
$LockDir = Join-Path $RepoRoot "data\live\locks"
$HeartbeatDir = Join-Path $RepoRoot "data\results"
$SafeSession = ($LogSession -replace '[^A-Za-z0-9_\-\.]+','_')

$LockPath = Join-Path $LockDir ("poll_{0}.lock" -f $SafeSession)
$HeartbeatPath = Join-Path $HeartbeatDir ("poll_{0}.json" -f $SafeSession)

New-Item -ItemType Directory -Force -Path $LockDir | Out-Null
New-Item -ItemType Directory -Force -Path $HeartbeatDir | Out-Null

# Acquire exclusive lock (fails fast if another poller is running)
$lockStream = $null
try {
  $lockStream = [System.IO.File]::Open(
    $LockPath,
    [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
  )

  # Stamp lock content
  $lockStream.SetLength(0)
  $lockWriter = New-Object System.IO.StreamWriter($lockStream, [System.Text.Encoding]::UTF8, 1024, $true)
  $lockWriter.WriteLine("pid=$PID")
  $lockWriter.WriteLine("log_session=$LogSession")
  $lockWriter.WriteLine("started_utc=$([DateTime]::UtcNow.ToString('o'))")
  $lockWriter.Flush()
}
catch {
  Write-Host ""
  Write-Host ("[BLOCK] Another poller is already running for log-session={0}" -f $LogSession) -ForegroundColor Red
  Write-Host ("        Lock file: {0}" -f $LockPath) -ForegroundColor DarkGray
  Write-Host ("        Stop the other poller (or delete the lock if it crashed), then retry.") -ForegroundColor DarkGray
  exit 2
}

# Ensure lock release on exit (best-effort)
$cleanup = {
  try {
    if ($lockStream) { $lockStream.Close(); $lockStream.Dispose() }
  } catch {}
  try {
    if (Test-Path $LockPath) { Remove-Item -Force $LockPath -ErrorAction SilentlyContinue }
  } catch {}
}
Register-EngineEvent PowerShell.Exiting -Action $cleanup | Out-Null

function Write-Heartbeat([int]$i, [string]$phase) {
  $obj = [ordered]@{
    log_session   = $LogSession
    pid           = $PID
    phase         = $phase
    loop          = $i
    loops         = $Loops
    sleep_sec     = $SleepSec
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    repo_root     = $RepoRoot.Path
    python        = $Py
  }
  $json = $obj | ConvertTo-Json -Depth 5
  # Atomic-ish write: write temp then move
  $tmp = $HeartbeatPath + ".tmp"
  $json | Out-File -FilePath $tmp -Encoding utf8 -Force
  Move-Item -Force -Path $tmp -Destination $HeartbeatPath
}

try {
  for ($i = 1; $i -le $Loops; $i++) {
    Write-Host ""
    Write-Host ("Polling ({0}/{1}) — log-session={2}" -f $i, $Loops, $LogSession) -ForegroundColor Yellow

    Write-Heartbeat -i $i -phase "poll_start"

    & $Py -m services.poll_order_status --mode paper --session $LogSession --refresh

    if ($LASTEXITCODE -ne 0) {
      Write-Heartbeat -i $i -phase ("poll_failed_exit_{0}" -f $LASTEXITCODE)
      throw ("poll_order_status failed (exit {0})" -f $LASTEXITCODE)
    }

    Write-Heartbeat -i $i -phase "poll_ok"

    if ($i -lt $Loops) {
      Start-Sleep -Seconds $SleepSec
    }
  }

  Write-Heartbeat -i $Loops -phase "done"
}
finally {
  & $cleanup
}
