$ErrorActionPreference = "Stop"

Set-Location "C:\Users\akimw\triton"

$lockFile = "C:\Users\akimw\triton\data\results\pipeline_scheduled.lock"
$logFile  = "C:\Users\akimw\triton\data\results\pipeline_scheduled.log"

if (Test-Path $lockFile) {
    $ageMinutes = ((Get-Date) - (Get-Item $lockFile).LastWriteTime).TotalMinutes
    if ($ageMinutes -lt 180) {
        Add-Content $logFile "$(Get-Date -Format s) [SKIP] Existing lock file detected; pipeline may already be running."
        exit 0
    } else {
        Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
        Add-Content $logFile "$(Get-Date -Format s) [WARN] Removed stale lock file."
    }
}

New-Item -ItemType File -Path $lockFile -Force | Out-Null

try {
    Add-Content $logFile "$(Get-Date -Format s) [START] Running pipeline..."
    $pipelineOut = & "C:\Users\akimw\triton\.venv\Scripts\python.exe" "C:\Users\akimw\triton\run_full_pipeline.py" --verbose 2>&1
    $code = $LASTEXITCODE
    $pipelineOut | Add-Content -Path $logFile
    Add-Content $logFile "$(Get-Date -Format s) [END] ExitCode=$code"
    exit $code
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}
