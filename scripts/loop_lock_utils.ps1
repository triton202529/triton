# TRITON loop lock utilities — Phase 148B orphan lock detection.

function Test-LoopProcessRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptName
    )

    $filter = "Name = 'powershell.exe' OR Name = 'pwsh.exe'"
    $procs = Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue
    if (-not $procs) {
        return $false
    }

    foreach ($proc in @($procs)) {
        $cmd = $proc.CommandLine
        if ($cmd -and $cmd -like "*$ScriptName*") {
            return $true
        }
    }

    return $false
}

function Clear-OrphanLoopLock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockFile,
        [Parameter(Mandatory = $true)]
        [string]$ScriptName
    )

    if (-not (Test-Path $LockFile)) {
        return $false
    }

    if (Test-LoopProcessRunning -ScriptName $ScriptName) {
        return $false
    }

    Write-Host "[LOCK_ORPHAN_DETECTED] lock=$LockFile script=$ScriptName"
    Remove-Item $LockFile -Force
    Write-Host "[LOCK_ORPHAN_CLEARED] lock=$LockFile"
    return $true
}
