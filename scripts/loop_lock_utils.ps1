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

function Update-LoopLockMetadata {
    <#
    Refresh loop lock metadata each cycle so audits distinguish active ownership
    from orphan stale locks. Preserves started_at; updates refreshed_at/pid/source/cycle.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockFile,
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [int]$Cycle = 0
    )

    $now = Get-Date -Format s
    $startedAt = $now

    if (Test-Path $LockFile) {
        try {
            $content = Get-Content -Path $LockFile -Raw -ErrorAction Stop
            if ($content -match 'started_at=([^\r\n]+)') {
                $startedAt = $Matches[1].Trim()
            }
        } catch {
            $startedAt = $now
        }
    }

    $meta = @(
        "started_at=$startedAt",
        "refreshed_at=$now",
        "pid=$PID",
        "source=$Source",
        "cycle=$Cycle"
    ) -join [Environment]::NewLine

    Set-Content -Path $LockFile -Value $meta -Encoding UTF8
}
