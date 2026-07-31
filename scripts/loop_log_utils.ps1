# scripts/loop_log_utils.ps1
# Bounded, fail-closed continuous-loop log writes.
# Retries only proven transient sharing/access contention. Never silently drops lines.

function Test-TritonLoopLogRetryableError {
    param(
        [Parameter(Mandatory = $true)]
        $ErrorRecord
    )

    $ex = $ErrorRecord.Exception
    while ($null -ne $ex) {
        $msg = [string]$ex.Message
        if ($ex -is [System.IO.IOException]) {
            if ($msg -match 'being used by another process|sharing violation|cannot access the file') {
                return $true
            }
            # ERROR_SHARING_VIOLATION = 32 → HRESULT 0x80070020
            try {
                if ($ex.HResult -eq -2147024864) { return $true }
            } catch {
                # Ignore HResult access failures; message match above is authoritative.
            }
        }
        if ($ex -is [System.UnauthorizedAccessException] -and ($msg -match 'being used by another process')) {
            return $true
        }
        $ex = $ex.InnerException
    }

    $text = [string]$ErrorRecord
    if ($text -match 'being used by another process|sharing violation') {
        return $true
    }
    return $false
}

function Write-TritonLoopLogLine {
    <#
    .SYNOPSIS
      Append one log line with bounded retries for transient file sharing violations.
    .NOTES
      Does not Write-Host. Caller is responsible for console/audit host output.
      On persistent contention, throws after MaxAttempts (fail closed).
    #>
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Line,

        [int]$MaxAttempts = 5,

        [int]$InitialBackoffMs = 50
    )

    if ($MaxAttempts -lt 1) { $MaxAttempts = 1 }
    if ($InitialBackoffMs -lt 0) { $InitialBackoffMs = 0 }

    $attempt = 0
    while ($true) {
        $attempt++
        try {
            Add-Content -LiteralPath $Path -Value $Line -ErrorAction Stop
            return @{
                ok           = $true
                attempts     = $attempt
                path         = $Path
                fatal        = $false
            }
        }
        catch {
            $retryable = Test-TritonLoopLogRetryableError -ErrorRecord $_
            if (-not $retryable) {
                throw
            }
            if ($attempt -ge $MaxAttempts) {
                $detail = $_.Exception.Message
                $fatal = "[LOG_WRITE_FATAL] exhausted $MaxAttempts attempts writing loop log; refusing further cycles. last_error=$detail"
                Write-Host "$(Get-Date -Format s) $fatal"
                $io = New-Object System.IO.IOException $fatal, $_.Exception
                throw $io
            }
            $backoff = $InitialBackoffMs * $attempt
            if ($backoff -gt 0) {
                Start-Sleep -Milliseconds $backoff
            }
        }
    }
}
