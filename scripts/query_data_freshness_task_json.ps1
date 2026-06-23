# scripts/query_data_freshness_task_json.ps1
param(
    [string]$TaskName = "TRITON_Data_Freshness_35Min"
)

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    @{ exists = $false; task_name = $TaskName } | ConvertTo-Json -Compress
    exit 0
}

$info = Get-ScheduledTaskInfo -TaskName $task.TaskName
@{
    exists      = $true
    task_name   = $task.TaskName
    state       = [string]$task.State
    last_result = $info.LastTaskResult
    last_run    = if ($info.LastRunTime) { $info.LastRunTime.ToString("o") } else { "" }
    next_run    = if ($info.NextRunTime) { $info.NextRunTime.ToString("o") } else { "" }
    arguments   = [string]$task.Actions[0].Arguments
    working_dir = [string]$task.Actions[0].WorkingDirectory
} | ConvertTo-Json -Compress
