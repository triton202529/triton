# tasks\run_fetch_news.ps1
# Usage: powershell -ExecutionPolicy Bypass -File .\tasks\run_fetch_news.ps1 --strategy aggregate --trusted-only ...

$proj = Split-Path -Parent $PSScriptRoot       # repo root (one level up from /tasks)
$script = Join-Path $proj 'services\fetch_news_sentiment.py'

if (-not (Test-Path $script)) {
  Write-Error "Can't find: $script"
  exit 1
}

# ensure API key present (optional warning)
if (-not $env:NEWSAPI_KEY -or [string]::IsNullOrWhiteSpace($env:NEWSAPI_KEY)) {
  Write-Warning "NEWSAPI_KEY environment variable is not set."
}

# Run and forward any arguments you pass to this script
& python $script @args
exit $LASTEXITCODE
