# scripts/run_operator_console.ps1
# Phase DO-002 — Launch TRITON Operator Console bound to localhost only.
# Does not start/stop trading loops. Does not arm paper or enable live execution.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
Set-Location $RepoRoot

$entry = Join-Path $RepoRoot "view_results.py"
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "Dashboard entry not found: $entry"
}

Write-Host "Starting TRITON Operator Console on http://127.0.0.1:$Port"
Write-Host "Entry: streamlit run view_results.py --server.address 127.0.0.1 --server.port $Port"
Write-Host "Advanced/Legacy dashboard remains available from the sidebar mode switch."

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

& $python -m streamlit run $entry `
    --server.address 127.0.0.1 `
    --server.port $Port `
    --browser.gatherUsageStats false
