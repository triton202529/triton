# scripts/triton_env_bootstrap.ps1
# Load git-ignored local env files before Python runtime steps (Phase 197F).

param(
    [string]$RepoRoot = ""
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$RepoRoot = $RepoRoot.TrimEnd('\', '/')

$envFiles = @(
    (Join-Path $RepoRoot ".env.local"),
    (Join-Path $RepoRoot ".env.paper.local"),
    (Join-Path $RepoRoot "config\local\runtime_env.local")
)

foreach ($path in $envFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        continue
    }
    Get-Content -LiteralPath $path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { return }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        if ($name -ne "") {
            Set-Item -Path "env:$name" -Value $value
        }
    }
}
