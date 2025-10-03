# tasks\wrap_sod_with_baseline.ps1
# PowerShell wrapper for SOD baseline build/validate/audit

$ErrorActionPreference = 'Stop'

$buildArgs = @(
  '--method','cap',
  '--top','20',
  '--max-weight','0.125',
  '--min-weight','0.0005',
  '--fund-whitelist','BITO','GBTC','GLD','ARKK','DIA','DBA',
  '--turnover-cap','0.05'
)

Write-Host "[info] Rebuilding baseline (cap, top 20, max-weight=0.125, min-weight=0.0005, turnover-cap=0.05)..."
python scripts\build_baseline.py @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build_baseline failed (exit $LASTEXITCODE)" }

Write-Host "[info] Validating baseline..."
python scripts\validate_baseline.py --path data\results\baseline\weights.csv
if ($LASTEXITCODE -ne 0) { throw "validate_baseline failed (exit $LASTEXITCODE)" }

Write-Host "[info] Audit summary:"
python scripts\audit_baseline.py --path data\results\baseline\weights.csv --top 10
if ($LASTEXITCODE -ne 0) { throw "audit_baseline failed (exit $LASTEXITCODE)" }

Write-Host "[ok] SOD wrapping complete."
