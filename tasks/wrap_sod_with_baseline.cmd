@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.."

echo [info] Rebuilding baseline (cap, top 20, keep-missing-equal=0.25, max-weight=0.15)...
python scripts\build_baseline.py ^
  --method cap ^
  --top 20 ^
  --keep-missing-equal ^
  --missing-share 0.25 ^
  --max-weight 0.15 ^
  --fund-whitelist BITO GBTC GLD ARKK DIA DBA
if %ERRORLEVEL% NEQ 0 exit /b 1

echo [info] Validating baseline...
python scripts\validate_baseline.py --path data\results\baseline\weights.csv
if %ERRORLEVEL% NEQ 0 exit /b 1

echo [info] Diff vs prior snapshot:
python scripts\diff_weights.py
if %ERRORLEVEL% NEQ 0 echo [warn] Diff skipped (not enough snapshots).

echo [info] Audit summary:
python scripts\audit_baseline.py --path data\results\baseline\weights.csv --top 10

echo [ok] SOD wrapping complete.
popd & exit /b 0
