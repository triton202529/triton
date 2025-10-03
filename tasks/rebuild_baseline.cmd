:: tasks\rebuild_baseline.cmd
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
set RC=%ERRORLEVEL%
if %RC% NEQ 0 (
  echo [err] Baseline rebuild failed rc=%RC%
  popd & exit /b %RC%
)

echo [ok] Baseline rebuilt successfully.
popd & exit /b 0
