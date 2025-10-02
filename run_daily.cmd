@echo off
setlocal

REM ─────────────────────────────────────────────────────────────
REM Triton — run_daily.cmd
REM Runs the daily baseline build with optional market rebuild.
REM - Uses repo path relative to this script (portable).
REM - Prefers venv Python; falls back to py/python.
REM - Logs to logs\run_daily.log and propagates exit code.
REM To force market rebuild, set env var REBUILD=1 before calling.
REM ─────────────────────────────────────────────────────────────

REM Repo root (folder of this script)
set "ROOT=%~dp0"
pushd "%ROOT%" >nul

REM Logging
set "LOGDIR=%ROOT%logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\run_daily.log"
echo ==== %DATE% %TIME% ==== >> "%LOG%"
echo [cd] %CD% >> "%LOG%"

REM Choose Python (venv -> py.exe -> python on PATH)
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  for %%P in (py.exe python.exe python) do (
    where %%P >nul 2>&1 && (set "PY=%%P" & goto :gotpy)
  )
  echo [error] No Python found. >> "%LOG%"
  set "EC=9009"
  goto :done
)
:gotpy
echo [python] Using: %PY% >> "%LOG%"

REM Config path
set "CFG=%ROOT%config\baseline.smart_weight.json"
if not exist "%CFG%" (
  echo [error] Missing config: "%CFG%" >> "%LOG%"
  set "EC=2"
  goto :done
)

REM Optional force-rebuild flag via REBUILD env var (1/true/on)
set "REBUILD_FLAG="
if /I "%REBUILD%"=="1"     set "REBUILD_FLAG=--rebuild-market"
if /I "%REBUILD%"=="true"  set "REBUILD_FLAG=--rebuild-market"
if /I "%REBUILD%"=="on"    set "REBUILD_FLAG=--rebuild-market"

REM Run (echo the command to the log)
echo [run] %PY% "%ROOT%run_daily.py" --config "%CFG%" %REBUILD_FLAG% >> "%LOG%"
%PY% "%ROOT%run_daily.py" --config "%CFG%" %REBUILD_FLAG% >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"

:done
echo [exit] %EC% at %DATE% %TIME% >> "%LOG%"
popd >nul
endlocal & exit /b %EC%
