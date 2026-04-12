@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  TRITON - Baseline Analyzer / Stress Diagnostics (STRICT)
REM ============================================================
set "ROOT=%~dp0.."
pushd "%ROOT%" >NUL
set PYTHONUTF8=1
set PYTHONIOENCODING=UTF-8
chcp 65001 >NUL

echo.
echo [TRITON] ============================================
echo [TRITON] Running Baseline Analyzer / STRICT Mode
echo [TRITON] ============================================
echo Time: %DATE% %TIME%
echo Root: %ROOT%
echo.

set "LOGDIR=%ROOT%\data\results\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOGFILE=%LOGDIR%\baseline_analyzer_strict.log"
> "%LOGFILE%" echo [INFO] Launching analyzer (STRICT)...

if not exist "%ROOT%\.venv\Scripts\activate.bat" (
    echo [ERR] Missing virtual environment: .venv\Scripts\activate.bat
    popd >NUL
    exit /b 1
)
call "%ROOT%\.venv\Scripts\activate.bat"

python -X utf8 services\baseline_analyzer.py ^
  --days 365 ^
  --shocks -0.05 -0.10 -0.20 ^
  --min-days-stats 30 ^
  >> "%LOGFILE%" 2>&1

set "RC=%ERRORLEVEL%"

echo.
echo [TRITON] Analyzer exit code: %RC%
echo [TRITON] Log written to: %LOGFILE%
echo.

if "%RC%"=="2" (
    echo [ALERT] One or more FAIL diagnostics detected. Skipping downstream tasks.
    popd >NUL
    exit /b 2
)

if not "%RC%"=="0" (
    echo [WARN] Non-zero exit code. Check logs for issues.
    popd >NUL
    exit /b %RC%
)

echo [OK] Baseline Analyzer (STRICT) completed successfully.
popd >NUL
exit /b 0
