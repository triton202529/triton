@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ====== CONFIG ======
set "VENV_PY=venv\Scripts\python.exe"
set "PY=%VENV_PY%"
set "LOG_DIR=logs"
set "RESULTS_DIR=data\results"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

REM Portable timestamp (locale-safe via PowerShell)
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd_HH-mm-ss\")"') do set "STAMP=%%i"
set "LOG_FILE=%LOG_DIR%\daily_%STAMP%.log"

if not exist "%PY%" (
  echo [WARN] venv python not found, falling back to system python >> "%LOG_FILE%"
  set "PY=python"
)

echo [START] Triton Daily Refresh @ %date% %time% > "%LOG_FILE%"

goto :pipeline


REM ================== FUNCTIONS ==================
:step
REM usage: call :step "Title" services\script.py [args...]
set "TITLE=%~1"
set "SCRIPT=%~2"
shift
shift
set "ARGS=%*"

echo.>> "%LOG_FILE%"
echo --- !TITLE! --- >> "%LOG_FILE%"

if not defined SCRIPT (
  echo [SKIP] No script provided >> "%LOG_FILE%"
  echo SKIP: !TITLE!  (no script)
  goto :eof
)

if not exist "%SCRIPT%" (
  echo [SKIP] Missing script: %SCRIPT% >> "%LOG_FILE%"
  echo SKIP: !TITLE!  (missing %SCRIPT%)
  goto :eof
)

echo [RUN ] %PY% "%SCRIPT%" %ARGS% >> "%LOG_FILE%"
"%PY%" "%SCRIPT%" %ARGS% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [FAIL] !TITLE! >> "%LOG_FILE%"
  echo FAILED: !TITLE! — check "%LOG_FILE%"
  exit /b 1
)
echo [OK  ] !TITLE! >> "%LOG_FILE%"
goto :eof


REM ================== PIPELINE ==================
:pipeline
call :step "Fetch OHLC data"                  services\fetch_data.py
call :step "Fetch fundamentals"               services\fetch_fundamentals.py
call :step "Score stocks"                     services\score_stocks.py
call :step "Train models + predictions"       services\train_models.py
call :step "Combine predictions"              services\combine_predictions.py
call :step "Generate signals + rationale"     services\generate_signals.py
call :step "News sentiment (RSS, 14d)"        services\fetch_news_sentiment_rss.py --tickers all --window 14
call :step "Economic calendar (RSS)"          services\generate_economic_calendar_rss.py
call :step "Smart alerts"                     services\generate_smart_alerts.py
call :step "Run paper trading"                services\run_paper_trading.py --out-prefix ""
call :step "Backtest from signals"            python services\backtest_signals.py
REM call :step "Portfolio summary"            services\portfolio_summary.py

echo.>> "%LOG_FILE%"
echo [DONE] %date% %time% >> "%LOG_FILE%"
echo ✅ Done. Log: %LOG_FILE%
exit /b 0
