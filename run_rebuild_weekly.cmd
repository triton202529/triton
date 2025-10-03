@echo off
setlocal
set REPO=%USERPROFILE%\triton
cd /d "%REPO%"

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd_HH-mm-ss\")"') do set TS=%%i
set LOGDIR=%REPO%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOG=%LOGDIR%\weekly_rebuild_%TS%.log

call "%REPO%\.venv\Scripts\activate.bat"

>> "%LOG%" 2>&1 echo [%DATE% %TIME%] Weekly rebuild starting...
>> "%LOG%" 2>&1 echo [%DATE% %TIME%] Step 1: Rebuild market_by_ticker.csv
python run_daily.py --rebuild-market >> "%LOG%" 2>&1

>> "%LOG%" 2>&1 echo [%DATE% %TIME%] Step 2: Rebuild smart-weight baseline
python run_daily.py --config "%REPO%\config\baseline.smart_weight.json" >> "%LOG%" 2>&1

>> "%LOG%" 2>&1 echo [%DATE% %TIME%] Weekly rebuild finished. ExitCode=%ERRORLEVEL%
endlocal
