@echo off
setlocal
REM repo root:
set REPO=C:\Users\akimw\triton
cd /d "%REPO%"
call "%REPO%\.venv\Scripts\activate.bat"
python run_daily.py --config "%REPO%\config\baseline.smart_weight.json"
endlocal
