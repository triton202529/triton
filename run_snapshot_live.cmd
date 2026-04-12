@echo off
setlocal
cd /d C:\Users\akimw\triton

REM Ensure results folder exists
if not exist data\results mkdir data\results

REM Always write something so we know the task ran
echo === TRITON_SnapshotLive %date% %time% ===>> data\results\scheduler_snapshot.log
echo PWD: %cd%>> data\results\scheduler_snapshot.log
where python>> data\results\scheduler_snapshot.log 2>>&1
echo Using venv python: %cd%\.venv\Scripts\python.exe>> data\results\scheduler_snapshot.log

REM Run snapshot
"%cd%\.venv\Scripts\python.exe" -m services.snapshot_live_orders --mode paper --verbose >> data\results\scheduler_snapshot.log 2>>&1

echo ExitCode: %ERRORLEVEL%>> data\results\scheduler_snapshot.log
echo.>> data\results\scheduler_snapshot.log
endlocal
