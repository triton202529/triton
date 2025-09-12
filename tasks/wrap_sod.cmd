@echo off
setlocal EnableExtensions

rem ===== anchor to repo root (parent of \tasks; works under Scheduler SYSTEM) =====
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul
set "ROOT=%CD%\"

rem ===== paths & logs =====
set "LOGDIR=%ROOT%data\results\reports\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1
set "LOG=%LOGDIR%\taskwrap_sod.log"

rem ===== single-instance lock =====
set "RUNDIR=%ROOT%data\run"
if not exist "%RUNDIR%" mkdir "%RUNDIR%" >nul 2>&1
set "LOCK=%RUNDIR%\wrap_sod.lock"

rem ----- early breadcrumb -----
echo [%DATE% %TIME%] BOOT: wrapper start CD=%CD% ROOT=%ROOT%>>"%LOG%"
echo [%DATE% %TIME%] PIN 01: after path setup>>"%LOG%"

rem ----- stale lock auto-clear (older than 30 min) -----
if exist "%LOCK%" powershell -NoProfile -ExecutionPolicy Bypass -Command "try{$f=Get-Item '%LOCK%'; if($f -and $f.LastWriteTime -lt (Get-Date).AddMinutes(-30)){ Remove-Item -Force '%LOCK%' }}catch{}" 1>nul 2>nul

rem ----- single-instance (no multi-line IF blocks) -----
if exist "%LOCK%" goto _already_running
>"%LOCK%" echo %DATE% %TIME%
echo [%DATE% %TIME%] PIN 02: lock created at "%LOCK%">>"%LOG%"

rem ----- env snapshot -----
set > "%RUNDIR%\wrap_sod.env" 2>nul
echo [%DATE% %TIME%] PIN 03: env snapshot written>>"%LOG%"

rem ===== discover Slack webhook (no paren grouping) =====
if not defined SLACK_WEBHOOK_URL if exist "%ROOT%data\secrets\slack_webhook.txt" set /p SLACK_WEBHOOK_URL=<"%ROOT%data\secrets\slack_webhook.txt"
if not defined SLACK_WEBHOOK_URL if exist "%ROOT%secrets\slack_webhook.txt" set /p SLACK_WEBHOOK_URL=<"%ROOT%secrets\slack_webhook.txt"
echo [%DATE% %TIME%] PIN 04: slack probe (%SLACK_WEBHOOK_URL:~0,4%...)>>"%LOG%"

rem ===== guardrails: weights must exist & be >= 10 bytes =====
set "WCSV=%ROOT%data\results\weights.csv"
echo [%DATE% %TIME%] ENTER wrap_sod CD=%CD% ROOT=%ROOT%>>"%LOG%"
echo === SOD start %DATE% %TIME% ===>>"%LOG%"
echo [%DATE% %TIME%] DEBUG: WCSV="%WCSV%">>"%LOG%"
echo [%DATE% %TIME%] PIN 05: pre-check weights existence>>"%LOG%"

if not exist "%WCSV%" goto _err_missing_weights

for %%F in ("%WCSV%") do set "WSZ=%%~zF"
echo [%DATE% %TIME%] DEBUG: weights size=%WSZ% bytes.>>"%LOG%"
echo [%DATE% %TIME%] PIN 06: size read>>"%LOG%"

rem ---- robust, paren-free numeric/size gate ----
if "%WSZ%"=="" goto _err_size_nonnum
set "WSZNUM=%WSZ%"
echo [%DATE% %TIME%] PIN 06a: numeric=%WSZNUM%>>"%LOG%"
if %WSZNUM% LSS 10 goto _err_too_small

echo [%DATE% %TIME%] DEBUG: guardrails passed.>>"%LOG%"
echo [%DATE% %TIME%] PIN 07: guard OK>>"%LOG%"

rem verify the job file exists (no grouped echo)
if not exist "%ROOT%sod_rebalance.bat" goto _err_missing_job
echo [%DATE% %TIME%] DEBUG: found "%ROOT%sod_rebalance.bat".>>"%LOG%"

rem ===== run job (direct CALL) =====
echo [%DATE% %TIME%] RUN sod_rebalance.bat (direct CALL)>>"%LOG%"
call "%ROOT%sod_rebalance.bat" >>"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%DATE% %TIME%] RETURN from sod_rebalance rc=%RC%>>"%LOG%"
goto _finish

:_err_missing_weights
echo [%DATE% %TIME%] ERROR: weights.csv missing at "%WCSV%".>>"%LOG%"
set "RC=2"
goto _finish

:_err_size_nonnum
echo [%DATE% %TIME%] ERROR: empty/non-numeric weight size "%WSZ%".>>"%LOG%"
set "RC=3"
goto _finish

:_err_too_small
echo [%DATE% %TIME%] ERROR: weights.csv too small (%WSZNUM% bytes).>>"%LOG%"
set "RC=3"
goto _finish

:_err_missing_job
echo [%DATE% %TIME%] ERROR: missing "%ROOT%sod_rebalance.bat".>>"%LOG%"
set "RC=6"
goto _finish

:_finish
if not defined RC set "RC=1"
echo === SOD end   %DATE% %TIME% rc=%RC% ===>>"%LOG%"

rem ===== Slack notify (single line; no paren blocks) =====
if defined SLACK_WEBHOOK_URL powershell -NoProfile -ExecutionPolicy Bypass -Command "$u=$env:SLACK_WEBHOOK_URL;$log=$env:LOG;$rc=$env:RC;$msg=($rc -eq 0)?('? SOD OK on '+$env:COMPUTERNAME):('? SOD FAILED rc='+$rc+' on '+$env:COMPUTERNAME);$tail=(Test-Path $log)?(Get-Content -Raw $log):'';if($tail.Length -gt 3000){$tail=$tail.Substring($tail.Length-3000)};$body=@{text=$msg+([string]::IsNullOrEmpty($tail)?'':(\"`n```\"+$tail+\"```\"))}|ConvertTo-Json -Compress;Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $body" 1>nul 2>nul

del "%LOCK%" >nul 2>&1
popd >nul
exit /b %RC%

:_already_running
echo [%DATE% %TIME%] wrap_sod: already running, skipping.>>"%LOG%"
popd >nul
exit /b 0
