@echo off
setlocal
pushd "%~dp0"
call ".venv\Scripts\activate.bat"
python run_daily.py --config "config\baseline.smart_weight.json"
popd
