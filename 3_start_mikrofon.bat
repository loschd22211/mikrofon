@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

set PY_EXE=python
if exist ".venv\Scripts\python.exe" set PY_EXE=".venv\Scripts\python.exe"

%PY_EXE% mikrofon.py run
if %errorlevel% neq 0 pause
