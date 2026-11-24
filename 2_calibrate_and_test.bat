@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

set PY_EXE=python
if exist ".venv\Scripts\python.exe" set PY_EXE=".venv\Scripts\python.exe"

%PY_EXE% mikrofon.py calibrate
if %errorlevel% neq 0 (
    pause
    exit /b %errorlevel%
)

echo.
echo Сейчас запишем 5 секунд голоса и прослушаем ДО и ПОСЛЕ обработки!
pause
%PY_EXE% mikrofon.py test
pause
