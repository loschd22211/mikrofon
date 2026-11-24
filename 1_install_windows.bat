@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ==============================================================
echo   MIKROFON - Установка зависимостей (один раз)
echo ==============================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден! Установите Python 3.10+ с https://www.python.org/downloads/
    echo При установке обязательно поставьте галочку "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Создаю виртуальное окружение .venv...
    python -m venv .venv
)

echo Устанавливаю аудио-библиотеки (numpy, scipy, sounddevice)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo [ГОТОВО] Установка успешно завершена!
echo Теперь запустите 2_calibrate_and_test.bat для автонастройки под ваш микрофон.
pause
