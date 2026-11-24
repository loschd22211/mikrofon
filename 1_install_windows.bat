@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ==============================================================
echo   MIKROFON 2.0 - Установка ИИ-шумодава и виртуального кабеля
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

echo Устанавливаю аудио-библиотеки и нейросеть RNNoise v0.2...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install --no-deps "pyrnnoise>=0.4.6"

echo.
echo Проверяю виртуальный микрофон VB-Cable для Telegram / ВК / Discord...
".venv\Scripts\python.exe" mikrofon.py install-cable

echo.
echo [ГОТОВО] Установка завершена!
echo Теперь запустите 2_calibrate_and_test.bat для настройки и проверки звука.
pause
