$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps "pyrnnoise>=0.4.6"

.\.venv\Scripts\python.exe mikrofon.py install-cable
.\.venv\Scripts\python.exe mikrofon.py calibrate --preset anti_barrel
.\.venv\Scripts\python.exe mikrofon.py test
.\.venv\Scripts\python.exe mikrofon.py autostart install
.\.venv\Scripts\python.exe mikrofon.py run --preset anti_barrel
