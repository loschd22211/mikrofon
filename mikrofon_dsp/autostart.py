from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def get_autostart_path(system_name: str | None = None, home_dir: Path | None = None) -> Path:
    """Return the OS-specific user autostart file path (no admin privileges required)."""
    sys_name = (system_name or platform.system()).lower()
    home = home_dir or Path.home()

    if sys_name == "windows":
        appdata_env = os.environ.get("APPDATA")
        appdata = Path(appdata_env) if appdata_env and home_dir is None else (home / "AppData" / "Roaming")
        return (
            appdata
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
            / "MikrofonDenoiser.vbs"
        )
    if sys_name == "darwin":
        return home / "Library" / "LaunchAgents" / "com.mikrofon.denoiser.plist"
    return home / ".config" / "systemd" / "user" / "mikrofon.service"


def build_autostart_content(
    script_path: Path,
    python_executable: Path | None = None,
    system_name: str | None = None,
) -> str:
    """Generate the silent background autostart launcher content for the target OS."""
    sys_name = (system_name or platform.system()).lower()
    script_abs = script_path.resolve()
    workdir = script_abs.parent
    log_path = workdir / "mikrofon.log"
    py_exe = (python_executable or Path(sys.executable)).resolve()

    if sys_name == "windows":
        pythonw = py_exe.with_name("pythonw.exe")
        runner = pythonw if pythonw.exists() else py_exe
        # VBScript escape: double any internal quotes
        wd_vbs = str(workdir).replace('"', '""')
        cmd_vbs = (
            f'""{runner}"" ""{script_abs}"" run --quiet --log-file ""{log_path}""'
        )
        return (
            "' Auto-generated silent launcher for Mikrofon Noise Suppression\r\n"
            'Set WshShell = CreateObject("WScript.Shell")\r\n'
            f'WshShell.CurrentDirectory = "{wd_vbs}"\r\n'
            f'WshShell.Run "{cmd_vbs}", 0, False\r\n'
        )

    if sys_name == "darwin":
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mikrofon.denoiser</string>
    <key>WorkingDirectory</key>
    <string>{workdir}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{py_exe}</string>
        <string>{script_abs}</string>
        <string>run</string>
        <string>--quiet</string>
        <string>--log-file</string>
        <string>{log_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"""

    return f"""[Unit]
Description=Mikrofon Real-Time Microphone Noise Suppression
After=pipewire.service pulseaudio.service sound.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStartPre=-{py_exe} {script_abs} setup-linux-mic
ExecStart={py_exe} {script_abs} run --quiet --log-file {log_path}
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
"""


def install_autostart(
    script_path: Path,
    system_name: str | None = None,
    home_dir: Path | None = None,
) -> tuple[Path, str]:
    sys_name = (system_name or platform.system()).lower()
    target = get_autostart_path(sys_name, home_dir=home_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = build_autostart_content(script_path=script_path, system_name=sys_name)
    target.write_text(content, encoding="utf-8")

    note = f"Автозапуск установлен: {target}"
    if sys_name == "linux" and home_dir is None and shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, capture_output=True)
        res = subprocess.run(
            ["systemctl", "--user", "enable", "--now", "mikrofon.service"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            note += "\nФоновая служба systemd (mikrofon.service) активирована и запущена."
    elif sys_name == "windows":
        note += (
            "\nСкрипт будет автоматически запускаться при входе в Windows в тихом фоновом режиме (без чёрного окна консоли)."
        )
    return target, note


def remove_autostart(
    system_name: str | None = None,
    home_dir: Path | None = None,
) -> tuple[bool, str]:
    sys_name = (system_name or platform.system()).lower()
    target = get_autostart_path(sys_name, home_dir=home_dir)

    if sys_name == "linux" and home_dir is None and shutil.which("systemctl"):
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", "mikrofon.service"],
            check=False,
            capture_output=True,
        )

    if target.exists():
        target.unlink()
        return True, f"Автозапуск удалён ({target})."
    return False, "Автозапуск не был установлен."


def autostart_status(
    system_name: str | None = None,
    home_dir: Path | None = None,
) -> tuple[bool, Path]:
    target = get_autostart_path(system_name, home_dir=home_dir)
    return target.exists(), target
