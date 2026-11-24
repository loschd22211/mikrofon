from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .config import DenoiserConfig
from .pipeline import VoiceEnhancerPipeline


logger = logging.getLogger("mikrofon")

VIRTUAL_CABLE_KEYWORDS = (
    "cable input",
    "cable output",
    "vb-audio",
    "voicemeeter",
    "virtual audio",
    "virtual cable",
    "blackhole",
    "soundflower",
    "mikrofon_clean",
    "virtmic",
)

VIRTUAL_INPUT_SKIP_KEYWORDS = VIRTUAL_CABLE_KEYWORDS + (
    "stereo mix",
    "стерео микшер",
    "what u hear",
    "loopback",
    ".monitor",
)


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: int
    is_default_input: bool
    is_default_output: bool
    is_virtual: bool


def _get_sd() -> Any:
    try:
        import sounddevice as sd

        return sd
    except OSError as exc:
        raise RuntimeError(
            "Не удалось загрузить аудио-библиотеку PortAudio.\n"
            "На Windows/macOS переустановите зависимости: pip install --force-reinstall sounddevice\n"
            "На Linux (Ubuntu/Debian) установите системный пакет: sudo apt install libportaudio2\n"
            f"Техническая деталь: {exc}"
        ) from exc


def refresh_audio_backend() -> None:
    """Re-initialize PortAudio so newly connected USB microphones are detected."""
    try:
        import sounddevice as sd

        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def is_virtual_device_name(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in VIRTUAL_CABLE_KEYWORDS)


def _should_skip_physical_input(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in VIRTUAL_INPUT_SKIP_KEYWORDS)


def list_audio_devices() -> list[AudioDeviceInfo]:
    sd = _get_sd()
    raw_devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_in, default_out = sd.default.device

    # Prefer primary OS host API when deduplicating / ranking
    result: list[AudioDeviceInfo] = []
    for idx, dev in enumerate(raw_devices):
        api_idx = int(dev.get("hostapi", 0))
        api_name = (
            hostapis[api_idx]["name"] if 0 <= api_idx < len(hostapis) else "Unknown"
        )
        name = str(dev.get("name", f"Device {idx}"))
        result.append(
            AudioDeviceInfo(
                index=idx,
                name=name,
                hostapi_name=str(api_name),
                max_input_channels=int(dev.get("max_input_channels", 0)),
                max_output_channels=int(dev.get("max_output_channels", 0)),
                default_samplerate=int(round(float(dev.get("default_samplerate", 48000)))),
                is_default_input=(idx == default_in),
                is_default_output=(idx == default_out),
                is_virtual=is_virtual_device_name(name),
            )
        )
    return result


def _hostapi_priority(api_name: str) -> int:
    """Rank low-latency / reliable host APIs higher when matching by name."""
    low = api_name.lower()
    if "wasapi" in low:
        return 0
    if "directsound" in low or "core audio" in low or "pipewire" in low or "pulse" in low:
        return 1
    if "mme" in low or "alsa" in low:
        return 2
    return 3


def resolve_input_device(
    requested: str | int | None,
    devices: list[AudioDeviceInfo] | None = None,
) -> AudioDeviceInfo:
    devs = devices if devices is not None else list_audio_devices()
    inputs = [d for d in devs if d.max_input_channels > 0]
    if not inputs:
        raise RuntimeError(
            "В системе не найдено ни одного микрофона (аудиовхода). "
            "Проверьте подключение микрофона к ПК."
        )

    # Explicit index
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        idx = int(requested)
        for d in inputs:
            if d.index == idx:
                return d
        raise RuntimeError(f"Микрофон с индексом {idx} не найден. Проверьте `python mikrofon.py devices`.")

    # Explicit name substring
    if isinstance(requested, str) and requested.strip():
        needle = requested.strip().lower()
        matches = [d for d in inputs if needle in d.name.lower()]
        if matches:
            matches.sort(key=lambda d: _hostapi_priority(d.hostapi_name))
            return matches[0]
        raise RuntimeError(
            f"Микрофон '{requested}' не найден. Запустите `python mikrofon.py devices` для просмотра списка."
        )

    # Auto-detect: prefer default input if it is NOT a virtual cable / stereo mix
    for d in inputs:
        if d.is_default_input and not _should_skip_physical_input(d.name):
            return d

    # Otherwise pick the best physical microphone
    physical = [d for d in inputs if not _should_skip_physical_input(d.name)]
    if physical:
        physical.sort(key=lambda d: _hostapi_priority(d.hostapi_name))
        return physical[0]

    return inputs[0]


def resolve_output_device(
    requested: str | int | None,
    allow_speakers_fallback: bool = False,
    devices: list[AudioDeviceInfo] | None = None,
) -> AudioDeviceInfo | None:
    devs = devices if devices is not None else list_audio_devices()
    outputs = [d for d in devs if d.max_output_channels > 0]
    if not outputs:
        return None

    # Explicit index
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        idx = int(requested)
        for d in outputs:
            if d.index == idx:
                return d
        raise RuntimeError(
            f"Устройство вывода с индексом {idx} не найдено. Проверьте `python mikrofon.py devices`."
        )

    # Explicit name substring
    if isinstance(requested, str) and requested.strip():
        needle = requested.strip().lower()
        matches = [d for d in outputs if needle in d.name.lower()]
        if matches:
            matches.sort(key=lambda d: _hostapi_priority(d.hostapi_name))
            return matches[0]
        raise RuntimeError(
            f"Устройство вывода '{requested}' не найдено. Запустите `python mikrofon.py devices`."
        )

    # Auto-detect virtual audio cable sink (VB-Cable, VoiceMeeter, BlackHole, Mikrofon_Clean)
    virtual_sinks = [d for d in outputs if d.is_virtual]
    if virtual_sinks:
        # Prefer "CABLE Input" or "mikrofon_clean" specifically
        def _sink_rank(d: AudioDeviceInfo) -> tuple[int, int]:
            low = d.name.lower()
            if "mikrofon_clean" in low or "cable input" in low or "blackhole" in low:
                pref = 0
            else:
                pref = 1
            return (pref, _hostapi_priority(d.hostapi_name))

        virtual_sinks.sort(key=_sink_rank)
        return virtual_sinks[0]

    if allow_speakers_fallback:
        for d in outputs:
            if d.is_default_output:
                return d
        return outputs[0]

    return None


def negotiate_sample_rate(
    in_dev: AudioDeviceInfo,
    out_dev: AudioDeviceInfo | None,
    preferred_sr: int = 48000,
) -> int:
    sd = _get_sd()
    candidates = []
    for sr in (
        preferred_sr,
        48000,
        44100,
        in_dev.default_samplerate,
        out_dev.default_samplerate if out_dev else 48000,
        32000,
        16000,
    ):
        if sr > 0 and sr not in candidates:
            candidates.append(sr)

    for sr in candidates:
        try:
            sd.check_input_settings(device=in_dev.index, channels=1, samplerate=sr, dtype="float32")
            if out_dev is not None:
                out_ch = min(2, max(1, out_dev.max_output_channels))
                sd.check_output_settings(
                    device=out_dev.index, channels=out_ch, samplerate=sr, dtype="float32"
                )
            return sr
        except Exception:
            continue
    return in_dev.default_samplerate or 48000


def record_snippet(
    duration_sec: float,
    in_dev: AudioDeviceInfo,
    sample_rate: int | None = None,
) -> tuple[int, np.ndarray]:
    """Record a mono float32 clip from `in_dev`."""
    sd = _get_sd()
    sr = sample_rate or negotiate_sample_rate(in_dev, None, 48000)
    frames = max(1, int(round(duration_sec * sr)))
    recording = sd.rec(
        frames,
        samplerate=sr,
        channels=1,
        dtype="float32",
        device=in_dev.index,
    )
    sd.wait()
    return sr, recording.reshape(-1)


def play_snippet(
    audio: np.ndarray,
    sample_rate: int,
    out_dev: AudioDeviceInfo | None = None,
) -> None:
    """Play a 1D float32 clip through default speakers/headphones."""
    sd = _get_sd()
    dev_idx = out_dev.index if out_dev is not None else None
    sd.play(np.asarray(audio, dtype=np.float32), samplerate=sample_rate, device=dev_idx)
    sd.wait()


def setup_linux_virtual_mic() -> tuple[bool, str]:
    """Create a PipeWire / PulseAudio virtual microphone `Mikrofon_Clean_Mic` on Linux."""
    if platform.system().lower() != "linux":
        return False, "Команда setup-linux-mic предназначена только для Linux (PipeWire/PulseAudio)."

    pactl = shutil.which("pactl")
    if not pactl:
        return False, "Утилита `pactl` не найдена. Установите пакет `pulseaudio-utils`."

    # Check if already created
    check = subprocess.run(
        [pactl, "list", "short", "sinks"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "mikrofon_clean" in check.stdout:
        return True, "Виртуальный микрофон `Mikrofon_Clean_Mic` уже создан и готов к работе!"

    r1 = subprocess.run(
        [
            pactl,
            "load-module",
            "module-null-sink",
            "sink_name=mikrofon_clean",
            "sink_properties=device.description=Mikrofon_Clean_Sink",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r1.returncode != 0:
        return False, f"Ошибка создания виртуального канала: {r1.stderr.strip()}"

    r2 = subprocess.run(
        [
            pactl,
            "load-module",
            "module-remap-source",
            "master=mikrofon_clean.monitor",
            "source_name=mikrofon_clean_mic",
            "source_properties=device.description=Mikrofon_Clean_Mic",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r2.returncode != 0:
        return (
            True,
            "Создан виртуальный приёмник `mikrofon_clean` (выбирайте Monitor of Mikrofon_Clean_Sink в приложениях).",
        )
    return (
        True,
        "Готово! Виртуальный микрофон `Mikrofon_Clean_Mic` создан. Выберите его как микрофон в Discord/играх.",
    )


def _format_meter_bar(db: float, width: int = 18) -> str:
    norm = float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))
    filled = int(round(norm * width))
    return "█" * filled + "░" * (width - filled)


def run_realtime_loop(
    config: DenoiserConfig,
    monitor_speakers: bool = False,
    quiet: bool = False,
) -> int:
    """
    Run the real-time noise suppression engine continuously with automatic reconnect.
    Designed to run 24/7 in the background without crashing if USB mic reconnects.
    """
    sd = _get_sd()

    # First check output routing before entering eternal loop so missing virtual cable gives clear guidance
    devices = list_audio_devices()
    in_dev = resolve_input_device(config.input_device, devices)
    out_dev = resolve_output_device(
        config.output_device,
        allow_speakers_fallback=monitor_speakers,
        devices=devices,
    )

    if out_dev is None:
        sys_name = platform.system().lower()
        print("\n[!] Не найден виртуальный аудиокабель для передачи очищенного звука в Discord/игры.")
        print(f"    Ваш физический микрофон успешно найден: [{in_dev.index}] {in_dev.name}\n")
        if sys_name == "windows":
            print("Чтобы очищенный микрофон работал во всех программах на Windows:")
            print("  1. Скачайте бесплатный драйвер VB-Audio Virtual Cable с официального сайта:")
            print("     https://vb-audio.com/Cable/  (нажмите Download -> запустите VBCABLE_Setup_x64.exe от имени администратора)")
            print("  2. Запустите этот скрипт снова — он сам найдёт 'CABLE Input'!")
            print("  3. В Discord / Telegram / играх выберите микрофон: 'CABLE Output (VB-Audio Virtual Cable)'.")
        elif sys_name == "linux":
            print("На Linux создайте виртуальный микрофон одной командой (без скачивания программ):")
            print("     python mikrofon.py setup-linux-mic")
            print("После этого снова запустите `python mikrofon.py`.")
        else:
            print("На macOS установите бесплатный драйвер BlackHole 2ch (brew install blackhole-2ch).")
        print("\nХотите прямо сейчас послушать, как звучит ваш очищенный голос в наушниках?")
        print("  -> Запустите тест До/После (5 секунд):  python mikrofon.py test")
        print("  -> Или включите живое прослушивание:    python mikrofon.py run --monitor\n")
        return 2

    while True:
        try:
            devices = list_audio_devices()
            in_dev = resolve_input_device(config.input_device, devices)
            out_dev = resolve_output_device(
                config.output_device,
                allow_speakers_fallback=monitor_speakers,
                devices=devices,
            )
            if out_dev is None:
                raise RuntimeError("Устройство вывода временно недоступно.")

            sr = negotiate_sample_rate(in_dev, out_dev, config.sample_rate)
            active_cfg = replace(config, sample_rate=sr)
            pipeline = VoiceEnhancerPipeline(active_cfg)

            in_ch = 1
            out_ch = min(2, max(1, out_dev.max_output_channels))
            blocksize = pipeline.hop_size

            logger.info(
                "Запущено шумоподавление | Вход: [%d] %s | Выход: [%d] %s | %d Гц | Пресет: %s",
                in_dev.index,
                in_dev.name,
                out_dev.index,
                out_dev.name,
                sr,
                active_cfg.preset,
            )
            if not quiet:
                print("=" * 72)
                print("  MIKROFON — Шумоподавление и улучшение голоса активно!")
                print(f"  Микрофон (вход) : [{in_dev.index}] {in_dev.name}")
                print(f"  Выходной канал  : [{out_dev.index}] {out_dev.name}")
                print(
                    f"  Частота / буфер : {sr} Гц / {active_cfg.frame_ms} мс "
                    f"| Режим: {active_cfg.preset}"
                )
                if out_dev.is_virtual:
                    print("  В Discord / играх выберите микрофон: 'CABLE Output' (или Mikrofon_Clean_Mic)")
                print("  Для остановки нажмите Ctrl+C")
                print("=" * 72)

            def _audio_callback(
                indata: np.ndarray,
                outdata: np.ndarray,
                frames: int,
                time_info: Any,
                status: Any,
            ) -> None:
                if status:
                    logger.debug("Audio stream status: %s", status)
                processed = pipeline.process_frame(indata, out_channels=out_ch)
                outdata[:] = processed

            with sd.Stream(
                device=(in_dev.index, out_dev.index),
                samplerate=sr,
                blocksize=blocksize,
                dtype="float32",
                channels=(in_ch, out_ch),
                latency="low",
                callback=_audio_callback,
            ):
                while True:
                    time.sleep(0.25)
                    if not quiet and sys.stdout.isatty():
                        in_bar = _format_meter_bar(pipeline.last_input_rms_db)
                        out_bar = _format_meter_bar(pipeline.last_output_rms_db)
                        red = pipeline.spectral.last_reduction_db
                        line = (
                            f"\r  Вход [{in_bar}] {pipeline.last_input_rms_db:5.1f} dB | "
                            f"Выход [{out_bar}] {pipeline.last_output_rms_db:5.1f} dB | "
                            f"Подавление шума: -{red:4.1f} dB   "
                        )
                        sys.stdout.write(line)
                        sys.stdout.flush()

        except KeyboardInterrupt:
            if not quiet:
                print("\nОстановлено пользователем.")
            return 0
        except Exception as exc:
            logger.warning("Поток аудио прерван (%s). Переподключение через 3 сек...", exc)
            if not quiet:
                print(f"\n[!] Аудиоустройство переподключается ({exc}). Ждём 3 сек...")
            time.sleep(3.0)
            refresh_audio_backend()
