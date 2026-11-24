#!/usr/bin/env python3
"""
Mikrofon 2.0 — ИИ-шумоподавление (RNNoise v0.2) + Анти-Бочка (Dereverb + Studio EQ)
для дешёвых микрофонов. Убирает скрип стула, стук клавиш, шипение и эхо комнаты.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import platform
import sys
import time

from mikrofon_dsp.audio_io import (
    install_windows_vbcable,
    list_audio_devices,
    play_snippet,
    record_snippet,
    resolve_input_device,
    resolve_output_device,
    run_realtime_loop,
    setup_linux_virtual_mic,
)
from mikrofon_dsp.autostart import (
    autostart_status,
    install_autostart,
    remove_autostart,
)
from mikrofon_dsp.config import (
    DEFAULT_CONFIG_PATH,
    PRESETS,
    PRESET_DESCRIPTIONS_RU,
    DenoiserConfig,
    preset_config,
)
from mikrofon_dsp.neural import is_neural_rnnoise_available
from mikrofon_dsp.pipeline import (
    build_calibrated_config,
    process_wav_file,
    read_wav_float32,
    write_wav_int16,
)
from mikrofon_dsp.spectral import analyze_microphone_noise


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _configure_logging(log_file: str | None, quiet: bool) -> None:
    handlers: list[logging.Handler] = []
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    elif not quiet:
        handlers.append(logging.StreamHandler(sys.stderr))
    else:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def _parse_device_arg(val: str | None) -> str | int | None:
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    if val.isdigit():
        return int(val)
    return val


def cmd_devices(_args: argparse.Namespace) -> int:
    devices = list_audio_devices()
    print("\n=== Аудиоустройства в системе ===")
    print(
        f"Движок ИИ RNNoise v0.2 : {'УСТАНОВЛЕН (Активен)' if is_neural_rnnoise_available() else 'Не установлен (используется Spectral DSP)'}"
    )
    print("\n[ВХОДЫ / МИКРОФОНЫ]:")
    inputs = [d for d in devices if d.max_input_channels > 0]
    if not inputs:
        print("  (не найдено ни одного микрофона)")
    else:
        try:
            auto_in = resolve_input_device(None, devices)
            auto_in_idx = auto_in.index
        except Exception:
            auto_in_idx = -1

        for d in inputs:
            tags = []
            if d.index == auto_in_idx:
                tags.append("АВТОВЫБОР ФИЗИЧЕСКОГО МИКРОФОНА")
            if d.is_default_input:
                tags.append("выбран по умолчанию в ОС")
            if d.is_virtual:
                tags.append("виртуальный (выбирать в Telegram/Discord!)")
            tag_str = f"  <-- {', '.join(tags)}" if tags else ""
            print(
                f"  [{d.index:2d}] {d.name} ({d.hostapi_name}, {d.default_samplerate} Гц){tag_str}"
            )

    print("\n[ВЫХОДЫ / ВИРТУАЛЬНЫЕ КАБЕЛИ]:")
    outputs = [d for d in devices if d.max_output_channels > 0]
    auto_out = resolve_output_device(None, allow_speakers_fallback=False, devices=devices)
    auto_out_idx = auto_out.index if auto_out is not None else -1

    for d in outputs:
        tags = []
        if d.index == auto_out_idx:
            tags.append("АВТОВЫБОР ВИРТУАЛЬНОГО КАБЕЛЯ")
        if d.is_default_output:
            tags.append("динамики/наушники по умолчанию")
        if d.is_virtual:
            tags.append("виртуальный кабель")
        tag_str = f"  <-- {', '.join(tags)}" if tags else ""
        print(
            f"  [{d.index:2d}] {d.name} ({d.hostapi_name}, {d.default_samplerate} Гц){tag_str}"
        )

    if auto_out is None:
        print(
            "\n[!] Виртуальный кабель НЕ установлен! "
            "На Windows запустите: `python mikrofon.py install-cable`"
        )
    print()
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    devices = list_audio_devices()
    in_dev = resolve_input_device(_parse_device_arg(args.input), devices)
    out_dev = resolve_output_device(None, allow_speakers_fallback=False, devices=devices)

    print("\n=== Диагностика вашего микрофона ===")
    print(f"ОС               : {platform.system()} {platform.release()}")
    print(f"Нейросеть ИИ     : {'RNNoise v0.2 АКТИВНА' if is_neural_rnnoise_available() else 'Выкл (установите pyrnnoise)'}")
    print(f"Выбран микрофон  : [{in_dev.index}] {in_dev.name} ({in_dev.hostapi_name})")
    print(f"Родная частота   : {in_dev.default_samplerate} Гц | Каналов: {in_dev.max_input_channels}")
    print(
        f"Виртуальный кабель: {'УСТАНОВЛЕН — ' + out_dev.name if out_dev else 'НЕ УСТАНОВЛЕН (запустите install-cable!)'}"
    )
    print("\nПожалуйста, помолчите 3 секунды — измеряю уровень фонового шума и наводок...")
    time.sleep(0.4)
    sr, silence = record_snippet(3.0, in_dev)
    diag = analyze_microphone_noise(silence, sr)

    print("\n--- Результаты анализа ---")
    print(f"Уровень фонового шума (RMS) : {diag.rms_db:.1f} dBFS (пики: {diag.peak_db:.1f} dBFS)")
    if diag.detected_hum_hz:
        print(
            f"Электрический гул сети      : Обнаружен фон {int(diag.detected_hum_hz)} Гц "
            f"(+{diag.hum_prominence_db:.1f} дБ)"
        )
    else:
        print("Электрический гул 50/60 Гц  : Явных наводок не обнаружено")
    print(f"Высокочастотное шипение     : {diag.hf_hiss_ratio_db:+.1f} дБ относительно средних частот")
    print(f"Рекомендуемый пресет        : {diag.recommended_preset} ({PRESET_DESCRIPTIONS_RU[diag.recommended_preset]})")
    print(f"Рекомендуемый порог гейта   : {diag.recommended_gate_threshold_db:.1f} dBFS\n")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    existing = DenoiserConfig.load(cfg_path) if cfg_path.exists() else DenoiserConfig()
    devices = list_audio_devices()
    req_in = _parse_device_arg(args.input) if args.input else existing.input_device
    in_dev = resolve_input_device(req_in, devices)

    print("\n=== Автонастройка под ваш микрофон (ИИ + Анти-Бочка) ===")
    print(f"Микрофон: [{in_dev.index}] {in_dev.name}\n")

    print("ШАГ 1 из 2: Помолчите 3 секунды (не трогайте клавиатуру и мышь)...")
    time.sleep(0.6)
    sr, silence = record_snippet(3.0, in_dev)
    print("  [OK] Фоновый шум записан.\n")

    print("ШАГ 2 из 2: Произнесите любую фразу обычным голосом в течение 4 секунд")
    print("            (например: «Раз, два, три, проверка микрофона, как меня слышно»)...")
    time.sleep(0.5)
    _, speech = record_snippet(4.0, in_dev, sample_rate=sr)
    print("  [OK] Образец голоса записан.\n")

    new_cfg, diag = build_calibrated_config(
        silence_audio=silence,
        sample_rate=sr,
        speech_audio=speech,
        mic_name=in_dev.name,
        base_config=existing,
    )
    if args.preset:
        preset_base = preset_config(args.preset)
        new_cfg = replace(
            preset_base,
            sample_rate=sr,
            input_device=new_cfg.input_device,
            output_device=new_cfg.output_device,
            input_gain_db=new_cfg.input_gain_db,
            hum_filter_hz=new_cfg.hum_filter_hz,
            gate_threshold_db=new_cfg.gate_threshold_db,
            calibrated_mic_name=in_dev.name,
            calibrated_noise_rms_db=diag.rms_db,
        )
    if args.input:
        new_cfg = replace(new_cfg, input_device=_parse_device_arg(args.input))
    saved = new_cfg.save(cfg_path)

    print("=== Настройка завершена! ===")
    print(f"  Движок ИИ              : {'RNNoise v0.2 Neural AI' if is_neural_rnnoise_available() else 'Spectral DSP'}")
    print(f"  Фоновый шум комнаты/ПК : {diag.rms_db:.1f} dBFS")
    print(f"  Выбранный режим        : {new_cfg.preset} (Анти-бочка включена)")
    print(f"  Предусиление микрофона : {new_cfg.input_gain_db:+.1f} дБ")
    print(f"  Порог шумового гейта   : {new_cfg.gate_threshold_db:.1f} dBFS")
    print(f"  Профиль сохранён в     : {saved.resolve()}\n")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    cfg = DenoiserConfig.load(args.config)
    if args.preset:
        cfg = replace(
            preset_config(args.preset),
            input_device=cfg.input_device,
            input_gain_db=cfg.input_gain_db,
        )
    if args.input:
        cfg = replace(cfg, input_device=_parse_device_arg(args.input))

    devices = list_audio_devices()
    in_dev = resolve_input_device(cfg.input_device, devices)
    duration = float(args.seconds)

    print("\n=== Тест микрофона «ДО и ПОСЛЕ» (ИИ RNNoise + Анти-Бочка) ===")
    print(f"Микрофон: [{in_dev.index}] {in_dev.name} | Режим: {cfg.preset}")
    print(
        f"\nГоворите в микрофон (можете поскрипеть стулом или пощёлкать мышкой!) в течение {duration:.0f} сек..."
    )
    time.sleep(0.4)
    sr, raw_audio = record_snippet(duration, in_dev)
    print("  [OK] Запись завершена! Очищаю нейросетью и убираю эффект бочки...")

    before_path = Path("test_before.wav")
    after_path = Path("test_after.wav")
    write_wav_int16(before_path, sr, raw_audio)
    process_wav_file(before_path, after_path, config=cfg)
    _, clean_audio = read_wav_float32(after_path)

    print(f"  Файлы сохранены: {before_path} (сырой) и {after_path} (очищенный)")
    if not args.no_playback:
        print("\n▶ 1. Слушаем СЫРОЙ ОРИГИНАЛ (как ваш микрофон звучит сам по себе)...")
        play_snippet(raw_audio, sr)
        time.sleep(0.5)
        print("▶ 2. Слушаем ПОСЛЕ ИИ-ОЧИСТКИ И АНТИ-БОЧКИ (то, что пойдёт в ГС/Discord)...")
        play_snippet(clean_audio, sr)
        print("\nГотово!")
    return 0


def cmd_install_cable(_args: argparse.Namespace) -> int:
    ok, msg = install_windows_vbcable()
    print(msg)
    return 0 if ok else 1


def cmd_process_file(args: argparse.Namespace) -> int:
    cfg = DenoiserConfig.load(args.config)
    if args.preset:
        cfg = preset_config(args.preset)
    sr, dur = process_wav_file(args.input_wav, args.output_wav, config=cfg)
    print(
        f"Готово: обработан файл '{args.input_wav}' -> '{args.output_wav}' "
        f"({dur:.2f} сек, {sr} Гц, пресет: {cfg.preset})."
    )
    return 0


def cmd_autostart(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    action = args.action
    if action == "install":
        _, note = install_autostart(script_path)
        print(note)
        return 0
    if action == "remove":
        _, note = remove_autostart()
        print(note)
        return 0
    installed, path = autostart_status()
    state_str = "ВКЛЮЧЁН" if installed else "ВЫКЛЮЧЕН"
    print(f"Автозапуск: {state_str} ({path})")
    return 0


def cmd_setup_linux_mic(_args: argparse.Namespace) -> int:
    ok, msg = setup_linux_virtual_mic()
    print(msg)
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = DenoiserConfig.load(cfg_path)
    if args.preset:
        base = preset_config(args.preset)
        cfg = replace(
            base,
            input_device=cfg.input_device,
            output_device=cfg.output_device,
            input_gain_db=cfg.input_gain_db,
        )
    if args.engine:
        cfg = replace(cfg, engine=args.engine)
    if args.input is not None:
        cfg = replace(cfg, input_device=_parse_device_arg(args.input))
    if args.output is not None:
        cfg = replace(cfg, output_device=_parse_device_arg(args.output))
    if args.no_eq:
        cfg = replace(cfg, voice_eq_enabled=False)
    if args.no_dereverb:
        cfg = replace(cfg, dereverb_enabled=False)
    if args.no_gate:
        cfg = replace(cfg, gate_enabled=False)
    if args.no_compressor:
        cfg = replace(cfg, compressor_enabled=False)
    if args.hum is not None:
        hum_val = None if args.hum <= 0 else float(args.hum)
        cfg = replace(cfg, hum_filter_hz=hum_val)

    _configure_logging(args.log_file, args.quiet)
    return run_realtime_loop(
        config=cfg,
        monitor_speakers=bool(args.monitor),
        quiet=bool(args.quiet),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mikrofon",
        description=(
            "ИИ-шумоподавление (RNNoise v0.2) и удаление эффекта бочки в реальном времени."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Путь к файлу конфигурации (по умолчанию: config.json)",
    )

    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Запустить шумоподавление в реальном времени (по умолчанию)")
    p_run.add_argument("--preset", choices=list(PRESETS.keys()), help="Пресет обработки звука")
    p_run.add_argument(
        "--engine",
        choices=["auto", "rnnoise", "spectral"],
        help="Движок шумоподавления (по умолчанию: auto -> нейросеть RNNoise v0.2)",
    )
    p_run.add_argument("--input", help="Индекс или часть названия входного микрофона")
    p_run.add_argument("--output", help="Индекс или часть названия виртуального кабеля / выхода")
    p_run.add_argument(
        "--monitor",
        action="store_true",
        help="Выводить звук в наушники по умолчанию (использовать ТОЛЬКО в наушниках для проверки)",
    )
    p_run.add_argument("--no-eq", action="store_true", help="Отключить анти-бочка эквалайзер")
    p_run.add_argument("--no-dereverb", action="store_true", help="Отключить подавление эха комнаты")
    p_run.add_argument("--no-gate", action="store_true", help="Отключить шумовой гейт")
    p_run.add_argument("--no-compressor", action="store_true", help="Отключить выравнивание громкости")
    p_run.add_argument(
        "--hum",
        type=int,
        choices=[0, 50, 60],
        help="Частота подавления сетевого гула: 50 Гц, 60 Гц или 0 (выкл)",
    )
    p_run.add_argument("--quiet", action="store_true", help="Тихий фоновый режим без вывода шкалы")
    p_run.add_argument("--log-file", help="Путь к лог-файлу (для фоновой работы)")

    # install-cable
    sub.add_parser(
        "install-cable",
        help="Скачать и установить виртуальный микрофон VB-Cable для Telegram / ВК / Discord (Windows)",
    )

    # devices
    sub.add_parser("devices", help="Показать список всех микрофонов и аудиоустройств")

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Проверить ваш микрофон и измерить уровень шума")
    p_diag.add_argument("--input", help="Индекс или название микрофона")

    # calibrate
    p_cal = sub.add_parser(
        "calibrate", help="Автоматически настроить ИИ-шумоподавление и анти-бочку под ваш микрофон"
    )
    p_cal.add_argument("--input", help="Индекс или название микрофона")
    p_cal.add_argument("--preset", choices=list(PRESETS.keys()), help="Принудительно задать пресет")

    # test
    p_test = sub.add_parser(
        "test", help="Записать 5 секунд голоса и прослушать разницу ДО и ПОСЛЕ"
    )
    p_test.add_argument("--preset", choices=list(PRESETS.keys()), help="Пресет обработки")
    p_test.add_argument("--input", help="Индекс или название микрофона")
    p_test.add_argument("--seconds", type=float, default=5.0, help="Длительность записи (сек)")
    p_test.add_argument("--no-playback", action="store_true", help="Только сохранить WAV без воспроизведения")

    # process-file
    p_wav = sub.add_parser("process-file", help="Очистить готовый WAV-файл от шумов и бочки")
    p_wav.add_argument("input_wav", help="Входной WAV-файл")
    p_wav.add_argument("output_wav", help="Выходной очищенный WAV-файл")
    p_wav.add_argument("--preset", choices=list(PRESETS.keys()), help="Пресет обработки")

    # autostart
    p_auto = sub.add_parser("autostart", help="Управление автозапуском при включении ПК")
    p_auto.add_argument(
        "action",
        choices=["install", "remove", "status"],
        nargs="?",
        default="status",
        help="install — включить автозапуск, remove — отключить, status — проверить",
    )

    # setup-linux-mic
    sub.add_parser(
        "setup-linux-mic",
        help="Создать виртуальный микрофон в Linux (PipeWire / PulseAudio)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_console()
    args_list = list(sys.argv[1:] if argv is None else argv)

    known_subcommands = {
        "run",
        "install-cable",
        "devices",
        "diagnose",
        "calibrate",
        "test",
        "process-file",
        "autostart",
        "setup-linux-mic",
        "-h",
        "--help",
    }
    if not args_list:
        args_list = ["run"]
    elif args_list[0] not in known_subcommands and not args_list[0].startswith("--config"):
        args_list = ["run"] + args_list

    parser = build_parser()
    args = parser.parse_args(args_list)

    dispatch = {
        "run": cmd_run,
        "install-cable": cmd_install_cable,
        "devices": cmd_devices,
        "diagnose": cmd_diagnose,
        "calibrate": cmd_calibrate,
        "test": cmd_test,
        "process-file": cmd_process_file,
        "autostart": cmd_autostart,
        "setup-linux-mic": cmd_setup_linux_mic,
    }
    handler = dispatch.get(args.command or "run", cmd_run)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
