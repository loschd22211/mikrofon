from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import pytest
from scipy import signal

import mikrofon
from mikrofon_dsp.audio_io import (
    AudioDeviceInfo,
    resolve_input_device,
    resolve_output_device,
)
from mikrofon_dsp.autostart import (
    autostart_status,
    build_autostart_content,
    install_autostart,
    remove_autostart,
)
from mikrofon_dsp.config import DenoiserConfig, preset_config
from mikrofon_dsp.dereverb import SpectralDereverb
from mikrofon_dsp.filters import build_voice_eq
from mikrofon_dsp.neural import is_neural_rnnoise_available
from mikrofon_dsp.pipeline import (
    VoiceEnhancerPipeline,
    build_calibrated_config,
    read_wav_float32,
    write_wav_int16,
)
from mikrofon_dsp.spectral import AdaptiveSpectralDenoiser, analyze_microphone_noise


def _make_speech_signal(sr: int, duration_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic glottal-excited human formant speech bursts with silent pauses."""
    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float64) / sr
    # Pitch contour 125..155 Hz -> rich Rosenberg-like glottal harmonics up to 4 kHz
    f0 = 135.0 + 15.0 * np.sin(2.0 * np.pi * 2.2 * t) + 3.0 * np.sin(2.0 * np.pi * 5.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    glottal = np.zeros(n, dtype=np.float64)
    for h in range(1, 30):
        glottal += (1.0 / (h ** 1.15)) * np.sin(h * phase)

    # Parallel vocal tract formant resonators (F1=520 Hz, F2=1480 Hz, F3=2550 Hz) + chest body
    f1 = signal.sosfilt(signal.tf2sos(*signal.iirpeak(520.0, Q=3.5, fs=sr)), glottal)
    f2 = signal.sosfilt(signal.tf2sos(*signal.iirpeak(1480.0, Q=5.0, fs=sr)), glottal)
    f3 = signal.sosfilt(signal.tf2sos(*signal.iirpeak(2550.0, Q=6.0, fs=sr)), glottal)
    voiced = 0.45 * glottal + 0.80 * f1 + 0.50 * f2 + 0.30 * f3
    voiced /= np.sqrt(np.mean(voiced * voiced)) + 1e-9

    env = np.zeros(n, dtype=np.float64)
    mask = ((t >= 0.8) & (t <= 1.8)) | ((t >= 2.4) & (t <= 3.4))
    syllable_mod = 0.55 + 0.45 * np.abs(np.sin(np.pi * 3.8 * t))
    env[mask] = syllable_mod[mask]
    smooth_len = max(4, int(0.015 * sr))
    kernel = np.hanning(smooth_len)
    kernel /= np.sum(kernel)
    env = np.convolve(env, kernel, mode="same")

    speech = np.clip(0.14 * voiced * env, -0.85, 0.85).astype(np.float32)
    return speech, mask


def test_spectral_cola_identity_and_variable_chunk_sizes() -> None:
    sr = 48000
    denoiser = AdaptiveSpectralDenoiser(
        sample_rate=sr,
        frame_ms=10,
        suppression_strength=0.0,
        spectral_floor_db=0.0,
    )
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, size=sr).astype(np.float32)

    chunk_sizes = [137, 480, 512, 3, 1024, 448, 999, 64, 777]
    outputs = []
    pos = 0
    idx = 0
    while pos < x.size:
        step = chunk_sizes[idx % len(chunk_sizes)]
        seg = x[pos : pos + step]
        out = denoiser.process(seg)
        assert out.shape == seg.shape
        assert np.all(np.isfinite(out))
        outputs.append(out)
        pos += step
        idx += 1

    reconstructed = np.concatenate(outputs)
    assert reconstructed.shape == x.shape

    delay = 2 * denoiser.hop_size
    interior_in = x[denoiser.hop_size : -delay]
    interior_out = reconstructed[delay + denoiser.hop_size :]
    max_err = float(np.max(np.abs(interior_in - interior_out)))
    assert max_err < 1e-5, f"COLA reconstruction error too large: {max_err}"


@pytest.mark.parametrize("engine", ["spectral", "rnnoise"])
def test_pipeline_suppresses_hiss_hum_and_rumble_while_preserving_speech(engine: str) -> None:
    if engine == "rnnoise" and not is_neural_rnnoise_available():
        pytest.skip("pyrnnoise not installed in test env")

    sr = 48000
    duration = 4.0
    speech, _ = _make_speech_signal(sr, duration)
    n = speech.size
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(123)

    rumble = 0.04 * np.sin(2.0 * np.pi * 35.0 * t)
    hum = 0.025 * np.sin(2.0 * np.pi * 50.0 * t) + 0.015 * np.sin(2.0 * np.pi * 100.0 * t)
    hiss = 0.018 * rng.standard_normal(n)
    noisy = (speech + rumble + hum + hiss).astype(np.float32)

    cfg = replace(preset_config("balanced"), engine=engine)
    pipeline = VoiceEnhancerPipeline(cfg)
    if engine == "spectral":
        pipeline.calibrate_silence(noisy[: int(0.6 * sr)])

    hop = pipeline.hop_size
    cleaned = np.concatenate(
        [pipeline.process_mono(noisy[i : i + hop]) for i in range(0, noisy.size, hop)]
    )
    assert cleaned.shape == noisy.shape
    assert np.all(np.isfinite(cleaned))
    assert float(np.max(np.abs(cleaned))) <= 1.0

    s0, s1 = int(3.65 * sr), int(3.95 * sr)
    in_silence_rms_db = 20.0 * np.log10(np.sqrt(np.mean(noisy[s0:s1] ** 2)) + 1e-12)
    out_silence_rms_db = 20.0 * np.log10(np.sqrt(np.mean(cleaned[s0:s1] ** 2)) + 1e-12)
    noise_reduction_db = in_silence_rms_db - out_silence_rms_db
    assert noise_reduction_db > 25.0, f"Expected >25 dB reduction ({engine}), got {noise_reduction_db:.1f} dB"

    v0, v1 = int(1.0 * sr), int(1.6 * sr)
    out_voice_rms_db = 20.0 * np.log10(np.sqrt(np.mean(cleaned[v0:v1] ** 2)) + 1e-12)
    assert -26.0 < out_voice_rms_db < -2.0, f"Unexpected voice level: {out_voice_rms_db:.1f} dBFS"


def test_neural_rnnoise_suppresses_chair_creak() -> None:
    if not is_neural_rnnoise_available():
        pytest.skip("pyrnnoise not installed in test env")

    sr = 48000
    rng = np.random.default_rng(7)
    n = sr * 2
    impulses = np.zeros(n, dtype=np.float64)
    pos = int(0.4 * sr)
    while pos < int(1.6 * sr):
        impulses[pos] = rng.uniform(0.5, 1.0)
        prog = (pos - 0.4 * sr) / (1.2 * sr)
        rate_hz = 28.0 + 45.0 * np.sin(np.pi * prog)
        pos += int(sr / rate_hz) + int(rng.integers(-30, 30))

    sos_creak = np.vstack(
        [
            signal.tf2sos(*signal.iirpeak(1150.0, Q=18.0, fs=sr)),
            signal.tf2sos(*signal.iirpeak(2450.0, Q=22.0, fs=sr)),
        ]
    )
    creak = signal.sosfilt(sos_creak, impulses)
    creak = (0.25 * creak / (np.max(np.abs(creak)) + 1e-9)).astype(np.float32)
    noisy = creak + (0.01 * rng.standard_normal(n)).astype(np.float32)

    pipeline = VoiceEnhancerPipeline(preset_config("anti_barrel"))
    hop = pipeline.hop_size
    cleaned = np.concatenate(
        [pipeline.process_mono(noisy[i : i + hop]) for i in range(0, noisy.size, hop)]
    )

    c0, c1 = int(0.5 * sr), int(1.5 * sr)
    in_db = 20.0 * np.log10(np.sqrt(np.mean(noisy[c0:c1] ** 2)) + 1e-12)
    out_db = 20.0 * np.log10(np.sqrt(np.mean(cleaned[c0:c1] ** 2)) + 1e-12)
    assert (in_db - out_db) > 32.0, f"Chair creak suppression too weak: {in_db - out_db:.1f} dB"


def test_dereverb_and_anti_barrel_eq_attenuate_boxy_room_tail() -> None:
    sr = 48000
    # 1. Verify anti-barrel EQ cuts 380 Hz boxiness and boosts 10 kHz studio air
    eq = build_voice_eq(
        sample_rate=sr,
        mud_cut_db=-5.0,
        honk_cut_db=-2.5,
        presence_boost_db=4.0,
        air_boost_db=3.0,
    )
    t = np.arange(sr // 2, dtype=np.float32) / sr
    tone_380 = np.sin(2.0 * np.pi * 380.0 * t).astype(np.float32) * 0.2
    tone_10k = np.sin(2.0 * np.pi * 10000.0 * t).astype(np.float32) * 0.2
    out_380 = eq.process(tone_380)
    eq.reset()
    out_10k = eq.process(tone_10k)

    gain_380_db = 20.0 * np.log10(np.sqrt(np.mean(out_380[4000:] ** 2)) / np.sqrt(np.mean(tone_380[4000:] ** 2)))
    gain_10k_db = 20.0 * np.log10(np.sqrt(np.mean(out_10k[4000:] ** 2)) / np.sqrt(np.mean(tone_10k[4000:] ** 2)))
    assert gain_380_db < -3.5, f"Expected 380 Hz barrel cut, got {gain_380_db:.2f} dB"
    assert gain_10k_db > 1.5, f"Expected 10 kHz air boost, got {gain_10k_db:.2f} dB"

    # 2. Verify SpectralDereverb suppresses late reverberant room tail after a speech burst
    dereverb = SpectralDereverb(sample_rate=sr, frame_ms=10, strength=1.2)
    n = sr
    burst = np.zeros(n, dtype=np.float32)
    t_b = np.arange(int(0.25 * sr), dtype=np.float32) / sr
    burst[: t_b.size] = 0.4 * np.sin(2.0 * np.pi * 420.0 * t_b)
    # Exponential room reverb tail starting right after burst ends at 0.25s
    tail_len = int(0.30 * sr)
    t_tail = np.arange(tail_len, dtype=np.float32) / sr
    burst[t_b.size : t_b.size + tail_len] = (
        0.06 * np.exp(-t_tail / 0.08) * np.sin(2.0 * np.pi * 420.0 * t_tail)
    )

    dry = dereverb.process(burst)
    r0, r1 = int(0.28 * sr), int(0.42 * sr)
    tail_in_db = 20.0 * np.log10(np.sqrt(np.mean(burst[r0:r1] ** 2)) + 1e-12)
    tail_out_db = 20.0 * np.log10(np.sqrt(np.mean(dry[r0:r1] ** 2)) + 1e-12)
    assert (tail_in_db - tail_out_db) > 2.5, f"Dereverb tail attenuation too weak: {tail_in_db - tail_out_db:.1f} dB"


def test_soft_limiter_prevents_clipping_on_loud_shouts() -> None:
    sr = 48000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    loud_shout = (2.5 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    pipeline = VoiceEnhancerPipeline(preset_config("gaming"))
    out = pipeline.process_mono(loud_shout)
    assert float(np.max(np.abs(out))) <= 0.99


@pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
def test_pipeline_supports_all_microphone_sample_rates(sr: int) -> None:
    cfg = DenoiserConfig(sample_rate=sr)
    pipeline = VoiceEnhancerPipeline(cfg)
    x = np.zeros(sr // 5, dtype=np.float32)
    out = pipeline.process_frame(x.reshape(-1, 1), out_channels=2)
    assert out.shape == (sr // 5, 2)


def test_diagnostics_detects_50hz_hum_and_calibrates_gain() -> None:
    sr = 48000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    rng = np.random.default_rng(7)
    silence = (
        0.002 * rng.standard_normal(t.size)
        + 0.025 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.010 * np.sin(2.0 * np.pi * 100.0 * t)
    ).astype(np.float32)
    quiet_speech = (0.02 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    diag = analyze_microphone_noise(silence, sr, speech_audio=quiet_speech)
    assert diag.detected_hum_hz == 50.0
    assert diag.recommended_input_gain_db > 5.0

    cfg, _ = build_calibrated_config(silence, sr, speech_audio=quiet_speech, mic_name="Cheap USB Mic")
    assert cfg.hum_filter_hz == 50.0
    assert cfg.calibrated_mic_name == "Cheap USB Mic"


def test_device_autoresolve_avoids_virtual_cable_loop_and_speaker_feedback() -> None:
    mock_devices = [
        AudioDeviceInfo(
            index=0,
            name="CABLE Output (VB-Audio Virtual Cable)",
            hostapi_name="Windows WASAPI",
            max_input_channels=2,
            max_output_channels=0,
            default_samplerate=48000,
            is_default_input=True,
            is_default_output=False,
            is_virtual=True,
        ),
        AudioDeviceInfo(
            index=1,
            name="Realtek High Definition Audio Mic",
            hostapi_name="Windows WASAPI",
            max_input_channels=2,
            max_output_channels=0,
            default_samplerate=48000,
            is_default_input=False,
            is_default_output=False,
            is_virtual=False,
        ),
        AudioDeviceInfo(
            index=2,
            name="Speakers (Realtek Audio)",
            hostapi_name="Windows WASAPI",
            max_input_channels=0,
            max_output_channels=2,
            default_samplerate=48000,
            is_default_input=False,
            is_default_output=True,
            is_virtual=False,
        ),
        AudioDeviceInfo(
            index=3,
            name="CABLE Input (VB-Audio Virtual Cable)",
            hostapi_name="Windows WASAPI",
            max_input_channels=0,
            max_output_channels=2,
            default_samplerate=48000,
            is_default_input=False,
            is_default_output=False,
            is_virtual=True,
        ),
    ]

    picked_in = resolve_input_device(None, mock_devices)
    assert picked_in.index == 1

    picked_out = resolve_output_device(None, allow_speakers_fallback=False, devices=mock_devices)
    assert picked_out is not None and picked_out.index == 3

    only_speakers = [mock_devices[1], mock_devices[2]]
    assert resolve_output_device(None, allow_speakers_fallback=False, devices=only_speakers) is None
    assert resolve_output_device(None, allow_speakers_fallback=True, devices=only_speakers) is not None


def test_cli_process_file_and_autostart_across_platforms(tmp_path: Path) -> None:
    sr = 48000
    speech, _ = _make_speech_signal(sr, 1.5)
    rng = np.random.default_rng(99)
    noisy = (speech + 0.02 * rng.standard_normal(speech.size)).astype(np.float32)

    in_wav = tmp_path / "noisy.wav"
    out_wav = tmp_path / "clean.wav"
    write_wav_int16(in_wav, sr, noisy)

    rc = mikrofon.main(["process-file", str(in_wav), str(out_wav), "--preset", "anti_barrel"])
    assert rc == 0
    out_sr, cleaned = read_wav_float32(out_wav)
    assert out_sr == sr
    assert cleaned.shape == noisy.shape

    fake_script = tmp_path / "mikrofon.py"
    fake_script.write_text("# stub\n", encoding="utf-8")

    for os_name in ("windows", "linux", "darwin"):
        os_home = tmp_path / f"home_{os_name}"
        target, _ = install_autostart(fake_script, system_name=os_name, home_dir=os_home)
        assert target.exists()
        is_on, status_path = autostart_status(system_name=os_name, home_dir=os_home)
        assert is_on and status_path == target
        content = build_autostart_content(fake_script, system_name=os_name)
        assert "mikrofon.py" in content
        removed, _ = remove_autostart(system_name=os_name, home_dir=os_home)
        assert removed
        assert not target.exists()


def test_uncalibrated_startup_during_speech_still_converges_and_suppresses_noise() -> None:
    sr = 48000
    duration = 4.0
    speech, _ = _make_speech_signal(sr, duration)
    t = np.arange(speech.size, dtype=np.float64) / sr
    early_voice = (0.22 * np.sin(2.0 * np.pi * 180.0 * t) * (t < 0.5)).astype(np.float32)
    rng = np.random.default_rng(55)
    noise = (0.02 * np.sin(2.0 * np.pi * 50.0 * t) + 0.015 * rng.standard_normal(t.size)).astype(np.float32)
    noisy = speech + early_voice + noise

    pipeline = VoiceEnhancerPipeline(replace(preset_config("balanced"), engine="spectral"))
    hop = pipeline.hop_size
    cleaned = np.concatenate(
        [pipeline.process_mono(noisy[i : i + hop]) for i in range(0, noisy.size, hop)]
    )
    s0, s1 = int(3.65 * sr), int(3.95 * sr)
    in_db = 20.0 * np.log10(np.sqrt(np.mean(noisy[s0:s1] ** 2)) + 1e-12)
    out_db = 20.0 * np.log10(np.sqrt(np.mean(cleaned[s0:s1] ** 2)) + 1e-12)
    assert (in_db - out_db) > 30.0, f"Uncalibrated reduction too weak: {in_db - out_db:.1f} dB"


def test_realtime_cpu_budget_10s_audio() -> None:
    sr = 48000
    pipeline = VoiceEnhancerPipeline(preset_config("anti_barrel"))
    hop = pipeline.hop_size
    rng = np.random.default_rng(0)
    frames = [rng.uniform(-0.1, 0.1, size=hop).astype(np.float32) for _ in range(1000)]

    t0 = time.perf_counter()
    for f in frames:
        pipeline.process_mono(f)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"Pipeline too slow for 24/7 background use: {elapsed:.2f}s for 10s audio"
