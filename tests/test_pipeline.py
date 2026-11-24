from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pytest

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
from mikrofon_dsp.pipeline import (
    VoiceEnhancerPipeline,
    build_calibrated_config,
    read_wav_float32,
    write_wav_int16,
)
from mikrofon_dsp.spectral import AdaptiveSpectralDenoiser, analyze_microphone_noise


def _make_speech_signal(sr: int, duration_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic voiced speech bursts with silent pauses and return (signal, speech_mask)."""
    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float64) / sr
    # Fundamental ~140 Hz with slight vibrato + speech-band harmonics up to 3.2 kHz
    phase = 2.0 * np.pi * (140.0 * t + 1.5 * np.sin(2.0 * np.pi * 4.5 * t))
    voiced = np.zeros(n, dtype=np.float64)
    for h, amp in ((1, 0.45), (2, 0.35), (3, 0.25), (5, 0.18), (8, 0.12), (14, 0.08), (22, 0.05)):
        voiced += amp * np.sin(h * phase)
    voiced /= np.max(np.abs(voiced))

    # Speech active between [0.8s..1.8s] and [2.4s..3.4s], silent elsewhere
    env = np.zeros(n, dtype=np.float64)
    mask = ((t >= 0.8) & (t <= 1.8)) | ((t >= 2.4) & (t <= 3.4))
    env[mask] = 1.0
    # Smooth 15ms attack/release on synthetic voice envelope
    smooth_len = max(4, int(0.015 * sr))
    kernel = np.hanning(smooth_len)
    kernel /= np.sum(kernel)
    env = np.convolve(env, kernel, mode="same")

    speech = (0.22 * voiced * env).astype(np.float32)
    return speech, mask


def test_spectral_cola_identity_and_variable_chunk_sizes() -> None:
    sr = 48000
    denoiser = AdaptiveSpectralDenoiser(
        sample_rate=sr,
        frame_ms=10,
        suppression_strength=0.0,
        spectral_floor_db=0.0,  # gain_floor = 1.0 -> exact unity pass-through
    )
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, size=sr).astype(np.float32)

    # Feed irregular WASAPI/ALSA callback chunk sizes
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

    # Account for exact 2 * hop_size FIFO + OLA delay; interior samples match to float32 precision
    delay = 2 * denoiser.hop_size
    interior_in = x[denoiser.hop_size : -delay]
    interior_out = reconstructed[delay + denoiser.hop_size :]
    max_err = float(np.max(np.abs(interior_in - interior_out)))
    assert max_err < 1e-5, f"COLA reconstruction error too large: {max_err}"


def test_pipeline_suppresses_hiss_hum_and_rumble_while_preserving_speech() -> None:
    sr = 48000
    duration = 4.0
    speech, speech_mask = _make_speech_signal(sr, duration)
    n = speech.size
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(123)

    # Cheap-mic noise cocktail: 35 Hz desk rumble + 50/100 Hz mains buzz + broadband ADC hiss
    rumble = 0.04 * np.sin(2.0 * np.pi * 35.0 * t)
    hum = 0.025 * np.sin(2.0 * np.pi * 50.0 * t) + 0.015 * np.sin(2.0 * np.pi * 100.0 * t)
    hiss = 0.018 * rng.standard_normal(n)
    noise = (rumble + hum + hiss).astype(np.float32)

    noisy = speech + noise

    cfg = preset_config("balanced")
    pipeline = VoiceEnhancerPipeline(cfg)
    pipeline.calibrate_silence(noisy[: int(0.6 * sr)])

    hop = pipeline.hop_size
    cleaned_parts = [
        pipeline.process_mono(noisy[i : i + hop]) for i in range(0, noisy.size, hop)
    ]
    cleaned = np.concatenate(cleaned_parts)
    assert cleaned.shape == noisy.shape
    assert np.all(np.isfinite(cleaned))
    assert float(np.max(np.abs(cleaned))) <= 1.0

    # Measure silent tail [3.65s..3.95s] RMS before vs after
    s0, s1 = int(3.65 * sr), int(3.95 * sr)
    in_silence_rms_db = 20.0 * np.log10(np.sqrt(np.mean(noisy[s0:s1] ** 2)) + 1e-12)
    out_silence_rms_db = 20.0 * np.log10(np.sqrt(np.mean(cleaned[s0:s1] ** 2)) + 1e-12)
    noise_reduction_db = in_silence_rms_db - out_silence_rms_db
    assert noise_reduction_db > 22.0, f"Expected >22 dB noise reduction, got {noise_reduction_db:.1f} dB"

    # Measure active voice [1.0s..1.6s] remains healthy and loud
    v0, v1 = int(1.0 * sr), int(1.6 * sr)
    out_voice_rms_db = 20.0 * np.log10(np.sqrt(np.mean(cleaned[v0:v1] ** 2)) + 1e-12)
    assert -24.0 < out_voice_rms_db < -3.0, f"Unexpected voice level: {out_voice_rms_db:.1f} dBFS"


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
        0.004 * rng.standard_normal(t.size)
        + 0.015 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.008 * np.sin(2.0 * np.pi * 100.0 * t)
    ).astype(np.float32)
    quiet_speech = (0.02 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)  # ~ -37 dBFS

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
            is_default_input=True,  # Windows often sets VB-Cable as default input by mistake!
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
    assert picked_in.index == 1  # Skipped CABLE Output and grabbed physical Realtek Mic!

    picked_out = resolve_output_device(None, allow_speakers_fallback=False, devices=mock_devices)
    assert picked_out is not None and picked_out.index == 3  # Picked CABLE Input!

    # Without virtual cable, allow_speakers_fallback=False must return None (prevent feedback howl)
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

    rc = mikrofon.main(["process-file", str(in_wav), str(out_wav), "--preset", "max_denoise"])
    assert rc == 0
    out_sr, cleaned = read_wav_float32(out_wav)
    assert out_sr == sr
    assert cleaned.shape == noisy.shape

    # Test autostart generation for Windows, Linux, and macOS in isolated tmp_path
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
    """Even if user launches `mikrofon.py` without calibration while speaking on frame 0, noise drops >30 dB."""
    sr = 48000
    duration = 4.0
    speech, _ = _make_speech_signal(sr, duration)
    # Force loud speech right from t=0.0s to t=0.6s as well!
    t = np.arange(speech.size, dtype=np.float64) / sr
    early_voice = (0.22 * np.sin(2.0 * np.pi * 180.0 * t) * (t < 0.5)).astype(np.float32)
    rng = np.random.default_rng(55)
    noise = (0.02 * np.sin(2.0 * np.pi * 50.0 * t) + 0.015 * rng.standard_normal(t.size)).astype(np.float32)
    noisy = speech + early_voice + noise

    pipeline = VoiceEnhancerPipeline(preset_config("balanced"))
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
    pipeline = VoiceEnhancerPipeline(preset_config("balanced"))
    hop = pipeline.hop_size  # 480 samples = 10 ms
    rng = np.random.default_rng(0)
    frames = [rng.uniform(-0.1, 0.1, size=hop).astype(np.float32) for _ in range(1000)]

    t0 = time.perf_counter()
    for f in frames:
        pipeline.process_mono(f)
    elapsed = time.perf_counter() - t0
    # 10.0 seconds of 48 kHz audio must process in well under 1.0s (<10% of 1 CPU core)
    assert elapsed < 1.0, f"Pipeline too slow for 24/7 background use: {elapsed:.2f}s for 10s audio"
