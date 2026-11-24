from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import DenoiserConfig, preset_config
from .dynamics import SmoothNoiseGate, VoiceCompressorLimiter
from .filters import StreamingSOSFilter, build_prefilter, build_voice_eq
from .spectral import AdaptiveSpectralDenoiser, MicDiagnostics, analyze_microphone_noise


class VoiceEnhancerPipeline:
    """
    Complete low-latency streaming audio processor for budget microphones:
      1. Input Pre-Gain
      2. Rumble High-Pass Filter (85 Hz) + Mains Hum Harmonic Notches (50/60 Hz)
      3. Ephraim-Malah Adaptive Spectral Noise Suppressor (STFT Overlap-Add)
      4. Smooth Voice-Aware Noise Gate / Downward Expander
      5. Studio Voice Clarity EQ (Anti-Mud + Presence Boost + HF Hiss Roll-Off)
      6. Broadcast Compressor + Makeup Gain + Soft-Knee Brickwall Limiter
    """

    def __init__(self, config: DenoiserConfig | None = None) -> None:
        self.config = config or DenoiserConfig()
        self.sample_rate = int(self.config.sample_rate)
        self.hop_size = max(
            64, int(round(self.sample_rate * self.config.frame_ms / 1000.0))
        )

        self._input_lin_gain = float(10.0 ** (self.config.input_gain_db / 20.0))

        self.prefilter: StreamingSOSFilter = build_prefilter(
            sample_rate=self.sample_rate,
            highpass_hz=self.config.highpass_hz,
            hum_filter_hz=self.config.hum_filter_hz,
        )

        self.spectral = AdaptiveSpectralDenoiser(
            sample_rate=self.sample_rate,
            frame_ms=self.config.frame_ms,
            suppression_strength=self.config.suppression_strength,
            spectral_floor_db=self.config.spectral_floor_db,
            vad_snr_threshold_db=self.config.vad_snr_threshold_db,
        )

        self.gate = SmoothNoiseGate(
            sample_rate=self.sample_rate,
            threshold_db=self.config.gate_threshold_db,
            attenuation_db=self.config.gate_attenuation_db,
            hold_ms=self.config.gate_hold_ms,
            release_ms=self.config.gate_release_ms,
        )

        self.voice_eq: StreamingSOSFilter = build_voice_eq(
            sample_rate=self.sample_rate,
            mud_cut_db=self.config.mud_cut_db,
            presence_boost_db=self.config.presence_boost_db,
            lowpass_hz=self.config.lowpass_hz,
        )

        self.compressor = VoiceCompressorLimiter(
            sample_rate=self.sample_rate,
            threshold_db=self.config.compressor_threshold_db,
            ratio=self.config.compressor_ratio,
            makeup_gain_db=self.config.makeup_gain_db,
        )

        # Live telemetry for UI/CLI meters
        self.last_input_rms_db: float = -96.0
        self.last_output_rms_db: float = -96.0

    def reset(self) -> None:
        self.prefilter.reset()
        self.spectral.reset()
        self.gate.reset()
        self.voice_eq.reset()
        self.compressor.reset()

    def calibrate_silence(self, silence_audio: np.ndarray) -> None:
        """Seed the spectral denoiser with a measured silence clip (pre-filtered)."""
        mono = self._to_mono(silence_audio)
        if abs(self._input_lin_gain - 1.0) > 1e-4:
            mono = mono * self._input_lin_gain
        filtered = self.prefilter.process(mono)
        self.prefilter.reset()
        self.spectral.set_noise_profile(filtered)

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        if arr.shape[1] == 1:
            return arr[:, 0]
        return np.mean(arr, axis=1, dtype=np.float32)

    def process_mono(self, block: np.ndarray) -> np.ndarray:
        """Process a 1D float32 block in [-1.0, 1.0] and return cleaned 1D float32 block."""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x.copy()

        eps = 1e-12
        in_rms = float(np.sqrt(np.mean(x * x) + eps))
        self.last_input_rms_db = float(20.0 * np.log10(in_rms + eps))

        # 1. Input pre-gain
        if abs(self._input_lin_gain - 1.0) > 1e-4:
            x = x * self._input_lin_gain

        # 2. Rumble high-pass + 50/60 Hz mains hum harmonic notch filter
        x = self.prefilter.process(x)

        # 3. Adaptive Ephraim-Malah spectral noise suppression
        x = self.spectral.process(x)

        # 4. Smooth voice-aware noise gate / downward expander
        if self.config.gate_enabled:
            x = self.gate.process(x, speech_prob=self.spectral.last_speech_prob)

        # 5. Studio voice clarity EQ
        if self.config.voice_eq_enabled:
            x = self.voice_eq.process(x)

        # 6. Broadcast compressor + brickwall soft limiter
        x = self.compressor.process(x, enabled=self.config.compressor_enabled)

        out_rms = float(np.sqrt(np.mean(x * x) + eps))
        self.last_output_rms_db = float(20.0 * np.log10(out_rms + eps))
        return x

    def process_frame(self, indata: np.ndarray, out_channels: int | None = None) -> np.ndarray:
        """
        Process a 1D or 2D `(frames, channels)` audio buffer and format to `out_channels`.
        Ideal for `sounddevice.Stream` real-time callbacks.
        """
        mono = self._to_mono(indata)
        clean = self.process_mono(mono)
        if indata.ndim == 1 and out_channels is None:
            return clean
        target_ch = out_channels if out_channels is not None else indata.shape[1]
        if target_ch <= 1:
            return clean.reshape(-1, 1)
        return np.repeat(clean.reshape(-1, 1), target_ch, axis=1)


def build_calibrated_config(
    silence_audio: np.ndarray,
    sample_rate: int,
    speech_audio: np.ndarray | None = None,
    mic_name: str | None = None,
    base_config: DenoiserConfig | None = None,
) -> tuple[DenoiserConfig, MicDiagnostics]:
    """Analyze recorded silence (+ optional speech) and return a tailored DenoiserConfig."""
    diag = analyze_microphone_noise(
        silence_audio=silence_audio,
        sample_rate=sample_rate,
        speech_audio=speech_audio,
    )
    cfg = preset_config(diag.recommended_preset)
    if base_config is not None:
        cfg = replace(
            cfg,
            input_device=base_config.input_device,
            output_device=base_config.output_device,
        )

    # Keep 50 Hz notch on by default for protection unless 60 Hz hum was specifically detected
    hum_hz = diag.detected_hum_hz if diag.detected_hum_hz is not None else 50.0
    cfg = replace(
        cfg,
        sample_rate=sample_rate,
        input_gain_db=diag.recommended_input_gain_db,
        hum_filter_hz=hum_hz,
        gate_threshold_db=diag.recommended_gate_threshold_db,
        calibrated_mic_name=mic_name,
        calibrated_noise_rms_db=diag.rms_db,
    )
    return cfg, diag


def read_wav_float32(path: Path | str) -> tuple[int, np.ndarray]:
    """Read a WAV file and normalize samples to float32 in [-1.0, 1.0]."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        f32 = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        f32 = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        f32 = (data.astype(np.float32) - 128.0) / 128.0
    else:
        f32 = data.astype(np.float32)
    return int(sr), f32


def write_wav_int16(path: Path | str, sample_rate: int, audio: np.ndarray) -> None:
    """Write float32 audio in [-1.0, 1.0] to a 16-bit PCM WAV file."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(target), int(sample_rate), pcm16)


def process_wav_file(
    input_path: Path | str,
    output_path: Path | str,
    config: DenoiserConfig | None = None,
) -> tuple[int, float]:
    """Clean an offline WAV recording and compensate for STFT framing delay."""
    sr, audio = read_wav_float32(input_path)
    cfg = replace(config or DenoiserConfig(), sample_rate=sr)
    pipeline = VoiceEnhancerPipeline(cfg)

    mono = pipeline._to_mono(audio)
    # Seed initial noise estimate from quietest 250ms segment in the file
    seg_len = max(int(0.25 * sr), pipeline.spectral.win_size)
    if mono.size >= seg_len * 2:
        step = seg_len // 2
        best_rms = float("inf")
        quiet_seg = mono[:seg_len]
        for start in range(0, mono.size - seg_len + 1, step):
            cand = mono[start : start + seg_len]
            r = float(np.mean(cand * cand))
            if r < best_rms:
                best_rms = r
                quiet_seg = cand
        pipeline.calibrate_silence(quiet_seg)

    # Pad tail by 2 hops and trim leading 2-hop STFT framing delay so output aligns in time
    delay = 2 * pipeline.hop_size
    padded = np.concatenate((mono, np.zeros(delay, dtype=np.float32)))
    out_chunks = []
    hop = pipeline.hop_size
    for pos in range(0, padded.size, hop):
        out_chunks.append(pipeline.process_mono(padded[pos : pos + hop]))
    full_out = np.concatenate(out_chunks)
    aligned = full_out[delay : delay + mono.size]

    # If original WAV was stereo, preserve channel count
    if audio.ndim == 2 and audio.shape[1] > 1:
        aligned = np.repeat(aligned.reshape(-1, 1), audio.shape[1], axis=1)

    write_wav_int16(output_path, sr, aligned)
    duration_sec = mono.size / float(sr)
    return sr, duration_sec
