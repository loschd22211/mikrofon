from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MicDiagnostics:
    """Measured acoustic characteristics of a microphone recording."""

    sample_rate: int
    rms_db: float
    peak_db: float
    detected_hum_hz: float | None
    hum_prominence_db: float
    hf_hiss_ratio_db: float
    recommended_preset: str
    recommended_input_gain_db: float
    recommended_gate_threshold_db: float


class AdaptiveSpectralDenoiser:
    """
    Real-time overlap-add Ephraim-Malah (Decision-Directed) spectral noise suppressor.

    Uses a 50%-overlapped root-Hann (sine) window satisfying exact COLA (w^2_0 + w^2_1 == 1)
    and supports arbitrary callback chunk sizes via an invariant FIFO queue.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_ms: int = 10,
        suppression_strength: float = 1.8,
        spectral_floor_db: float = -28.0,
        vad_snr_threshold_db: float = 4.5,
        initial_noise_psd: np.ndarray | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_size = max(64, int(round(self.sample_rate * frame_ms / 1000.0)))
        self.win_size = 2 * self.hop_size
        self.n_bins = self.win_size // 2 + 1

        self.suppression_strength = float(suppression_strength)
        self.gain_floor = float(10.0 ** (spectral_floor_db / 20.0))
        self.vad_snr_threshold_db = float(vad_snr_threshold_db)

        # Vorbis / WebRTC root-Hann (sine) window -> exact perfect reconstruction at 50% overlap
        idx = np.arange(self.win_size, dtype=np.float64)
        self.window = np.sin(np.pi * (idx + 0.5) / self.win_size).astype(np.float32)

        # Frequency axis & band masks
        freqs = np.fft.rfftfreq(self.win_size, d=1.0 / self.sample_rate)
        self.freqs = freqs
        self.speech_mask = (freqs >= 250.0) & (freqs <= 3800.0)
        if not np.any(self.speech_mask):
            self.speech_mask = np.ones_like(freqs, dtype=bool)

        # Extra over-subtraction taper above 5.5 kHz to tame cheap-ADC white hiss
        hf_weight = np.ones(self.n_bins, dtype=np.float32)
        hf_bins = freqs > 5500.0
        if np.any(hf_bins):
            span = np.clip((freqs[hf_bins] - 5500.0) / 6500.0, 0.0, 1.0)
            hf_weight[hf_bins] = (1.0 + 0.65 * span).astype(np.float32)
        self.freq_oversub_weight = hf_weight

        self._initial_noise_psd = (
            initial_noise_psd.astype(np.float32, copy=True)
            if initial_noise_psd is not None and initial_noise_psd.shape == (self.n_bins,)
            else None
        )

        # Diagnostics exposed for live meter
        self.last_speech_prob: float = 0.0
        self.last_reduction_db: float = 0.0

        self.reset()

    def reset(self) -> None:
        self._analysis_frame = np.zeros(self.win_size, dtype=np.float32)
        self._ola_tail = np.zeros(self.hop_size, dtype=np.float32)
        self._in_fifo = np.zeros(0, dtype=np.float32)
        # Pre-seed out_fifo with hop_size zeros so arbitrary driver block sizes never underflow
        self._out_fifo = np.zeros(self.hop_size, dtype=np.float32)

        self._prev_clean_psd = np.zeros(self.n_bins, dtype=np.float32)
        self._prev_gain = np.ones(self.n_bins, dtype=np.float32)
        self._smooth_psd = np.zeros(self.n_bins, dtype=np.float32)

        if self._initial_noise_psd is not None:
            self.noise_psd = self._initial_noise_psd.copy()
            self._frames_seen = 100
        else:
            self.noise_psd = np.full(self.n_bins, 1e-7, dtype=np.float32)
            self._frames_seen = 0

        # Rolling minimum-statistics tracker (6 sub-windows x 30 frames ≈ 1.8 s)
        self._subwin_len = 30
        self._subwin_counter = 0
        self._curr_min = np.full(self.n_bins, np.inf, dtype=np.float32)
        self._min_ring = np.full((6, self.n_bins), np.inf, dtype=np.float32)
        self._ring_idx = 0

    def set_noise_profile(self, audio_silence: np.ndarray) -> np.ndarray:
        """Estimate stationary noise PSD directly from a calibration silence snippet."""
        mono = np.asarray(audio_silence, dtype=np.float32).reshape(-1)
        if mono.size < self.win_size:
            return self.noise_psd.copy()

        n_hops = (mono.size - self.win_size) // self.hop_size + 1
        psds = []
        for i in range(n_hops):
            start = i * self.hop_size
            frame = mono[start : start + self.win_size] * self.window
            spec = np.fft.rfft(frame)
            psds.append((spec.real * spec.real + spec.imag * spec.imag).astype(np.float32))

        # Use median across frames so accidental mouse clicks during calibration are ignored.
        # For complex Gaussian noise, |FFT|^2 is Exponentially distributed where median = ln(2) * mean;
        # dividing by ln(2) (~0.69315) yields an unbiased noise PSD estimate.
        est_psd = (np.median(np.stack(psds, axis=0), axis=0) / np.log(2.0)).astype(np.float32)
        est_psd = np.maximum(est_psd, 1e-11)
        self._initial_noise_psd = est_psd.copy()
        self.noise_psd = est_psd.copy()
        self._frames_seen = max(self._frames_seen, 100)
        return est_psd

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        H = self.hop_size
        self._analysis_frame[:H] = self._analysis_frame[H:]
        self._analysis_frame[H:] = hop

        windowed = self._analysis_frame * self.window
        spec = np.fft.rfft(windowed)
        power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

        eps = 1e-12
        self._frames_seen += 1

        # Exponentially smoothed power (alpha=0.88, ~85ms) for stable minimum-statistics tracking
        if self._frames_seen == 1:
            self._smooth_psd[:] = power
            if self._initial_noise_psd is None:
                self.noise_psd[:] = np.maximum(power, eps)
        else:
            self._smooth_psd = 0.88 * self._smooth_psd + 0.12 * power

        # Update minimum-statistics ring buffer
        self._curr_min = np.minimum(self._curr_min, self._smooth_psd)
        self._subwin_counter += 1
        if self._subwin_counter >= self._subwin_len:
            self._min_ring[self._ring_idx] = self._curr_min
            self._ring_idx = (self._ring_idx + 1) % self._min_ring.shape[0]
            self._curr_min.fill(np.inf)
            self._subwin_counter = 0

        running_min = np.minimum(self._curr_min, np.min(self._min_ring, axis=0))
        finite_mask = np.isfinite(running_min)

        # Speech-band a posteriori SNR for VAD
        speech_pwr = float(np.sum(power[self.speech_mask])) + eps
        noise_speech_pwr = float(np.sum(self.noise_psd[self.speech_mask])) + eps
        snr_speech_db = 10.0 * np.log10(speech_pwr / noise_speech_pwr)

        is_speech = snr_speech_db > self.vad_snr_threshold_db
        self.last_speech_prob = float(
            np.clip((snr_speech_db - 1.5) / max(self.vad_snr_threshold_db, 1.0), 0.0, 1.0)
        )

        # Update noise PSD estimate
        if self._frames_seen <= 12 and self._initial_noise_psd is None:
            # Fast bootstrap during initial ambient frames, capped if sudden loud speech starts
            if speech_pwr < 10.0 * noise_speech_pwr or self._frames_seen <= 3:
                alpha_n = 0.65
                self.noise_psd = alpha_n * self.noise_psd + (1.0 - alpha_n) * power
            else:
                self.noise_psd = np.minimum(self.noise_psd, self._smooth_psd)
        elif not is_speech:
            alpha_n = 0.95
            self.noise_psd = alpha_n * self.noise_psd + (1.0 - alpha_n) * power
        else:
            # During speech, never let voice inflate noise_psd; use minimum-statistics bounds
            # to recover if initialized during loud speech (5.0x cap) or if fan speed rose (1.8x floor).
            if np.any(finite_mask):
                min_cap = running_min * 5.0
                min_floor = running_min * 1.8
                bounded = np.clip(self.noise_psd[finite_mask], min_floor[finite_mask], min_cap[finite_mask])
                self.noise_psd[finite_mask] = np.maximum(bounded, eps)

        self.noise_psd = np.maximum(self.noise_psd, eps)

        # Effective over-subtracted noise PSD (2.0x accounts for 86th-percentile Exponential FFT bin variance)
        oversub = 2.0 * self.suppression_strength if self.suppression_strength > 0.0 else 0.0
        eff_noise = self.noise_psd * (oversub * self.freq_oversub_weight)
        eff_noise = np.maximum(eff_noise, eps)

        # Ephraim-Malah Decision-Directed a priori SNR estimation
        gamma = power / eff_noise  # a posteriori SNR
        dd_alpha = 0.95
        xi = dd_alpha * (self._prev_clean_psd / eff_noise) + (1.0 - dd_alpha) * np.maximum(
            gamma - 1.0, 0.0
        )

        # Wiener-type gain curve with perceptual exponent & spectral floor
        raw_gain = (xi / (1.0 + xi)) ** 0.85
        raw_gain = np.clip(raw_gain, self.gain_floor, 1.0).astype(np.float32)

        # 3-bin frequency smoothing to suppress isolated musical-noise tonal artifacts
        if self.n_bins >= 3:
            smoothed_gain = raw_gain.copy()
            smoothed_gain[1:-1] = (
                0.25 * raw_gain[:-2] + 0.50 * raw_gain[1:-1] + 0.25 * raw_gain[2:]
            )
        else:
            smoothed_gain = raw_gain

        # Asymmetric temporal gain smoothing (instant attack for consonants, smooth decay)
        gain = np.where(
            smoothed_gain > self._prev_gain,
            0.75 * smoothed_gain + 0.25 * self._prev_gain,
            0.45 * smoothed_gain + 0.55 * self._prev_gain,
        ).astype(np.float32)
        self._prev_gain = gain

        clean_spec = spec * gain
        self._prev_clean_psd = (
            clean_spec.real * clean_spec.real + clean_spec.imag * clean_spec.imag
        ).astype(np.float32)

        mean_speech_gain = float(np.mean(gain[self.speech_mask]))
        self.last_reduction_db = float(-20.0 * np.log10(max(mean_speech_gain, 1e-4)))

        synth = np.fft.irfft(clean_spec, n=self.win_size).astype(np.float32) * self.window
        out_hop = self._ola_tail + synth[:H]
        self._ola_tail = synth[H:].copy()
        return out_hop

    def process(self, block: np.ndarray) -> np.ndarray:
        """Process a 1D float32 audio block of ANY length and return exact same length."""
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        n_samples = mono.size
        if n_samples == 0:
            return mono.copy()

        if self._in_fifo.size == 0:
            work = mono
        else:
            work = np.concatenate((self._in_fifo, mono))

        n_hops = work.size // self.hop_size
        produced: list[np.ndarray] = [self._out_fifo]
        for idx in range(n_hops):
            start = idx * self.hop_size
            hop = work[start : start + self.hop_size]
            produced.append(self._process_hop(hop))

        rem_start = n_hops * self.hop_size
        self._in_fifo = work[rem_start:].copy()

        all_out = np.concatenate(produced) if len(produced) > 1 else produced[0]
        result = all_out[:n_samples].copy()
        self._out_fifo = all_out[n_samples:].copy()
        return result


def analyze_microphone_noise(
    silence_audio: np.ndarray,
    sample_rate: int,
    speech_audio: np.ndarray | None = None,
) -> MicDiagnostics:
    """
    Analyze recorded ambient noise (and optional speech) from the user's microphone
    to detect 50/60 Hz mains hum, hiss level, noise floor, and optimal parameters.
    """
    eps = 1e-12
    mono = np.asarray(silence_audio, dtype=np.float64).reshape(-1)
    if mono.size == 0:
        mono = np.zeros(sample_rate, dtype=np.float64)

    # Remove DC offset before measuring
    mono = mono - np.mean(mono)
    rms = float(np.sqrt(np.mean(mono * mono) + eps))
    peak = float(np.max(np.abs(mono)) + eps)
    rms_db = float(20.0 * np.log10(rms))
    peak_db = float(20.0 * np.log10(peak))

    # High-resolution Welch PSD to detect narrow 50 Hz / 60 Hz mains hum
    nperseg = min(max(1024, sample_rate // 2), max(mono.size, 1024))
    if mono.size < nperseg:
        padded = np.pad(mono, (0, nperseg - mono.size))
    else:
        padded = mono

    win = np.hanning(nperseg)
    n_seg = max(1, (padded.size - nperseg // 2) // max(1, nperseg // 2))
    psd_acc = np.zeros(nperseg // 2 + 1, dtype=np.float64)
    step = max(1, nperseg // 2)
    count = 0
    for start in range(0, padded.size - nperseg + 1, step):
        seg = padded[start : start + nperseg] * win
        sp = np.fft.rfft(seg)
        psd_acc += sp.real * sp.real + sp.imag * sp.imag
        count += 1
    psd = psd_acc / max(count, 1)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)

    def _hum_prominence(base_hz: float) -> float:
        prominences = []
        for h in (1, 2, 3):
            f0 = base_hz * h
            if f0 >= 0.45 * sample_rate:
                break
            tone_mask = np.abs(freqs - f0) <= 3.0
            side_mask = (np.abs(freqs - f0) > 5.0) & (np.abs(freqs - f0) <= 22.0)
            if np.any(tone_mask) and np.any(side_mask):
                tone_pwr = float(np.max(psd[tone_mask])) + eps
                side_pwr = float(np.median(psd[side_mask])) + eps
                prominences.append(10.0 * np.log10(tone_pwr / side_pwr))
        return max(prominences) if prominences else 0.0

    prom_50 = _hum_prominence(50.0)
    prom_60 = _hum_prominence(60.0)
    best_prom = max(prom_50, prom_60)
    if best_prom >= 8.0:
        detected_hum: float | None = 50.0 if prom_50 >= prom_60 else 60.0
    else:
        detected_hum = None

    # High-frequency hiss vs mid-band noise ratio
    mid_mask = (freqs >= 300.0) & (freqs <= 3000.0)
    hf_mask = (freqs >= 5500.0) & (freqs <= min(14000.0, 0.45 * sample_rate))
    if np.any(mid_mask) and np.any(hf_mask):
        mid_db = 10.0 * np.log10(float(np.mean(psd[mid_mask])) + eps)
        hf_db = 10.0 * np.log10(float(np.mean(psd[hf_mask])) + eps)
        hf_hiss_ratio_db = float(hf_db - mid_db)
    else:
        hf_hiss_ratio_db = -10.0

    # Recommend input pre-gain if speech snippet was provided
    recommended_gain_db = 0.0
    speech_rms_db: float | None = None
    if speech_audio is not None and speech_audio.size > 0:
        sp_mono = np.asarray(speech_audio, dtype=np.float64).reshape(-1)
        sp_mono = sp_mono - np.mean(sp_mono)
        # Measure active top-35% energy frames so pauses don't bias speech RMS downward
        frame_len = max(160, sample_rate // 50)
        n_f = sp_mono.size // frame_len
        if n_f >= 3:
            frames = sp_mono[: n_f * frame_len].reshape(n_f, frame_len)
            f_rms = np.sqrt(np.mean(frames * frames, axis=1) + eps)
            cutoff = np.percentile(f_rms, 65)
            active_rms = float(np.mean(f_rms[f_rms >= cutoff]))
        else:
            active_rms = float(np.sqrt(np.mean(sp_mono * sp_mono) + eps))
        speech_rms_db = float(20.0 * np.log10(active_rms + eps))
        # Target comfortable pre-compressor speech level around -20 dBFS
        if speech_rms_db < -24.0:
            recommended_gain_db = float(np.clip(-20.0 - speech_rms_db, 0.0, 18.0))
        elif speech_rms_db > -10.0:
            recommended_gain_db = float(np.clip(-14.0 - speech_rms_db, -10.0, 0.0))

    effective_noise_db = rms_db + recommended_gain_db
    if effective_noise_db > -38.0 or best_prom >= 12.0:
        recommended_preset = "max_denoise"
    elif effective_noise_db < -54.0 and best_prom < 8.0:
        recommended_preset = "natural"
    else:
        recommended_preset = "balanced"

    # Place gate threshold safely above calibrated noise floor (+7 dB) while below speech
    gate_thresh = float(np.clip(effective_noise_db + 7.5, -54.0, -28.0))
    if speech_rms_db is not None:
        eff_speech_db = speech_rms_db + recommended_gain_db
        gate_thresh = float(min(gate_thresh, eff_speech_db - 8.0))
        gate_thresh = float(np.clip(gate_thresh, -56.0, -26.0))

    return MicDiagnostics(
        sample_rate=sample_rate,
        rms_db=round(rms_db, 1),
        peak_db=round(peak_db, 1),
        detected_hum_hz=detected_hum,
        hum_prominence_db=round(best_prom, 1),
        hf_hiss_ratio_db=round(hf_hiss_ratio_db, 1),
        recommended_preset=recommended_preset,
        recommended_input_gain_db=round(recommended_gain_db, 1),
        recommended_gate_threshold_db=round(gate_thresh, 1),
    )
