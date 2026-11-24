from __future__ import annotations

import math
import numpy as np


class SpectralDereverb:
    """
    Real-time Lebart/Habets late-reverberation suppressor ("Анти-бочка").

    Attenuates hollow room reflections and desk-bounce tails (especially in the
    180–1500 Hz "barrel" zone) while keeping direct close-mic speech transients crisp.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_ms: int = 10,
        strength: float = 0.55,
        rt60_sec: float = 0.26,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_size = max(64, int(round(self.sample_rate * frame_ms / 1000.0)))
        self.win_size = 2 * self.hop_size
        self.n_bins = self.win_size // 2 + 1
        self.strength = float(np.clip(strength, 0.0, 1.5))

        dt_sec = self.hop_size / float(self.sample_rate)
        # Exponential energy decay rate corresponding to RT60 (-60 dB decay time)
        self.decay = float(math.exp(-2.0 * dt_sec * math.log(1000.0) / max(rt60_sec, 0.08)))
        self.delay_frames = max(2, int(round(0.030 / dt_sec)))  # ~30 ms early-to-late boundary

        idx = np.arange(self.win_size, dtype=np.float64)
        self.window = np.sin(np.pi * (idx + 0.5) / self.win_size).astype(np.float32)

        freqs = np.fft.rfftfreq(self.win_size, d=1.0 / self.sample_rate)
        # Focus dereverberation on 180..1500 Hz room modes without attenuating direct vowels
        band_profile = np.full(self.n_bins, 0.08, dtype=np.float32)
        barrel_zone = (freqs >= 160.0) & (freqs <= 1600.0)
        mid_zone = (freqs > 1600.0) & (freqs <= 4000.0)
        band_profile[barrel_zone] = 0.24
        band_profile[mid_zone] = 0.14
        self.reverb_scale = (band_profile * self.strength).astype(np.float32)

        self.reset()

    def reset(self) -> None:
        self._analysis_frame = np.zeros(self.win_size, dtype=np.float32)
        self._ola_tail = np.zeros(self.hop_size, dtype=np.float32)
        self._in_fifo = np.zeros(0, dtype=np.float32)
        self._out_fifo = np.zeros(self.hop_size, dtype=np.float32)

        self._smooth_psd = np.zeros(self.n_bins, dtype=np.float32)
        self._late_reverb_psd = np.zeros(self.n_bins, dtype=np.float32)
        self._psd_history = np.zeros((self.delay_frames, self.n_bins), dtype=np.float32)
        self._hist_idx = 0
        self._prev_gain = np.ones(self.n_bins, dtype=np.float32)

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        H = self.hop_size
        self._analysis_frame[:H] = self._analysis_frame[H:]
        self._analysis_frame[H:] = hop

        spec = np.fft.rfft(self._analysis_frame * self.window)
        if self.strength <= 1e-3:
            synth = np.fft.irfft(spec, n=self.win_size).astype(np.float32) * self.window
            out = self._ola_tail + synth[:H]
            self._ola_tail = synth[H:].copy()
            return out

        power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
        self._smooth_psd = 0.55 * self._smooth_psd + 0.45 * power

        delayed_psd = self._psd_history[self._hist_idx]
        self._psd_history[self._hist_idx] = self._smooth_psd
        self._hist_idx = (self._hist_idx + 1) % self.delay_frames

        # Late reverberant energy model
        self._late_reverb_psd = (
            self.decay * self._late_reverb_psd
            + (1.0 - self.decay) * (self.reverb_scale * delayed_psd)
        )

        eps = 1e-12
        gain = power / (power + self._late_reverb_psd + eps)
        gain = np.clip(gain, 0.22, 1.0).astype(np.float32)

        if self.n_bins >= 3:
            sm = gain.copy()
            sm[1:-1] = 0.25 * gain[:-2] + 0.50 * gain[1:-1] + 0.25 * gain[2:]
            gain = sm

        # Instant attack on direct voice transients, smooth release on tails
        gain = np.where(
            gain > self._prev_gain,
            0.85 * gain + 0.15 * self._prev_gain,
            0.55 * gain + 0.45 * self._prev_gain,
        ).astype(np.float32)
        self._prev_gain = gain

        clean_spec = spec * gain
        synth = np.fft.irfft(clean_spec, n=self.win_size).astype(np.float32) * self.window
        out_hop = self._ola_tail + synth[:H]
        self._ola_tail = synth[H:].copy()
        return out_hop

    def process(self, block: np.ndarray) -> np.ndarray:
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        n = mono.size
        if n == 0:
            return mono.copy()

        # Zero-allocation fast path when callback block equals exact hop_size
        if n == self.hop_size and self._in_fifo.size == 0 and self._out_fifo.size == self.hop_size:
            prev_out = self._out_fifo
            self._out_fifo = self._process_hop(mono)
            return prev_out

        work = mono if self._in_fifo.size == 0 else np.concatenate((self._in_fifo, mono))
        n_hops = work.size // self.hop_size
        produced: list[np.ndarray] = [self._out_fifo]
        for idx in range(n_hops):
            start = idx * self.hop_size
            produced.append(self._process_hop(work[start : start + self.hop_size]))

        self._in_fifo = work[n_hops * self.hop_size :].copy()
        all_out = np.concatenate(produced) if len(produced) > 1 else produced[0]
        result = all_out[:n].copy()
        self._out_fifo = all_out[n:].copy()
        return result
