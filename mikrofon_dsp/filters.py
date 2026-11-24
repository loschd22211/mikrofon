from __future__ import annotations

import math
import numpy as np
from scipy import signal


def design_peaking_biquad_sos(
    freq_hz: float,
    gain_db: float,
    q: float,
    sample_rate: int,
) -> np.ndarray | None:
    """Design a single RBJ Audio EQ Cookbook peaking EQ filter as SOS (shape 1x6)."""
    if abs(gain_db) < 1e-3 or freq_hz <= 10.0 or freq_hz >= 0.48 * sample_rate:
        return None

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sample_rate
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / a_amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_amp

    sos = np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )
    return sos


class StreamingSOSFilter:
    """Stateful streaming IIR filter backed by cascaded second-order sections."""

    def __init__(self, sos_sections: list[np.ndarray]) -> None:
        valid = [s for s in sos_sections if s is not None and s.size > 0]
        if valid:
            self.sos: np.ndarray | None = np.vstack(valid).astype(np.float64)
            self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)
        else:
            self.sos = None
            self._zi = np.empty((0, 2), dtype=np.float64)

    def reset(self) -> None:
        if self.sos is not None:
            self._zi.fill(0.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.sos is None or block.size == 0:
            return block.astype(np.float32, copy=True)
        out, self._zi = signal.sosfilt(
            self.sos,
            block.astype(np.float64, copy=False),
            zi=self._zi,
        )
        return out.astype(np.float32)


def build_prefilter(
    sample_rate: int,
    highpass_hz: float = 85.0,
    hum_filter_hz: float | None = 50.0,
) -> StreamingSOSFilter:
    """Build stage 1+2 prefilter: rumble high-pass + mains hum harmonic notches."""
    sections: list[np.ndarray] = []
    nyquist_limit = 0.48 * sample_rate

    if highpass_hz and 15.0 <= highpass_hz < nyquist_limit:
        hp_sos = signal.butter(
            4,
            highpass_hz,
            btype="highpass",
            fs=sample_rate,
            output="sos",
        )
        sections.append(hp_sos)

    if hum_filter_hz and hum_filter_hz >= 40.0:
        # Attenuate fundamental (50/60 Hz) and first 3 harmonics (100/150/200 or 120/180/240 Hz)
        # where motherboard / cheap 3.5mm ground-loop buzz concentrates.
        for harmonic in (1, 2, 3, 4):
            f_notch = hum_filter_hz * harmonic
            if f_notch >= nyquist_limit:
                break
            b, a = signal.iirnotch(f_notch, Q=30.0, fs=sample_rate)
            notch_sos = signal.tf2sos(b, a)
            sections.append(notch_sos)

    return StreamingSOSFilter(sections)


def build_voice_eq(
    sample_rate: int,
    mud_cut_db: float = -2.5,
    presence_boost_db: float = 3.0,
    lowpass_hz: float = 13500.0,
) -> StreamingSOSFilter:
    """Build stage 5 voice clarity EQ tailored for budget microphones."""
    sections: list[np.ndarray] = []

    # 1. Cut boxy/muffled resonance typical of cheap plastic headset/desktop housings
    mud_sos = design_peaking_biquad_sos(
        freq_hz=280.0,
        gain_db=mud_cut_db,
        q=1.2,
        sample_rate=sample_rate,
    )
    if mud_sos is not None:
        sections.append(mud_sos)

    # 2. Boost speech intelligibility & consonant clarity
    presence_sos = design_peaking_biquad_sos(
        freq_hz=3400.0,
        gain_db=presence_boost_db,
        q=1.0,
        sample_rate=sample_rate,
    )
    if presence_sos is not None:
        sections.append(presence_sos)

    # 3. Roll off ultrasonic / high-frequency ADC white hiss where voice has little energy
    if lowpass_hz and 3500.0 < lowpass_hz < 0.48 * sample_rate:
        lp_sos = signal.butter(
            2,
            lowpass_hz,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        sections.append(lp_sos)

    return StreamingSOSFilter(sections)
