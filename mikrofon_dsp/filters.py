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

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


def design_highshelf_biquad_sos(
    freq_hz: float,
    gain_db: float,
    sample_rate: int,
    slope: float = 1.0,
) -> np.ndarray | None:
    """Design an RBJ Audio EQ Cookbook high-shelf filter ('Air' boost) as SOS (shape 1x6)."""
    if abs(gain_db) < 1e-3 or freq_hz <= 100.0 or freq_hz >= 0.46 * sample_rate:
        return None

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = (sin_w0 / 2.0) * math.sqrt(max(0.01, (a_amp + 1.0 / a_amp) * (1.0 / slope - 1.0) + 2.0))
    two_sqrt_a_alpha = 2.0 * math.sqrt(a_amp) * alpha

    b0 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0)
    b2 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0)
    a2 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


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
    highpass_hz: float = 95.0,
    hum_filter_hz: float | None = None,
) -> StreamingSOSFilter:
    """
    Build stage 1+2 prefilter: clean rumble high-pass + optional fundamental mains notch.
    Avoids placing notches on 100/150/200 Hz voice harmonics so voice never sounds hollow.
    """
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

    if hum_filter_hz and 40.0 <= hum_filter_hz < nyquist_limit:
        # Notch ONLY the sub-vocal mains fundamental (50 or 60 Hz) to prevent phase ringing in speech
        b, a = signal.iirnotch(hum_filter_hz, Q=30.0, fs=sample_rate)
        sections.append(signal.tf2sos(b, a))

    return StreamingSOSFilter(sections)


def build_voice_eq(
    sample_rate: int,
    mud_cut_db: float = -4.5,
    honk_cut_db: float = -2.0,
    presence_boost_db: float = 3.5,
    air_boost_db: float = 2.5,
    lowpass_hz: float = 16500.0,
) -> StreamingSOSFilter:
    """
    Build studio 'Anti-Barrel' voice clarity EQ tailored for budget desk & headset mics:
      - 380 Hz wide cut (-4.5 dB): eliminates 'звук как в бочке' (hollow room & plastic housing resonance)
      - 680 Hz cut (-2.0 dB): removes tinny/can-like honk
      - 3800 Hz boost (+3.5 dB): crisp close-mic speech presence
      - 8200 Hz high-shelf (+2.5 dB): open studio 'Air' so voice sounds bright and natural
    """
    sections: list[np.ndarray] = []

    barrel_sos = design_peaking_biquad_sos(
        freq_hz=380.0,
        gain_db=mud_cut_db,
        q=1.0,
        sample_rate=sample_rate,
    )
    if barrel_sos is not None:
        sections.append(barrel_sos)

    honk_sos = design_peaking_biquad_sos(
        freq_hz=680.0,
        gain_db=honk_cut_db,
        q=1.4,
        sample_rate=sample_rate,
    )
    if honk_sos is not None:
        sections.append(honk_sos)

    presence_sos = design_peaking_biquad_sos(
        freq_hz=3800.0,
        gain_db=presence_boost_db,
        q=0.9,
        sample_rate=sample_rate,
    )
    if presence_sos is not None:
        sections.append(presence_sos)

    air_sos = design_highshelf_biquad_sos(
        freq_hz=8200.0,
        gain_db=air_boost_db,
        sample_rate=sample_rate,
    )
    if air_sos is not None:
        sections.append(air_sos)

    if lowpass_hz and 5000.0 < lowpass_hz < 0.46 * sample_rate:
        lp_sos = signal.butter(
            2,
            lowpass_hz,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        sections.append(lp_sos)

    return StreamingSOSFilter(sections)
