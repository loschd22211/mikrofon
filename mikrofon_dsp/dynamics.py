from __future__ import annotations

import math
import numpy as np


class SmoothNoiseGate:
    """
    Click-free downward expander / noise gate with Attack, Hold, and Release.
    Uses 2 ms micro-frames with sample-level linear gain ramps for zero zipper noise.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold_db: float = -44.0,
        attenuation_db: float = -22.0,
        attack_ms: float = 4.0,
        hold_ms: float = 160.0,
        release_ms: float = 150.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.sub_len = max(16, int(round(self.sample_rate * 0.002)))  # 2 ms micro-block
        dt_ms = 1000.0 * self.sub_len / self.sample_rate

        self.threshold_db = float(threshold_db)
        self.max_atten_db = abs(float(attenuation_db))
        self.min_gain = float(10.0 ** (-self.max_atten_db / 20.0))

        self.attack_coeff = float(math.exp(-dt_ms / max(attack_ms, 0.5)))
        self.release_coeff = float(math.exp(-dt_ms / max(release_ms, 5.0)))
        self.hold_steps = max(1, int(round(hold_ms / dt_ms)))

        self.reset()

    def reset(self) -> None:
        self._curr_gain: float = self.min_gain
        self._hold_left: int = 0

    def process(self, block: np.ndarray, speech_prob: float = 0.0) -> np.ndarray:
        if block.size == 0 or self.max_atten_db <= 0.1:
            return block.astype(np.float32, copy=True)

        out = np.empty_like(block, dtype=np.float32)
        n = block.size
        eps = 1e-12
        pos = 0

        while pos < n:
            end = min(pos + self.sub_len, n)
            seg = block[pos:end]
            rms = float(np.sqrt(np.mean(seg * seg) + eps))
            level_db = 20.0 * math.log10(rms + eps)

            is_active_voice = (
                (level_db >= (self.threshold_db - 3.0) and speech_prob >= 0.55)
                or (level_db >= self.threshold_db and speech_prob >= 0.25)
                or level_db >= (self.threshold_db + 10.0)
            )
            if is_active_voice:
                self._hold_left = self.hold_steps
                target_gain = 1.0
            elif self._hold_left > 0:
                self._hold_left -= 1
                target_gain = 1.0
            else:
                deficit_db = max(4.0, self.threshold_db + 6.0 - level_db)
                atten_db = min(self.max_atten_db, deficit_db * 1.8)
                target_gain = max(self.min_gain, 10.0 ** (-atten_db / 20.0))

            if target_gain > self._curr_gain:
                next_gain = (
                    self.attack_coeff * self._curr_gain
                    + (1.0 - self.attack_coeff) * target_gain
                )
            else:
                next_gain = (
                    self.release_coeff * self._curr_gain
                    + (1.0 - self.release_coeff) * target_gain
                )

            ramp = np.linspace(self._curr_gain, next_gain, end - pos, dtype=np.float32)
            out[pos:end] = seg * ramp
            self._curr_gain = next_gain
            pos = end

        return out


class VoiceCompressorLimiter:
    """
    Broadcast-style speech compressor + makeup gain + transparent soft-knee limiter.
    Evens out quiet speech vs loud shouts without ever allowing digital clipping.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold_db: float = -18.0,
        ratio: float = 3.0,
        makeup_gain_db: float = 3.5,
        attack_ms: float = 6.0,
        release_ms: float = 120.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.sub_len = max(16, int(round(self.sample_rate * 0.002)))
        dt_ms = 1000.0 * self.sub_len / self.sample_rate

        self.threshold_db = float(threshold_db)
        self.ratio = max(1.0, float(ratio))
        self.makeup_gain_db = float(makeup_gain_db)

        self.attack_coeff = float(math.exp(-dt_ms / max(attack_ms, 0.5)))
        self.release_coeff = float(math.exp(-dt_ms / max(release_ms, 5.0)))

        self.reset()

    def reset(self) -> None:
        self._gr_db: float = 0.0
        self._curr_lin_gain: float = float(10.0 ** (self.makeup_gain_db / 20.0))

    @staticmethod
    def soft_limit(samples: np.ndarray) -> np.ndarray:
        """
        Linear below |x| <= 0.82 (-1.7 dBFS), smooth tanh knee up to ceiling 0.98 (-0.17 dBFS).
        Guarantees zero distortion on normal voice and zero digital clipping on sudden peaks.
        """
        knee = 0.82
        ceiling = 0.98
        headroom = ceiling - knee
        abs_x = np.abs(samples)
        over_mask = abs_x > knee
        if not np.any(over_mask):
            return samples
        out = samples.copy()
        over = abs_x[over_mask] - knee
        compressed = knee + headroom * np.tanh(over / headroom)
        out[over_mask] = np.sign(samples[over_mask]) * compressed
        return out

    def process(self, block: np.ndarray, enabled: bool = True) -> np.ndarray:
        if block.size == 0:
            return block.astype(np.float32, copy=True)

        if not enabled:
            return self.soft_limit(block.astype(np.float32, copy=True))

        out = np.empty_like(block, dtype=np.float32)
        n = block.size
        eps = 1e-12
        slope = 1.0 - (1.0 / self.ratio)
        pos = 0

        while pos < n:
            end = min(pos + self.sub_len, n)
            seg = block[pos:end]
            rms = float(np.sqrt(np.mean(seg * seg) + eps))
            peak = float(np.max(np.abs(seg)) + eps)
            det = 0.65 * rms + 0.35 * peak
            det_db = 20.0 * math.log10(det + eps)

            if det_db > self.threshold_db:
                target_gr_db = (det_db - self.threshold_db) * slope
            else:
                target_gr_db = 0.0

            if target_gr_db > self._gr_db:
                self._gr_db = (
                    self.attack_coeff * self._gr_db
                    + (1.0 - self.attack_coeff) * target_gr_db
                )
            else:
                self._gr_db = (
                    self.release_coeff * self._gr_db
                    + (1.0 - self.release_coeff) * target_gr_db
                )

            next_lin_gain = 10.0 ** ((self.makeup_gain_db - self._gr_db) / 20.0)
            ramp = np.linspace(
                self._curr_lin_gain, next_lin_gain, end - pos, dtype=np.float32
            )
            out[pos:end] = seg * ramp
            self._curr_lin_gain = next_lin_gain
            pos = end

        return self.soft_limit(out)
