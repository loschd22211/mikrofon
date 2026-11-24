from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass
class DenoiserConfig:
    """Configuration for the real-time microphone noise suppression pipeline."""

    preset: str = "balanced"
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 48000
    frame_ms: int = 10

    # Pre-gain (useful for quiet cheap 3.5mm / USB mics)
    input_gain_db: float = 0.0

    # Stage 1 & 2: Rumble high-pass & electrical mains hum notch (50/60 Hz)
    highpass_hz: float = 85.0
    hum_filter_hz: float | None = 50.0

    # Stage 3: Adaptive Ephraim-Malah spectral noise suppression
    suppression_strength: float = 1.8
    spectral_floor_db: float = -28.0
    vad_snr_threshold_db: float = 4.5

    # Stage 4: Smooth voice-aware noise gate / downward expander
    gate_enabled: bool = True
    gate_threshold_db: float = -44.0
    gate_attenuation_db: float = -22.0
    gate_hold_ms: float = 160.0
    gate_release_ms: float = 150.0

    # Stage 5: Cheap-mic -> studio voice clarity EQ + HF hiss cut
    voice_eq_enabled: bool = True
    mud_cut_db: float = -2.5
    presence_boost_db: float = 3.0
    lowpass_hz: float = 13500.0

    # Stage 6: Broadcast compressor & brickwall soft limiter
    compressor_enabled: bool = True
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 3.0
    makeup_gain_db: float = 3.5

    # Optional diagnostic metadata saved by `calibrate`
    calibrated_mic_name: str | None = None
    calibrated_noise_rms_db: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "DenoiserConfig":
        target = Path(path)
        if not target.exists():
            return preset_config("balanced")
        raw = json.loads(target.read_text(encoding="utf-8"))
        preset_name = raw.get("preset", "balanced")
        base = preset_config(preset_name) if preset_name in PRESETS else cls()
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return replace(base, **filtered)


PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "preset": "balanced",
        "highpass_hz": 85.0,
        "hum_filter_hz": 50.0,
        "suppression_strength": 1.8,
        "spectral_floor_db": -28.0,
        "gate_enabled": True,
        "gate_threshold_db": -44.0,
        "gate_attenuation_db": -22.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -2.5,
        "presence_boost_db": 3.0,
        "lowpass_hz": 13500.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 3.0,
        "makeup_gain_db": 3.5,
    },
    "max_denoise": {
        "preset": "max_denoise",
        "highpass_hz": 95.0,
        "hum_filter_hz": 50.0,
        "suppression_strength": 2.8,
        "spectral_floor_db": -36.0,
        "gate_enabled": True,
        "gate_threshold_db": -40.0,
        "gate_attenuation_db": -32.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -3.0,
        "presence_boost_db": 3.5,
        "lowpass_hz": 12000.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 3.5,
        "makeup_gain_db": 4.0,
    },
    "natural": {
        "preset": "natural",
        "highpass_hz": 75.0,
        "hum_filter_hz": 50.0,
        "suppression_strength": 1.2,
        "spectral_floor_db": -20.0,
        "gate_enabled": True,
        "gate_threshold_db": -48.0,
        "gate_attenuation_db": -14.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -1.5,
        "presence_boost_db": 2.0,
        "lowpass_hz": 15000.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -16.0,
        "compressor_ratio": 2.2,
        "makeup_gain_db": 2.5,
    },
    "gaming": {
        "preset": "gaming",
        "highpass_hz": 90.0,
        "hum_filter_hz": 50.0,
        "suppression_strength": 2.2,
        "spectral_floor_db": -32.0,
        "gate_enabled": True,
        "gate_threshold_db": -41.0,
        "gate_attenuation_db": -28.0,
        "gate_hold_ms": 140.0,
        "gate_release_ms": 110.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -3.0,
        "presence_boost_db": 4.0,
        "lowpass_hz": 12500.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -20.0,
        "compressor_ratio": 4.0,
        "makeup_gain_db": 5.0,
    },
}

PRESET_DESCRIPTIONS_RU: dict[str, str] = {
    "balanced": "Оптимальный баланс: убирает шум ПК/шипение и улучшает голос без роботизации (по умолчанию)",
    "max_denoise": "Максимальное шумоподавление: для очень шумной комнаты, гудящих кулеров и дешёвых 3.5мм микрофонов",
    "natural": "Мягкая очистка: минимум вмешательства, естественный тембр голоса",
    "gaming": "Для игр и Discord: чёткая разборчивость речи, подавление кликов клавиатуры и выравнивание громкости",
}


def preset_config(name: str) -> DenoiserConfig:
    if name not in PRESETS:
        valid = ", ".join(PRESETS)
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
    return DenoiserConfig(**PRESETS[name])
