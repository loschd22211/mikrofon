from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass
class DenoiserConfig:
    """Configuration for the real-time microphone noise suppression & anti-barrel pipeline."""

    preset: str = "balanced"
    engine: str = "auto"  # "auto" (Neural RNNoise v0.2 if available), "rnnoise", or "spectral"
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 48000
    frame_ms: int = 10

    # Pre-gain (useful for quiet cheap 3.5mm / USB mics)
    input_gain_db: float = 0.0

    # Stage 1 & 2: Rumble high-pass & optional fundamental mains hum notch
    highpass_hz: float = 95.0
    hum_filter_hz: float | None = None

    # Stage 3: Neural RNNoise v0.2 + Adaptive spectral suppression fallback
    neural_wet_mix: float = 1.0
    suppression_strength: float = 1.8
    spectral_floor_db: float = -30.0
    vad_snr_threshold_db: float = 4.5

    # Stage 4: Real-time room dereverberation ("Анти-бочка" / room echo killer)
    dereverb_enabled: bool = True
    dereverb_strength: float = 0.65

    # Stage 5: Smooth voice-aware noise gate / downward expander
    gate_enabled: bool = True
    gate_threshold_db: float = -43.0
    gate_attenuation_db: float = -26.0
    gate_hold_ms: float = 150.0
    gate_release_ms: float = 130.0

    # Stage 6: Studio Anti-Barrel Voice EQ + Open Air boost
    voice_eq_enabled: bool = True
    mud_cut_db: float = -4.5
    honk_cut_db: float = -2.0
    presence_boost_db: float = 3.5
    air_boost_db: float = 2.5
    lowpass_hz: float = 16500.0

    # Stage 7: Broadcast compressor & brickwall soft limiter
    compressor_enabled: bool = True
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 3.0
    makeup_gain_db: float = 5.5

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
        "highpass_hz": 95.0,
        "hum_filter_hz": None,
        "neural_wet_mix": 1.0,
        "suppression_strength": 1.8,
        "spectral_floor_db": -30.0,
        "dereverb_enabled": True,
        "dereverb_strength": 0.65,
        "gate_enabled": True,
        "gate_threshold_db": -43.0,
        "gate_attenuation_db": -26.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -4.5,
        "honk_cut_db": -2.0,
        "presence_boost_db": 3.5,
        "air_boost_db": 2.5,
        "lowpass_hz": 16500.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 3.0,
        "makeup_gain_db": 5.5,
    },
    "max_denoise": {
        "preset": "max_denoise",
        "highpass_hz": 105.0,
        "hum_filter_hz": None,
        "neural_wet_mix": 1.0,
        "suppression_strength": 2.8,
        "spectral_floor_db": -38.0,
        "dereverb_enabled": True,
        "dereverb_strength": 0.85,
        "gate_enabled": True,
        "gate_threshold_db": -39.0,
        "gate_attenuation_db": -34.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -5.5,
        "honk_cut_db": -2.5,
        "presence_boost_db": 4.0,
        "air_boost_db": 2.5,
        "lowpass_hz": 15500.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 3.5,
        "makeup_gain_db": 6.0,
    },
    "anti_barrel": {
        "preset": "anti_barrel",
        "highpass_hz": 105.0,
        "hum_filter_hz": None,
        "neural_wet_mix": 1.0,
        "suppression_strength": 2.2,
        "spectral_floor_db": -32.0,
        "dereverb_enabled": True,
        "dereverb_strength": 1.05,
        "gate_enabled": True,
        "gate_threshold_db": -41.0,
        "gate_attenuation_db": -28.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -6.0,
        "honk_cut_db": -3.0,
        "presence_boost_db": 4.2,
        "air_boost_db": 3.2,
        "lowpass_hz": 17000.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 3.2,
        "makeup_gain_db": 6.5,
    },
    "natural": {
        "preset": "natural",
        "highpass_hz": 80.0,
        "hum_filter_hz": None,
        "neural_wet_mix": 0.92,
        "suppression_strength": 1.2,
        "spectral_floor_db": -22.0,
        "dereverb_enabled": True,
        "dereverb_strength": 0.40,
        "gate_enabled": True,
        "gate_threshold_db": -48.0,
        "gate_attenuation_db": -16.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -2.5,
        "honk_cut_db": -1.0,
        "presence_boost_db": 2.2,
        "air_boost_db": 1.5,
        "lowpass_hz": 17500.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -16.0,
        "compressor_ratio": 2.2,
        "makeup_gain_db": 2.5,
    },
    "gaming": {
        "preset": "gaming",
        "highpass_hz": 100.0,
        "hum_filter_hz": None,
        "neural_wet_mix": 1.0,
        "suppression_strength": 2.4,
        "spectral_floor_db": -34.0,
        "dereverb_enabled": True,
        "dereverb_strength": 0.75,
        "gate_enabled": True,
        "gate_threshold_db": -40.0,
        "gate_attenuation_db": -30.0,
        "gate_hold_ms": 130.0,
        "gate_release_ms": 100.0,
        "voice_eq_enabled": True,
        "mud_cut_db": -5.0,
        "honk_cut_db": -2.2,
        "presence_boost_db": 4.0,
        "air_boost_db": 2.5,
        "lowpass_hz": 16000.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -20.0,
        "compressor_ratio": 4.0,
        "makeup_gain_db": 4.5,
    },
}

PRESET_DESCRIPTIONS_RU: dict[str, str] = {
    "balanced": "ИИ-шумоподавление (RNNoise) + анти-бочка + студийный голос (по умолчанию)",
    "max_denoise": "Максимальное подавление скрипа стула, клавиатуры, кулеров и гула комнаты",
    "anti_barrel": "Усиленное удаление эффекта «звука из бочки» / эха пустой комнаты + открытый верх",
    "natural": "Мягкая деликатная очистка с максимально естественным тембром",
    "gaming": "Для игр, ГС и Discord: плотный близкий голос и жёсткое отсечение посторонних звуков",
}


def preset_config(name: str) -> DenoiserConfig:
    if name not in PRESETS:
        valid = ", ".join(PRESETS)
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
    return DenoiserConfig(**PRESETS[name])
