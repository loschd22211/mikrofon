from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import platform
from typing import Any

import numpy as np
from scipy import signal


RNNOISE_SAMPLE_RATE = 48000
RNNOISE_FRAME_SIZE = 480  # 10 ms at 48 kHz


def _load_rnnoise_cdll() -> Any | None:
    """
    Locate and load the compiled C RNNoise v0.2 shared library bundled inside `pyrnnoise`
    without importing `pyrnnoise/__init__.py` (avoids heavy CLI-only audiolab/matplotlib deps).
    """
    spec = importlib.util.find_spec("pyrnnoise")
    if spec is None or not spec.submodule_search_locations:
        return None

    pkg_dir = spec.submodule_search_locations[0]
    sys_name = platform.system()
    if sys_name == "Windows":
        lib_name = "rnnoise.dll"
    elif sys_name == "Darwin":
        lib_name = "librnnoise.dylib"
    else:
        lib_name = "librnnoise.so"

    lib_path = os.path.join(pkg_dir, lib_name)
    if not os.path.exists(lib_path):
        return None

    try:
        lib = ctypes.CDLL(lib_path)
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.restype = None
        lib.rnnoise_get_frame_size.argtypes = []
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
        lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        return lib
    except Exception:
        return None


_RNNOISE_LIB = _load_rnnoise_cdll()


def is_neural_rnnoise_available() -> bool:
    return _RNNOISE_LIB is not None


class NeuralRNNoiseDenoiser:
    """
    Streaming 48 kHz Xiph RNNoise v0.2 recurrent neural network noise suppressor.

    Suppresses both stationary noise (hiss, PC fan hum) and non-stationary noises
    (squeaky chairs, keyboard clicks, mouse clicks, desk knocks, coughs) in real time
    using pure float32 precision and arbitrary callback block sizes.
    """

    def __init__(self, sample_rate: int = 48000, wet_mix: float = 1.0) -> None:
        if _RNNOISE_LIB is None:
            raise RuntimeError(
                "Модуль нейросетевого шумоподавления `pyrnnoise` не установлен. "
                "Установите его командой: pip install --no-deps pyrnnoise"
            )
        self._lib = _RNNOISE_LIB
        self.sample_rate = int(sample_rate)
        self.wet_mix = float(np.clip(wet_mix, 0.0, 1.0))
        self.frame_size = int(self._lib.rnnoise_get_frame_size())  # 480 @ 48 kHz

        # Compute rational resampling ratio if microphone runs at e.g. 44100 or 16000 Hz
        g = math.gcd(self.sample_rate, RNNOISE_SAMPLE_RATE)
        self._up = RNNOISE_SAMPLE_RATE // g
        self._down = self.sample_rate // g
        self._needs_resample = self.sample_rate != RNNOISE_SAMPLE_RATE

        self._state: ctypes.c_void_p | None = self._lib.rnnoise_create(None)
        self._c_buf = np.zeros(self.frame_size, dtype=np.float32)
        self._c_ptr = self._c_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.last_speech_prob: float = 0.0
        self.last_reduction_db: float = 0.0
        self.reset_buffers()

    def reset_buffers(self) -> None:
        self._in_fifo = np.zeros(0, dtype=np.float32)
        self._out_fifo = np.zeros(self.frame_size, dtype=np.float32)

    def reset(self) -> None:
        if self._state is not None and self._lib is not None:
            self._lib.rnnoise_destroy(self._state)
        self._state = self._lib.rnnoise_create(None)
        self.last_speech_prob = 0.0
        self.last_reduction_db = 0.0
        self.reset_buffers()

    def close(self) -> None:
        if getattr(self, "_state", None) is not None and getattr(self, "_lib", None) is not None:
            self._lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _process_48k_hop(self, hop_48k: np.ndarray) -> np.ndarray:
        # RNNoise expects float32 scaled to [-32768.0, 32767.0]
        np.multiply(hop_48k, 32768.0, out=self._c_buf)
        in_energy = float(np.dot(hop_48k, hop_48k)) + 1e-12

        prob = float(self._lib.rnnoise_process_frame(self._state, self._c_ptr, self._c_ptr))
        self.last_speech_prob = prob

        clean_48k = (self._c_buf * (1.0 / 32768.0)).astype(np.float32)
        if self.wet_mix < 0.999:
            clean_48k = self.wet_mix * clean_48k + (1.0 - self.wet_mix) * hop_48k

        out_energy = float(np.dot(clean_48k, clean_48k)) + 1e-12
        red_db = max(0.0, 10.0 * math.log10(in_energy / out_energy))
        self.last_reduction_db = 0.7 * self.last_reduction_db + 0.3 * red_db
        return clean_48k

    def process(self, block: np.ndarray) -> np.ndarray:
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        n_orig = mono.size
        if n_orig == 0:
            return mono.copy()

        if not self._needs_resample and n_orig == self.frame_size and self._in_fifo.size == 0 and self._out_fifo.size == self.frame_size:
            prev_out = self._out_fifo
            self._out_fifo = self._process_48k_hop(mono)
            return prev_out

        if self._needs_resample:
            x_48k = signal.resample_poly(mono, self._up, self._down).astype(np.float32)
        else:
            x_48k = mono

        n_48k = x_48k.size
        work = x_48k if self._in_fifo.size == 0 else np.concatenate((self._in_fifo, x_48k))

        F = self.frame_size
        n_hops = work.size // F
        produced: list[np.ndarray] = [self._out_fifo]
        for idx in range(n_hops):
            start = idx * F
            produced.append(self._process_48k_hop(work[start : start + F]))

        self._in_fifo = work[n_hops * F :].copy()
        all_out = np.concatenate(produced) if len(produced) > 1 else produced[0]
        out_48k = all_out[:n_48k].copy()
        self._out_fifo = all_out[n_48k:].copy()

        if self._needs_resample:
            resampled = signal.resample_poly(out_48k, self._down, self._up).astype(np.float32)
            if resampled.size >= n_orig:
                return resampled[:n_orig]
            return np.pad(resampled, (0, n_orig - resampled.size))
        return out_48k
