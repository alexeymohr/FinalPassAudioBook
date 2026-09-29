"""Synthesized narration-like audio for tests. Never real recordings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from finalpass.audio_io import AudioFile
from scipy.signal import butter, sosfilt

from finalpass_audiobook.chapter import Chapter

SR = 44100
RNG = np.random.default_rng(20260928)


def word(seconds: float, f0: float = 140.0, glide: float = 25.0) -> np.ndarray:
    """A voiced "word": harmonics on a gliding pitch, with 15 ms fades."""
    t = np.arange(int(seconds * SR)) / SR
    f = f0 + glide * np.sin(2 * np.pi * t / max(seconds, 0.2))
    phase = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 14))
    x = 0.25 * x / np.max(np.abs(x))
    n = int(0.015 * SR)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def phrase(n_words: int = 4) -> np.ndarray:
    parts = []
    for i in range(n_words):
        parts += [word(0.32 + 0.05 * (i % 3), f0=120 + 12 * (i % 4)), room(0.06)]
    return np.concatenate(parts)


def room(seconds: float, dbfs: float = -72.0) -> np.ndarray:
    y = RNG.standard_normal(int(seconds * SR))
    return y * 10 ** (dbfs / 20)


def band_noise(seconds: float, lo: float, hi: float, dbfs: float) -> np.ndarray:
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), RNG.standard_normal(int(seconds * SR)))
    return y / np.sqrt(np.mean(y ** 2)) * 10 ** (dbfs / 20)


def chapter(x: np.ndarray, name: str = "chapter.wav") -> Chapter:
    data = x[:, None]
    audio = AudioFile(path=Path(name), data=data, sample_rate=SR, bit_depth=24, channel_count=1,
                      duration_seconds=len(x) / SR)
    return Chapter(path=Path(name), audio=audio, x=x, sr=SR)
