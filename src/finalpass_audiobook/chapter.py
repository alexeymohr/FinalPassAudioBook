"""One chapter file, loaded once, with the measurements every check shares.

Numbers only: levels, envelopes and spans. Nothing here listens to or
interprets the words.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from finalpass.audio_io import AudioFile, read_wav
from finalpass.breath_features import Frames, compute
from finalpass.breath_detect import BreathParams, narration_level
from finalpass.timecode import samples_to_clock
from scipy.signal import resample_poly

DUAL_MONO_TOLERANCE = 4 / 32768   # a few 16-bit steps: per-channel dither still counts as dual mono
ENV_BIN_MS = 1.0          # fine envelope hop
BLOCK = 1 << 22           # samples per block when squaring the whole file
LOW_RATE_HZ = 4400.0      # decimated signal for low-frequency checks
LOCAL_SPEECH_PAD_S = 10.0 # speech level around a finding: this far either side
LOCAL_SPEECH_MIN_FRAMES = 20


class ChapterError(ValueError):
    """The file cannot be analysed as a narration chapter."""


def to_mono(audio: AudioFile) -> tuple[np.ndarray, list[str]]:
    data = audio.data
    if data.shape[0] == 0:
        raise ChapterError(f"{audio.path.name}: empty file (no samples)")
    if data.shape[1] == 1:
        return data[:, 0], []
    finite = np.nan_to_num(data)
    if float(np.max(np.abs(finite - finite[:, :1]))) <= DUAL_MONO_TOLERANCE:
        return data[:, 0], [f"{data.shape[1]} identical channels: analysed the first"]
    raise ChapterError(f"{audio.path.name}: needs mono narration (got {data.shape[1]} channels that differ)")


def dbfs(power: np.ndarray | float) -> np.ndarray | float:
    """Mean-square power to dBFS (full scale = 1.0)."""
    return 10.0 * np.log10(np.maximum(power, 1e-20))


@dataclass
class Chapter:
    path: Path
    audio: AudioFile
    x: np.ndarray
    sr: int
    notes: list[str] = field(default_factory=list)
    invalid_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def load(cls, path: Path) -> "Chapter":
        if not path.is_file():
            raise ChapterError(f"{path.name}: file not found")
        audio = read_wav(path)
        x, notes = to_mono(audio)
        bad = np.flatnonzero(~np.isfinite(x))
        if bad.size:                    # one NaN would otherwise silently disable most checks
            x = np.where(np.isfinite(x), x, 0.0)
            notes.append(f"{bad.size} invalid (NaN/Inf) samples set to zero for analysis")
        return cls(path=path, audio=audio, x=x, sr=audio.sample_rate, notes=notes, invalid_samples=bad)

    @property
    def mono_audio(self) -> AudioFile:
        """This chapter as a one-channel AudioFile (what FinalPass's breath check is given)."""
        return AudioFile(path=self.path, data=self.x[:, None], sample_rate=self.sr, bit_depth=self.audio.bit_depth,
                         channel_count=1, duration_seconds=len(self.x) / self.sr)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def duration_s(self) -> float:
        return len(self.x) / self.sr

    def clock(self, sample: int) -> str:
        return samples_to_clock(int(sample), self.sr)

    # --- shared measurements, computed on first use -------------------------

    @cached_property
    def frames(self) -> Frames:
        return compute(self.x, self.sr)

    @cached_property
    def narration_dbfs(self) -> float:
        return narration_level(self.frames)

    @cached_property
    def _speech_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Start sample and level of every pitched narration frame (as `narration_level` counts them)."""
        f = self.frames
        loud = (~f.zero) & (f.voicing > BreathParams().speech_voicing) & (f.rms_db > -50)
        idx = np.flatnonzero(loud)
        return idx * f.hop, f.rms_db[idx]

    def local_speech_dbfs(self, start: int, end: int, pad_s: float = LOCAL_SPEECH_PAD_S) -> float:
        """Median level of the narration within `pad_s` of a span; the chapter's level if too little speech."""
        at, level = self._speech_frames
        pad = int(pad_s * self.sr)
        lo, hi = np.searchsorted(at, start - pad), np.searchsorted(at, end + pad)
        return float(np.median(level[lo:hi])) if hi - lo >= LOCAL_SPEECH_MIN_FRAMES else self.narration_dbfs

    @cached_property
    def bin_samples(self) -> int:
        return max(1, int(round(ENV_BIN_MS * self.sr / 1000.0)))

    @cached_property
    def bin_energy(self) -> np.ndarray:
        """Sum of squares in consecutive ``bin_samples`` blocks (~1 ms)."""
        h = self.bin_samples
        n = len(self.x) // h
        out = np.empty(n)
        step = (BLOCK // h) * h
        for s in range(0, n * h, step):
            e = min(n * h, s + step)
            seg = self.x[s:e]
            out[s // h:e // h] = np.einsum("ij,ij->i", seg.reshape(-1, h), seg.reshape(-1, h))
        return out

    def envelope_db(self, window_bins: int) -> np.ndarray:
        """RMS (dBFS) over ``window_bins`` bins, one value per bin start."""
        c = np.concatenate(([0.0], np.cumsum(self.bin_energy)))
        w = max(1, int(window_bins))
        ms = (c[w:] - c[:-w]) / (w * self.bin_samples)
        return dbfs(np.concatenate([ms, np.full(w - 1, ms[-1] if ms.size else 0.0)]))

    @cached_property
    def low(self) -> tuple[np.ndarray, float]:
        """The signal decimated to about 4.4 kHz, for checks below 1 kHz."""
        q = max(1, int(self.sr // LOW_RATE_HZ))
        return (resample_poly(self.x, 1, q) if q > 1 else self.x.copy()), self.sr / q

    def bin_to_sample(self, i: int) -> int:
        return int(i) * self.bin_samples
