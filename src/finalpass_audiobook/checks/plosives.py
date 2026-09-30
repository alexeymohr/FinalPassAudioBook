"""Plosive pop: a low thump that stands on its own just before a word. Numbers only.

What the operator heard in the QC-noted pops (often breath, then pop, then word): a vertical
column of energy below 100 Hz, fading out by a few hundred hertz, sitting in the gap before the
word — often a stray pre-word artifact of a "b" or "p" (removing it leaves the word intact). A
normal p/b burst is part of the word's own onset: its high end arrives with it. So a pop is:

* a burst below 100 Hz (20-100 Hz, 5 ms RMS) at least -42 dBFS, at most 40 ms wide (within
  10 dB of its peak);
* low-dominated: at its peak the 20-100 Hz band stands >= 22 dB over the 500-8000 Hz band;
* before a word: the word's high end (loudest 500-8000 Hz level 30-250 ms later) is >= 20 dB
  above the high end at the pop, and reaches within 10 dB of that level 75-150 ms after it;
* over before the word: the low band falls >= 30 dB between the pop and the word's onset.

Severity by the pop's level below 100 Hz, in the whole dB the text shows: 1 from -42 dBFS, 2
from -34, 3 from -26. A pop overlapping a mouth-click inhale or within 200 ms after it belongs to
that breath (operator: the plosive is then a component of the T-inhale) and is not listed
separately.

Evidence (one title): 12 QC-noted pops; on them this lists 10 (the other two: one inside a
mouth-click inhale, one at -41 dBFS with little low-band dominance). A blind round of 32
candidates from a looser rule: 11 wanted (8 breath-then-plosive, 3 consonants with too much
plosive energy), 11 normal consonants, 10 with nothing; this rule lists 10 of the 11 and 1 of the
other 21. The limits were set on those same clips; held out, 32 listings never heard before were
29 plosives by ear (22 breath-then-plosive, 5 consonants with too much plosive energy, 2 inside a
word), 2 normal consonants and 1 nothing; the operator's "large"/"nasty" pops were severity 3.
About 11 per 15-minute chapter (6 / 4 / 1 at severity 1 / 2 / 3). The earlier rule (20-65 Hz, louder than the speech, never near a breath)
found none of the QC-noted pops.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt, sosfiltfilt

from ..chapter import Chapter
from ..findings import Finding, ladder


@dataclass(frozen=True)
class PlosiveTunables:
    low_band_hz: tuple[float, float] = (20.0, 100.0)
    high_band_hz: tuple[float, float] = (500.0, 8000.0)
    block_ms: float = 5.0
    min_dbfs: float = -42.0            # listed from here (severity 1)...
    sev2_dbfs: float = -34.0
    sev3_dbfs: float = -26.0
    one_per_ms: float = 50.0           # the loudest low-band peak within this, either side
    max_width_ms: float = 40.0
    low_over_high_db: float = 22.0
    word_over_pop_db: float = 20.0     # the word's high end over the high end at the pop
    word_from_ms: float = 30.0
    word_to_ms: float = 250.0
    word_within_db: float = 10.0       # the word has started when its high end is within this
    min_ms_to_word: float = 75.0
    max_ms_to_word: float = 150.0
    min_fall_db: float = 30.0          # the low band's fall between the pop and the word
    mouth_click_after_ms: float = 200.0

    def as_dict(self) -> dict:
        return asdict(self)


def _block_db(y: np.ndarray, n: int) -> np.ndarray:
    k = len(y) // n
    ms = np.mean(y[:k * n].reshape(k, n) ** 2, axis=1) if k else np.zeros(0)
    return 10 * np.log10(np.maximum(ms, 1e-20))


def plosive_findings(ch: Chapter, mouth_click_spans: tuple[tuple[int, int], ...] = (),
                     t: PlosiveTunables = PlosiveTunables()) -> list[Finding]:
    """`mouth_click_spans`: (start, end) samples of breaths listed as mouth-click inhales."""
    y, fs = ch.low
    q = max(1, int(round(ch.sr / fs)))
    n = max(1, int(round(t.block_ms / 1000 * fs)))
    if len(y) < 60 * n or t.high_band_hz[1] >= ch.sr / 2:
        return []
    low = _block_db(sosfiltfilt(butter(8, list(t.low_band_hz), btype="band", fs=fs, output="sos"), y), n)
    high = _block_db(sosfilt(butter(4, list(t.high_band_hz), btype="band", fs=ch.sr, output="sos"), ch.x), n * q)
    m = min(len(low), len(high))
    low, high = low[:m], high[:m]
    block_s = n * q / ch.sr
    blocks = lambda ms: max(1, int(round(ms / 1000 / block_s)))  # noqa: E731
    w_from, w_to = blocks(t.word_from_ms), blocks(t.word_to_ms)
    near = blocks(t.one_per_ms)
    peaks = np.flatnonzero((low >= t.min_dbfs) & (low == maximum_filter1d(low, 2 * near + 1, mode="nearest")))
    after = int(t.mouth_click_after_ms / 1000 * ch.sr)
    out: list[Finding] = []
    for j in peaks:
        if j < 1 or j + w_to >= m:
            continue
        lo, hi = j, j
        while lo > 0 and low[lo - 1] >= low[j] - 10:
            lo -= 1
        while hi + 1 < m and low[hi + 1] >= low[j] - 10:
            hi += 1
        if (hi - lo + 1) * block_s * 1000 > t.max_width_ms:
            continue
        high_at = float(high[j - 1:j + 2].max())
        if low[j] - high_at < t.low_over_high_db:
            continue
        word = float(high[j + w_from:j + w_to].max())
        if word - high_at < t.word_over_pop_db:
            continue
        onset = next((k for k in range(j + 1, j + w_to) if high[k] >= word - t.word_within_db), None)
        if onset is None:
            continue
        to_word = (onset - j) * block_s * 1000
        if not (t.min_ms_to_word <= to_word <= t.max_ms_to_word):
            continue
        if low[j] - float(low[j + 1:onset].min()) < t.min_fall_db:
            continue
        s, e = lo * n * q, (hi + 1) * n * q
        if any(s <= b + after and e >= a for a, b in mouth_click_spans):
            continue                  # part of a mouth-click inhale: the breath's entry covers it
        level = round(float(low[j]))          # graded on the whole-dB value the text shows
        out.append(Finding(
            file=ch.name, check="plosive", start_sample=s, end_sample=e, start_time=ch.clock(s), end_time=ch.clock(e),
            severity=ladder(level, (t.min_dbfs, t.sev2_dbfs, t.sev3_dbfs)),
            problem=f"plosive pop before a word, {level} dBFS below 100 Hz",
            measures={"low_dbfs": round(float(low[j]), 1), "low_over_high_db": round(float(low[j] - high_at), 1),
                      "ms_to_word": int(round(to_word)), "width_ms": int(round((hi - lo + 1) * block_s * 1000))},
        ))
    return out
