"""Digital tick: a sample-level spike that no voice made. Numbers only.

A guard for edit and processing glitches. Rendered narration stops at about
16 kHz (58-74 dB down above it on the calibration title), so a spike above
16.5 kHz did not come from the voice. Listed only when it is all three:

* extremely short: at most 8 samples at 44.1 kHz (scaled) at half its peak;
* loud: at least -60 dBFS above 16.5 kHz (a one-sample spike of about -40 dBFS);
* alone: at least 20 dB over everything else above 16.5 kHz within 0.5-10 ms on
  both sides, and the sharpest sample-to-sample step in that span. A tick inside
  a consonant blends in (operator), and a consonant's own high end is never this
  short or this isolated.

Severity 3. Evidence: 32 candidates from a looser rule (median-based isolation)
were all consonants by ear; under this rule none of 11,892 candidates on 12
chapters remains. Its reach was set with one-sample spikes planted in memory in
those chapters: in pauses 100 % found at -30 dBFS and 99 % at -40; inside speech
about half at -30 (the rest sit where the voice's own waveform steps as sharply).
The shipped code on two chapters: 40/40 planted in pauses (-40 dBFS), 23/40 inside
speech (-30 dBFS), nothing else listed. Whole title (41 files): none found. No
real example has been heard yet.

The step into and out of a dropout is the dropout's own edge: a tick within 3 ms
of one is not listed again; a word that ends in a tick where its clip stops is
listed once, as the tick (run.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..chapter import Chapter
from ..findings import Finding

TICK_SEVERITY = 3


@dataclass(frozen=True)
class TickTunables:
    above_hz: float = 16500.0
    max_width_samples_44k: int = 8
    min_hf_dbfs: float = -60.0
    min_isolation_db: float = 20.0
    side_ms: float = 10.0
    own_ms: float = 0.5                  # the spike's own ringing, left out of "the rest"

    def as_dict(self) -> dict:
        return asdict(self)


def _db(v: float) -> float:
    return 20.0 * np.log10(max(float(v), 1e-12))


HP_BLOCK, HP_PAD = 1 << 21, 8192


TOP_HZ = 22050.0                        # the band the limits were set on (44.1/48 kHz audio)


def _highpass(x: np.ndarray, sr: float, hz: float) -> np.ndarray:
    """Zero-phase high-pass in overlapping blocks: the same result (the filter rings for far less
    than the overlap), without the many full-length copies one pass over a long file needs. At high
    rates the band also stops at 22.05 kHz, so it is the band the limits were set on."""
    sos = butter(8, hz, "highpass", fs=sr, output="sos")
    if sr / 2 > 1.1 * TOP_HZ:
        sos = np.concatenate([sos, butter(8, TOP_HZ, "lowpass", fs=sr, output="sos")])
    if len(x) <= HP_BLOCK + 2 * HP_PAD:
        return sosfiltfilt(sos, x)
    out = np.empty_like(x)
    for a in range(0, len(x), HP_BLOCK):
        b = min(len(x), a + HP_BLOCK)
        lo, hi = max(0, a - HP_PAD), min(len(x), b + HP_PAD)
        out[a:b] = sosfiltfilt(sos, x[lo:hi])[a - lo:a - lo + (b - a)]
    return out


def tick_findings(ch: Chapter, t: TickTunables = TickTunables()) -> list[Finding]:
    """At 88.2 kHz and up only 16.5-22.05 kHz is looked at (see _highpass): a step into silence
    carries far more above that at a high rate, which no 44.1/48 kHz tick limit was set for."""
    sr, x = ch.sr, np.asarray(ch.x, dtype=np.float64)          # no copy when it already is
    if t.above_hz >= 0.95 * sr / 2 or len(x) < sr // 10:
        return []                        # nothing above the voice to look at at this rate
    hp = _highpass(x, sr, t.above_hz)
    thr = 10 ** (t.min_hf_dbfs / 20)
    loud = np.flatnonzero((hp >= thr) | (hp <= -thr))          # no full-length |hp| copy
    if not loud.size:
        return []
    own, side = max(1, int(t.own_ms * sr / 1000)), int(t.side_ms * sr / 1000)
    width_max = max(1, round(t.max_width_samples_44k * sr / 44100))
    # one candidate per burst of loud samples
    groups = np.split(loud, np.flatnonzero(np.diff(loud) > own) + 1)
    out: list[Finding] = []
    for g in groups:
        j = int(g[np.argmax(np.abs(hp[g]))])
        # judged on both sides; within 10 ms of the file's start or end, on the one side there is
        left, right = j - side - own - 1 >= 0, j + side + own + 2 <= len(x)
        if not (left or right):
            continue
        p = abs(hp[j])
        w2 = int(0.002 * sr)
        near = np.abs(hp[max(0, j - w2):j + w2 + 1])
        half = np.flatnonzero(near >= 0.5 * p)
        width = int(half[-1] - half[0] + 1)
        if width > width_max:
            continue
        rest = max(([np.abs(hp[j - side:j - own]).max()] if left else [])
                   + ([np.abs(hp[j + own:j + side]).max()] if right else []))
        iso = _db(p) - _db(rest)
        if iso < t.min_isolation_db:
            continue
        # sample steps over the span of one 44.1 kHz sample (the same as plain steps at 44.1/48 kHz):
        # at a high rate a voice moves less per sample, which no limit set at 44.1 kHz expects
        lag = 1 if sr <= 48000 else int(np.ceil(sr / 44100))   # at least one 44.1 kHz sample long
        a0 = max(0, j - side - lag)
        seg = x[a0:min(len(x), j + side + 1)]
        dy = np.abs(seg[lag:] - seg[:-lag])
        c = j - a0
        step = dy[max(0, c - 2 - lag):c + 2].max()
        other = max((d.max() for d in (dy[:max(0, c - own - lag + 1)] if left else dy[:0],
                                        dy[c + own:] if right else dy[:0]) if d.size), default=0.0)
        if step < other:
            continue
        pk = _db(np.abs(x[max(0, j - own):j + own + 1]).max())
        out.append(Finding(
            file=ch.name, check="ticks", start_sample=j, end_sample=j + 1, start_time=ch.clock(j),
            end_time=ch.clock(j + 1), severity=TICK_SEVERITY, problem="digital tick",
            measures={"above_16k_dbfs": round(_db(p), 1), "width_samples": width, "isolation_db": round(iso, 1),
                      "peak_dbfs": round(pk, 1)},
        ))
    return out
