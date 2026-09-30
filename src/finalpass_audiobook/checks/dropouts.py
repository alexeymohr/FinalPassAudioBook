"""Dropout: sound falling to dead silence for a frame or less, then coming straight back. Numbers only.

The operator's definition (a classic dropout is a frame or less at 29.97 fps, ~33 ms):

* it truly drops to dead silence at once: a run of exact digital zeros;
* against a signal loud enough to hear it go: at least -45 dBFS in the 5 ms before;
* and recovers at once: the silence lasts at most 33 ms and the 5 ms after it are back at
  -45 dBFS or louder.

Always severity 3. A run of at least 10 zero samples (at 44.1 kHz, scaled) counts: at these
levels a slow zero crossing leaves 2-4 exact zeros (about 40 per 15 minutes on one title) and
at most 9; a real hole is longer.

Generated narration has no room, so digital silence after a word is simply a pause (operator);
cuts at the end of a word into a pause are not dropouts. Evidence: none on the calibration
title's 12 chapters or a second title's 5 (the longest recovering zero run there was 9 samples);
a guard for edit and render glitches. Planted in real narration (in memory, 6 chapters): holes of
10 samples, 1, 5 and 15 ms inside words listed 60/60 each, 30 ms 58/60 (the other two ran past the
word, so nothing loud came back); 8-sample holes, 40 ms holes and holes in quiet audio 0/60 each.
It replaces two earlier rules the operator rejected by ear: a 20 dB step from about -50 dBFS of
background, and any drop to zero from -45 dBFS (which listed only natural pauses after words).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..chapter import Chapter
from ..findings import Finding

DROPOUT_SEVERITY = 3


@dataclass(frozen=True)
class DropoutTunables:
    min_level_dbfs: float = -45.0        # before and after the silence
    level_ms: float = 5.0
    min_zeros_44k: int = 10              # exact zeros, scaled with the rate
    max_silence_ms: float = 33.4         # one 29.97 fps frame

    def as_dict(self) -> dict:
        return asdict(self)


def _db(y: np.ndarray) -> float:
    return 10 * np.log10(max(float(np.mean(y.astype(np.float64) ** 2)), 1e-20))


def dropout_findings(ch: Chapter, t: DropoutTunables = DropoutTunables()) -> list[Finding]:
    x, sr = ch.x, ch.sr
    k = max(1, int(round(t.level_ms / 1000 * sr)))
    shortest = max(1, round(t.min_zeros_44k * sr / 44100))
    longest = t.max_silence_ms / 1000 * sr
    d = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    out: list[Finding] = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if not (shortest <= b - a <= longest) or a < k or b + k > len(x):
            continue
        before, after = _db(x[a - k:a]), _db(x[b:b + k])
        if min(before, after) < t.min_level_dbfs:
            continue
        ms = (b - a) * 1000 / sr
        out.append(Finding(
            file=ch.name, check="dropout", start_sample=int(a), end_sample=int(b), start_time=ch.clock(a),
            end_time=ch.clock(b), severity=DROPOUT_SEVERITY,
            problem=f"dropout: {ms:.1f} ms of dead silence inside the sound",
            measures={"silence_ms": round(ms, 1), "level_before_dbfs": round(before, 1),
                      "level_after_dbfs": round(after, 1)},
        ))
    return out
