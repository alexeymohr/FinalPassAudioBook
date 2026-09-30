"""Sound cutting out to digital silence, audibly. Numbers only.

Generated narration has no room: its quiet background is whatever the renderer left, and
stretches of exact digital zero are everywhere in edited deliveries (about 60 short ones per
15 minutes on one title, most inside very quiet background). What draws a QC fail is a drop to
nothing that is heard (operator): the audio just before the cut must be loud enough. So a dropout
is a run of exact zeros at least 300 samples long (at 44.1 kHz; scaled with the rate) whose last
5 ms before it are at least -45 dBFS (operator's floor), and it is always severity 3. Sound
cutting in from silence is not listed: a word starting after inserted silence is normal.

Evidence: 32 blind holes in quiet background (-64 to -98 dBFS in the 5 ms before) were heard as
drops to nothing but not as noticeable; on the calibration title this rule lists 1 cut in 12
chapters (a word's tail at -29.5 dBFS cut to 374 ms of silence), on a second title 1 in 5.
The rule it replaces (a 20 dB step to under the chapter's floor, from about -50 dBFS of
background) listed quiet background edits the operator does not count.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..chapter import Chapter
from ..findings import Finding

DROPOUT_SEVERITY = 3


@dataclass(frozen=True)
class DropoutTunables:
    min_silence_samples_44k: int = 300   # exact zeros, scaled with the rate
    min_level_dbfs: float = -45.0        # the last `level_ms` before the cut
    level_ms: float = 5.0

    def as_dict(self) -> dict:
        return asdict(self)


def dropout_findings(ch: Chapter, t: DropoutTunables = DropoutTunables()) -> list[Finding]:
    x, sr = ch.x, ch.sr
    k = max(1, int(round(t.level_ms / 1000 * sr)))
    need = max(1, round(t.min_silence_samples_44k * sr / 44100))
    d = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out: list[Finding] = []
    for a, b in zip(starts, ends):
        if b - a < need or a < k:
            continue
        level = 10 * np.log10(max(float(np.mean(x[a - k:a].astype(np.float64) ** 2)), 1e-20))
        if level < t.min_level_dbfs:
            continue
        silence_ms = (b - a) * 1000 / sr
        out.append(Finding(
            file=ch.name, check="dropout", start_sample=int(a), end_sample=int(b), start_time=ch.clock(a),
            end_time=ch.clock(b), severity=DROPOUT_SEVERITY,
            problem=f"sound cuts to silence from {level:.0f} dBFS",
            measures={"level_before_dbfs": round(level, 1), "silence_ms": int(round(silence_ms))},
        ))
    return out
