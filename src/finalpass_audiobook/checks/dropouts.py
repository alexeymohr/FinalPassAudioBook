"""Hard floor dropout: the background cuts out (or in) abruptly. Numbers only.

A natural decay ends in the room's own floor; it cannot fall below it. A bad
edit can: the background — room tone, a breath, the tail of a word — stops dead
and the level lands well under the chapter's floor, often at digital black. So a
dropout is a step, within a few milliseconds, from audible background down to
well below the floor, that then stays there. The mirror image (background
switching on out of near-silence) is available but off by default: on real
chapters it is mostly words starting after an inserted digital-black pause.

Word bodies are excluded — whether a word was chopped is the truncation check's
question — by requiring the level before the step to sit clearly under the
narration.

Only a noticeable hard cut is listed (operator): what cuts out must be within
31 dB of the speech around it (about -50 dBFS at the usual narration level);
steady room tone at -55 stepping to black is not worth a note. Severity by the
level of what cuts out, relative to the speech: 3 from -18 dB (a breath or a
word's tail, louder than about -37 dBFS), 2 from -26 dB, else 1. A first guess:
on the first real chapter this lists 1 cut where the looser -70 dBFS gate
listed 19; 16 across 12 chapters (1 / 3 / 12 at severity 3 / 2 / 1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..chapter import Chapter
from ..findings import Finding, ladder


@dataclass(frozen=True)
class DropoutTunables:
    min_step_db: float = 20.0          # size of the step
    within_ms: int = 5                 # how fast it must happen
    audible_dbfs: float = -70.0        # candidate steps: the background is at least this loud
    min_vs_speech_db: float = -31.0    # listed: what cuts out is at least this loud vs the speech
    sev2_vs_speech_db: float = -26.0
    sev3_vs_speech_db: float = -18.0
    below_floor_db: float = 10.0       # and land at least this far under the chapter floor
    below_narration_db: float = 15.0   # background, not the body of a word
    settle_ms: int = 40                # the new level must hold this long
    context_ms: int = 20               # level measured over this long before the step
    cut_in: bool = False               # also report sound starting abruptly out of silence
    room_tone_db: float = 10.0         # background within this of the floor is "room tone"

    def as_dict(self) -> dict:
        return asdict(self)


def _steps(env: np.ndarray, t: DropoutTunables) -> np.ndarray:
    """Bins where the level falls by at least the step size within `within_ms`."""
    k = t.within_ms
    drop = env[:-k] - env[k:]
    return np.flatnonzero(drop >= t.min_step_db)


def _scan(env: np.ndarray, floor: float, narration: float, t: DropoutTunables) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    c, k, s = t.context_ms, t.within_ms, t.settle_ms
    last = -10**9
    for i in _steps(env, t):
        if i < c or i + k + s > len(env) or i - last < c + s:
            continue
        before = float(np.median(env[i - c:i + 1]))
        after = float(np.median(env[i + k:i + k + s]))
        if (before >= max(t.audible_dbfs, floor - 3.0) and before <= narration - t.below_narration_db
                and after <= min(floor - t.below_floor_db, before - t.min_step_db)
                and float(env[i + k:i + k + s].max()) <= before - t.min_step_db + 6.0):
            out.append((i, before, after))
            last = i
    return out


def dropout_findings(ch: Chapter, floor_dbfs: float | None,
                     t: DropoutTunables = DropoutTunables()) -> list[Finding]:
    if floor_dbfs is None or not np.isfinite(ch.narration_dbfs):
        return []
    env = ch.envelope_db(2)
    h = ch.bin_samples
    out = []
    directions = (("out", env), ("in", env[::-1])) if t.cut_in else (("out", env),)
    for direction, series in directions:
        for i, before, after in _scan(series, floor_dbfs, ch.narration_dbfs, t):
            b = i if direction == "out" else len(env) - 1 - i - t.within_ms   # where the step begins
            sample = b * h
            rel = round(before - ch.local_speech_dbfs(sample, sample), 1)   # graded on the value shown
            if rel < t.min_vs_speech_db:
                continue
            what = "room tone" if before <= floor_dbfs + t.room_tone_db else "sound"
            where = "cuts out" if direction == "out" else "cuts in"
            out.append(Finding(
                file=ch.name, check="dropout", start_sample=sample, end_sample=sample + t.within_ms * h,
                start_time=ch.clock(sample), end_time=ch.clock(sample + t.within_ms * h),
                severity=ladder(rel, (t.min_vs_speech_db, t.sev2_vs_speech_db, t.sev3_vs_speech_db)),
                problem=f"{what} {where} abruptly ({before:.0f} → {max(after, -150):.0f} dBFS)" if direction == "out"
                else f"{what} {where} abruptly ({max(after, -150):.0f} → {before:.0f} dBFS)",
                measures={"direction": direction, "background_dbfs": round(before, 1),
                          "background_vs_speech_db": round(rel, 1),
                          "silence_side_dbfs": round(max(after, -150.0), 1),
                          "chapter_floor_dbfs": round(floor_dbfs, 1)},
            ))
    return sorted(out, key=lambda f: f.start_sample)
