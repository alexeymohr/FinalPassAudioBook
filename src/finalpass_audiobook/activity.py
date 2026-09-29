"""Where the narration is sounding and where it pauses. Numbers only.

A pause is measured word to word, the way a mixer measures it: breaths, mouth
clicks and room tone inside a gap all count as pause. Levels are judged against
the chapter's own narration level and its own quiet floor, so the same rule
works on a digital-black edit and on one with room tone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from finalpass.breath_check import BreathTunables, followed_by_pause
from finalpass.breath_detect import BreathParams, detect

from .chapter import Chapter

SOUND_BELOW_NARRATION_DB = 45.0   # quieter than this under the narration is pause...
SOUND_ABOVE_FLOOR_DB = 8.0        # ...unless the floor itself is that loud
ENV_WINDOW_MS = 5
CLICK_BRIDGE_MS = 30              # isolated sound shorter than this is a click, not a word
MIN_PAUSE_MS = 250
BREATH_MASK_MIN_MS = 80.0
BLACK_DBFS = -120.0


@dataclass(frozen=True)
class Activity:
    threshold_dbfs: float
    floor_dbfs: float | None
    first_sound: int | None        # sample
    last_sound: int | None         # sample (exclusive)
    pauses: tuple[tuple[int, int], ...]   # internal pauses, samples [start, end)
    breath_spans: tuple[tuple[int, int], ...]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def breath_spans(ch: Chapter) -> list[tuple[int, int]]:
    """Breaths (including short ones) that fade into a pause: they belong to the gap."""
    raw, level = detect(ch.frames, BreathParams(min_body_s=BREATH_MASK_MIN_MS / 1000.0))
    if not raw:
        return []
    kept, _ = followed_by_pause(ch.x, ch.frames, level, raw, BreathTunables(min_breath_ms=BREATH_MASK_MIN_MS))
    return [(b.start_sample, b.end_sample) for b in kept]


def measure(ch: Chapter) -> Activity:
    env = ch.envelope_db(ENV_WINDOW_MS)
    audible = env[env > BLACK_DBFS]
    floor = float(np.percentile(audible, 10)) if audible.size else None
    level = ch.narration_dbfs
    candidates = [floor + SOUND_ABOVE_FLOOR_DB] if floor is not None else []
    if np.isfinite(level):
        candidates.append(level - SOUND_BELOW_NARRATION_DB)
    thr = max(candidates) if candidates else BLACK_DBFS
    sound = env >= thr

    spans = breath_spans(ch)
    h = ch.bin_samples
    for s, e in spans:
        sound[s // h:(e + h - 1) // h] = False
    for a, b in _runs(sound):                       # clicks alone in a gap are not words
        if b - a < CLICK_BRIDGE_MS and a > 0 and b < len(sound):
            sound[a:b] = False

    runs = _runs(sound)
    if not runs:
        return Activity(thr, floor, None, None, (), tuple(spans))
    bin_db = 10 * np.log10(np.maximum(ch.bin_energy / h, 1e-20))
    first = _first_loud(bin_db, runs[0][0], thr) * h
    last = (_last_loud(bin_db, runs[-1][1], thr) + 1) * h
    pauses = []
    min_pause = MIN_PAUSE_MS * ch.sr / 1000
    for (_, end_a), (start_b, _) in zip(runs[:-1], runs[1:]):
        p0 = (_last_loud(bin_db, end_a, thr) + 1) * h
        p1 = _first_loud(bin_db, start_b, thr) * h
        if p1 - p0 >= min_pause:
            pauses.append((p0, p1))
    return Activity(thr, floor, first, last, tuple(pauses), tuple(spans))


def _last_loud(bin_db: np.ndarray, run_end: int, thr: float) -> int:
    """Last ~1 ms bin at or above the threshold where a run of sound ends."""
    lo, hi = max(0, run_end - 1), min(len(bin_db), run_end + ENV_WINDOW_MS - 1)
    loud = np.flatnonzero(bin_db[lo:hi] >= thr)
    return lo + int(loud[-1]) if loud.size else max(0, run_end - 1)


def _first_loud(bin_db: np.ndarray, run_start: int, thr: float) -> int:
    """First ~1 ms bin at or above the threshold where a run of sound starts."""
    lo, hi = run_start, min(len(bin_db), run_start + ENV_WINDOW_MS)
    loud = np.flatnonzero(bin_db[lo:hi] >= thr)
    return lo + int(loud[0]) if loud.size else run_start
