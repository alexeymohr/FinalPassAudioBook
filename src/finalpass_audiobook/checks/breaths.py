"""Breaths, from FinalPass's breath check. Only what QC is likely to notice is listed.

Mouth-click inhale (internally "T-inhale"): a breath that opens with a mouth
click. On its own a word-final consonant running into a breath sounds almost the
same; what makes one a defect is where the click stands (operator): a consonant's
release follows its vowel directly, completing the word; a mouth click stands
alone, after some pause, serving no word. So:

* the click: FinalPass flags the breath, or a sharp click (the click-in-a-pause
  detector's rise, at least 8 dB) sits from 100 ms before to 40 ms after the
  breath's start;
* the pause: at least 100 ms since the word — since the level (5 ms RMS) was last
  within 25 dB of the narration — before the click's onset. A consonant's stop
  closure is shorter;
* severity 3 when the click (0.73 ms RMS peak) is at least narration -20 dB,
  else 2. A click right after its word is a consonant and is not listed.

Evidence, on the operator's breaths heard in context (one audiobook, 35+
chapters): mouth-click inhales 41/49, consonant-then-inhale 0/29, plain breaths
3/127 (all 3 also flagged by FinalPass's own score), tiny ticks the operator
called barely audible 0/4. FinalPass's score alone: 30/39, 5/12 and 8/127 of
those it was measured on. Any pause from 100 to 150 ms gives the same result.
Held-out check: 32 listings on 12 chapters, none heard before, all 32 confirmed
by ear as mouth-click inhales (some milder than others).
The three clicks the operator called "small" in the first study sat at narration
-20.7 to -23.9 dB, the other 29 at -18.9 dB or louder.

Loud breath (FinalPass grade 3, "quite noticeable") is severity 2: an artistic
call, not a mechanical defect, but clients dislike them. Grades 1-2 are counted,
not listed. A breath that is both is one finding at the higher severity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from finalpass.breath_check import BreathAssetResult, BreathTunables, analyze_breaths
from finalpass.breath_edges import CLICK_RMS_SAMPLES

from ..chapter import Chapter
from ..findings import Finding
from .clicks import ClickTunables, rise_db

SILENCE_LOOK_S = 1.5         # how far back to look for the word
CLICK_OWN_RISE_MS = 15       # the click's own rise, walked back over before the pause is measured


@dataclass(frozen=True)
class BreathSeverity:
    click_before_ms: float = 100.0       # look for the click from this long before the breath's start...
    click_after_ms: float = 40.0         # ...to this long after it
    click_min_rise_db: float = 8.0       # a click FinalPass did not flag: at least this sharp
    word_vs_narration_db: float = -25.0  # "the word": level within this of the narration
    silence_before_click_ms: float = 100.0
    harsh_click_vs_narration_db: float = -20.0
    loud_breath: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


def mouth_click_severity(silence_ms: float, click_rel_db: float, s: BreathSeverity = BreathSeverity()) -> int:
    """0 when the click follows its word directly (a consonant), else 3 (harsh) or 2."""
    if silence_ms < s.silence_before_click_ms:
        return 0
    return 3 if click_rel_db >= s.harsh_click_vs_narration_db else 2


MOUTH_CLICK_TEXT = {
    3: "mouth-click inhale: a click after a pause, before the inhale",
    2: "small mouth-click inhale: a click after a pause, before the inhale",
}


def _click_near(rise: np.ndarray, n: int, hop: int, sr: int, start: int, s: BreathSeverity) -> tuple[int, float]:
    """The sharpest moment around a breath's start: (sample, rise dB)."""
    f0 = max(0, (start - int(s.click_before_ms * sr / 1000) - n // 2) // hop)
    f1 = min(len(rise), (start + int(s.click_after_ms * sr / 1000) - n // 2) // hop + 1)
    if f1 <= f0:
        return start, float("-inf")
    k = f0 + int(np.argmax(rise[f0:f1]))
    return int(k * hop + n // 2), float(rise[k])


def silence_before(ch: Chapter, click: int, s: BreathSeverity = BreathSeverity()) -> int:
    """Milliseconds from the click's onset back to the word (0 if sound runs straight into it)."""
    env = ch.envelope_db(5)
    b = click // ch.bin_samples
    thr = ch.narration_dbfs + s.word_vs_narration_db
    i = min(len(env), max(0, b - 3))            # before the 5 ms window reaches the click
    walked = 0
    while i > 0 and env[i - 1] >= thr and walked < CLICK_OWN_RISE_MS:
        i -= 1
        walked += 1
    if walked >= CLICK_OWN_RISE_MS:
        return 0
    lo = max(0, i - int(SILENCE_LOOK_S * 1000))
    loud = np.flatnonzero(env[lo:i] >= thr)
    ms = (i - (lo + int(loud[-1]))) if loud.size else i - lo
    return int(round(ms * ch.bin_samples * 1000 / ch.sr))


def click_level(ch: Chapter, click: int) -> float:
    """The click's peak (0.73 ms RMS, as FinalPass measures it) relative to the narration."""
    sr = ch.sr
    n_rms = max(1, round(CLICK_RMS_SAMPLES * sr / 44100))
    w = int(0.0015 * sr)
    y = ch.x[max(0, click - w - n_rms):click + w + n_rms].astype(np.float64)
    if y.size == 0:
        return float("-inf")
    power = np.convolve(y ** 2, np.ones(n_rms) / n_rms, mode="same")
    return float(10 * np.log10(max(float(power.max()), 1e-20)) - ch.narration_dbfs)


def breath_findings(ch: Chapter, tunables: BreathTunables = BreathTunables(),
                    sev: BreathSeverity = BreathSeverity(), rise: tuple[np.ndarray, int, int] | None = None,
                    ) -> tuple[list[Finding], BreathAssetResult]:
    """`rise`: the click detector's per-frame rise (clicks.rise_db), if already computed."""
    result = analyze_breaths(ch.mono_audio, tunables)
    r, n, hop = rise if rise is not None else rise_db(ch.x, ch.sr, ClickTunables())
    narration_ok = bool(np.isfinite(ch.narration_dbfs))
    out: list[Finding] = []
    for e in result.breaths:
        parts: list[tuple[int, str]] = []
        start = e.start_sample
        measures: dict = {"duration_ms": e.duration_ms, "grade": e.grade, "peak_db_vs_narration": e.peak_db,
                          "mouth_click": "no"}
        if narration_ok and r.size:
            at, sharp = _click_near(r, n, hop, ch.sr, e.start_sample, sev)
            if e.t_inhale or sharp >= sev.click_min_rise_db:
                quiet = silence_before(ch, at, sev)
                level = click_level(ch, at)
                k = mouth_click_severity(quiet, level, sev)
                measures.update({"click_rise_db": round(sharp, 1), "ms_since_word": quiet,
                                 "click_db_vs_narration": round(level, 1)})
                if k:
                    parts.append((k, MOUTH_CLICK_TEXT[k]))
                    measures["mouth_click"] = "yes"
                    start = min(start, at)
        if e.grade == 3:
            parts.append((sev.loud_breath, "loud breath"))
        if not parts:
            continue
        measures["finalpass_t_inhale_score"] = e.t_inhale_score if e.t_inhale_score is not None else "off"
        out.append(Finding(
            file=ch.name, check="breaths", start_sample=start, end_sample=e.end_sample,
            start_time=ch.clock(start), end_time=e.end_time, severity=max(k for k, _ in parts),
            problem="; ".join(text for _, text in parts), measures=measures,
        ))
    return out, result
