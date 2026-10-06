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
-20.7 to -23.9 dB, the other 29 at -18.9 dB or louder. A click more than 40 dB under the narration is no mouth
click: on a very clean V4 render two transients at -50.7 and -50.6 dB passed as "small mouth-click inhales" and the
operator heard no click in either; the faintest confirmed one sat at -36.6 dB (4 of 228 on the calibration title
fall below -40, none heard before).

Every other breath is listed too (operator: no breath left behind), scored by
loudness: FinalPass's noticeability (the breath's median level relative to the
narration, weighted by its length; its grades were fitted to the operator's own).
Below -31.6 dB a breath is quiet and informational; from -31.6 it is severity 1,
from -26.4 severity 2, from -22.5 severity 3. The scale is fixed, not relative to
each chapter, so a book of quiet breaths lists few: on the calibration title the
lines sit at its 35th (the operator's suggested cut, which lands on FinalPass's
very-minor/noticeable boundary), 75th and 95th percentiles, about 73 quiet /
68 / 38 / 9 breaths per chapter; on a second title, whose breaths sit about
10 dB lower, about 61 / 2 / 1 / 3. A breath that is also a mouth-click inhale is
one finding at the higher severity.

With the breath model (breath_np), a breath is listed — as a problem or as a quiet breath — only if the model says
breath (>= 0.5) for at least 100 ms inside it; a mouth-click inhale is a breath, so its click needs that confirmed
breath too. On voices other than the calibration title's, FinalPass's detector also finds a consonant left alone
at a word's end before a pause ("t", "k", "p", "ch", "s"); the model, trained with words as "not breath", tells
them apart. Held-out evidence: breaths kept 18/18 and 16/16 on two titles, consonants listed 5/28 (4 of them
"breathy"), 0/14 on a breathless test voice; mouth-click inhales kept 43/43 + 8/8; at 150 ms one breath would
be lost (docs/PLAN.md 3.1).

Breath cut off into silence (with the model only; informational): a breath that runs straight into a hole of
exact digital silence (50-80 ms, sound after it) — a generated clip that ended mid-breath — is reported if nothing
above lists it (it replaces a quiet breath at the same spot). On the calibration title's QC report, the breaths
QC removed by cutting them to silence mostly sat right at such a hole (63 of 69 within 0.3 s; in 44 the removed
breath ends exactly where the hole begins). Holes alone are everywhere (about 45 of 40-60 ms per 15 minutes, the
generator's clip gaps), so only a breath running into one counts: 374 on that title, 48 of the QC ones among
them, a QC share of 17 % against at most 2.6 % for the breaths listed by loudness. Nothing measured (the breath's
length or level, how abruptly it meets the silence, the hole's length, the timing) told QC's apart from the rest,
so all are listed. Not listed by loudness: 137 there (4.5 per 15 minutes), holding 21 of the 33 QC breath cut-outs
missed before; a second title 12, the test book none. The model finds breaths shorter than FinalPass's 150 ms.
Heard by the operator: of 20 such breaths QC had not flagged, most holes were "too short to really matter" (QC's
own had distinct gaps), so the minimum became 50 ms (31 there, 1.0 per 15 minutes; 8 of QC's); of the next 24
none was "truly rejectable" (one held a plosive, not the cut) — so it is informational, not a problem.
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
    min_click_vs_narration_db: float = -40.0   # fainter: no mouth click (inaudible; the breath is graded alone)
    breath_sev1_db: float = -31.6        # loudness (noticeability) from which a breath is severity 1...
    breath_sev2_db: float = -26.4        # ...2
    breath_sev3_db: float = -22.5        # ...3; below sev1 it is quiet: informational

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BreathConfirm:
    probability: float = 0.5             # the breath model says breath from this probability...
    min_ms: float = 100.0                # ...for at least this long inside the breath

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BreathCutTunables:
    min_silence_ms: float = 50.0         # a hole of exact digital silence this long...
    max_silence_ms: float = 80.0         # ...up to this long, with sound after it
    probability: float = 0.5             # the breath model says breath from this probability
    reach_ms: float = 40.0               # the breath ends at most this long before the hole
    min_breath_ms: float = 50.0          # and lasts at least this long
    listed_within_ms: float = 400.0      # a breath already listed ending this close before the hole covers it
    severity: int = 0                    # informational (operator, after two rounds by ear)

    def as_dict(self) -> dict:
        return asdict(self)


CUT_OFF_TEXT = "breath cut off into silence"


def breath_cut_findings(ch: Chapter, probs: np.ndarray, listed: list[tuple[int, int]],
                        t: BreathCutTunables = BreathCutTunables()) -> list[Finding]:
    """Breaths running straight into a short hole of exact digital silence, not already listed. `probs`: the breath
    model's probability per 10 ms (frame i at i * 10 ms); `listed`: the spans of the breaths already listed."""
    x, sr = ch.x, ch.sr
    d = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    reach, need = int(round(t.reach_ms / 10)), int(round(t.min_breath_ms / 10))
    within = int(round(t.listed_within_ms * sr / 1000))
    out: list[Finding] = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        silence = (b - a) * 1000 / sr
        if a == 0 or b >= len(x) or not t.min_silence_ms <= silence <= t.max_silence_ms:
            continue
        fa = min(int(a / sr * 100), len(probs))          # the frame at the hole
        j = fa - 1
        while j >= max(0, fa - reach) and probs[j] < t.probability:
            j -= 1
        if j < max(0, fa - reach) or probs[j] < t.probability:
            continue                                     # no breath ends at the hole
        k = j
        while k > 0 and (probs[k - 1] >= t.probability or (k > 1 and probs[k - 2] >= t.probability)):
            k -= 1                                       # back to the breath's start, over one-frame dips
        if j - k + 1 < need:
            continue
        if any(s < a + int(0.02 * sr) and e > a - within for s, e in listed):
            continue                                     # already listed as a breath
        start = int(k * sr / 100)
        seg = x[start:a].astype(np.float64)
        level = float(10 * np.log10(max(float(np.mean(seg ** 2)) if seg.size else 0.0, 1e-20)))
        out.append(Finding(
            file=ch.name, check="breaths", start_sample=start, end_sample=int(b), start_time=ch.clock(start),
            end_time=ch.clock(int(b)), severity=t.severity, problem=CUT_OFF_TEXT,
            measures={"duration_ms": (j - k + 1) * 10, "silence_ms": round(silence, 1), "breath_dbfs": round(level, 1),
                      "level_vs_narration_db": round(level - ch.narration_dbfs, 1)
                      if np.isfinite(ch.narration_dbfs) else "n/a",
                      "cut_into_silence": "yes", "mouth_click": "no"}))
    return out


def confirmed_ms(confirm: np.ndarray, sr: int, start: int, end: int, rule: BreathConfirm = BreathConfirm()) -> int:
    """How long (ms) the breath model says breath inside samples start..end (its frames are 10 ms, frame i at i*10 ms)."""
    a = int(round(start / sr * 100))
    b = max(a + 1, int(round(end / sr * 100)))
    return 10 * int(np.count_nonzero(confirm[a:b] >= rule.probability))


def mouth_click_severity(silence_ms: float, click_rel_db: float, s: BreathSeverity = BreathSeverity()) -> int:
    """0 when the click follows its word directly (a consonant) or is too faint to hear, else 3 (harsh) or 2."""
    if silence_ms < s.silence_before_click_ms or click_rel_db < s.min_click_vs_narration_db:
        return 0
    return 3 if click_rel_db >= s.harsh_click_vs_narration_db else 2


BREATH_TEXT = {0: "quiet breath", 1: "noticeable breath", 2: "loud breath", 3: "very loud breath"}


def breath_loudness_severity(loudness_db: float, s: BreathSeverity = BreathSeverity()) -> int:
    """0 (quiet: informational) to 3, on the fixed loudness scale."""
    return sum(loudness_db >= c for c in (s.breath_sev1_db, s.breath_sev2_db, s.breath_sev3_db))


MOUTH_CLICK_TEXT = {
    3: "mouth-click inhale",
    2: "small mouth-click inhale",
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
                    confirm: np.ndarray | None = None, rule: BreathConfirm = BreathConfirm(),
                    ) -> tuple[list[Finding], list[Finding], BreathAssetResult]:
    """Every breath: (problem findings, quiet breaths as informational events, FinalPass's result).
    `rise`: the click detector's per-frame rise (clicks.rise_db), if already computed. `confirm`: the breath model's
    probability per 10 ms (breath_np), or None to list FinalPass's breaths as they are."""
    result = analyze_breaths(ch.mono_audio, tunables)
    r, n, hop = rise if rise is not None else rise_db(ch.x, ch.sr, ClickTunables())
    narration_ok = bool(np.isfinite(ch.narration_dbfs))
    problems: list[Finding] = []
    quiet_breaths: list[Finding] = []
    for e in result.breaths:
        held = None
        if confirm is not None:
            held = confirmed_ms(confirm, ch.sr, e.start_sample, e.end_sample, rule)
            if held < rule.min_ms:
                continue                     # the model hears no breath here: a consonant, or nothing
        parts: list[tuple[int, str]] = []
        start = e.start_sample
        loudness = round(float(e.noticeability_db), 1)        # graded on the value the report shows
        loud = breath_loudness_severity(loudness, sev)
        measures: dict = {"loudness_db": loudness, "duration_ms": e.duration_ms,
                          "peak_db_vs_narration": e.peak_db, "mouth_click": "no"}
        if held is not None:
            measures["breath_model_ms"] = held
        if narration_ok and r.size:
            at, sharp = _click_near(r, n, hop, ch.sr, e.start_sample, sev)
            if e.t_inhale or sharp >= sev.click_min_rise_db:
                quiet = silence_before(ch, at, sev)
                level = round(click_level(ch, at), 1)
                k = mouth_click_severity(quiet, level, sev)
                measures.update({"click_rise_db": round(sharp, 1), "ms_since_word": quiet,
                                 "click_db_vs_narration": level})
                if k:
                    parts.append((k, MOUTH_CLICK_TEXT[k]))
                    measures["mouth_click"] = "yes"
                    start = min(start, at)
        if loud or not parts:
            parts.append((loud, BREATH_TEXT[loud]))
        severity = max(k for k, _ in parts)
        f = Finding(file=ch.name, check="breaths", start_sample=start, end_sample=e.end_sample,
                    start_time=ch.clock(start), end_time=e.end_time, severity=severity,
                    problem="; ".join(text for _, text in parts), measures=measures)
        (problems if severity else quiet_breaths).append(f)
    return problems, quiet_breaths, result
