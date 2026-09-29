"""Breaths, from FinalPass's breath check. Only what QC is likely to notice is listed.

Mouth-click inhale (internally "T-inhale": FinalPass flags a breath that opens
with a mouth-release burst). How clearly it is a defect depends on whether the
click stands apart from the word before it:

* 3 — a gap of at least 250 samples at 44.1 kHz (5.7 ms) at or below -60 dBFS
  just before the click, and the click at least narration -20 dB;
* 2 — the gap, but a quieter click;
* 1 — no such gap: the word's own hard consonant may run into the inhale.

The gap is measured with a 0.73 ms RMS (32 samples at 44.1 kHz, scaled with the
rate), so 250 measured samples correspond to about 6.4 ms of true silence. It is
counted at or below -60 dBFS: in a room whose tone sits above that, no gap can
be seen and every mouth-click inhale grades 1.

Evidence, on the operator's breaths heard in context (one audiobook): mouth-click
inhales had the gap 9/11 in one chapter and 15/19 in others; breaths after a
word's hard consonant 2/11 (their stop closure is often 180-260 samples). The
three the operator called "small" sat at narration -20.7 to -23.9 dB; the other
29 at -18.9 dB or louder.

Loud breath (FinalPass grade 3, "quite noticeable") is severity 2: an artistic
call, not a mechanical defect, but clients dislike them. Grades 1-2 are counted,
not listed. A breath that is both is one finding at the higher severity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from finalpass.breath_check import BreathAssetResult, BreathEvent, BreathTunables, analyze_breaths

from ..chapter import Chapter
from ..findings import Finding


@dataclass(frozen=True)
class BreathSeverity:
    click_gap_ms: float = 250 / 44.1       # 250 samples at 44.1 kHz
    harsh_click_vs_narration_db: float = -20.0
    loud_breath: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


def mouth_click_severity(e: BreathEvent, sr: int, s: BreathSeverity = BreathSeverity()) -> int:
    gap = e.click_gap_samples if e.click_gap_samples is not None else 0
    if gap < s.click_gap_ms * sr / 1000.0 - 1e-9:
        return 1
    loud = e.click_rel_db is not None and e.click_rel_db >= s.harsh_click_vs_narration_db
    return 3 if loud else 2


MOUTH_CLICK_TEXT = {
    3: "mouth-click inhale: inhale starts with a mouth click after a gap",
    2: "small mouth-click inhale after a gap",
    1: "hard consonant runs into inhale (possible mouth-click inhale)",
}


def breath_findings(ch: Chapter, tunables: BreathTunables = BreathTunables(),
                    sev: BreathSeverity = BreathSeverity()) -> tuple[list[Finding], BreathAssetResult]:
    result = analyze_breaths(ch.mono_audio, tunables)
    out: list[Finding] = []
    for e in result.breaths:
        parts: list[tuple[int, str]] = []
        if e.t_inhale:
            k = mouth_click_severity(e, ch.sr, sev)
            parts.append((k, MOUTH_CLICK_TEXT[k]))
        if e.grade == 3:
            parts.append((sev.loud_breath, "loud breath"))
        if not parts:
            continue
        out.append(Finding(
            file=ch.name, check="breaths", start_sample=e.start_sample, end_sample=e.end_sample,
            start_time=e.start_time, end_time=e.end_time, severity=max(k for k, _ in parts),
            problem="; ".join(text for _, text in parts),
            measures={"duration_ms": e.duration_ms, "grade": e.grade, "peak_db_vs_narration": e.peak_db,
                      "t_inhale_score": e.t_inhale_score if e.t_inhale_score is not None else "off",
                      "click_gap_samples": e.click_gap_samples if e.click_gap_samples is not None else "n/a",
                      "click_db_vs_narration": e.click_rel_db if e.click_rel_db is not None else "n/a"},
        ))
    return out, result
