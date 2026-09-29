"""Pause map: every pause, its length, and a guess at what it is. Informational.

The tool cannot see the text, so "section break" or "paragraph break" is a guess
from the length alone, checked against the chosen pause rule set.
"""
from __future__ import annotations

from ..activity import Activity
from ..chapter import Chapter
from ..findings import Pause
from ..rules import RuleSet, guess_internal, head_or_tail_note

LIST_FROM_S = 0.5     # shorter pauses are kept in the JSON only


def pause_map(ch: Chapter, act: Activity, rules: RuleSet) -> list[Pause]:
    n = len(ch.x)
    out: list[Pause] = []
    if act.first_sound is None:
        return [Pause(start_sample=0, end_sample=n, start_time=ch.clock(0),
                      duration_ms=int(round(n * 1000 / ch.sr)), kind="head",
                      guess="no narration found")]

    def add(s: int, e: int, kind: str, guess: str) -> None:
        out.append(Pause(start_sample=s, end_sample=e, start_time=ch.clock(s),
                         duration_ms=int(round((e - s) * 1000 / ch.sr)), kind=kind, guess=guess))

    head_s = act.first_sound / ch.sr
    add(0, act.first_sound, "head", head_or_tail_note(head_s, rules.chapter_head_s, "chapter start"))
    heading_seen = False
    for s, e in act.pauses:
        guess = guess_internal((e - s) / ch.sr, s / ch.sr, rules, heading_seen)
        heading_seen = heading_seen or guess.startswith("after chapter heading")
        add(s, e, "internal", guess)
    tail_s = (n - act.last_sound) / ch.sr
    add(act.last_sound, n, "tail", head_or_tail_note(tail_s, rules.chapter_tail_s, "chapter end"))
    return out


def listed(pauses: list[Pause]) -> list[Pause]:
    """The pauses worth reading: head, tail, and internal ones from LIST_FROM_S."""
    return [p for p in pauses if p.kind != "internal" or p.duration_ms >= LIST_FROM_S * 1000]
