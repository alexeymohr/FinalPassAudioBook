"""Pause map: measured word to word, guessed from the rules. Synthetic audio only."""
from __future__ import annotations

import numpy as np

from finalpass_audiobook.activity import measure
from finalpass_audiobook.checks.pauses import listed, pause_map
from finalpass_audiobook.rules import RULE_SETS, guess_internal, head_or_tail_note
from synth import SR, band_noise, chapter, phrase, room


def _layout():
    parts = [("pause", 1.0), ("speech", 2), ("pause", 2.5), ("speech", 4), ("pause", 0.6),
             ("speech", 4), ("pause", 1.5), ("speech", 3), ("pause", 2.5)]
    x, marks = [], []
    for kind, v in parts:
        seg = room(v) if kind == "pause" else phrase(v)
        if kind == "speech":
            seg = seg[: len(seg) - int(0.06 * SR)]       # end on the word, not its trailing gap
        marks.append((kind, sum(len(s) for s in x), len(seg)))
        x.append(seg)
    return np.concatenate(x), marks


def test_pauses_are_measured_and_guessed() -> None:
    x, marks = _layout()
    ch = chapter(x)
    pauses = pause_map(ch, measure(ch), RULE_SETS["standard"])
    got = [(p.kind, p.duration_ms / 1000, p.guess) for p in listed(pauses)]
    kinds = [g[0] for g in got]
    assert kinds[0] == "head" and kinds[-1] == "tail"
    durations = [g[1] for g in got]
    expected = [m[2] / SR for m in marks if m[0] == "pause"]
    assert np.allclose(durations, expected, atol=0.03), (durations, expected)
    guesses = [g[2] for g in got]
    assert guesses[0] == "chapter start: matches the 1 s rule"
    assert guesses[1] == "after chapter heading (likely)"
    assert guesses[2] == "paragraph break"
    assert guesses[3] == "between paragraph and section length"
    assert guesses[4] == "chapter end: matches the 2.5 s rule"


def test_a_breath_inside_a_gap_counts_as_pause() -> None:
    gap = np.concatenate([room(0.25), band_noise(0.3, 900, 3200, -48), room(0.25)])
    x = np.concatenate([room(1.0), phrase(3)[: -int(0.06 * SR)], gap, phrase(3), room(2.5)])
    ch = chapter(x)
    internal = [p for p in pause_map(ch, measure(ch), RULE_SETS["standard"]) if p.kind == "internal"]
    long_ones = [p for p in internal if p.duration_ms >= 500]
    assert len(long_ones) == 1
    assert abs(long_ones[0].duration_ms / 1000 - len(gap) / SR) < 0.04


def test_the_no_paragraph_set_has_no_paragraph_rule() -> None:
    hc = RULE_SETS["no-paragraph"]
    assert guess_internal(0.6, 30.0, hc, heading_seen=True) == "sentence or paragraph pause"
    assert guess_internal(2.5, 30.0, hc, heading_seen=True) == "section break"
    assert guess_internal(4.0, 30.0, hc, heading_seen=True) == "longer than a section break"


def test_head_and_tail_tolerance() -> None:
    assert head_or_tail_note(1.08, 1.0, "chapter start") == "chapter start: matches the 1 s rule"
    assert head_or_tail_note(1.43, 1.0, "chapter start") == "chapter start: 0.43 s longer than the 1 s rule"
    assert head_or_tail_note(2.0, 2.5, "chapter end") == "chapter end: 0.50 s shorter than the 2.5 s rule"


def test_silence_only_file() -> None:
    ch = chapter(room(3.0))
    (p,) = pause_map(ch, measure(ch), RULE_SETS["standard"])
    assert p.guess == "no narration found"


def test_tolerance_is_symmetric_to_the_millisecond() -> None:
    from finalpass_audiobook.rules import HEAD_TAIL_TOLERANCE_S, head_or_tail_note
    assert HEAD_TAIL_TOLERANCE_S == 0.1
    for s in (0.9, 1.1):
        assert "matches" in head_or_tail_note(s, 1.0, "chapter start")
    assert "longer" in head_or_tail_note(1.101, 1.0, "chapter start")
