"""Hum: a fixed-frequency line that persists. Synthetic audio only."""
from __future__ import annotations

import numpy as np
import pytest

from finalpass_audiobook.checks.hum import hum_findings
from synth import SR, chapter, phrase, room, word


def _speech(seconds: float) -> np.ndarray:
    parts, n = [], 0
    while n < seconds * SR:
        p = np.concatenate([phrase(5), room(0.5)])
        parts.append(p)
        n += len(p)
    return np.concatenate(parts)[: int(seconds * SR)]


def test_hum_fading_up_under_speech_is_found() -> None:
    x = _speech(20.0)
    t = np.arange(len(x)) / SR
    ramp = np.clip((t - 4.0) / 8.0, 0, 1)                     # silent, then builds over 8 s
    level = 10 ** ((-90 + 32 * ramp) / 20) * np.sqrt(2)       # -90 -> -58 dBFS RMS: clear of the -55 line
    x = x + level * np.sin(2 * np.pi * 60.0 * t) * (t >= 4.0)
    found = hum_findings(chapter(x))
    assert len(found) == 1
    f = found[0]
    assert f.severity == 1                                     # low-level (builds to -58, under the strong limit)
    assert abs(f.measures["frequency_hz"] - 60.0) < 0.3
    assert "building" in f.problem
    assert f.measures["level_end_dbfs"] > -62
    assert 3.0 <= f.start_sample / SR <= 12.0


def test_harmonics_are_grouped_with_their_hum() -> None:
    """A harmonic inside the voice's range is masked while the narrator speaks;
    in the pause it shows, and joins its fundamental rather than being a second hum."""
    x = np.concatenate([_speech(6.0), room(6.0)])
    t = np.arange(len(x)) / SR
    x = x + 10 ** (-60 / 20) * np.sqrt(2) * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t))
    (f,) = hum_findings(chapter(x))
    assert abs(f.measures["frequency_hz"] - 50.0) < 0.3
    assert f.measures["harmonics_hz"] == "150"


def test_grouping_needs_a_whole_multiple_at_the_same_time() -> None:
    from finalpass_audiobook.checks.hum import _group, _Track

    base = _Track(0, 20, [60.0] * 21, [-60.0] * 21, [20.0] * 21)
    harmonic = _Track(5, 15, [120.1] * 11, [-66.0] * 11, [15.0] * 11)
    unrelated = _Track(0, 20, [97.0] * 21, [-70.0] * 21, [12.0] * 21)
    later = _Track(30, 40, [180.0] * 11, [-70.0] * 11, [12.0] * 11)     # a multiple, but not at the same time
    groups = _group([harmonic, base, unrelated, later])
    assert [(round(g.freqs[0]), [round(h.freqs[0]) for h in hs]) for g, hs in groups] == [
        (60, [120]), (97, []), (180, [])]


def test_a_held_note_and_ordinary_speech_are_not_hum() -> None:
    held = word(1.0, f0=150.0, glide=0.0)                      # a sung/held note, 1 s
    x = np.concatenate([_speech(6.0), held, _speech(6.0)])
    assert hum_findings(chapter(x)) == []
    assert hum_findings(chapter(_speech(15.0))) == []


# --- added after the audit ----------------------------------------------------------------


def _tone(n: int, hz: float, dbfs: float, on_s: float = 0.0, off_s: float = 1e9) -> np.ndarray:
    t = np.arange(n) / SR
    return ((t >= on_s) & (t < off_s)) * 10 ** (dbfs / 20) * np.sqrt(2) * np.sin(2 * np.pi * hz * t)


def _hums(x: np.ndarray) -> list:
    return [f for f in hum_findings(chapter(x))]


def test_a_hum_loud_enough_to_fill_every_pause_is_still_a_hum() -> None:
    x = _speech(30.0)
    for dbfs in (-40.0, -30.0):
        found = _hums(x + _tone(len(x), 60.0, dbfs))
        assert [round(f.measures["frequency_hz"]) for f in found] == [60], (dbfs, [f.problem for f in found])


def test_overlapping_hums_with_different_spans_are_reported_separately() -> None:
    x = _speech(30.0)
    x = x + _tone(len(x), 60.0, -55.0, 10.0, 16.0) + _tone(len(x), 97.0, -55.0)
    spans = sorted((round(f.measures["frequency_hz"]), f.start_sample / SR, f.end_sample / SR) for f in _hums(x))
    assert [s[0] for s in spans] == [60, 97], spans
    assert abs(spans[0][1] - 10.0) < 0.6 and abs(spans[0][2] - 16.0) < 0.6
    assert spans[1][1] < 3.0 and spans[1][2] > 27.0          # 97 Hz sits in the synthetic voice's pitch range


@pytest.mark.parametrize("hz", [40.0, 40.2])
def test_a_hum_at_the_40_hz_edge_is_tracked(hz: float) -> None:
    x = _speech(20.0)
    (f,) = _hums(x + _tone(len(x), hz, -55.0))
    assert abs(f.measures["frequency_hz"] - hz) < 0.3 and f.measures["found_by"] == "tracking"


def test_start_and_end_follow_the_tone_not_the_analysis_windows() -> None:
    x = _speech(30.0)
    (f,) = _hums(x + _tone(len(x), 60.0, -55.0, 8.0, 20.0))
    assert abs(f.start_sample / SR - 8.0) < 0.3 and abs(f.end_sample / SR - 20.0) < 0.3


def test_a_steady_tone_is_one_finding_not_several() -> None:
    x = _speech(30.0)
    found = _hums(x + _tone(len(x), 998.0, -60.0))
    assert len(found) == 1 and found[0].end_sample / SR - found[0].start_sample / SR > 25.0


def test_a_loud_hum_that_starts_and_stops_mid_chapter_is_a_hum() -> None:
    """Loud enough to fill the pauses inside it; the pauses around it are clean."""
    x = _speech(40.0)
    t = np.arange(len(x)) / SR
    x = x + ((t >= 15.0) & (t < 19.0)) * 10 ** (-45 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)
    (f,) = hum_findings(chapter(x))
    assert f.severity == 3 and "abruptly" in f.problem
    assert 14.5 <= f.start_sample / SR <= 15.5 and 18.5 <= f.end_sample / SR <= 19.5


def test_hum_severity_follows_the_level_its_text_shows() -> None:
    import re
    x = _speech(20.0)
    t = np.arange(len(x)) / SR
    x = x + 10 ** (-55.3 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)
    (f,) = hum_findings(chapter(x))
    shown = int(re.search(r"(-?\d+) dBFS", f.problem).group(1))
    assert shown == -55 and f.severity == 2                           # "-55 dBFS" reads as strong


def test_pieces_of_one_hum_join_whichever_way_their_frequencies_round() -> None:
    from finalpass_audiobook.checks.hum import HumTunables, _join, _Track
    for f1, f2 in ((60.28, 60.32), (60.32, 60.28)):
        a = _Track(first=0, last=16, freqs=[f1] * 17, levels=[-60.0] * 17, proms=[20.0] * 17)
        b = _Track(first=20, last=36, freqs=[f2] * 17, levels=[-60.0] * 17, proms=[20.0] * 17)
        assert len(_join([a, b], HumTunables())) == 1
