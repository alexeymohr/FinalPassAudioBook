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


def test_a_building_hum_is_shown_and_graded_at_the_level_it_reaches() -> None:
    x = _speech(30.0)
    t = np.arange(len(x)) / SR
    ramp = np.clip((t - 2.0) / 26.0, 0, 1)
    x = x + 10 ** ((-80 + 35 * ramp) / 20) * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t) * (t >= 2.0)
    (f,) = hum_findings(chapter(x))
    assert "building" in f.problem and f.problem.split("→ ")[1].startswith(("-45", "-46", "-44"))
    assert f.severity >= 2 and f.measures["level_dbfs"] >= -47


def test_a_held_voiced_note_is_not_confirmed_in_gaps_as_a_hum() -> None:
    """A line as loud as the voice never takes the filled-gap route."""
    from finalpass_audiobook.checks.hum import HumTunables, _filled_gaps
    ch = chapter(_speech(10.0))
    assert _filled_gaps(ch, HumTunables(), 0.0, 10.0, ch.narration_dbfs - 2.0) == []


# --- added after the 0.2.0 review ---------------------------------------------------------


@pytest.mark.parametrize("hz, dbfs", [(120.0, -47.0), (120.0, -40.0), (180.0, -47.0), (150.0, -40.0), (100.0, -40.0)])
def test_a_hum_in_the_voices_range_loud_enough_to_fill_the_pauses_is_one_hum(hz: float, dbfs: float) -> None:
    """Too loud for the 30 dB pauses, buried in the voice's own pitch for tracking (or tracked in pieces):
    once missed, split into pieces that each "started abruptly", or (100 Hz) read 5 dB high and "cut off
    abruptly" at the end of the file — a word in the tone's band is not its edge."""
    x = _speech(40.0)
    (f,) = _hums(x + _tone(len(x), hz, dbfs))
    assert abs(f.measures["frequency_hz"] - hz) < 0.3 and abs(f.measures["level_dbfs"] - dbfs) < 1.5
    assert "abruptly" not in f.problem and f.severity == 2
    assert f.start_sample / SR < 3.0 and f.end_sample / SR > 37.0


def test_a_loud_hum_in_the_voices_range_mid_chapter_is_one_hum_over_its_span() -> None:
    """Its edges fall under words, where the voice fills the tone's band: they are placed between the
    pauses either side (2.6 s apart here), not to the tenth of a second."""
    x = _speech(40.0)
    (f,) = _hums(x + _tone(len(x), 180.0, -47.0, 10.0, 30.0))
    assert abs(f.measures["level_dbfs"] + 47.0) < 1.5 and f.severity == 3
    assert abs(f.start_sample / SR - 10.0) < 2.0 and abs(f.end_sample / SR - 30.0) < 2.0


def _gappy(gaps: list[tuple[float, float]]) -> np.ndarray:
    """Phrases with 0.5 s gaps; gap i holds a steady tone (Hz, dBFS) when given, room tone otherwise."""
    parts = []
    for i in range(6):
        parts.append(phrase(4))
        g = room(0.5)
        if i < len(gaps) and gaps[i]:
            hz, dbfs = gaps[i]
            g = g + _tone(len(g), hz, dbfs)
        parts.append(g)
    return np.concatenate(parts)


def test_a_filled_pause_must_have_a_twin_nearby() -> None:
    """A gap a loud hum fills is a pause; one soft held voiced sound at the narrator's pitch is not (on the
    calibration title 4 such gaps were flat and one line), and neither are two at different pitches."""
    from finalpass_audiobook.checks.hum import HumTunables, _pauses
    t = HumTunables()

    def filled(gaps):
        ch = chapter(_gappy(gaps))
        return [p for p in _pauses(ch, t) if p not in _pauses(chapter(_gappy([])), t)]

    assert len(filled([(99.0, -38.0), (99.0, -38.0)])) == 2
    assert filled([(99.0, -38.0)]) == []
    assert filled([(99.0, -38.0), (99.9, -38.0)]) == []                  # 0.9 Hz apart: not one hum
    assert filled([(99.0, -38.0), (99.0, -43.0)]) == []                  # 5 dB apart: not one hum


def test_a_harmonic_louder_than_its_tone_sets_the_level() -> None:
    """Operator: a strong hum is one whose loudest line reaches -55 dBFS — a weak 60 Hz under strong
    180/300 Hz lines read -70 dBFS, severity 1."""
    x = _speech(40.0)
    n = len(x)
    (f,) = _hums(x + _tone(n, 60.0, -70.0) + _tone(n, 180.0, -50.0) + _tone(n, 300.0, -52.0))
    assert f.problem.startswith("hum 60.0 Hz, -50 dBFS at 180 Hz") and f.severity == 2
    assert f.measures["loudest_line_hz"] == 180 and abs(f.measures["level_dbfs"] + 50.0) < 1.0


def test_a_later_hum_at_a_multiple_is_its_own_hum() -> None:
    x = _speech(40.0)
    x = x + _tone(len(x), 50.0, -55.0, 0.0, 25.0) + _tone(len(x), 150.0, -55.0, 15.0, 40.0)
    spans = sorted((round(f.measures["frequency_hz"]), f.start_sample / SR, f.end_sample / SR) for f in _hums(x))
    assert [s[0] for s in spans] == [50, 150], spans
    assert spans[0][2] < 26.0 and abs(spans[1][1] - 15.0) < 1.5 and spans[1][2] > 38.0


def test_an_unrelated_tone_near_a_high_multiple_is_not_a_harmonic() -> None:
    from finalpass_audiobook.checks.hum import HumTunables, _group, _is_multiple, _Track
    base = _Track(0, 20, [41.0] * 21, [-60.0] * 21, [20.0] * 21)
    far = _Track(0, 20, [987.0] * 21, [-60.0] * 21, [20.0] * 21)          # 41 x 24 = 984
    real = _Track(0, 20, [984.1] * 21, [-70.0] * 21, [12.0] * 21)
    late = _Track(12, 40, [123.0] * 29, [-60.0] * 29, [20.0] * 29)        # x 3, but mostly after the hum
    groups = _group([base, far, real, late], HumTunables())
    assert [(round(g.freqs[0]), [round(h.freqs[0]) for h in hs]) for g, hs in groups] == [
        (41, [984]), (123, []), (987, [])]
    assert not _is_multiple(987.0, 41.0) and _is_multiple(984.4, 41.0) and _is_multiple(198.6, 99.2)
