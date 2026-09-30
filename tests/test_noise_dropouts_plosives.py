"""Noise floor, floor dropouts and plosive pops. Synthetic audio only."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from finalpass_audiobook.checks.dropouts import dropout_findings
from finalpass_audiobook.checks.noise import floor_track, noise_findings
from finalpass_audiobook.checks.plosives import plosive_findings
from synth import RNG, SR, chapter, phrase, room


def _narration(seconds: float, room_dbfs: float = -72.0) -> np.ndarray:
    parts, n = [], 0
    while n < seconds * SR:
        p = np.concatenate([phrase(4), room(0.7, room_dbfs)])
        parts.append(p)
        n += len(p)
    return np.concatenate(parts)[: int(seconds * SR)]


def _pink(n: int) -> np.ndarray:
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, RNG.standard_normal(n))


# --- noise floor ----------------------------------------------------------


@pytest.mark.parametrize("sr", [44100, 48000])
@pytest.mark.parametrize("shape", ["white", "pink"])
def test_floor_estimate_matches_steady_noise(sr: int, shape: str) -> None:
    y = RNG.standard_normal(sr * 20) if shape == "white" else _pink(sr * 20)
    y = y / np.sqrt(np.mean(y ** 2)) * 10 ** (-55 / 20)
    track, _ = floor_track(y, sr)
    f = np.fft.rfftfreq(len(y), 1 / sr)
    band = (f >= 56.1) & (f < min(17818, sr / 2))
    true = 10 * np.log10(2 * np.sum(np.abs(np.fft.rfft(y)[band]) ** 2) / len(y) ** 2)
    assert abs(float(np.median(track[300:-300])) - true) < 0.6


def _noisy(noise_dbfs: float) -> np.ndarray:
    x = _narration(30.0)
    t = np.arange(len(x)) / SR
    return x + ((t >= 10) & (t < 20)) * RNG.standard_normal(len(x)) * 10 ** (noise_dbfs / 20)


def test_noise_under_continuous_narration_is_found() -> None:
    found, median_floor = noise_findings(chapter(_noisy(-50)))
    assert len(found) == 1
    f = found[0]
    assert abs(f.start_sample / SR - 10) < 1.5 and abs(f.end_sample / SR - 20) < 1.5
    assert -53 < f.measures["floor_median_dbfs"] < -47
    assert median_floor is not None and median_floor < -60


@pytest.mark.parametrize("noise_dbfs, severity", [(-40, 3), (-47, 2), (-55, 1), (-62, None)])
def test_noise_severity_follows_the_gap_under_the_speech(noise_dbfs: float, severity: int | None) -> None:
    """Speech about -18 dBFS: under 25 dB gap -> 3, under 32 -> 2, within 41 -> 1, else not listed."""
    found, _ = noise_findings(chapter(_noisy(noise_dbfs)))
    assert [f.severity for f in found] == ([severity] if severity else [])
    for f in found:
        gap = f.measures["floor_under_speech_db"]
        assert f.severity == (3 if gap < 25 else 2 if gap < 32 else 1)


def test_clean_narration_is_not_noisy() -> None:
    found, median_floor = noise_findings(chapter(_narration(30.0)))
    assert found == []
    assert median_floor < -65


# --- dropouts --------------------------------------------------------------


def _hole(level_dbfs: float, zeros: int, after_dbfs: float | None = None) -> tuple[np.ndarray, int]:
    """A steady tone at `level_dbfs` with `zeros` samples of dead silence punched into it."""
    t = np.arange(int(0.5 * SR)) / SR
    tone = lambda db: np.sqrt(2) * 10 ** (db / 20) * np.sin(2 * np.pi * 1000.0 * t)  # noqa: E731
    x = np.concatenate([_narration(3.0), tone(level_dbfs), np.zeros(zeros),
                        tone(level_dbfs if after_dbfs is None else after_dbfs), _narration(3.0)])
    return x, int(3.0 * SR) + len(t)


def _at(x: np.ndarray, at: int) -> list:
    return [f for f in dropout_findings(chapter(x)) if abs(f.start_sample - at) < 5]


@pytest.mark.parametrize("level_dbfs, listed", [(-20.0, True), (-44.0, True), (-47.0, False)])
def test_a_frame_of_dead_silence_inside_the_sound_is_a_dropout(level_dbfs: float, listed: bool) -> None:
    x, at = _hole(level_dbfs, int(0.02 * SR))
    found = _at(x, at)
    assert len(found) == int(listed)
    for f in found:
        assert f.severity == 3 and f.measures["silence_ms"] == 20.0


def test_it_must_recover_within_a_frame() -> None:
    x, at = _hole(-20.0, int(0.040 * SR))                      # 40 ms: longer than a frame
    assert _at(x, at) == []


def test_it_must_come_back_loud_enough() -> None:
    x, at = _hole(-20.0, int(0.02 * SR), after_dbfs=-60.0)     # sound cut to a quiet tail: not a dropout
    assert _at(x, at) == []


def test_a_zero_crossing_is_not_a_dropout() -> None:
    x, at = _hole(-20.0, 4)                                    # a few exact zeros happen naturally
    assert _at(x, at) == []
    x, at = _hole(-20.0, 10)                                   # ten is a hole
    assert len(_at(x, at)) == 1


def test_natural_endings_into_room_tone_are_not_dropouts() -> None:
    assert dropout_findings(chapter(_narration(12.0, room_dbfs=-45.0))) == []


# --- plosives --------------------------------------------------------------


def _thump(amp: float, hz: float = 60.0) -> np.ndarray:
    """A pop: a low sine dying away in a few tens of milliseconds."""
    t = np.arange(int(0.03 * SR)) / SR
    return amp * np.sin(2 * np.pi * hz * t) * np.exp(-t / 0.006)


def _before_word(pop: np.ndarray | None, lead_s: float = 0.11) -> tuple[np.ndarray, int]:
    """A phrase, a pause, then a phrase whose first word starts at `word`; the pop `lead_s` before it."""
    x = np.concatenate([phrase(3), room(0.8), phrase(3), room(1.0)])
    word = len(phrase(3)) + int(0.8 * SR)
    if pop is not None:
        at = word - int(lead_s * SR)
        x[at:at + len(pop)] += pop
    return x, word


def test_a_pop_just_before_a_word_is_found_and_speech_alone_is_not() -> None:
    x, _ = _before_word(None)
    assert plosive_findings(chapter(x)) == []
    x, word = _before_word(_thump(0.2))
    (f,) = plosive_findings(chapter(x))
    assert f.check == "plosive" and abs(f.start_sample - (word - int(0.11 * SR))) < int(0.02 * SR)
    assert f.severity == 3 and 75 <= f.measures["ms_to_word"] <= 150


@pytest.mark.parametrize("amp, severity", [(0.2, 3), (0.1, 2), (0.05, 1), (0.002, None)])
def test_plosive_severity_follows_its_level(amp: float, severity: int | None) -> None:
    """1 from -42 dBFS below 100 Hz, 2 from -34, 3 from -26."""
    x, _ = _before_word(_thump(amp))
    found = plosive_findings(chapter(x))
    assert [f.severity for f in found] == ([severity] if severity else [])
    for f in found:
        lv = f.measures["low_dbfs"]
        assert f.severity == (3 if lv >= -26 else 2 if lv >= -34 else 1)


def test_a_normal_plosive_onset_is_not_a_pop() -> None:
    """The same thump at the word's own onset: the word's high end arrives with it."""
    x, word = _before_word(None)
    pop = _thump(0.2)
    x[word:word + len(pop)] += pop
    assert plosive_findings(chapter(x)) == []


def test_a_thump_with_a_high_end_is_not_a_pop() -> None:
    noise = RNG.standard_normal(int(0.03 * SR)) * 0.05 * np.exp(-np.arange(int(0.03 * SR)) / (0.006 * SR))
    x, _ = _before_word(_thump(0.2) + noise)
    assert plosive_findings(chapter(x)) == []


def test_a_thump_long_before_the_next_word_is_not_this_pop() -> None:
    x, _ = _before_word(_thump(0.2), lead_s=0.2)            # 200 ms: the word starts too late
    assert plosive_findings(chapter(x)) == []


def test_a_pop_that_is_part_of_a_mouth_click_inhale_is_left_to_it() -> None:
    x, word = _before_word(_thump(0.2))
    breath = (word - int(0.5 * SR), word - int(0.2 * SR))      # the pop falls within 200 ms after it
    assert plosive_findings(chapter(x), (breath,)) == []
    assert len(plosive_findings(chapter(x))) == 1


# --- added after the audit ----------------------------------------------------------------


def test_dc_offset_is_not_a_noisy_section() -> None:
    for dc in (0.003, 0.01, 0.03):
        found, _ = noise_findings(chapter(_narration(30.0) + dc))
        assert found == [], (dc, [f.problem for f in found])


def test_a_steady_tone_reads_at_its_true_level_not_inflated() -> None:
    t = np.arange(SR * 20) / SR
    y = RNG.standard_normal(len(t)) * 10 ** (-85 / 20) + np.sqrt(2) * 10 ** (-60 / 20) * np.sin(2 * np.pi * 60 * t)
    track, _ = floor_track(y, SR)
    assert abs(float(np.median(track[300:-300])) - (-60.0)) < 3.0       # was +7 dB before; now within 3 dB


def test_a_noisy_stretch_is_reported_at_its_true_times() -> None:
    x = _narration(30.0)
    t = np.arange(len(x)) / SR
    x = x + ((t >= 10) & (t < 16.5)) * RNG.standard_normal(len(x)) * 10 ** (-40 / 20)
    (f,) = noise_findings(chapter(x))[0]
    assert abs(f.start_sample / SR - 10.0) < 0.6 and abs(f.end_sample / SR - 16.5) < 0.6


def test_noise_shorter_than_the_calibrated_length_is_not_listed() -> None:
    x = _narration(30.0)
    t = np.arange(len(x)) / SR
    x = x + ((t >= 10) & (t < 14.0)) * RNG.standard_normal(len(x)) * 10 ** (-40 / 20)
    assert noise_findings(chapter(x))[0] == []


def test_a_listed_hum_is_not_also_a_noisy_section() -> None:
    from finalpass_audiobook.run import RunOptions, analyze_file
    import soundfile as sf
    import tempfile
    x = _narration(30.0)
    t = np.arange(len(x)) / SR
    x = x + np.sqrt(2) * 10 ** (-50 / 20) * np.sin(2 * np.pi * 60 * t)
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/hum.wav"
        sf.write(p, x, SR, subtype="PCM_24")
        from pathlib import Path
        fr = analyze_file(Path(p), RunOptions(truncation=False))
    checks = [f.check for f in fr.findings]
    assert "hum" in checks and "noise" not in checks, [f.problem for f in fr.findings]


def test_noise_in_a_file_without_speech_says_so() -> None:
    x = RNG.standard_normal(SR * 10) * 10 ** (-40 / 20)
    (f,) = noise_findings(chapter(x))[0]
    assert "no narration in this file" in f.problem and f.severity == 3
