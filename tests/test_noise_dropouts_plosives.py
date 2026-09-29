"""Noise floor, floor dropouts and plosive pops. Synthetic audio only."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from finalpass_audiobook.activity import measure
from finalpass_audiobook.checks.dropouts import DropoutTunables, dropout_findings
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


def _cut(background_dbfs: float) -> tuple[np.ndarray, int, int]:
    lead = np.concatenate([_narration(5.0, room_dbfs=background_dbfs), phrase(3)])
    gap = room(1.2, background_dbfs)
    x = np.concatenate([lead, gap, phrase(3), _narration(5.0, room_dbfs=background_dbfs)])
    cut0, cut1 = len(lead) + int(0.4 * SR), len(lead) + int(0.8 * SR)
    x[cut0:cut1] = 0.0
    return x, cut0, cut1


def test_background_cut_to_black_and_back_is_found() -> None:
    lead = np.concatenate([_narration(5.0, room_dbfs=-45.0), phrase(3)])
    gap = room(1.2, -45.0)
    x = np.concatenate([lead, gap, phrase(3), _narration(5.0, room_dbfs=-45.0)])
    cut0, cut1 = len(lead) + int(0.4 * SR), len(lead) + int(0.8 * SR)
    x[cut0:cut1] = 0.0                                   # the room tone drops out to black
    ch = chapter(x)
    floor = measure(ch).floor_dbfs
    (out,) = dropout_findings(ch, floor)
    assert abs(out.start_sample - cut0) < 0.01 * SR and out.problem.startswith("room tone cuts out")
    both = dropout_findings(ch, floor, DropoutTunables(cut_in=True))
    near = lambda at: [f.measures["direction"] for f in both if abs(f.start_sample - at) < 0.01 * SR]
    assert near(cut0) == ["out"] and near(cut1) == ["in"]


def test_natural_endings_into_room_tone_are_not_dropouts() -> None:
    ch = chapter(_narration(12.0, room_dbfs=-45.0))
    assert dropout_findings(ch, measure(ch).floor_dbfs) == []


@pytest.mark.parametrize("background_dbfs, severity", [(-35, 3), (-40, 2), (-47, 1), (-62, None)])
def test_dropout_severity_follows_what_cuts_out(background_dbfs: float, severity: int | None) -> None:
    """Speech about -18 dBFS: from -18 dB -> 3, -26 -> 2, -31 -> 1; quieter room tone is not listed."""
    x, cut0, _ = _cut(background_dbfs)
    ch = chapter(x)
    found = dropout_findings(ch, measure(ch).floor_dbfs)
    assert [f.severity for f in found] == ([severity] if severity else [])
    for f in found:
        assert abs(f.start_sample - cut0) < 0.01 * SR


# --- plosives --------------------------------------------------------------


def _pop(amp: float, hz: float = 40.0) -> np.ndarray:
    t = np.arange(int(0.06 * SR)) / SR
    return amp * np.sin(2 * np.pi * hz * t) * np.exp(-t / 0.02)


def test_low_frequency_pop_is_found_and_speech_alone_is_not() -> None:
    x = _narration(10.0)
    assert plosive_findings(chapter(x), ()) == []
    pop = _pop(0.75)
    at = int(4.0 * SR)
    y = x.copy()
    y[at:at + len(pop)] += pop
    (f,) = plosive_findings(chapter(y), ())
    assert abs(f.start_sample / SR - 4.0) < 0.05


@pytest.mark.parametrize("amp, severity", [(1.0, 3), (0.75, 2), (0.5, 1), (0.2, None)])
def test_plosive_severity_follows_the_margin_over_the_speech(amp: float, severity: int | None) -> None:
    """Listed from speech +3 dB; 2 from +6, 3 from +9. The loudest peak is kept, not the filter's pre-ring."""
    x = _narration(10.0)
    pop = _pop(amp)
    x[int(4.0 * SR):int(4.0 * SR) + len(pop)] += pop
    found = plosive_findings(chapter(x), ())
    assert [f.severity for f in found] == ([severity] if severity else [])
    for f in found:
        rel = f.measures["burst_vs_speech_db"]
        assert f.severity == (3 if rel >= 9 else 2 if rel >= 6 else 1)


def test_energy_above_65_hz_is_not_a_pop() -> None:
    """The voice's own low pitch lives above 65 Hz; the steep band keeps it out."""
    x = _narration(10.0)
    thump = _pop(1.0, hz=110.0)
    x[int(4.0 * SR):int(4.0 * SR) + len(thump)] += thump
    assert plosive_findings(chapter(x), ()) == []


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
    assert "no narration measured" in f.problem and f.severity == 3
