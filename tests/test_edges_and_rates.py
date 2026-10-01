"""File edges, odd sample rates and odd headers: what the final review found. Synthetic audio only."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from finalpass.audio_io import AudioFile

from finalpass_audiobook.chapter import Chapter
from finalpass_audiobook.run import RunOptions, analyze_file
from synth import SR, chapter, phrase, room
from test_hum import _speech
from test_ticks import _band_limited


def _narration() -> np.ndarray:
    return np.concatenate([room(1.0), phrase(4), room(0.9), phrase(4), room(2.5)])


def _analyse(tmp_path: Path, x: np.ndarray, sr: int = SR, subtype: str = "PCM_24"):
    p = tmp_path / "c.wav"
    sf.write(str(p), x, sr, subtype=subtype)
    return analyze_file(p, RunOptions(truncation=False))


def _at_rate(x: np.ndarray, sr: int) -> Chapter:
    audio = AudioFile(path=Path("c.wav"), data=x[:, None], sample_rate=sr, bit_depth=24, channel_count=1,
                      duration_seconds=len(x) / sr)
    return Chapter(path=Path("c.wav"), audio=audio, x=x, sr=sr)


@pytest.mark.parametrize("where", ["start", "end"])
def test_a_spike_at_the_very_edge_is_a_tick_not_a_word(tmp_path: Path, where: str) -> None:
    x = _band_limited(_narration())
    at = int(0.002 * SR) if where == "start" else len(x) - int(0.003 * SR)
    x[at] += 10 ** (-30 / 20)
    fr = _analyse(tmp_path, x)
    head, tail = fr.pauses[0], fr.pauses[-1]
    assert abs(head.duration_ms - 1000) < 30 and abs(tail.duration_ms - 2500) < 80     # the pause map holds
    assert [f.check for f in fr.findings if abs(f.start_sample - at) < 0.005 * SR] == ["ticks"]


def test_clicks_in_the_head_and_tail_are_found_through_the_whole_run(tmp_path: Path) -> None:
    from test_clicks import _tick
    x = _narration()
    for at in (int(0.5 * SR), len(x) - int(1.0 * SR)):
        x[at:at + int(0.003 * SR)] += _tick(-35.0)
    problems = [f.problem for f in _analyse(tmp_path, x).findings if f.check == "clicks"]
    assert problems == ["click before the first word", "click after the last word"]


def test_a_loud_hum_interrupted_by_a_patch_of_clean_room_tone_is_still_a_hum() -> None:
    from finalpass_audiobook.checks.hum import hum_findings
    x = _speech(60.0)
    t = np.arange(len(x)) / SR
    hum = ((t >= 15.0) & (t < 45.0)) * 10 ** (-45 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)
    patch = (t >= 30.0) & (t < 30.6)
    x = np.where(patch, room(60.0)[: len(x)], x + hum)            # an edit patch with no hum in it
    found = hum_findings(chapter(x))
    assert [f.check for f in found] == ["hum"] and found[0].severity == 3


def test_a_hum_as_loud_in_the_words_but_quieter_in_the_gaps_is_not_confirmed_there() -> None:
    from finalpass_audiobook.checks.hum import hum_findings
    x = _speech(40.0)
    t = np.arange(len(x)) / SR
    talking = np.convolve(np.abs(x) > 1e-3, np.ones(int(0.05 * SR)), mode="same") > 0
    level = np.where(talking, 10 ** (-34 / 20), 10 ** (-46 / 20))
    x = x + ((t >= 15.0) & (t < 25.0)) * level * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)
    assert hum_findings(chapter(x)) == []                         # it follows the voice: not a steady hum


def test_a_hum_that_starts_abruptly_is_not_called_building() -> None:
    from finalpass_audiobook.checks.hum import hum_findings
    x = _speech(60.0)
    t = np.arange(len(x)) / SR
    x = x + (t >= 30.0) * 10 ** (-45 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)
    (f,) = hum_findings(chapter(x))
    assert "starts abruptly" in f.problem and "building" not in f.problem


def test_an_absurd_float_sample_is_corrupt_audio_not_a_disaster(tmp_path: Path) -> None:
    x = _narration()
    x[SR + 100] = 1e10
    fr = _analyse(tmp_path, x, subtype="FLOAT")
    assert [f.check for f in fr.findings] == ["file"] and len(fr.pauses) == 3   # the analysis is unharmed


def test_low_rate_audio_skips_the_breath_check_with_a_note(tmp_path: Path) -> None:
    from scipy.signal import resample_poly
    x = resample_poly(_narration(), 1, 4)                         # 11.025 kHz
    fr = _analyse(tmp_path, x, sr=SR // 4)
    assert all(f.check != "breaths" for f in fr.findings + fr.informational)
    assert any(n.startswith("breath check skipped: 11.025 kHz") for n in fr.notes)


@pytest.mark.parametrize("claimed, cut", [(2_500_000_000, True), (0xFFFFFFFF, False), ("plus2", False)])
def test_header_sizes_that_are_and_are_not_a_file_cut_short(tmp_path: Path, claimed, cut: bool) -> None:  # noqa: ANN001
    p = tmp_path / "full.wav"
    sf.write(str(p), _narration(), SR, subtype="PCM_24")
    b = bytearray(p.read_bytes())
    k = b.index(b"data") + 4
    size = struct.unpack("<I", b[k:k + 4])[0]
    b[k:k + 4] = struct.pack("<I", size + 2 if claimed == "plus2" else claimed)   # 2 bytes: under a 3-byte frame
    q = tmp_path / "h.wav"
    q.write_bytes(bytes(b))
    events = [f.problem for f in analyze_file(q, RunOptions(truncation=False)).findings if f.check == "file"]
    assert bool(events) == cut and all("cut short" in e for e in events)


def test_a_file_just_under_a_second_says_so(tmp_path: Path) -> None:
    x = np.concatenate([phrase(1), room(1.0)])[: int(0.9998 * SR)]
    assert [f.problem for f in _analyse(tmp_path, x).findings] == ["file is only 999 ms long"]


def test_ticks_behave_at_high_rates_as_at_44k() -> None:
    """The same audio at 44.1 and 96 kHz: a one-sample spike is found at both, and the edges of a
    40 ms hole of black inside a word are judged the same way at both."""
    from finalpass_audiobook.checks.ticks import tick_findings
    x44 = _band_limited(np.concatenate([phrase(4), room(1.0), phrase(4), room(0.5)]))
    edges_listed = {}
    for sr in (44100, 96000):
        t = np.arange(int(len(x44) * sr / SR)) / sr
        x = _band_limited(np.interp(t, np.arange(len(x44)) / SR, x44), sr)
        hole = int(0.12 * sr)
        x[hole:hole + int(0.04 * sr)] = 0.0
        spike = int((len(phrase(4)) / SR + 0.5) * sr)
        x[spike] += 10 ** (-30 / 20)
        found = [f.start_sample for f in tick_findings(_at_rate(x, sr))]
        assert any(abs(s - spike) <= 4 for s in found), sr
        edges_listed[sr] = [any(abs(s - e) <= int(0.001 * sr) for s in found) for e in (hole, hole + int(0.04 * sr))]
    assert edges_listed[96000] == edges_listed[44100]


def test_a_pause_is_guessed_from_the_length_the_map_shows() -> None:
    from finalpass_audiobook.activity import Activity
    from finalpass_audiobook.checks.pauses import pause_map
    from finalpass_audiobook.rules import RULE_SETS
    rules = RULE_SETS["standard"]
    lo = rules.paragraph_s[0]
    ch = chapter(np.zeros(10 * SR))
    s = 3 * SR
    act = Activity(-60.0, -80.0, SR, 9 * SR, ((s, s + int((lo - 0.0004) * SR)),), ())
    (_, internal, _) = pause_map(ch, act, rules)
    assert f"{internal.duration_ms / 1000:.2f}" == f"{lo:.2f}" and internal.guess == "paragraph break"
