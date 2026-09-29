"""Clicks in pauses. Synthetic audio only."""
from __future__ import annotations

import numpy as np
import pytest

from finalpass_audiobook.activity import measure
from finalpass_audiobook.checks.clicks import click_findings
from synth import RNG, SR, band_noise, chapter, phrase, room


@pytest.fixture(autouse=True)
def _leave_shared_noise_alone():
    """These tests draw from synth's shared RNG; put its state back so tests in later files
    see the same noise whether or not these ran."""
    state = RNG.bit_generator.state
    yield
    RNG.bit_generator.state = state


def _tick(peak_dbfs: float, sr: int = SR) -> np.ndarray:
    """A 3 ms decaying broadband tick."""
    n = int(0.003 * sr)
    y = RNG.standard_normal(n) * np.exp(-np.arange(n) / (0.0006 * sr))
    return y / np.max(np.abs(y)) * 10 ** (peak_dbfs / 20)


def _with(insert: np.ndarray | None, at_ms: float, pause_s: float = 1.5, sr: int = SR) -> tuple[np.ndarray, int]:
    """phrase, pause, phrase, pause, phrase; `insert` added `at_ms` into the first pause."""
    p = phrase(4)
    gap = room(pause_s)
    x = np.concatenate([p, gap, phrase(4), room(0.7), phrase(3)])
    at = len(p) + int(at_ms * sr / 1000)
    if insert is not None:
        x[at:at + len(insert)] += insert
    return x, at


def _clicks(x: np.ndarray, spans=()):
    ch = chapter(x)
    return click_findings(ch, measure(ch).pauses, list(spans))


def test_a_tick_in_the_middle_of_a_pause_is_found() -> None:
    x, at = _with(_tick(-35.0), 700)
    found = _clicks(x)
    assert len(found) == 1
    f = found[0]
    assert f.severity == 3 and f.check == "clicks"
    assert abs(f.start_sample - at) < int(0.004 * SR)
    assert f.measures["peak_dbfs"] == pytest.approx(-35.0, abs=0.5)


def test_clean_narration_lists_nothing() -> None:
    x, _ = _with(None, 0)
    assert _clicks(x) == []


def test_a_quiet_tick_is_not_listed() -> None:
    x, _ = _with(_tick(-50.0), 700)
    assert _clicks(x) == []


@pytest.mark.parametrize("at_ms", [40.0, 1500.0 - 60.0])
def test_a_tick_close_to_a_word_is_not_listed(at_ms: float) -> None:
    x, _ = _with(_tick(-35.0), at_ms)
    assert _clicks(x) == []


def test_a_tick_just_before_a_breath_is_left_to_the_breath_check() -> None:
    x, at = _with(_tick(-35.0), 700)
    breath = (at + int(0.06 * SR), at + int(0.40 * SR))
    assert _clicks(x, [breath]) == []
    assert len(_clicks(x)) == 1


def test_a_sustained_hiss_in_a_pause_is_not_a_click() -> None:
    x, _ = _with(band_noise(0.08, 4000, 8000, -30.0), 700)
    assert _clicks(x) == []


def test_found_at_48k_too() -> None:
    sr = 48000
    x44, _ = _with(None, 0)
    t = np.arange(int(len(x44) * sr / SR)) / sr
    x = np.interp(t, np.arange(len(x44)) / SR, x44)
    at = int((len(phrase(4)) / SR + 0.7) * sr)
    x[at:at + int(0.003 * sr)] += _tick(-35.0, sr)
    from pathlib import Path
    from finalpass.audio_io import AudioFile
    from finalpass_audiobook.chapter import Chapter
    audio = AudioFile(path=Path("c.wav"), data=x[:, None], sample_rate=sr, bit_depth=24, channel_count=1,
                      duration_seconds=len(x) / sr)
    ch = Chapter(path=Path("c.wav"), audio=audio, x=x, sr=sr)
    assert len(click_findings(ch, measure(ch).pauses, [])) == 1
