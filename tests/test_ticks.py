"""Digital ticks. Synthetic audio only."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from finalpass_audiobook.checks.ticks import tick_findings
from synth import RNG, SR, chapter, phrase, room


@pytest.fixture(autouse=True)
def _leave_shared_noise_alone():
    state = RNG.bit_generator.state
    yield
    RNG.bit_generator.state = state


def _band_limited(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Like the rendered narration: nothing above 16 kHz."""
    return sosfiltfilt(butter(12, 16000, "lowpass", fs=sr, output="sos"), x)


def _narration() -> tuple[np.ndarray, int, int]:
    """Two phrases around a 1 s pause; returns (audio, a point in the pause, a point in a word)."""
    a = phrase(4)
    x = _band_limited(np.concatenate([a, room(1.0), phrase(4)]))
    return x, len(a) + int(0.5 * SR), int(0.15 * SR)


def test_a_one_sample_spike_in_a_pause_is_a_tick() -> None:
    x, at, _ = _narration()
    x[at] += 10 ** (-30 / 20)
    (f,) = tick_findings(chapter(x))
    assert f.severity == 3 and f.check == "ticks" and f.problem == "digital tick"
    assert abs(f.start_sample - at) <= 2
    assert f.measures["width_samples"] <= 8 and f.measures["isolation_db"] >= 20


def test_band_limited_narration_has_no_ticks() -> None:
    x, _, _ = _narration()
    assert tick_findings(chapter(x)) == []


def test_a_quiet_spike_is_not_listed() -> None:
    """In digital black, where only the level limit can keep it off the list."""
    x, at, _ = _narration()
    x[at - int(0.1 * SR):at + int(0.1 * SR)] = 0.0
    y = x.copy()
    y[at] += 10 ** (-70 / 20)
    assert tick_findings(chapter(y)) == []
    y[at] += 10 ** (-40 / 20)
    assert len(tick_findings(chapter(y))) == 1


def test_a_voice_made_click_is_not_a_digital_tick() -> None:
    """A 3 ms decaying click that stops at 16 kHz, like the model's own ticks."""
    x, at, _ = _narration()
    n = int(0.003 * SR)
    click = RNG.standard_normal(n) * np.exp(-np.arange(n) / (0.0006 * SR))
    x[at:at + n] += _band_limited(click / np.max(np.abs(click)) * 10 ** (-20 / 20))
    assert tick_findings(chapter(x)) == []


def test_a_spike_with_company_is_not_alone() -> None:
    """Broadband crackle: several spikes within a few ms of each other."""
    x, at, _ = _narration()
    for k in range(5):
        x[at + k * int(0.002 * SR)] += 10 ** (-30 / 20)
    assert tick_findings(chapter(x)) == []


def test_found_at_48k_too() -> None:
    sr = 48000
    x44, at44, _ = _narration()
    t = np.arange(int(len(x44) * sr / SR)) / sr
    x = np.interp(t, np.arange(len(x44)) / SR, x44)
    x = sosfiltfilt(butter(12, 16000, "lowpass", fs=sr, output="sos"), x)
    at = int(at44 * sr / SR)
    x[at] += 10 ** (-30 / 20)
    from pathlib import Path
    from finalpass.audio_io import AudioFile
    from finalpass_audiobook.chapter import Chapter
    audio = AudioFile(path=Path("c.wav"), data=x[:, None], sample_rate=sr, bit_depth=24, channel_count=1,
                      duration_seconds=len(x) / sr)
    assert len(tick_findings(Chapter(path=Path("c.wav"), audio=audio, x=x, sr=sr))) == 1


def test_low_sample_rates_are_skipped() -> None:
    from pathlib import Path
    from finalpass.audio_io import AudioFile
    from finalpass_audiobook.chapter import Chapter
    sr = 22050
    x = np.zeros(sr)
    x[sr // 2] = 0.5
    audio = AudioFile(path=Path("c.wav"), data=x[:, None], sample_rate=sr, bit_depth=16, channel_count=1,
                      duration_seconds=1.0)
    assert tick_findings(Chapter(path=Path("c.wav"), audio=audio, x=x, sr=sr)) == []
