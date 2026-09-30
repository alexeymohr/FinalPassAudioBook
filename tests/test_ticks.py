"""Digital ticks. Synthetic audio only."""
from __future__ import annotations

import numpy as np
import pytest

from finalpass_audiobook.checks.ticks import tick_findings
from synth import RNG, SR, chapter, phrase, room


def _band_limited(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Like the rendered narration: nothing above about 16 kHz (a brick wall at 15.5 kHz)."""
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1 / sr) > 15500] = 0
    return np.fft.irfft(spec, len(x))


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
    x = _band_limited(x, sr)
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


def test_a_digital_tick_in_a_pause_is_listed_once(tmp_path) -> None:
    import soundfile as sf
    from finalpass_audiobook.run import RunOptions, analyze_file
    x, at, _ = _narration()
    x[at] += 10 ** (-30 / 20)
    p = tmp_path / "c.wav"
    sf.write(p, x, SR, subtype="PCM_24")
    found = [f for f in analyze_file(p, RunOptions(truncation=False)).findings if f.check in ("ticks", "clicks")]
    assert [f.check for f in found] == ["ticks"]


def test_a_dropout_is_listed_once_not_again_as_ticks_at_its_edges(tmp_path) -> None:        # noqa: ANN001
    """The step into and out of a hole is the dropout's own edge (the raw tick check does fire there)."""
    import soundfile as sf
    from finalpass_audiobook.run import RunOptions, analyze_file
    x = np.concatenate([room(1.0), phrase(4), room(1.0), phrase(4), room(1.0)])
    at = SR + int(0.1 * SR)                                          # inside the first word
    x[at:at + int(0.012 * SR)] = 0.0
    assert any(abs(f.start_sample - at) < 5 for f in tick_findings(chapter(x)))
    p = tmp_path / "c.wav"
    sf.write(p, x, SR, subtype="PCM_24")
    near = [f.check for f in analyze_file(p, RunOptions(truncation=False)).findings if abs(f.start_sample - at) < 0.05 * SR]
    assert near == ["dropout"]


def test_a_tick_in_digital_black_is_still_a_tick(tmp_path) -> None:        # noqa: ANN001
    """Black follows the spike itself: that is not a clip end that could excuse it."""
    import soundfile as sf
    from finalpass_audiobook.run import RunOptions, analyze_file
    x = np.concatenate([_band_limited(phrase(4)), np.zeros(SR), _band_limited(phrase(4)), np.zeros(int(0.5 * SR))])
    at = len(phrase(4)) + int(0.3 * SR)
    x[at] = 10 ** (-40 / 20)
    p = tmp_path / "c.wav"
    sf.write(p, x, SR, subtype="PCM_24")
    ticks = [f for f in analyze_file(p, RunOptions(truncation=False)).findings if f.check == "ticks"]
    assert len(ticks) == 1 and abs(ticks[0].start_sample - at) <= 2


def test_a_word_that_ends_in_a_tick_is_listed_once_as_the_tick(tmp_path) -> None:        # noqa: ANN001
    import soundfile as sf
    from finalpass_audiobook.run import RunOptions, analyze_file
    from test_truncation_model_cli import _FakeModel
    word_end = len(phrase(4)) - int(0.2 * SR)
    x = np.concatenate([_band_limited(phrase(4))[:word_end], np.zeros(int(0.7 * SR)), _band_limited(phrase(4)),
                        np.zeros(int(0.5 * SR))])
    x[word_end - 1] = 0.25                                        # the cut clicks
    p = tmp_path / "c.wav"
    sf.write(p, x, SR, subtype="PCM_24")
    fr = analyze_file(p, RunOptions(), model=_FakeModel([0.999, 0.999]))
    at_cut = [f.check for f in fr.findings if abs(f.start_sample - word_end) < 0.003 * SR]
    assert at_cut == ["ticks"]
