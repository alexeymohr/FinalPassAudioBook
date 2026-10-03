"""Whole-file problems are listed, never passed off as a clean file, and corrupt samples do not show
up again as other events. Synthetic audio only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from finalpass_audiobook.run import RunOptions, analyze_file
from synth import SR, phrase, room


def _analyse(tmp_path: Path, x: np.ndarray, subtype: str = "PCM_24", name: str = "c.wav"):
    p = tmp_path / name
    sf.write(str(p), x, SR, subtype=subtype)
    return analyze_file(p, RunOptions(truncation=False))


def _file_events(fr) -> list[tuple[int, str]]:        # noqa: ANN001
    return [(f.severity, f.problem) for f in fr.findings if f.check == "file"]


def _narration() -> np.ndarray:
    return np.concatenate([room(1.0), phrase(4), room(0.9), phrase(4), room(1.5)])


def test_a_silent_file_is_a_problem_not_a_clean_result(tmp_path: Path) -> None:
    for x in (np.zeros(SR * 5), room(5.0)):
        assert _file_events(_analyse(tmp_path, x)) == [(3, "no narration found in the file")]


def test_a_file_of_a_few_samples_is_a_problem(tmp_path: Path) -> None:
    assert _file_events(_analyse(tmp_path, np.concatenate([phrase(1), room(0.5)])[: SR // 2])) == [
        (3, "file is only 500 ms long")]


def test_a_file_cut_short_is_a_problem(tmp_path: Path) -> None:
    p = tmp_path / "full.wav"
    sf.write(str(p), _narration(), SR, subtype="PCM_16")
    cut = tmp_path / "cut.wav"
    cut.write_bytes(p.read_bytes()[: p.stat().st_size * 2 // 3])
    fr = analyze_file(cut, RunOptions(truncation=False))
    assert [f.problem for f in fr.findings if f.check == "file"] == [
        "file is cut short: its audio ends before its header says it should"]
    assert all(f.check != "file" for f in analyze_file(p, RunOptions(truncation=False)).findings)


def test_normal_narration_has_no_file_event(tmp_path: Path) -> None:
    assert _file_events(_analyse(tmp_path, _narration())) == []


def test_a_nan_in_one_channel_of_dual_mono_is_corruption_not_a_difference(tmp_path: Path) -> None:
    x = _narration()
    st = np.stack([x, x], axis=1)
    st[SR + 1000, 1] = np.nan
    fr = _analyse(tmp_path, st, subtype="FLOAT")
    assert fr.sample_rate == SR
    assert _file_events(fr) == [(3, "file contains 1 invalid (NaN, Inf or out-of-range) sample — corrupt audio")]


def test_a_run_of_nan_samples_is_listed_once_not_again_as_a_dropout_or_ticks(tmp_path: Path) -> None:
    x = _narration()
    at = SR + int(0.1 * SR)                                  # inside the first word
    x[at:at + int(0.012 * SR)] = np.nan
    fr = _analyse(tmp_path, x, subtype="FLOAT")
    near = [f for f in fr.findings if abs(f.start_sample - at) < 0.05 * SR]
    assert [f.check for f in near] == ["file"]


def test_eight_bit_audio_skips_the_exact_zero_checks_with_a_note(tmp_path: Path) -> None:
    """8-bit steps round quiet audio to exact zero: no dropout and no clip-end (chopped-word) test."""
    from test_truncation_model_cli import _FakeModel
    p = tmp_path / "c.wav"
    sf.write(str(p), _narration(), SR, subtype="PCM_U8")
    fr = analyze_file(p, RunOptions(), model=_FakeModel([0.999] * 50))
    assert all(f.check not in ("dropout", "truncation") for f in fr.findings) and fr.counts["phrase_ends_scored"] == 0
    assert "dropout and chopped-word checks skipped: 8-bit audio" in fr.notes


def test_one_corrupt_sample_does_not_hide_a_hum_or_a_noisy_section(tmp_path: Path) -> None:
    x = np.concatenate([_narration()] * 4)
    t = np.arange(len(x)) / SR
    x = x + 10 ** (-45 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)
    x[len(x) // 2] = np.nan
    fr = _analyse(tmp_path, x, subtype="FLOAT")
    assert {"hum", "file"} <= {f.check for f in fr.findings}


def test_a_header_off_by_less_than_a_frame_is_not_cut_short(tmp_path: Path) -> None:
    import struct
    p = tmp_path / "full.wav"
    sf.write(str(p), _narration(), SR, subtype="PCM_16")
    b = bytearray(p.read_bytes())
    k = b.index(b"data") + 4
    size = struct.unpack("<I", b[k:k + 4])[0]
    b[k:k + 4] = struct.pack("<I", size + 1)                     # one byte of a two-byte frame
    q = tmp_path / "plus1.wav"
    q.write_bytes(bytes(b))
    assert _file_events(analyze_file(q, RunOptions(truncation=False))) == []
