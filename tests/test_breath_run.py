"""The breath model end to end, as the review of 0.2.0 asked: the audio the model really gets (16 kHz, in time), the
run and the command line with the model on, off, missing, unloadable and failing, what every report says, short
files, the reported rule taking effect, the speech-loud switch, and installing and downloading the weights.
Synthetic audio only; models are fakes or a tiny random network."""
from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from finalpass_audiobook import breath_model, breath_np
from finalpass_audiobook import run as run_mod
from finalpass_audiobook.checks import breaths as breaths_mod
from finalpass_audiobook.checks.breaths import BreathConfirm, breath_findings
from finalpass_audiobook.cli import main
from finalpass_audiobook.model import ModelError
from report_csv import read_report
from synth import SR, POSIX, chapter, phrase, room
from test_breath_np import TINY, TINY_W
from test_severity import _clicky_narration, _event, _fake


class _EnergyModel:
    """A stand-in breath model that hears the 16 kHz audio it is given: 'breath' wherever the 25 ms level is above
    -60 dBFS, so it confirms a planted noise burst only when the audio arrives at 16 kHz and in time."""
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def probs(self, y16: np.ndarray) -> np.ndarray:
        self.lengths.append(len(y16))
        n = 1 + len(y16) // 160
        pad = np.pad(np.asarray(y16, np.float64), 200)
        e = np.array([np.mean(pad[i * 160:i * 160 + 400] ** 2) for i in range(n)])
        return (10 * np.log10(np.maximum(e, 1e-20)) > -60).astype(np.float32)


def _burst_at(seconds: float, sr: int, burst_s: tuple[float, float]) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.where((t >= burst_s[0]) & (t < burst_s[1]), 0.05, 0.0) * np.sin(2 * np.pi * 1700 * t)


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
def test_the_model_gets_16_khz_audio_in_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sr: int) -> None:
    x = _burst_at(4.0, sr, (2.0, 2.3))
    p = tmp_path / "c.wav"
    sf.write(str(p), x, sr, subtype="PCM_24")
    start, end = int(2.0 * sr), int(2.3 * sr)
    fake_breath = _event(start)
    fake_breath = fake_breath.model_copy(update={"end_sample": end})
    from finalpass.breath_check import BreathAssetResult, BreathCounts
    res = BreathAssetResult(path="c.wav", sample_rate=sr, duration_seconds=4.0, narration_dbfs=-18.0,
                            counts=BreathCounts(), breaths=[fake_breath])
    monkeypatch.setattr(breaths_mod, "analyze_breaths", lambda audio, tunables: res)
    model = _EnergyModel()
    fr = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=model)
    assert model.lengths == [int(np.ceil(len(x) * 16000 / sr))]                # resampled to 16 kHz first
    listed = [f for f in fr.findings + fr.informational if f.check == "breaths"]
    assert len(listed) == 1 and listed[0].measures["breath_model_ms"] >= 280   # the burst lands on its own frames
    late = res.breaths[0].model_copy(update={"start_sample": int(3.0 * sr), "end_sample": int(3.3 * sr)})
    res.breaths[:] = [late]                                                    # a "breath" where the audio is silent
    fr = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=_EnergyModel())
    assert [f for f in fr.findings + fr.informational if f.check == "breaths"] == []


def _speech_wav(path: Path, seconds: float = 8.0, sr: int = SR) -> Path:
    from test_hum import _speech
    x = _speech(seconds)
    if sr != SR:
        from scipy.signal import resample_poly
        x = resample_poly(x, sr, SR)
    sf.write(str(path), x, sr, subtype="PCM_24")
    return path


def _check(tmp_path: Path, wavs: list[Path], *extra: str):
    r = CliRunner().invoke(main, ["check", "--csv-per-file", "--no-truncation", "--out", str(tmp_path / "rep"),
                                  *extra, *map(str, wavs)])
    report = json.loads((tmp_path / "rep" / "report.json").read_text()) if (tmp_path / "rep" / "report.json").exists() else None
    return r, report


def test_the_run_and_the_csv_say_on_when_the_model_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _EnergyModel()
    monkeypatch.setattr(breath_model, "load", lambda: model)
    wav = _speech_wav(tmp_path / "ch.wav")
    r, rep = _check(tmp_path, [wav])
    assert r.exit_code == 0, r.output
    assert model.lengths, "the model never ran"                                # really ran on the file
    summary = read_report(wav.with_suffix(".csv"))[0]
    assert summary["breath model"] == "on" and rep["files"][0]["breath_model"] == "on"
    assert rep["tunables"]["breath_model"] == {"probability": 0.5, "min_ms": 100.0}
    assert rep["tunables"]["breaths"]["speech_loud_rule"] == "off where the breath model ran"
    assert "breath model: on" in r.output and "Breath model: on." in (tmp_path / "rep" / "issues.txt").read_text()


def test_a_model_that_cannot_be_loaded_is_reported_as_such(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken():
        raise ModelError(f"{breath_model.weights_path()}: SHA-256 abc does not match")
    monkeypatch.setattr(breath_model, "load", broken)
    monkeypatch.setattr(breath_model, "installed", lambda: True)
    wav = _speech_wav(tmp_path / "ch.wav")
    r, rep = _check(tmp_path, [wav])
    assert r.exit_code == 0, r.output
    want = "off (model could not be loaded: SHA-256 abc does not match)"
    assert read_report(wav.with_suffix(".csv"))[0]["breath model"] == want
    assert rep["files"][0]["breath_model"] == want and rep["tunables"]["breath_model"] == want
    assert rep["tunables"]["breaths"]["speech_loud_rule"] is True                # the model never ran: the rule did


def test_no_breath_model_never_loads_it_and_every_report_says_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_load():
        raise AssertionError("loaded despite --no-breath-model")
    monkeypatch.setattr(breath_model, "load", must_not_load)
    wav = _speech_wav(tmp_path / "ch.wav")
    r, rep = _check(tmp_path, [wav], "--no-breath-model")
    assert r.exit_code == 0, r.output
    assert read_report(wav.with_suffix(".csv"))[0]["breath model"] == "off"
    assert rep["tunables"]["breath_model"] == "off" and rep["files"][0]["breath_model"] == "off"
    assert "Breath model: off." in (tmp_path / "rep" / "issues.txt").read_text() and "breath model: off" in r.output


def test_a_file_below_16_khz_says_the_model_did_not_run_on_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(breath_model, "load", lambda: _EnergyModel())
    hi, lo = _speech_wav(tmp_path / "hi.wav"), _speech_wav(tmp_path / "lo.wav", sr=12000)
    r, rep = _check(tmp_path, [hi, lo])
    assert r.exit_code == 0, r.output
    assert read_report(hi.with_suffix(".csv"))[0]["breath model"] == "on"
    assert read_report(lo.with_suffix(".csv"))[0]["breath model"] == "off (not run: 12 kHz audio)"
    issues = (tmp_path / "rep" / "issues.txt").read_text()
    assert "breath model: off (not run: 12 kHz audio)" in issues
    low = next(f for f in rep["files"] if f["file"] == "lo.wav")
    assert low["counts"]["breaths"] == 0                                         # the breath check was skipped


def test_a_model_that_fails_on_a_file_costs_only_the_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["none"], loudness=-24.0)])
    p = tmp_path / "c.wav"
    sf.write(str(p), x, SR, subtype="PCM_24")

    class Failing:
        def probs(self, y16):  # noqa: ANN001, ANN201
            raise MemoryError("out of memory")
    fr = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=Failing())
    assert fr.breath_model == "off (failed on this file: MemoryError: out of memory)"
    assert [f.problem for f in fr.findings if f.check == "breaths"] == ["loud breath"]       # listed without it
    assert any("breath model failed on this file" in n for n in fr.notes)


@pytest.mark.parametrize("n", [0, 1, 159, 160, 479, 480, 481])
def test_the_model_takes_audio_of_any_length(n: int) -> None:
    m = breath_np.BreathModel(TINY_W, TINY)
    p = m.probs(np.sin(np.arange(n) / 7.0).astype(np.float32) * 0.1)
    assert p.shape == (1 + n // 160,) and np.all(np.isfinite(p))


def test_a_file_of_a_few_milliseconds_keeps_its_too_short_finding(tmp_path: Path) -> None:
    p = tmp_path / "tiny.wav"
    sf.write(str(p), (0.1 * np.sin(np.arange(240) / 5)).astype(np.float32), 48000, subtype="PCM_24")
    fr = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=breath_np.BreathModel(TINY_W, TINY))
    assert [f.problem for f in fr.findings] == ["file is only 5 ms long"] and fr.breath_model == "on"


def test_non_finite_input_reads_as_silence() -> None:
    m = breath_np.BreathModel(TINY_W, TINY)
    y = (0.1 * np.sin(np.arange(16000) / 9.0)).astype(np.float32)
    clean = y.copy()
    clean[5000:5010] = 0.0
    y[5000:5005], y[5005:5010] = np.nan, np.inf
    assert np.array_equal(m.probs(y), m.probs(clean))


def test_the_reported_rule_is_the_rule_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["none"], loudness=-24.0)])
    s = at["none"] / SR
    p = np.zeros(1 + int(len(x) / SR * 100), np.float32)
    p[int(s * 100):int(s * 100) + 15] = 0.6                                     # 150 ms at 0.6
    assert len(breath_findings(chapter(x), confirm=p)[0]) == 1
    assert breath_findings(chapter(x), confirm=p, rule=BreathConfirm(probability=0.7))[0] == []
    assert breath_findings(chapter(x), confirm=p, rule=BreathConfirm(min_ms=200.0))[0] == []


def test_a_mouth_click_inhale_flagged_by_finalpass_needs_the_model_too(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["closure"] + int(0.02 * SR), t_inhale=True)])
    probs, quiet, _ = breath_findings(chapter(x))
    assert len(probs) + len(quiet) == 1                                         # listed without the model
    none = np.zeros(1 + int(len(x) / SR * 100), np.float32)
    probs, quiet, _ = breath_findings(chapter(x), confirm=none)
    assert probs == [] and quiet == []


def test_the_speech_loud_rule_is_off_only_where_the_model_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    real = breaths_mod.analyze_breaths

    def spy(audio, tunables):  # noqa: ANN001, ANN202
        seen.append(tunables.speech_loud_rule)
        return real(audio, tunables)
    monkeypatch.setattr(breaths_mod, "analyze_breaths", spy)
    p = _speech_wav(tmp_path / "c.wav", seconds=4.0)
    run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=_EnergyModel())
    run_mod.analyze_file(p, run_mod.RunOptions(truncation=False))
    assert seen == [False, True]


def test_the_csv_breath_count_adds_up_and_says_what_the_model_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["none"], loudness=-24.0), _event(at["pause_small"], loudness=-40.0)])
    p = tmp_path / "c.wav"
    sf.write(str(p), x, SR, subtype="PCM_24")
    s = at["none"] / SR
    keep = type("Keep", (), {"probs": lambda self, y: (np.arange(1 + len(y) // 160) / 100 >= s).astype(np.float32)
                             * (np.arange(1 + len(y) // 160) / 100 < s + 0.3)})()
    fr = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=keep)
    c = fr.counts
    assert c["breaths"] == c["breaths_listed"] + c["quiet_breaths"] == 1
    assert c["breaths_detected"] == 2 and c["breaths_not_confirmed"] == 1
    from finalpass_audiobook.output import summary_rows
    row = next(r for r in summary_rows(fr, None) if r[0] == "breaths")
    assert row[1] == 1 and row[-1] == "1 not confirmed by the breath model"


def _fake_weights(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    import hashlib
    monkeypatch.setattr(breath_model, "WEIGHTS_BYTES", len(payload))
    monkeypatch.setattr(breath_model, "WEIGHTS_SHA256", hashlib.sha256(payload).hexdigest())


def test_a_breath_model_download_is_verified_and_capped_before_install(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(breath_model, "WEIGHTS_BYTES", 64)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"y" * 64))
    with pytest.raises(ModelError, match="SHA-256"):
        breath_model.download()
    assert not breath_model.installed() and list(breath_model.model_dir().iterdir()) == []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"y" * 65))
    with pytest.raises(ModelError, match="larger"):
        breath_model.download()
    _fake_weights(monkeypatch, b"y" * 64)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"y" * 64))
    dest = breath_model.download()
    assert dest == breath_model.weights_path() and dest.name == breath_model.FILENAME and dest.read_bytes() == b"y" * 64
    if POSIX:
        assert os.stat(dest).st_mode & 0o044 == 0o044                          # readable by every account


def test_setup_model_installs_each_file_as_its_own_model_and_skips_what_is_installed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from finalpass_audiobook import model as model_mod
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "models"))
    trunc, breath = tmp_path / "t.bin", tmp_path / "b.bin"
    trunc.write_bytes(b"t" * 32)
    breath.write_bytes(b"b" * 48)
    import hashlib
    monkeypatch.setattr(model_mod, "WEIGHTS_BYTES", 32)
    monkeypatch.setattr(model_mod, "WEIGHTS_SHA256", hashlib.sha256(b"t" * 32).hexdigest())
    _fake_weights(monkeypatch, b"b" * 48)
    r = CliRunner().invoke(main, ["setup-model", "--from-file", str(trunc), "--breath-from-file", str(breath)])
    assert r.exit_code == 0, r.output
    assert model_mod.weights_path().read_bytes() == b"t" * 32 and breath_model.weights_path().read_bytes() == b"b" * 48

    def offline(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("went online")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    r = CliRunner().invoke(main, ["setup-model", "--from-file", str(trunc)])     # breath model already there
    assert r.exit_code == 0, r.output
    assert "breath model is already installed and verified" in r.output


def test_weights_carrying_metadata_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import struct
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "models"))
    header = json.dumps({"__metadata__": {"title": "x"}, "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    raw = struct.pack("<Q", len(header)) + header.encode() + b"\0\0\0\0"
    assert breath_model.has_metadata(raw)
    _fake_weights(monkeypatch, raw)
    src = tmp_path / "w.safetensors"
    src.write_bytes(raw)
    breath_model.install_from_file(src)
    with pytest.raises(ModelError, match="metadata"):
        breath_model.load()


@pytest.mark.parametrize("sr", [44100, 48000])
def test_resampling_keeps_speech_and_stops_what_would_alias(sr: int) -> None:
    t = np.arange(2 * sr) / sr
    keep = breath_np.to_16k(np.sin(2 * np.pi * 1000 * t), sr)[2000:-2000]
    stop = breath_np.to_16k(np.sin(2 * np.pi * 10000 * t), sr)[2000:-2000]   # above the new 8 kHz Nyquist
    assert abs(20 * np.log10(np.sqrt(np.mean(keep ** 2)) * np.sqrt(2))) < 0.1
    assert 20 * np.log10(np.sqrt(np.mean(stop ** 2)) * np.sqrt(2) + 1e-12) < -90


def test_an_odd_sample_rate_keeps_the_filter_small() -> None:
    sr = 47999                                                                  # shares no factor with 16000
    y = breath_np.to_16k(np.zeros(sr // 2), sr)
    assert len(y) == int(np.ceil((sr // 2) * 16000 / sr))


def test_the_full_size_network_runs() -> None:
    rng = np.random.default_rng(0)
    w = {k: (rng.standard_normal(s) * 0.05).astype(np.float32) for k, s in breath_np.Config().expected_shapes().items()}
    for k in w:
        if k.endswith("running_var"):
            w[k] = np.abs(w[k]) + 1.0
    t = np.arange(45 * 16000) / 16000
    p = breath_np.BreathModel(w).probs((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    assert p.shape == (1 + 45 * 16000 // 160,) and np.all(np.isfinite(p))


def test_invalid_samples_at_both_ends_are_two_events_not_one_over_the_file(tmp_path: Path) -> None:
    x = np.concatenate([room(1.0), phrase(4), room(2.0), phrase(4), room(1.0)]).astype(np.float32)
    x[100] = np.nan
    x[-100] = np.nan
    p = tmp_path / "c.wav"
    sf.write(str(p), x, SR, subtype="FLOAT")
    ev = [f for f in run_mod.analyze_file(p, run_mod.RunOptions(truncation=False)).findings if f.check == "file"]
    assert [f.problem for f in ev] == ["file contains 1 invalid (NaN, Inf or out-of-range) sample — corrupt audio"] * 2
    assert ev[0].end_sample < SR and ev[1].start_sample > len(x) - SR


@pytest.mark.skipif(not POSIX, reason="a newline cannot be in a Windows file name")
def test_a_newline_in_a_file_name_does_not_break_the_text_report(tmp_path: Path) -> None:
    wav = _speech_wav(tmp_path / "a\nb.wav", seconds=3.0)
    r, _ = _check(tmp_path, [wav], "--no-breath-model")
    assert r.exit_code == 0, r.output
    assert "== a\\nb.wav" in (tmp_path / "rep" / "issues.txt").read_text()


# --- a breath cut off into silence -------------------------------------------------------------


def _cut_breath(hole_ms: float = 60.0, breath_s: float = 0.25, breath_dbfs: float = -50.0, after: bool = True):
    """A phrase, room tone, a breath-like noise running straight into exact digital silence, then a phrase."""
    from synth import band_noise
    pre = np.concatenate([phrase(4), room(0.3)])
    breath = band_noise(breath_s, 1500, 8000, breath_dbfs)
    hole = np.zeros(int(hole_ms * SR / 1000))
    x = np.concatenate([pre, breath, hole] + ([phrase(3), room(0.5)] if after else []))
    return x, len(pre), len(pre) + len(breath)


def _probs(x: np.ndarray, on: tuple[int, int]) -> np.ndarray:
    p = np.zeros(1 + len(x) // (SR // 100), np.float32)
    p[int(on[0] / SR * 100):int(on[1] / SR * 100)] = 1.0
    return p


def test_a_breath_running_into_a_hole_is_reported_as_informational() -> None:
    from finalpass_audiobook.checks.breaths import CUT_OFF_TEXT, breath_cut_findings
    x, b0, b1 = _cut_breath()
    (f,) = breath_cut_findings(chapter(x), _probs(x, (b0, b1)), [])
    assert f.check == "breaths" and f.severity == 0 and f.problem == CUT_OFF_TEXT
    assert abs(f.start_sample - b0) <= SR // 100 and abs(f.end_sample - (b1 + int(0.06 * SR))) <= 2   # a word may start on a zero
    assert f.measures["silence_ms"] == pytest.approx(60.0, abs=0.1) and 230 <= f.measures["duration_ms"] <= 250
    assert f.measures["breath_dbfs"] == pytest.approx(-50.0, abs=1.0)


@pytest.mark.parametrize("case", ["hole too long", "hole too short", "nothing after", "breath stops early",
                                  "breath too short", "already listed"])
def test_only_a_breath_running_straight_into_a_short_hole_counts(case: str) -> None:
    from finalpass_audiobook.checks.breaths import breath_cut_findings
    x, b0, b1 = _cut_breath(hole_ms={"hole too long": 100.0, "hole too short": 45.0}.get(case, 60.0),
                            breath_s=0.03 if case == "breath too short" else 0.25, after=case != "nothing after")
    on = (b0, b1 - int(0.06 * SR)) if case == "breath stops early" else (b0, b1)
    listed = [(b0, b1)] if case == "already listed" else []
    assert breath_cut_findings(chapter(x), _probs(x, on), listed) == []


def test_the_run_reports_a_quiet_breath_cut_off_into_silence_once(tmp_path: Path) -> None:
    """A breath too quiet for the loudness scale moves up from informational; without the model nothing changes."""
    x, b0, b1 = _cut_breath(breath_dbfs=-55.0)
    p = tmp_path / "c.wav"
    sf.write(str(p), x, SR, subtype="PCM_24")
    with_model = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=_EnergyModel(),
                                      breath_state="on")
    assert not [f for f in with_model.findings if f.problem == breaths_mod.CUT_OFF_TEXT]      # not a problem
    near = [q for q in with_model.informational if q.start_sample < b1 + SR // 10 and q.end_sample > b0]
    assert [q.problem for q in near] == [breaths_mod.CUT_OFF_TEXT] and near[0].severity == 0
    assert with_model.counts["breaths_cut_off"] == 1
    without = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False, breath_model=False))
    assert not [f for f in without.findings + without.informational if f.problem == breaths_mod.CUT_OFF_TEXT]
    assert run_mod.RunOptions().tunables("off")["breath_cut_off"] == "off (needs the breath model)"


def test_a_quiet_breath_that_is_cut_off_is_listed_once_not_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from finalpass_audiobook.findings import Finding
    x, b0, b1 = _cut_breath(breath_dbfs=-55.0)
    p = tmp_path / "c.wav"
    sf.write(str(p), x, SR, subtype="PCM_24")
    real = run_mod.breath_findings

    def with_a_quiet_one(ch, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        problems, quiet, result = real(ch, *args, **kwargs)
        q = Finding(file=ch.name, check="breaths", start_sample=b0, end_sample=b1, start_time=ch.clock(b0),
                    end_time=ch.clock(b1), severity=0, problem="quiet breath", measures={})
        return problems, quiet + [q], result

    monkeypatch.setattr(run_mod, "breath_findings", with_a_quiet_one)
    fr = run_mod.analyze_file(p, run_mod.RunOptions(truncation=False), breath_model=_EnergyModel(), breath_state="on")
    assert [q.problem for q in fr.informational if q.start_sample < b1 and q.end_sample > b0] == [breaths_mod.CUT_OFF_TEXT]
    assert fr.counts["quiet_breaths"] == len(fr.informational) - fr.counts["breaths_cut_off"]
