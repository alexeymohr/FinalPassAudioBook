"""Breaths confirmed by the breath model: the 100 ms rule, mouth-click inhales, the run's counts and the reports'
"breath model" row, with and without the model. Synthetic audio only; the model's probabilities are made by hand."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from finalpass_audiobook import run as run_mod
from finalpass_audiobook.checks.breaths import BreathConfirm, breath_findings, confirmed_ms
from finalpass_audiobook.cli import main
from report_csv import read_report as _report
from synth import SR, chapter, phrase, room
from test_severity import _clicky_narration, _event, _fake


def _probs(x: np.ndarray, spans_s: list[tuple[float, float]]) -> np.ndarray:
    """A breath model's output: 1.0 inside the given spans (seconds), 0 elsewhere, one value per 10 ms."""
    p = np.zeros(1 + int(len(x) / SR * 100), np.float32)
    for a, b in spans_s:
        p[int(round(a * 100)):int(round(b * 100))] = 1.0
    return p


def test_confirmed_ms_counts_the_frames_at_or_over_the_probability() -> None:
    p = np.array([0.0, 0.5, 0.49, 0.9, 1.0, 0.2], np.float32)
    assert confirmed_ms(p, 100, 0, 6) == 30                     # frames of 10 ms; 0.5 counts, 0.49 does not
    assert confirmed_ms(p, 100, 3, 4) == 10
    assert confirmed_ms(p, 100, 4, 4) == 10                      # an empty span still looks at its frame


def test_a_breath_needs_100_ms_of_the_model_saying_breath(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    start = at["none"]                                            # a loud breath, no click (250 ms long)
    _fake(monkeypatch, x, [_event(start, loudness=-24.0)])
    s = start / SR
    as_before, _, _ = breath_findings(chapter(x))
    assert [f.problem for f in as_before] == ["loud breath"]
    for held_s, listed in ((0.0, False), (0.09, False), (0.10, True), (0.25, True)):
        probs, quiet, _ = breath_findings(chapter(x), confirm=_probs(x, [(s, s + held_s)]))
        assert bool(probs) == listed and not quiet, held_s
        if listed:
            assert probs[0].measures["breath_model_ms"] == round(held_s * 1000)
            assert probs[0].problem == "loud breath" and probs[0].severity == as_before[0].severity


def test_quiet_breaths_need_it_too(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["none"], loudness=-40.0)])
    s = at["none"] / SR
    assert len(breath_findings(chapter(x))[1]) == 1
    assert breath_findings(chapter(x), confirm=_probs(x, []))[1] == []
    assert len(breath_findings(chapter(x), confirm=_probs(x, [(s, s + 0.2)]))[1]) == 1


def test_a_mouth_click_inhale_needs_a_confirmed_breath_after_the_click(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    lead = int(0.02 * SR)
    breath = at["pause_harsh"] + lead
    _fake(monkeypatch, x, [_event(breath)])
    (f,), _, _ = breath_findings(chapter(x))
    assert f.problem.startswith("mouth-click inhale") and f.severity == 3
    b = breath / SR
    assert breath_findings(chapter(x), confirm=_probs(x, [(b - 0.02, b + 0.05)]))[0] == []   # click + 50 ms
    (g,), _, _ = breath_findings(chapter(x), confirm=_probs(x, [(b, b + 0.2)]))
    assert g.problem == f.problem and g.start_sample == f.start_sample                      # listed at the click


def test_the_rule_is_a_reported_tunable() -> None:
    opts = run_mod.RunOptions()
    assert opts.tunables()["breath_model"] == {"probability": 0.5, "min_ms": 100.0}
    assert run_mod.RunOptions(breath_model=False).tunables()["breath_model"] == "off"
    assert BreathConfirm().min_ms == 100.0


class _FakeModel:
    def __init__(self, spans_s): self.spans_s = spans_s          # noqa: E704, ANN001

    def probs(self, y16: np.ndarray) -> np.ndarray:
        p = np.zeros(1 + len(y16) // 160, np.float32)
        for a, b in self.spans_s:
            p[int(a * 100):int(b * 100)] = 1.0
        return p


def _narration(path: Path) -> Path:
    from test_hum import _speech
    sf.write(str(path), _speech(12.0), SR, subtype="PCM_24")
    return path


def test_the_run_passes_the_model_to_the_breath_check_and_counts_what_it_drops(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["none"], loudness=-24.0), _event(at["pause_small"], loudness=-24.0)])
    p = tmp_path / "c.wav"
    sf.write(str(p), x, SR, subtype="PCM_24")
    keep = at["none"] / SR
    opts = run_mod.RunOptions(truncation=False)
    fr = run_mod.analyze_file(p, opts, breath_model=_FakeModel([(keep, keep + 0.3)]))
    assert [f.start_sample for f in fr.findings if f.check == "breaths"] == [at["none"]]
    assert fr.counts["breaths_not_confirmed"] == 1
    assert run_mod.analyze_file(p, opts).counts["breaths_not_confirmed"] == 0               # no model: as before


def test_without_the_model_the_reports_say_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "no-models"))
    wav = _narration(tmp_path / "ch.wav")
    r = CliRunner().invoke(main, ["check", "--csv-per-file", "--out", str(tmp_path / "rep"), str(wav)])
    assert r.exit_code == 0, r.output
    summary, _, _ = _report(wav.with_suffix(".csv"))
    assert summary["breath model"] == "off (model not installed)"
    assert any(n.startswith("breath model off: breath model weights not installed")
               for n in __import__("json").loads((tmp_path / "rep" / "report.json").read_text())["notes"])
    wav2 = _narration(tmp_path / "ch2.wav")
    r = CliRunner().invoke(main, ["check", "--csv-per-file", "--no-breath-model", "--no-truncation", str(wav2)])
    assert r.exit_code == 0, r.output
    assert _report(wav2.with_suffix(".csv"))[0]["breath model"] == "off"


def test_setup_model_installs_each_model_and_reports_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "models"))
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\0" * 10)
    r = CliRunner().invoke(main, ["setup-model", "--from-file", str(bad), "--breath-from-file", str(bad)])
    assert r.exit_code == 2
    assert "truncated-word model:" in r.output and "breath model:" in r.output
