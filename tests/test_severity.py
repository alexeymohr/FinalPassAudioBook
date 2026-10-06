"""Severity 1-3: ladders, mouth-click inhales, hum edges and harmonics, output. Synthetic audio only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from finalpass.breath_check import BreathAssetResult, BreathCounts, BreathEvent

from finalpass_audiobook.checks import breaths as breaths_mod
from finalpass_audiobook.checks.breaths import BreathSeverity, breath_findings, mouth_click_severity
from finalpass_audiobook.checks.hum import hum_findings
from finalpass_audiobook.cli import main
from finalpass_audiobook.findings import FileResult, Finding, RunReport, ladder
from finalpass_audiobook.output import issues_text, write
from synth import SR, chapter, phrase, room
from test_hum import _speech


def test_ladder_boundaries_are_inclusive() -> None:
    cuts = (3.0, 6.0, 9.0)
    assert [ladder(v, cuts) for v in (3.0, 5.99, 6.0, 8.99, 9.0, 20.0)] == [1, 1, 2, 2, 3, 3]


# --- mouth-click inhale ----------------------------------------------------


def _event(start: int, t_inhale: bool = False, loudness: float = -40.0) -> BreathEvent:
    return BreathEvent(start_sample=start, end_sample=start + 11025, start_time="0:00:01.000",
                       end_time="0:00:01.250", duration_ms=250, peak_db=-20.0, body_db=-30.0,
                       noticeability_db=loudness, grade=2, t_inhale=t_inhale, t_inhale_score=0.5,
                       click_gap_samples=None, click_gap_ms=None, click_rel_db=None)


@pytest.mark.parametrize("loudness, severity", [
    (-50.0, 0), (-31.61, 0),                  # quiet: informational
    (-31.6, 1), (-26.41, 1), (-26.4, 2), (-22.51, 2), (-22.5, 3), (-10.0, 3),
])
def test_breath_loudness_scale(loudness, severity) -> None:
    from finalpass_audiobook.checks.breaths import breath_loudness_severity
    assert breath_loudness_severity(loudness) == severity


@pytest.mark.parametrize("silence_ms, click_db, severity", [
    (100, -20.0, 3), (900, -5.0, 3),         # after a pause, harsh click
    (100, -20.01, 2), (400, -30.0, 2),       # after a pause, small click
    (99, -5.0, 0), (0, -5.0, 0),             # right after the word: a consonant, not listed
])
def test_mouth_click_severity(silence_ms, click_db, severity) -> None:
    assert mouth_click_severity(silence_ms, click_db) == severity


def _tick(peak_dbfs: float) -> np.ndarray:
    n = int(0.003 * SR)
    y = np.random.default_rng(7).standard_normal(n) * np.exp(-np.arange(n) / (0.0006 * SR))
    return y / np.max(np.abs(y)) * 10 ** (peak_dbfs / 20)


def _clicky_narration() -> tuple[np.ndarray, dict[str, int]]:
    """Phrases; after some, a tick then (where the fake breath check puts it) a breath."""
    parts, at, n = [], {}, 0

    def add(y: np.ndarray) -> None:
        nonlocal n
        parts.append(y)
        n += len(y)

    for name, silence_s, peak in (("pause_harsh", 0.30, -12.0), ("closure", 0.04, -12.0),
                                  ("pause_small", 0.30, -40.0), ("none", None, None)):
        add(phrase(3)[: -int(0.06 * SR)])            # ends on the word itself
        if silence_s is None:
            add(room(1.0))
            at[name] = n - int(0.6 * SR)
            continue
        add(room(silence_s))
        at[name] = n
        add(_tick(peak))
        add(room(1.0))
    add(phrase(3))
    return np.concatenate(parts), at


def _fake(monkeypatch: pytest.MonkeyPatch, x: np.ndarray, events: list) -> None:
    fake = BreathAssetResult(path="c.wav", sample_rate=SR, duration_seconds=len(x) / SR, narration_dbfs=-18.0,
                             counts=BreathCounts(), breaths=events)
    monkeypatch.setattr(breaths_mod, "analyze_breaths", lambda audio, tunables: fake)


def test_every_breath_is_listed_mouth_clicks_need_a_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    lead = int(0.02 * SR)
    _fake(monkeypatch, x, [_event(at["pause_harsh"] + lead),                    # click after a pause: 3
                           _event(at["closure"] + lead, t_inhale=True),         # right after the word, quiet
                           _event(at["pause_small"] + lead),                    # quiet click after a pause: 2
                           _event(at["none"], loudness=-24.0)])                 # no click, loud: 2
    problems, quiet, _ = breath_findings(chapter(x))
    assert [(f.severity, f.problem.split(":")[0]) for f in problems] == [
        (3, "mouth-click inhale"), (2, "small mouth-click inhale"), (2, "loud breath")]
    assert [(f.severity, f.problem) for f in quiet] == [(0, "quiet breath")]    # nothing left behind
    assert abs(problems[0].start_sample - at["pause_harsh"]) < int(0.004 * SR)  # listed at the click
    assert problems[0].measures["ms_since_word"] >= 250
    assert all("T-inhale" not in f.problem for f in problems + quiet)


def test_a_click_that_is_also_a_loud_breath_is_one_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    x, at = _clicky_narration()
    _fake(monkeypatch, x, [_event(at["pause_small"] + int(0.02 * SR), loudness=-20.0)])
    (f,), quiet, _ = breath_findings(chapter(x))
    assert not quiet
    assert f.severity == 3 and f.problem.startswith("small mouth-click inhale") and f.problem.endswith("; very loud breath")


# --- hum ---------------------------------------------------------------------


def _with_hum(x: np.ndarray, level_dbfs: float, on_s: float, off_s: float, fade_s: float = 0.0,
              partials: tuple[tuple[float, float], ...] = ((60.0, 1.0),)) -> np.ndarray:
    t = np.arange(len(x)) / SR
    gain = ((t >= on_s) & (t < off_s)).astype(float)
    if fade_s:
        gain = np.clip((off_s - t) / fade_s, 0, 1) * (t >= on_s)
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in partials)
    return x + gain * tone * 10 ** (level_dbfs / 20) * np.sqrt(2)


def test_any_steady_hum_is_listed_even_very_quiet() -> None:
    x = np.concatenate([_speech(6.0), room(8.0, -110.0)])
    (f,) = hum_findings(chapter(_with_hum(x, -95.0, 0.0, len(x) / SR)))
    assert f.severity == 1 and abs(f.measures["frequency_hz"] - 60.0) < 0.3


@pytest.mark.parametrize("loudest, abrupt, severity", [
    (-80.0, True, 1), (-55.01, True, 1),        # low-level: 1, abrupt or not
    (-55.0, False, 2), (-40.0, False, 2),       # strong
    (-55.0, True, 3), (-30.0, True, 3),         # strong and abrupt
])
def test_hum_severity_ladder(loudest, abrupt, severity) -> None:
    from finalpass_audiobook.checks.hum import hum_severity
    assert hum_severity(loudest, abrupt) == severity


def test_strong_hums_grade_by_their_edges() -> None:
    x = np.concatenate([_speech(6.0), room(8.0, -90.0)])
    (cut,) = hum_findings(chapter(_with_hum(x, -50.0, 1.0, 10.0)))
    assert "cuts off abruptly" in cut.problem and cut.severity == 3
    (faded,) = hum_findings(chapter(_with_hum(x, -50.0, 0.0, 12.0, fade_s=5.0)))
    assert "abruptly" not in faded.problem and faded.severity == 2
    (quiet,) = hum_findings(chapter(_with_hum(x, -65.0, 1.0, 10.0)))
    assert "cuts off abruptly" in quiet.problem and quiet.severity == 1


def test_a_hum_that_stops_dead_says_so_and_a_fade_does_not() -> None:
    x = np.concatenate([_speech(6.0), room(8.0, -90.0)])
    (cut,) = hum_findings(chapter(_with_hum(x, -65.0, 1.0, 10.0)))
    assert "cuts off abruptly" in cut.problem and cut.measures["end_drop_db"] >= 20
    assert "starts abruptly" in cut.problem                   # switched on hard at 1 s
    (faded,) = hum_findings(chapter(_with_hum(x, -65.0, 1.0, 12.0, fade_s=5.0)))
    assert "cuts off abruptly" not in faded.problem


def test_harmonic_stack_is_measured_even_when_weak() -> None:
    """A 43 Hz hum whose harmonics are too weak to track on their own."""
    x = np.concatenate([_speech(6.0), room(8.0, -85.0)])
    stack = ((43.0, 1.0), (86.0, 0.12), (129.0, 0.12), (172.0, 0.08))
    (f,) = hum_findings(chapter(_with_hum(x, -68.0, 0.0, len(x) / SR, partials=stack)))
    assert f.measures["harmonics_hz"] == "86,129,172"


# --- output ----------------------------------------------------------------


def _report() -> RunReport:
    def fnd(sample: int, sev: int, problem: str) -> Finding:
        return Finding(file="c.wav", check="x", start_sample=sample, end_sample=sample, start_time=f"0:00:0{sample}.000",
                       end_time=f"0:00:0{sample}.000", problem=problem, severity=sev)
    fr = FileResult(file="c.wav", path="c.wav", sample_rate=SR, duration_seconds=9.0, narration_dbfs=-18.0,
                    noise_floor_dbfs=-70.0, findings=[fnd(1, 1, "one"), fnd(2, 3, "three"), fnd(3, 2, "two")],
                    pauses=[])
    return RunReport(version="t", finalpass_version="t", run_id="r", run_started_at="t", rules="standard",
                     tunables={}, network_attempts=0, files=[fr])


def test_issue_list_shows_severity_and_a_tally() -> None:
    text = issues_text(_report())
    assert "== c.wav — 1 × sev 3, 1 × sev 2, 1 × sev 1" in text
    assert "0:00:02.000  sev 3  three" in text
    assert "REJECT" not in text and "NOTE" not in text


def test_min_sev_filters_the_lists_but_json_keeps_everything(tmp_path: Path) -> None:
    write(_report(), tmp_path, min_sev=2)
    rows = (tmp_path / "issues.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].startswith("file,start_time,end_time,event,severity")
    assert [r.split(",")[4] for r in rows[1:]] == ["3", "2"]
    assert "one" not in (tmp_path / "issues.txt").read_text()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema_version"] == 5
    assert [f["severity"] for f in report["files"][0]["findings"]] == [1, 3, 2]


def test_cli_min_sev_option(tmp_path: Path) -> None:
    """A severity-1 plosive pop and a severity-3 digital tick: --min-sev 3 lists only the tick."""
    from test_noise_dropouts_plosives import _before_word, _thump
    x, word = _before_word(_thump(0.05))
    x[word - int(0.5 * SR)] += 10 ** (-30 / 20)                    # a one-sample spike in the pause
    sf.write(str(tmp_path / "ch.wav"), x, SR, subtype="PCM_24")
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--min-sev", "3", "--out", str(tmp_path / "o"),
                                  str(tmp_path / "ch.wav")])
    assert r.exit_code == 0, r.output
    assert "Listing severity 3 and above only." in (tmp_path / "o" / "issues.txt").read_text()
    rows = (tmp_path / "o" / "issues.csv").read_text(encoding="utf-8-sig").splitlines()[1:]
    assert [r.split(",")[5] for r in rows] == ["ticks"]
    everything = json.loads((tmp_path / "o" / "report.json").read_text())["files"][0]["findings"]
    assert sorted((f["check"], f["severity"]) for f in everything) == [("plosive", 1), ("ticks", 3)]
    assert CliRunner().invoke(main, ["check", "--min-sev", "4", str(tmp_path / "ch.wav")]).exit_code != 0


# --- hum heard in the pauses (PLAN §3.3) --------------------------------------


def _low_speech(seconds: float) -> np.ndarray:
    """Narration pitched around 100 Hz, with 0.8 s pauses of quiet room tone."""
    from synth import RNG, word
    parts, n = [], 0
    while n < seconds * SR:                   # vary pitch and length: identical phrases make steady harmonics
        p = np.concatenate([word(RNG.uniform(0.3, 0.5), f0=RNG.uniform(92, 108), glide=RNG.uniform(15, 30)),
                            room(0.05, -95.0),
                            word(RNG.uniform(0.35, 0.55), f0=RNG.uniform(92, 108), glide=RNG.uniform(15, 30)),
                            room(RNG.uniform(0.7, 0.9), -95.0)])
        parts.append(p)
        n += len(p)
    return np.concatenate(parts)[: int(seconds * SR)]


def _tones(n: int, parts: tuple[tuple[float, float], ...]) -> np.ndarray:
    t = np.arange(n) / SR
    return sum(10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * f * t) for f, db in parts)


def test_a_hum_is_described_from_its_pauses_with_every_line() -> None:
    """Two unrelated tones plus their sum and difference tones (a real artifact seen in AI narration)."""
    x = _low_speech(14.0)
    x = x + _tones(len(x), ((43.0, -65.0), (99.0, -56.0), (142.0, -80.0), (185.0, -89.0), (86.0, -74.0)))
    found = hum_findings(chapter(x))
    assert len(found) == 1
    f = found[0]
    import re
    m = re.match(r"hum ([\d.]+) \+ ([\d.]+) Hz", f.problem)
    assert m and abs(float(m[1]) - 43.0) < 0.3 and abs(float(m[2]) - 99.0) < 0.3, f.problem
    assert f.measures["harmonics_hz"].split(",")[0] == "86"
    assert f.measures["other_lines_hz"] == "142,185"


def test_a_hum_cut_on_the_next_word_is_named() -> None:
    lead = _low_speech(8.0)
    x = np.concatenate([lead, _low_speech(8.0)])
    cut = len(lead) - int(0.8 * SR) + int(0.9 * SR)          # stops as the next phrase's word starts
    tone = _tones(len(x), ((43.0, -62.0),))
    tone[cut:] = 0.0
    (f,) = [h for h in hum_findings(chapter(x + tone)) if h.measures["frequency_hz"] < 50]
    assert "cuts off abruptly" in f.problem


def test_pause_line_hums_need_to_be_loud_enough() -> None:
    from finalpass_audiobook.checks.hum import HumTunables, _hidden_hums, _pauses
    t = HumTunables()
    for db, expected in ((-62.0, 1), (-80.0, 0)):
        x = _low_speech(14.0)
        ch = chapter(x + _tones(len(x), ((99.0, db),)))
        y, fs = ch.low
        hidden = _hidden_hums(y, fs, t, _pauses(ch, t), [])
        assert len(hidden) == expected, (db, hidden)
        if hidden:
            assert abs(hidden[0][0] - 99.0) < 0.5 and hidden[0][3] - hidden[0][2] >= t.pause_hum_min_span_s


def test_pause_line_already_part_of_a_hum_is_not_found_twice() -> None:
    from finalpass_audiobook.checks.hum import HumTunables, _hidden_hums, _pauses
    t = HumTunables()
    x = _low_speech(14.0)
    ch = chapter(x + _tones(len(x), ((99.0, -60.0),)))
    y, fs = ch.low
    assert _hidden_hums(y, fs, t, _pauses(ch, t), [(0.0, 14.0, [99.0])]) == []


def test_a_line_steady_only_while_words_sound_is_not_a_hum() -> None:
    """Tracking alone listed such lines on real chapters; the operator heard no tone. It must be in the pauses."""
    x = _low_speech(14.0)
    speaking = (np.abs(x) > 1e-3).astype(float)
    kernel = np.ones(int(0.02 * SR)) / int(0.02 * SR)
    gate = np.convolve(speaking, kernel, mode="same") > 0.5
    found = hum_findings(chapter(x + gate * _tones(len(x), ((41.5, -60.0),))))
    assert not any(abs(f.measures["frequency_hz"] - 41.5) < 1 for f in found)
    (steady,) = [f for f in hum_findings(chapter(x + _tones(len(x), ((41.5, -60.0),))))
                 if abs(f.measures["frequency_hz"] - 41.5) < 1]
    assert steady.severity == 1


@pytest.mark.parametrize("click_db, severity", [(-39.0, 2), (-41.0, 0), (-50.7, 0)])
def test_a_click_too_faint_to_hear_makes_no_mouth_click_inhale(click_db: float, severity: int) -> None:
    """Operator heard no click at -50.7 / -50.6 dB under the narration; the faintest confirmed was -36.6."""
    assert mouth_click_severity(500.0, click_db) == severity
