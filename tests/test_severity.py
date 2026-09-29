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


def _event(gap: int | None = 0, click_db: float | None = -10.0, t_inhale: bool = True, grade: int = 2,
           start: int = 44100) -> BreathEvent:
    return BreathEvent(start_sample=start, end_sample=start + 11025, start_time="0:00:01.000",
                       end_time="0:00:01.250", duration_ms=250, peak_db=-20.0, body_db=-30.0,
                       noticeability_db=-28.0, grade=grade, t_inhale=t_inhale, t_inhale_score=0.5,
                       click_gap_samples=gap, click_gap_ms=None if gap is None else gap / 44.1,
                       click_rel_db=click_db)


@pytest.mark.parametrize("gap, click_db, severity", [
    (250, -20.0, 3), (1323, -5.0, 3),        # clear gap, harsh click
    (250, -20.01, 2), (600, -23.9, 2),       # clear gap, small click
    (249, -5.0, 1), (0, -5.0, 1), (None, None, 1),
])
def test_mouth_click_severity_at_44k1(gap, click_db, severity) -> None:
    assert mouth_click_severity(_event(gap, click_db), 44100) == severity


def test_gap_limit_scales_with_the_sample_rate() -> None:
    """250 samples at 44.1 kHz is 5.67 ms: 272 samples at 48 kHz."""
    assert mouth_click_severity(_event(271), 48000) == 1
    assert mouth_click_severity(_event(273), 48000) == 3


def test_breath_findings_wording_and_combined_severity(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [_event(0, -5.0, start=44100),                            # run-on consonant: 1
              _event(400, -5.0, start=4 * 44100),                      # mouth-click inhale: 3
              _event(0, -5.0, t_inhale=False, grade=3, start=8 * 44100),   # loud breath: 2
              _event(0, -5.0, grade=3, start=12 * 44100),              # both: the higher, 2
              _event(0, -5.0, t_inhale=False, grade=2, start=16 * 44100)]  # ordinary: not listed
    fake = BreathAssetResult(path="c.wav", sample_rate=SR, duration_seconds=20.0, narration_dbfs=-18.0,
                             counts=BreathCounts(), breaths=events)
    monkeypatch.setattr(breaths_mod, "analyze_breaths", lambda audio, tunables: fake)
    found, _ = breath_findings(chapter(room(1.0)))
    assert [f.severity for f in found] == [1, 3, 2, 2]
    assert found[0].problem == "hard consonant runs into inhale (possible mouth-click inhale)"
    assert found[1].problem.startswith("mouth-click inhale")
    assert found[2].problem == "loud breath"
    assert found[3].problem == "hard consonant runs into inhale (possible mouth-click inhale); loud breath"
    assert all("T-inhale" not in f.problem for f in found)
    assert BreathSeverity().loud_breath == 2


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
    assert f.severity == 3 and abs(f.measures["frequency_hz"] - 60.0) < 0.3


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
    assert rows[0].startswith("file,time,problem,severity")
    assert [r.split(",")[3] for r in rows[1:]] == ["3", "2"]
    assert "one" not in (tmp_path / "issues.txt").read_text()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema_version"] == 2
    assert [f["severity"] for f in report["files"][0]["findings"]] == [1, 3, 2]


def test_cli_min_sev_option(tmp_path: Path) -> None:
    x = np.concatenate([room(1.0), phrase(4), room(2.5)])
    sf.write(str(tmp_path / "ch.wav"), x, SR, subtype="PCM_24")
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--min-sev", "3", "--out", str(tmp_path / "o"),
                                  str(tmp_path / "ch.wav")])
    assert r.exit_code == 0, r.output
    assert "Listing severity 3 and above only." in (tmp_path / "o" / "issues.txt").read_text()
    assert CliRunner().invoke(main, ["check", "--min-sev", "4", str(tmp_path / "ch.wav")]).exit_code != 0


# --- hum heard in the pauses (PLAN §8) --------------------------------------


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
    m = re.match(r"hum at ([\d.]+) Hz and ([\d.]+) Hz", f.problem)
    assert m and abs(float(m[1]) - 43.0) < 0.3 and abs(float(m[2]) - 99.0) < 0.3, f.problem
    assert f.measures["harmonics_hz"].split(",")[0] == "86"
    assert "other steady lines at 142, 185 Hz in the pauses" in f.problem


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
    assert steady.severity == 3
