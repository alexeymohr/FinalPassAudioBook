"""`fpab check` options the macOS app uses: one CSV per WAV, pause rows, JSON-lines progress.
Synthetic audio only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from finalpass_audiobook.cli import main
from finalpass_audiobook.run import STAGES
from finalpass_audiobook.output import REPORT_MARKER
from report_csv import read_report as _report
from synth import SR, phrase, room


def _wav(path: Path, spike: bool = False) -> Path:
    """Two phrases; with `spike`, a one-sample digital tick in the pause between them (severity 3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([room(1.2), phrase(4), room(0.9), phrase(3), room(3.0)])
    if spike:
        x[int(1.2 * SR) + len(phrase(4)) + int(0.45 * SR)] += 10 ** (-30 / 20)
    sf.write(str(path), x, SR, subtype="PCM_24")
    return path


def _rows(path: Path) -> list[dict]:
    _, problems, info = _report(path)
    return problems + info


def test_one_csv_beside_each_wav_with_progress_lines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = _wav(tmp_path / "one" / "ch01.wav"), _wav(tmp_path / "two" / "ch02.wav")
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-per-file", "--progress", "jsonl", str(a), str(b)])
    assert r.exit_code == 0, r.output
    events = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("file_done") == 2
    without_model = [s for s in STAGES if s != "chopped words"]            # --no-truncation
    assert events[0]["files"] == 2 and events[0]["stages"] == without_model
    stages = [e["stage"] for e in events if e["event"] == "stage" and e["index"] == 0]
    assert stages == without_model and {e["steps"] for e in events if e["event"] == "stage"} == {len(without_model)}
    assert events[-1]["network_attempts"] == 0
    done = [e for e in events if e["event"] == "file_done"]
    assert [Path(e["csv"]) for e in done] == [a.with_suffix(".csv"), b.with_suffix(".csv")]
    for wav in (a, b):
        rows = _rows(wav.with_suffix(".csv"))
        assert all(r["check"] != "pause" for r in rows)
    with open(a.with_suffix(".csv"), encoding="utf-8-sig") as fh:
        assert fh.readline().startswith(REPORT_MARKER + ",")
    summary, problems, info = _report(a.with_suffix(".csv"))
    assert summary["file"] == "ch01.wav" and summary["duration"].startswith("0:00:")
    assert "44.1 kHz" in summary["format"] and "1 channel" in summary["format"] and "24 bit" in summary["format"]
    assert summary["problem events"].startswith(f"{len(problems)} ") and "× sev 3" in summary["problem events"]
    assert summary["informational events"].endswith("pause rows off")
    assert info[-1]["event"].startswith("pause map not included")
    assert summary["breaths"].split()[0].isdigit() and summary["breaths"].endswith(" quiet")
    assert summary["chopped-word check"] == "off" and summary["rules"] == "standard"
    assert not (tmp_path / "fpab-report").exists()                         # no run report unless asked


def test_pause_rows_are_added_on_request_in_time_order(tmp_path: Path) -> None:
    a = _wav(tmp_path / "ch01.wav", spike=True)
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-per-file", "--with-pauses", str(a)])
    assert r.exit_code == 0, r.output
    summary, problems, info = _report(a.with_suffix(".csv"))
    assert [p["check"] for p in problems] == ["ticks"]
    pauses = [r for r in info if r["check"] == "pause"]
    assert len(pauses) >= 3 and all(r["check"] in ("pause", "breaths") for r in info)
    assert all(r["severity"] == "" and r["event"].startswith("pause ") for r in pauses)
    assert all(r["severity"] in ("1", "2", "3") for r in problems)                 # problems first, then the rest
    assert any("chapter start" in r["event"] for r in pauses)
    quiet = len(info) - len(pauses)
    assert summary["informational events"] == f"{len(info)} {quiet} quiet breaths {len(pauses)} pauses"
    for part in (problems, info):
        times = [r["start_time"] for r in part]
        assert times == sorted(times)
        assert all(r["end_time"] >= r["start_time"] for r in part)


def test_csv_dir_collects_them_and_keeps_same_names_apart(tmp_path: Path) -> None:
    a, b = _wav(tmp_path / "one" / "ch01.wav"), _wav(tmp_path / "two" / "ch01.wav")
    out = tmp_path / "csvs"
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-dir", str(out), "--out", str(tmp_path / "rep"),
                                  str(a), str(b)])
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in out.iterdir()) == ["ch01 (2).csv", "ch01.csv"]
    assert (tmp_path / "rep" / "report.json").is_file()                   # --out still writes the report
    assert not a.with_suffix(".csv").exists()


def test_our_old_and_new_csvs_are_recognised_and_someone_elses_is_not(tmp_path: Path) -> None:
    from finalpass_audiobook.output import LEGACY_HEADER, is_ours
    new, old, other = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    new.write_text(REPORT_MARKER + ",0.1.0\n", encoding="utf-8-sig")
    old.write_text(LEGACY_HEADER + "\n", encoding="utf-8-sig")
    other.write_text("file,time,notes\n", encoding="utf-8-sig")
    assert is_ours(new) and is_ours(old) and not is_ours(other)
    a = _wav(tmp_path / "ch01.wav")
    old.replace(a.with_suffix(".csv"))                    # a CSV an earlier version wrote: kept, like any file
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-per-file", str(a)])
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in tmp_path.glob("ch01*.csv")) == ["ch01 (2).csv", "ch01.csv"]
    assert (tmp_path / "ch01.csv").read_text(encoding="utf-8-sig") == LEGACY_HEADER + "\n"
    assert _report(tmp_path / "ch01 (2).csv")[0]["file"] == "ch01.wav"


def test_problems_and_informational_events_never_mix(tmp_path: Path) -> None:
    """A severity-2 finding goes to PROBLEM EVENTS; a severity-0 quiet breath only to INFORMATIONAL."""
    from finalpass_audiobook.findings import FileResult, Finding
    from finalpass_audiobook.output import file_csv

    def ev(sev: int, start: int, text: str) -> Finding:
        return Finding(file="c.wav", check="breaths", start_sample=start, end_sample=start + 100,
                       start_time=f"0:00:0{start // 1000}.000", end_time=f"0:00:0{start // 1000}.100",
                       severity=sev, problem=text, measures={"loudness_db": -30.0})
    fr = FileResult(file="c.wav", path=str(tmp_path / "c.wav"), sample_rate=SR, duration_seconds=10.0,
                    narration_dbfs=-19.0, noise_floor_dbfs=-70.0, findings=[ev(2, 3000, "loud breath")],
                    informational=[ev(0, 1000, "quiet breath"), ev(0, 5000, "quiet breath")], pauses=[])
    summary, problems, info = _report(file_csv(fr, tmp_path / "c.csv"))
    assert [(p["event"], p["severity"]) for p in problems] == [("loud breath", "2")]
    assert [(i["event"], i["severity"]) for i in info[:-1]] == [("quiet breath", ""), ("quiet breath", "")]
    assert summary["problem events"].startswith("1 ") and summary["informational events"].startswith("2 ")
    raw = (tmp_path / "c.csv").read_bytes().decode("utf-8-sig")              # spacer rows exactly as written
    assert "\r\n,,,,,,\r\nPROBLEM EVENTS" in raw and "\r\n,,,,,,\r\n,,,,,,\r\nINFORMATIONAL EVENTS" in raw
