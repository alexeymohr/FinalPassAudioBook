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


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([room(1.2), phrase(4), room(0.9), phrase(3), room(3.0)])
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
    assert all(r["event"] == "quiet breath" and r["severity"] == "" for r in info[:-1])
    assert summary["breaths"].split()[0].isdigit() and summary["breaths"].endswith(" quiet")
    assert summary["chopped-word check"] == "off" and summary["rules"] == "standard"
    assert not (tmp_path / "fpab-report").exists()                         # no run report unless asked


def test_pause_rows_are_added_on_request_in_time_order(tmp_path: Path) -> None:
    a = _wav(tmp_path / "ch01.wav")
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-per-file", "--with-pauses", str(a)])
    assert r.exit_code == 0, r.output
    summary, problems, info = _report(a.with_suffix(".csv"))
    pauses = [r for r in info if r["check"] == "pause"]
    assert pauses and all(r["check"] in ("pause", "breaths") for r in info)
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
    old.replace(a.with_suffix(".csv"))                    # a CSV an earlier version wrote: replaced, not "(2)"
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-per-file", str(a)])
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in tmp_path.glob("ch01*.csv")) == ["ch01.csv"]
    assert _report(a.with_suffix(".csv"))[0]["file"] == "ch01.wav"
