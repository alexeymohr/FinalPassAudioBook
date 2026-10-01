"""One bad file never stops a batch; CSVs never overwrite each other or someone else's file.
Synthetic audio only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from finalpass_audiobook.chapter import Chapter
from finalpass_audiobook.cli import main
from synth import CAN_LOCK, LOCKS, SR, phrase, room


def _wav(path: Path, x: np.ndarray | None = None, subtype: str = "PCM_24") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if x is None:
        x = np.concatenate([room(1.2), phrase(4), room(0.9), phrase(3), room(3.0)])
    sf.write(str(path), x, SR, subtype=subtype)
    return path


def _events(r) -> list[dict]:        # noqa: ANN001
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


def _check(*args: str):
    return CliRunner().invoke(main, ["check", "--no-truncation", "--progress", "jsonl", *args])


def test_a_missing_file_and_a_folder_named_like_the_csv_are_not_fatal(tmp_path: Path) -> None:
    a, b = _wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")
    out = tmp_path / "csvs"
    (out / "a.csv").mkdir(parents=True)                       # a folder where a.csv should go
    r = _check("--csv-dir", str(out), str(a), str(tmp_path / "gone.wav"), str(b))
    assert r.exit_code == 0, r.output
    done = [e for e in _events(r) if e["event"] == "file_done"]
    assert [e["csv"] is None for e in done] == [False, True, False]
    assert done[0]["csv"].endswith("a (2).csv") and (out / "a.csv").is_dir()
    assert any("not found" in n for n in done[1]["notes"])
    assert _events(r)[-1]["event"] == "done"


def test_same_name_in_any_case_gets_its_own_csv(tmp_path: Path) -> None:
    a, b = _wav(tmp_path / "one" / "Ch01.wav"), _wav(tmp_path / "two" / "ch01.WAV")
    c = _wav(tmp_path / "three" / "ch01.wav")
    out = tmp_path / "csvs"
    r = _check("--csv-dir", str(out), str(a), str(b), str(c))
    assert r.exit_code == 0, r.output
    assert sorted(p.name.casefold() for p in out.iterdir()) == ["ch01 (2).csv", "ch01 (3).csv", "ch01.csv"]


def test_a_csv_we_did_not_write_is_kept(tmp_path: Path) -> None:
    a = _wav(tmp_path / "ch01.wav")
    mine = tmp_path / "ch01.csv"
    mine.write_text("my own notes\n", encoding="utf-8")
    r = _check("--csv-per-file", str(a))
    assert r.exit_code == 0, r.output
    assert mine.read_text(encoding="utf-8") == "my own notes\n"
    assert (tmp_path / "ch01 (2).csv").is_file()
    r = _check("--csv-per-file", str(a))                      # a re-run never replaces: a new name
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["ch01 (2).csv", "ch01 (3).csv", "ch01.csv"]


def test_invalid_samples_are_a_finding_and_do_not_disable_the_checks(tmp_path: Path) -> None:
    x = np.concatenate([room(1.2), phrase(4), room(0.9), phrase(3), room(3.0)])
    x[1000] = np.nan
    a = _wav(tmp_path / "nan.wav", x, subtype="FLOAT")
    r = _check("--csv-per-file", "--with-pauses", str(a))
    assert r.exit_code == 0, r.output
    from report_csv import read_report
    _, problems, info = read_report(a.with_suffix(".csv"))
    assert any(r["check"] == "file" and r["severity"] == "3" and "NaN" in r["event"] for r in problems)
    assert any(r["check"] == "pause" for r in info)            # the rest of the analysis still ran


def test_empty_file_is_skipped_with_a_reason(tmp_path: Path) -> None:
    a = _wav(tmp_path / "empty.wav", np.zeros(0))
    r = _check("--csv-per-file", str(a))
    (done,) = [e for e in _events(r) if e["event"] == "file_done"]
    assert done["csv"] is None and any("empty" in n for n in done["notes"])


def test_near_dual_mono_and_identical_channels_are_accepted(tmp_path: Path) -> None:
    x = np.concatenate([room(1.2), phrase(4), room(2.0)])
    st = np.stack([x, x], axis=1)
    st[5000, 1] += 1.5 / 32768                                 # one dither step apart
    ch = Chapter.load(_wav(tmp_path / "st.wav", st, subtype="PCM_16"))
    assert ch.notes == ["2 identical channels: analysed the first"]
    ch3 = Chapter.load(_wav(tmp_path / "three.wav", np.stack([x, x, x], axis=1)))
    assert ch3.notes == ["3 identical channels: analysed the first"]


def test_file_names_are_safe_in_spreadsheets_and_in_the_terminal(tmp_path: Path) -> None:
    a = _wav(tmp_path / "=HYPERLINK(1).wav")
    b = _wav(tmp_path / "[red]x.wav")
    out = tmp_path / "rep"
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--out", str(out), str(a), str(b)])
    assert r.exit_code == 0, r.output
    assert "[red]x.wav" in r.output
    text = (out / "issues.csv").read_text(encoding="utf-8-sig")
    assert ",=HYPERLINK" not in text and "\n=HYPERLINK" not in text


def test_skipped_files_appear_in_issues_csv_and_in_the_exit_code(tmp_path: Path) -> None:
    out = tmp_path / "rep"
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not audio")
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--out", str(out), str(bad)])
    assert r.exit_code == 1, r.output                       # the report is written; a script still learns of it
    rows = (out / "issues.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 2 and rows[1].startswith("broken.wav") and "skipped" in rows[1]


@pytest.mark.skipif(not CAN_LOCK, reason=LOCKS)
def test_a_read_only_destination_is_a_note_and_the_batch_goes_on(tmp_path: Path) -> None:
    a, b = _wav(tmp_path / "in" / "a.wav"), _wav(tmp_path / "in" / "b.wav")
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o555)
    try:
        r = _check("--csv-dir", str(out), str(a), str(b))
    finally:
        out.chmod(0o755)
    assert r.exit_code == 0, r.output
    done = [e for e in _events(r) if e["event"] == "file_done"]
    assert len(done) == 2 and all(e["csv"] is None and any("could not write" in n for n in e["notes"]) for e in done)
    assert _events(r)[-1]["event"] == "done"
