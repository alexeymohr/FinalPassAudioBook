"""Reports land only where they belong: never over a file this tool did not write, never through a
symlink, never over another WAV's report; a bad input or output never loses the run. Synthetic audio only."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from finalpass_audiobook.cli import main
from finalpass_audiobook.output import _cell, is_ours, report_source
from report_csv import read_report
from synth import CAN_LOCK, LINKS, LOCKS, POSIX, SR, phrase, room


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.concatenate([room(1.2), phrase(4), room(0.9), phrase(3), room(3.0)]), SR, subtype="PCM_24")
    return path


def _check(*args: str):
    return CliRunner().invoke(main, ["check", "--no-truncation", *args])


def _done(r) -> list[dict]:        # noqa: ANN001
    return [json.loads(line) for line in r.stdout.splitlines()
            if line.startswith("{") and json.loads(line)["event"] == "file_done"]


def test_the_run_report_never_replaces_files_it_did_not_write(tmp_path: Path) -> None:
    a = _wav(tmp_path / "ch01.wav")
    out = tmp_path / "out"
    out.mkdir()
    (out / "issues.txt").write_text("my notes\n")
    r = _check("--out", str(out), str(a))
    assert r.exit_code == 2 and "did not write" in r.output
    assert (out / "issues.txt").read_text() == "my notes\n" and not (out / "report.json").exists()
    (out / "issues.txt").unlink()
    assert _check("--out", str(out), str(a)).exit_code == 0
    assert _check("--out", str(out), str(a)).exit_code == 0             # its own report is replaced on a re-run


def test_a_per_file_csv_never_takes_a_run_report_name(tmp_path: Path) -> None:
    a = _wav(tmp_path / "issues.wav")
    r = _check("--csv-per-file", "--out", str(tmp_path), str(a))
    assert r.exit_code == 0, r.output
    assert is_ours(tmp_path / "issues (2).csv")
    assert (tmp_path / "issues.csv").read_text(encoding="utf-8-sig").startswith("file,start_time")


@pytest.mark.skipif(not POSIX, reason=LINKS)
@pytest.mark.parametrize("dangling", [True, False])
def test_a_csv_is_never_written_through_a_symlink(tmp_path: Path, dangling: bool) -> None:
    a = _wav(tmp_path / "in" / "ch01.wav")
    elsewhere = tmp_path / "elsewhere.txt"
    if not dangling:
        elsewhere.write_text("keep\n")
    (tmp_path / "in" / "ch01.csv").symlink_to(elsewhere)
    r = _check("--csv-per-file", str(a))
    assert r.exit_code == 0, r.output
    assert is_ours(tmp_path / "in" / "ch01 (2).csv")
    assert not elsewhere.exists() if dangling else elsewhere.read_text() == "keep\n"


def test_two_books_chapter_01_never_share_a_report_across_runs(tmp_path: Path) -> None:
    a, b = _wav(tmp_path / "bookA" / "ch01.wav"), _wav(tmp_path / "bookB" / "ch01.wav")
    out = tmp_path / "csvs"
    assert _check("--csv-dir", str(out), str(a), str(b)).exit_code == 0
    first, second = (out / "ch01.csv").read_bytes(), (out / "ch01 (2).csv").read_bytes()
    assert report_source(out / "ch01.csv") == ("ch01.wav", str((tmp_path / "bookA").resolve()))
    assert _check("--csv-dir", str(out), str(b)).exit_code == 0           # book B alone, later
    assert (out / "ch01.csv").read_bytes() == first and (out / "ch01 (2).csv").read_bytes() == second
    assert report_source(out / "ch01 (3).csv") == ("ch01.wav", str((tmp_path / "bookB").resolve()))
    summary, _, _ = read_report(out / "ch01 (3).csv")
    assert summary["folder"] == str((tmp_path / "bookB").resolve())


def test_a_rerun_never_replaces_the_earlier_report(tmp_path: Path) -> None:
    a = _wav(tmp_path / "ch01.wav")
    assert _check("--csv-per-file", str(a)).exit_code == 0
    first = (tmp_path / "ch01.csv").read_bytes()
    for _ in range(2):
        assert _check("--csv-per-file", str(a)).exit_code == 0
    assert (tmp_path / "ch01.csv").read_bytes() == first
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["ch01 (2).csv", "ch01 (3).csv", "ch01.csv"]


def test_a_numbered_name_is_never_another_wavs_own(tmp_path: Path) -> None:
    a, a2 = _wav(tmp_path / "a.wav"), _wav(tmp_path / "a (2).wav")
    (tmp_path / "a.csv").write_text("mine\n")
    assert _check("--csv-per-file", str(a)).exit_code == 0
    assert not (tmp_path / "a (2).csv").exists()                           # that is "a (2).wav"'s own name
    assert report_source(tmp_path / "a (3).csv")[0] == "a.wav"
    assert _check("--csv-per-file", str(a2)).exit_code == 0
    assert report_source(tmp_path / "a (2).csv")[0] == "a (2).wav"
    assert (tmp_path / "a.csv").read_text() == "mine\n"


def test_a_csv_in_another_encoding_is_someone_elses(tmp_path: Path) -> None:
    a = _wav(tmp_path / "ch01.wav")
    (tmp_path / "ch01.csv").write_bytes(b"caf\xe9,notes\n")                 # Latin-1, as Excel may save
    assert _check("--csv-per-file", str(a)).exit_code == 0
    assert (tmp_path / "ch01.csv").read_bytes() == b"caf\xe9,notes\n" and is_ours(tmp_path / "ch01 (2).csv")


def test_a_header_that_only_starts_like_ours_is_not_ours(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("file,time,problem,severity,end_time,check,measures,my notes\n", encoding="utf-8-sig")
    assert not is_ours(p)
    p.write_text("FinalPass AudioBook reports,1\n", encoding="utf-8-sig")
    assert not is_ours(p)


@pytest.mark.skipif(not CAN_LOCK, reason=LOCKS)
def test_an_unreadable_file_is_skipped_and_the_batch_goes_on(tmp_path: Path) -> None:
    a, b = _wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")
    b.chmod(0)
    try:
        r = _check("--csv-dir", str(tmp_path / "c"), "--progress", "jsonl", str(a), str(b))
    finally:
        b.chmod(0o644)
    assert r.exit_code == 0, r.output
    done = _done(r)
    assert done[0]["csv"] and done[1]["csv"] is None and any("skipped" in n for n in done[1]["notes"])


def test_an_unusable_out_folder_stops_before_the_analysis(tmp_path: Path) -> None:
    a = _wav(tmp_path / "a.wav")
    (tmp_path / "file").write_text("x")
    r = _check("--out", str(tmp_path / "file" / "sub"), str(a))
    assert r.exit_code == 2 and "cannot write the run report" in r.output


def test_a_run_report_that_cannot_be_written_still_shows_the_results(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.output as output

    def full_disk(*a, **k):        # noqa: ANN002, ANN003
        raise OSError("No space left on device")
    monkeypatch.setattr(output, "write", full_disk)
    a = _wav(tmp_path / "a.wav")
    r = _check("--out", str(tmp_path / "rep"), str(a))
    assert r.exit_code == 1
    assert "a.wav" in r.output and "No space left" in r.output and "network attempts: 0" in r.output


def test_a_csv_that_fails_to_write_for_any_reason_is_a_note(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.output as output

    def broken(*a, **k):           # noqa: ANN002, ANN003
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")
    monkeypatch.setattr(output, "file_csv", broken)
    a, b = _wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")
    r = _check("--csv-per-file", "--progress", "jsonl", str(a), str(b))
    done = _done(r)
    assert len(done) == 2 and all(e["csv"] is None and any("could not write" in n for n in e["notes"]) for e in done)


def test_formula_starts_in_file_names_are_neutralised() -> None:
    for bad in ("=1+1.wav", "+1.wav", "-1.wav", "@x.wav", "\t=1.wav", "\r=1.wav"):
        assert _cell(bad) == "'" + bad
    assert _cell("chapter 1.wav") == "chapter 1.wav"


def test_one_file_named_twice_is_checked_once(tmp_path: Path) -> None:
    a = _wav(tmp_path / "in" / "ch01.wav")
    r = _check("--csv-per-file", "--progress", "jsonl", str(a), str(tmp_path / "in"), str(tmp_path / "in" / "." / "ch01.wav"))
    assert r.exit_code == 0 and len(_done(r)) == 1
    assert sorted(p.name for p in (tmp_path / "in").glob("*.csv")) == ["ch01.csv"]


def test_a_csv_that_appears_during_the_run_is_not_replaced(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.run as run_mod
    a, b = _wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")
    real = run_mod.analyze_file

    def analyze(path, *args, **kw):        # noqa: ANN001, ANN002, ANN003, ANN202
        if path.name == "a.wav":
            (tmp_path / "b.csv").write_text("my notes, saved mid-run\n")
        return real(path, *args, **kw)
    monkeypatch.setattr(run_mod, "analyze_file", analyze)
    assert _check("--csv-per-file", str(a), str(b)).exit_code == 0
    assert (tmp_path / "b.csv").read_text() == "my notes, saved mid-run\n"
    assert report_source(tmp_path / "b (2).csv")[0] == "b.wav"


def test_a_report_without_a_folder_is_kept_away_from_its_wav(tmp_path: Path) -> None:
    """A CSV an earlier version wrote into a shared folder cannot say which book it was for."""
    a = _wav(tmp_path / "bookA" / "ch01.wav")
    out = tmp_path / "csvs"
    out.mkdir()
    old = "FinalPass AudioBook report,0.0.9\r\nfile,ch01.wav\r\n,,,,,,\r\n".encode("utf-8-sig")
    (out / "ch01.csv").write_bytes(old)
    assert _check("--csv-dir", str(out), str(a)).exit_code == 0
    assert (out / "ch01.csv").read_bytes() == old and is_ours(out / "ch01 (2).csv")


def test_a_name_starting_with_an_apostrophe_reads_back(tmp_path: Path) -> None:
    from finalpass_audiobook.output import _uncell
    for name in ("'-01.wav", "'plain.wav", "-01.wav", "=01.wav", "ch01.wav"):
        assert _uncell(_cell(name)) == name
    a = _wav(tmp_path / "'-01.wav")
    assert _check("--csv-per-file", str(a)).exit_code == 0
    assert report_source(tmp_path / "'-01.csv")[0] == "'-01.wav"


def test_a_csv_that_fails_part_way_is_not_left_behind(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.output as output
    from finalpass_audiobook.findings import FileResult

    def broken(*a, **k):           # noqa: ANN002, ANN003
        raise OSError("No space left on device")
    monkeypatch.setattr(output, "summary_rows", broken)
    fr = FileResult(file="c.wav", path=str(tmp_path / "c.wav"), sample_rate=SR, duration_seconds=1.0,
                    narration_dbfs=None, noise_floor_dbfs=None, findings=[], pauses=[])
    with pytest.raises(OSError):
        output.file_csv(fr, tmp_path / "c.csv")
    assert not (tmp_path / "c.csv").exists()
    (tmp_path / "c.csv").write_text("mine\n")
    with pytest.raises(FileExistsError):                          # never over an existing file
        output.file_csv(fr, tmp_path / "c.csv")
    assert (tmp_path / "c.csv").read_text() == "mine\n"


def test_running_out_of_names_is_a_note_and_exit_1(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.output as output

    def always_taken(*a, **k):        # noqa: ANN002, ANN003
        raise FileExistsError("taken")
    monkeypatch.setattr(output, "file_csv", always_taken)
    a = _wav(tmp_path / "a.wav")
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--csv-per-file", str(a)])
    assert r.exit_code == 1 and "every name tried was taken" in r.output


@pytest.mark.skipif(not POSIX, reason=LINKS)
def test_a_csv_folder_that_is_a_broken_link_is_a_note_and_exit_1(tmp_path: Path) -> None:
    a = _wav(tmp_path / "a.wav")
    (tmp_path / "csvs").symlink_to(tmp_path / "unmounted")
    r = _check("--csv-dir", str(tmp_path / "csvs"), str(a))
    assert r.exit_code == 1 and "is not a folder" in r.output


@pytest.mark.skipif(not CAN_LOCK, reason=LOCKS)
def test_a_csv_that_cannot_be_written_makes_text_mode_exit_1(tmp_path: Path) -> None:
    a = _wav(tmp_path / "in" / "a.wav")
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o555)
    try:
        r = _check("--csv-dir", str(out), str(a))
    finally:
        out.chmod(0o755)
    assert r.exit_code == 1 and "could not write the CSV" in r.output


@pytest.mark.parametrize("name", ["issues.txt", "issues.csv", "pauses.txt", "pauses.csv", "report.json"])
def test_any_run_report_name_held_by_someone_else_stops_the_run(tmp_path: Path, name: str) -> None:
    a = _wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_text('{"a": 1}\n' if name.endswith("json") else "mine\n")
    r = _check("--out", str(out), str(a))
    assert r.exit_code == 2 and (out / name).read_text() in ('{"a": 1}\n', "mine\n")


def test_a_run_report_file_that_appears_during_the_run_is_kept(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.run as run_mod
    a = _wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    real = run_mod.analyze_file

    def analyze(path, *args, **kw):        # noqa: ANN001, ANN002, ANN003, ANN202
        out.mkdir(exist_ok=True)
        (out / "issues.txt").write_text("mine, saved mid-run\n")
        return real(path, *args, **kw)
    monkeypatch.setattr(run_mod, "analyze_file", analyze)
    r = _check("--out", str(out), str(a))
    assert r.exit_code == 1 and (out / "issues.txt").read_text() == "mine, saved mid-run\n"


def test_an_interrupted_run_report_does_not_lock_the_folder(tmp_path: Path, monkeypatch) -> None:
    import finalpass_audiobook.output as output
    a = _wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    assert _check("--out", str(out), str(a)).exit_code == 0
    real = output.pauses_text

    def disk_full(*args, **kw):        # noqa: ANN002, ANN003, ANN202
        raise OSError("No space left on device")
    monkeypatch.setattr(output, "pauses_text", disk_full)
    assert _check("--out", str(out), str(a)).exit_code == 1
    assert not list(out.glob(".*part"))                          # no half-written file left behind
    monkeypatch.setattr(output, "pauses_text", real)
    assert _check("--out", str(out), str(a)).exit_code == 0       # still ours: the next run goes ahead


def test_the_issue_list_counts_every_finding_whatever_min_sev_hides(tmp_path: Path) -> None:
    from finalpass_audiobook.findings import FileResult, Finding, RunReport
    from finalpass_audiobook.output import issues_text
    f = Finding(file="c.wav", check="plosive", start_sample=0, end_sample=1, start_time="0:00:00.000",
                end_time="0:00:00.000", severity=1, problem="pop")
    fr = FileResult(file="c.wav", path="c.wav", sample_rate=SR, duration_seconds=1.0, narration_dbfs=None,
                    noise_floor_dbfs=None, findings=[f], pauses=[])
    rep = RunReport(version="t", finalpass_version="t", run_id="r", run_started_at="t", rules="standard",
                    tunables={}, network_attempts=0, files=[fr])
    text = issues_text(rep, min_sev=3)
    assert "0 × sev 3, 0 × sev 2, 1 × sev 1" in text and "pop" not in text


def test_a_shared_csv_folder_never_takes_another_wavs_own_name(tmp_path: Path) -> None:
    a = _wav(tmp_path / "bookA" / "ch01.wav")
    _wav(tmp_path / "bookB" / "ch01.wav")
    assert _check("--csv-dir", str(tmp_path / "bookB"), str(a)).exit_code == 0
    assert not (tmp_path / "bookB" / "ch01.csv").exists()                 # bookB's ch01.wav's own name
    assert report_source(tmp_path / "bookB" / "ch01 (2).csv")[1] == str((tmp_path / "bookA").resolve())


@pytest.mark.skipif(not POSIX, reason=LINKS)
def test_a_name_too_long_to_number_is_a_note_and_the_batch_goes_on(tmp_path: Path) -> None:
    long = _wav(tmp_path / ("x" * 250 + ".wav"))
    other = _wav(tmp_path / "b.wav")
    assert _check("--csv-per-file", str(long)).exit_code == 0             # the plain name still fits
    r = _check("--csv-per-file", "--progress", "jsonl", str(long), str(other))
    done = _done(r)
    assert r.exit_code == 0 and done[0]["csv"] is None and any("too long" in n for n in done[0]["notes"])
    assert done[1]["csv"]


@pytest.mark.skipif(not POSIX, reason=LINKS)
def test_names_are_planned_past_files_links_and_this_run(tmp_path: Path) -> None:
    from finalpass_audiobook.cli import _csv_targets
    a, b = _wav(tmp_path / "one" / "ch01.wav"), _wav(tmp_path / "two" / "ch01.wav")
    out = tmp_path / "csvs"
    out.mkdir()
    (out / "ch01.csv").write_text("x")
    (out / "ch01 (2).csv").symlink_to(tmp_path / "nowhere")
    assert [p.name for p in _csv_targets([a, b], out)] == ["ch01 (3).csv", "ch01 (4).csv"]


@pytest.mark.skipif(not CAN_LOCK, reason=LOCKS)
def test_an_unreadable_folder_is_a_skipped_entry_and_the_rest_runs(tmp_path: Path) -> None:
    a = _wav(tmp_path / "a.wav")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        r = _check("--csv-dir", str(tmp_path / "c"), "--progress", "jsonl", str(locked), str(a))
    finally:
        locked.chmod(0o755)
    done = _done(r)
    assert r.exit_code == 0 and len(done) == 2 and done[1]["csv"] and any("folder" in n for n in done[0]["notes"])
