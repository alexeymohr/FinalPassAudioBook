from fpab_gui.engine import EngineEvent, build_command, find_engine, repo_root


def test_parse_start_event():
    event = EngineEvent.parse('{"event": "start", "files": 2, "stages": ["loading", "hum"], '
                              '"paths": ["/a.wav", "/b.wav"]}')
    assert event is not None
    assert event.event == "start"
    assert event.files == 2
    assert event.stages == ("loading", "hum")
    assert event.paths == ("/a.wav", "/b.wav")


def test_parse_file_done_event():
    event = EngineEvent.parse('{"event": "file_done", "index": 0, "path": "/a.wav", '
                              '"csv": "/tmp/a.csv", "notes": ["n"], "sev3": 1, "sev2": 2, "sev1": 3}')
    assert event is not None
    assert (event.index, event.sev3, event.sev2, event.sev1) == (0, 1, 2, 3)
    assert event.notes == ("n",)


def test_parse_ignores_unknown_fields_and_junk():
    assert EngineEvent.parse('{"event": "done", "network_attempts": 0, "extra": 1}').event == "done"
    assert EngineEvent.parse("not json") is None
    assert EngineEvent.parse('{"no": "event"}') is None
    assert EngineEvent.parse('{"event": "x", "notes": 5}') is None


def test_build_command():
    from pathlib import Path
    argv = build_command([Path("a.wav"), Path("b.wav")], Path("/out"), True, ["/eng/fpab"])
    assert argv == ["/eng/fpab", "check", "--csv-dir", "/out", "--progress", "jsonl",
                    "--with-pauses", "--", "a.wav", "b.wav"]


def test_find_engine_explicit_and_env(monkeypatch, tmp_path):
    assert find_engine("/x/fpab") == ["/x/fpab"]
    monkeypatch.setenv("FPAB_ENGINE", "/y/fpab")
    assert find_engine(None) == ["/y/fpab"]


def test_find_engine_prefers_project_venv(monkeypatch):
    monkeypatch.delenv("FPAB_ENGINE", raising=False)
    (repo_root() / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    found = find_engine(None)
    assert found[-1].endswith("fpab")
