import os
import stat
import sys
import time

import pytest

from fpab_gui.config import Settings
from fpab_gui.runmodel import RunModel
from fpab_gui import runmodel

FAKE_ENGINE = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
args = sys.argv[1:]
csv_dir = Path(args[args.index("--csv-dir") + 1])
files = [Path(a) for a in args[args.index("--") + 1:]]
print(json.dumps({"event": "start", "files": len(files), "stages": ["loading"],
                  "paths": [str(f) for f in files]}), flush=True)
for i, f in enumerate(files):
    print(json.dumps({"event": "file", "index": i, "total": len(files), "path": str(f)}), flush=True)
    print(json.dumps({"event": "stage", "index": i, "stage": "loading", "step": 0, "steps": 1}), flush=True)
    target = csv_dir / (f.stem + ".csv")
    target.write_text("summary\\n")
    print(json.dumps({"event": "file_done", "index": i, "path": str(f), "csv": str(target),
                      "notes": [], "sev3": 1, "sev2": 0, "sev1": 2}), flush=True)
print(json.dumps({"event": "done", "network_attempts": 0, "notes": []}), flush=True)
'''

FAIL_ENGINE = '''#!/usr/bin/env python3
import sys
print("boom: engine exploded", file=sys.stderr)
sys.exit(3)
'''

MODEL_DIR_ENGINE = '''#!/usr/bin/env python3
import json, os
print(json.dumps({"event": "done", "network_attempts": 0,
                  "notes": [os.environ.get("FPAB_MODEL_DIR", "")]}), flush=True)
'''


def _write_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _argv(script):
    """Run a fake engine script with this Python (a .py file is not executable on Windows)."""
    return [sys.executable, str(script)]


@pytest.fixture
def engine(tmp_path):
    return _write_script(tmp_path, "fake-fpab", FAKE_ENGINE)


def _wait(model, timeout=10.0):
    deadline = time.time() + timeout
    while model.running and time.time() < deadline:
        time.sleep(0.02)
    assert not model.running, "run did not finish"


def _settings(tmp_path, **kwargs):
    return Settings(output=kwargs.pop("output", "beside"),
                    folder=kwargs.pop("folder", None),
                    with_pauses=kwargs.pop("with_pauses", False))


def test_run_places_csv_beside(engine, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    model = RunModel(_argv(engine), settings=_settings(tmp_path), confirm=lambda c, a: True)
    model.add([wav])
    model.go()
    _wait(model)
    item = model.items[0]
    assert item.state == "done"
    assert item.csv == tmp_path / "a.csv"
    assert item.summary == "1 × sev 3 · 0 × sev 2 · 2 × sev 1"
    assert item.csv.exists()


def test_numbered_when_name_taken(engine, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    (tmp_path / "a.csv").write_text("old\n")
    model = RunModel(_argv(engine), settings=_settings(tmp_path), confirm=lambda c, a: True)
    model.add([wav])
    model.go()
    _wait(model)
    assert model.items[0].csv == tmp_path / "a (2).csv"
    assert (tmp_path / "a.csv").read_text() == "old\n"


def test_folder_output(engine, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    dest = tmp_path / "out"
    dest.mkdir()
    model = RunModel(_argv(engine),
                     settings=_settings(tmp_path, output="folder", folder=str(dest)),
                     confirm=lambda c, a: True)
    model.add([wav])
    model.go()
    _wait(model)
    assert model.items[0].csv == dest / "a.csv"


def test_folder_required_before_go(tmp_path):
    model = RunModel(["fpab"], settings=_settings(tmp_path, output="folder", folder=None),
                     confirm=lambda c, a: True)
    assert not model.can_go


def test_engine_error_surfaces(engine, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    failing = _write_script(tmp_path, "fail-fpab", FAIL_ENGINE)
    model = RunModel(_argv(failing), settings=_settings(tmp_path), confirm=lambda c, a: True)
    model.add([wav])
    model.go()
    _wait(model)
    assert model.items[0].state == "failed"
    assert model.error_text and "boom" in model.error_text


def test_dedupes_same_file(engine, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    model = RunModel(_argv(engine), settings=_settings(tmp_path), confirm=lambda c, a: True)
    model.add([wav, wav])
    assert len(model.items) == 1
    assert model.items[0].path == wav


def test_clear_needs_confirmation_only_when_unsaved(engine, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    calls = []
    model = RunModel(_argv(engine), settings=_settings(tmp_path),
                     confirm=lambda c, a: calls.append((c, a)) or False)
    model.add([wav])
    model.clear()
    assert model.items == []
    assert calls == []


def test_points_engine_at_bundled_models(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    engine = _write_script(tmp_path, "md-fpab", MODEL_DIR_ENGINE)
    monkeypatch.setattr(runmodel, "bundled_model_dir", lambda: models)
    model = RunModel(_argv(engine), settings=_settings(tmp_path), confirm=lambda c, a: True)
    model.add([wav])
    model.go()
    _wait(model)
    assert model.run_notes == [str(models)]


def test_the_engine_never_opens_a_console_window(engine, tmp_path, monkeypatch):
    """On Windows the bundled fpab.exe is a console program; started from the GUI it must not open a window."""
    seen = {}
    real = runmodel.subprocess.Popen

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(runmodel.subprocess, "Popen", spy)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    model = RunModel(_argv(engine), settings=_settings(tmp_path), confirm=lambda c, a: True)
    model.add([wav])
    model.go()
    _wait(model)
    assert seen["creationflags"] == getattr(runmodel.subprocess, "CREATE_NO_WINDOW", 0)
    if sys.platform == "win32":
        assert seen["creationflags"] == 0x08000000
