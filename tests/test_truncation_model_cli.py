"""Truncation gate, model install/verification, network guard, CLI. Synthetic audio only."""
from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from finalpass_audiobook import model as model_mod
from finalpass_audiobook.checks.truncation import (TRUNCATION_SEVERITY, TruncationTunables, clip_ends,
                                                   score_phrase_ends, truncation_findings)
from finalpass_audiobook.cli import main
from finalpass_audiobook.netguard import NetworkAccessDenied, NetworkGuard
from synth import SR, chapter, phrase, room


class _FakeModel:
    """Stands in for the real model: returns the scores it is given, records its inputs."""

    def __init__(self, scores: list[float]) -> None:
        self.scores, self.calls = scores, []

    def predict_from_audio(self, *, audio, sampling_rate, decision_window_ms):
        self.calls.append((len(audio), sampling_rate, decision_window_ms, [len(a) for a in audio]))
        s = np.array(self.scores[: len(audio)])
        self.scores = self.scores[len(audio):]
        return SimpleNamespace(truncation_score=SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: s))))


def test_flag_needs_a_confident_model_and_an_audible_ending() -> None:
    loud_end = np.concatenate([room(1.0), phrase(3)[: -int(0.06 * SR)]])      # ends mid-word, loud
    quiet_end = np.concatenate([room(1.0), room(0.5, -40.0)])                 # ends quietly
    x = np.concatenate([loud_end, room(1.0), quiet_end, room(1.0)])
    ch = chapter(x)
    e1, e2 = len(loud_end), len(loud_end) + int(1.0 * SR) + len(quiet_end)
    fake = _FakeModel([0.999, 0.999])
    records = score_phrase_ends(ch, [e1, e2], fake)
    assert fake.calls[0][:3] == (2, SR, 10.0)
    assert fake.calls[0][3] == [int(5.0 * SR)] * 2               # always 5 s, left-padded when short
    assert [r["flagged"] for r in records] == [True, False]      # quiet ending fails the audibility gate
    fake = _FakeModel([0.5])
    assert score_phrase_ends(ch, [e1], fake)[0]["flagged"] is False
    (f,) = truncation_findings(ch, records)
    assert f.start_sample == e1 and f.severity == TRUNCATION_SEVERITY and f.problem == "word ends abruptly"


def test_tunables_default_to_the_evaluated_gate() -> None:
    t = TruncationTunables()
    assert (t.threshold, t.decision_window_ms, t.peak_gate_dbfs, t.peak_window_ms, t.context_s) == \
        (0.979795, 10.0, -23.5, 30.0, 5.0)


def test_weights_are_refused_unless_they_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "models"))
    good = tmp_path / "w.safetensors"
    good.write_bytes(b"x" * 64)
    monkeypatch.setattr(model_mod, "WEIGHTS_BYTES", 64)
    monkeypatch.setattr(model_mod, "WEIGHTS_SHA256", model_mod.sha256(good))
    dest = model_mod.install_from_file(good)
    assert dest == model_mod.weights_path() and dest.read_bytes() == good.read_bytes()
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"y" * 64)
    with pytest.raises(model_mod.ModelError, match="SHA-256"):
        model_mod.install_from_file(bad)


def test_network_guard_refuses_connections() -> None:
    with NetworkGuard() as g:
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", 9))
    assert g.attempts


def test_check_command_writes_everything_and_touches_no_network(tmp_path: Path) -> None:
    x = np.concatenate([room(1.0), phrase(4), room(0.6), phrase(4), room(2.5)])
    sf.write(str(tmp_path / "ch01.wav"), x, SR, subtype="PCM_24")
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["check", "--no-truncation", "--out", str(out), str(tmp_path)])
    assert r.exit_code == 0, r.output
    for name in ("issues.txt", "issues.csv", "pauses.txt", "pauses.csv", "report.json"):
        assert (out / name).is_file()
    assert (out / "issues.csv").read_text(encoding="utf-8-sig").splitlines()[0].startswith("file,start_time,end_time,event")
    report = json.loads((out / "report.json").read_text())
    assert report["network_attempts"] == 0 and report["rules"] == "standard"
    (fr,) = report["files"]
    assert fr["pauses"][0]["kind"] == "head"
    assert "chapter start" in (out / "pauses.txt").read_text()


def test_missing_model_is_a_note_not_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path / "empty"))
    x = np.concatenate([room(1.0), phrase(3), room(2.5)])
    sf.write(str(tmp_path / "ch.wav"), x, SR, subtype="PCM_24")
    r = CliRunner().invoke(main, ["check", "--out", str(tmp_path / "o"), str(tmp_path / "ch.wav")])
    assert r.exit_code == 0, r.output
    assert "truncation check skipped" in (tmp_path / "o" / "issues.txt").read_text()


def test_only_clip_ends_into_digital_black_are_scored() -> None:
    """The model was evaluated where a generated clip stops and digital black begins."""
    x = np.concatenate([room(0.5), phrase(3), np.zeros(int(0.2 * SR)),      # a clip end: sound, then black
                        phrase(3), np.zeros(int(0.04 * SR)),                 # 40 ms of black: too short
                        phrase(3), room(0.8), phrase(2)])                    # an ordinary pause: not a clip end
    end = int(0.5 * SR) + len(phrase(3))
    assert clip_ends(chapter(x)) == [end]
    assert clip_ends(chapter(np.concatenate([np.zeros(SR), phrase(3), room(1.0)]))) == []   # black at the file start
