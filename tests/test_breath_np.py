"""The breath model in numpy: features against librosa, the network against the authors' torch code (fixtures made
by tests/fixtures/make_breath_fixtures.py), the 30 s windows, and strict weights. Synthetic data only."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from finalpass_audiobook import breath_model, breath_np
from finalpass_audiobook.model import ModelError
from finalpass_audiobook.truncation_np import WeightsError

FIX = np.load(Path(__file__).parent / "fixtures" / "breath_np.npz")
TINY = breath_np.Config(**{k[4:]: int(FIX[k]) for k in FIX.files if k.startswith("cfg:")})
TINY_W = {k[2:]: FIX[k] for k in FIX.files if k.startswith("w:")}


def test_features_match_librosa() -> None:
    f = breath_np.features(FIX["signal"])
    assert f.shape == (3, 128, FIX["mel_db"].shape[1])
    assert np.abs(f[0] - FIX["mel_db"]).max() < 2e-3                    # dB
    assert np.abs(f[1, 0] - FIX["vms"]).max() < 2e-3
    assert np.abs(f[2, 0] - FIX["zcr"]).max() < 1e-7
    assert np.all(f[1] == f[1, :1]) and np.all(f[2] == f[2, :1])         # one value per frame, across the bands


def test_network_matches_the_torch_reference() -> None:
    m = breath_np.BreathModel(TINY_W, TINY)
    p = m.probs_from_features(FIX["tiny_input"])
    assert p.shape == FIX["tiny_output"].shape
    assert np.abs(p - FIX["tiny_output"]).max() < 1e-5


def test_a_chapter_is_scored_in_30_s_windows_keeping_the_middle_20() -> None:
    m = breath_np.BreathModel(TINY_W, TINY)
    sr = breath_np.SR
    t = np.arange(45 * sr) / sr
    y = (0.2 * np.sin(2 * np.pi * (180 + 40 * np.sin(t)) * t) * (np.sin(2 * np.pi * 0.3 * t) > 0)).astype(np.float32)
    p = m.probs(y)
    assert p.shape == (1 + len(y) // 160,)
    first = m.probs_from_features(breath_np.features(y[:25 * sr], TINY.n_mels))   # window 0-25 s keeps 0-20 s
    assert np.array_equal(p[:2000], first[:2000])
    second = m.probs_from_features(breath_np.features(y[15 * sr:45 * sr], TINY.n_mels))  # 15-45 s keeps 20-40 s
    assert np.array_equal(p[2000:4000], second[500:2500])
    third = m.probs_from_features(breath_np.features(y[35 * sr:], TINY.n_mels))   # 35 s-end keeps 40 s-end
    assert np.array_equal(p[4000:], third[500:])
    short = y[: 7 * sr]                                                       # one window: the whole thing
    assert np.array_equal(m.probs(short), m.probs_from_features(breath_np.features(short, TINY.n_mels)))


def test_resampling_to_16k_keeps_the_length_and_the_level() -> None:
    sr = 44100
    t = np.arange(3 * sr) / sr
    y = breath_np.to_16k(np.sin(2 * np.pi * 440 * t) * 0.5, sr)
    assert len(y) == 3 * 16000 and abs(np.sqrt(np.mean(y[1000:-1000] ** 2)) - 0.5 / np.sqrt(2)) < 1e-3
    assert breath_np.to_16k(y, 16000) is not None and len(breath_np.to_16k(y, 16000)) == len(y)


def test_the_weights_must_be_exactly_the_networks() -> None:
    breath_np.BreathModel(TINY_W, TINY)                                       # exact set: accepted
    with pytest.raises(WeightsError, match="missing"):
        breath_np.BreathModel({k: v for k, v in TINY_W.items() if k != "fc.bias"}, TINY)
    with pytest.raises(WeightsError, match="unexpected"):
        breath_np.BreathModel(dict(TINY_W, extra=np.zeros(1, np.float32)), TINY)
    with pytest.raises(WeightsError, match="shape"):
        breath_np.BreathModel(dict(TINY_W, **{"fc.bias": np.zeros(2, np.float32)}), TINY)
    assert breath_np.Config().expected_shapes()["linear.weight"] == (128, 31)  # the published network's sizes
    assert len(breath_np.Config().expected_shapes()) == 276


def test_load_refuses_anything_but_the_recorded_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path))
    assert not breath_model.installed()
    with pytest.raises(ModelError, match="not installed"):
        breath_model.load()
    wrong = tmp_path / "wrong.safetensors"
    wrong.write_bytes(b"\0" * 64)
    with pytest.raises(ModelError, match="bytes, expected"):
        breath_model.install_from_file(wrong)
    same_size = tmp_path / "same_size.safetensors"
    same_size.write_bytes(b"\0" * breath_model.WEIGHTS_BYTES)
    with pytest.raises(ModelError, match="SHA-256"):
        breath_model.install_from_file(same_size)
    breath_model.weights_path().parent.mkdir(parents=True)
    os.replace(same_size, breath_model.weights_path())                       # placed by hand: still refused
    with pytest.raises(ModelError, match="SHA-256"):
        breath_model.load()
    assert breath_model.WEIGHTS_URL.startswith("https://github.com/alexeymohr/FinalPassAudioBook/releases/download/")
