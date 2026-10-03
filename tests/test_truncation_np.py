"""The chopped-word model in numpy: strict weight loading, and equivalence with the audited torch
reference (skipped where the `truncation` extra or the weights are not installed). Synthetic audio only."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from finalpass_audiobook import model as model_mod
from finalpass_audiobook import truncation_np as T
from synth import SR, phrase, room

CONFIG = model_mod.VENDOR / "config.json"


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray], dtype: str = "F32") -> Path:
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return path


def _tiny() -> T.Config:
    return T.Config(n_mels=4, hidden=8, intermediate=12, layers=1, heads=2, vocab=3)


def _weights(cfg: T.Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {k: rng.standard_normal(s).astype(np.float32) * 0.1 for k, s in cfg.expected_shapes().items()}


def test_reader_round_trips_and_the_model_refuses_anything_unexpected(tmp_path: Path) -> None:
    cfg = _tiny()
    w = _weights(cfg)
    back = T.read_safetensors(_write_safetensors(tmp_path / "w.safetensors", w))
    assert set(back) == set(w) and all(np.array_equal(back[k], w[k]) for k in w)
    T.TruncationModel(back, cfg)                                           # exact set: accepted
    extra = dict(w, **{"backbone.layers.0.extra.weight": np.zeros(2, np.float32)})
    with pytest.raises(T.WeightsError, match="unexpected"):
        T.TruncationModel(extra, cfg)
    missing = {k: v for k, v in w.items() if k != "classifier.bias"}
    with pytest.raises(T.WeightsError, match="missing"):
        T.TruncationModel(missing, cfg)
    wrong = dict(w, **{"classifier.bias": np.zeros(3, np.float32)})
    with pytest.raises(T.WeightsError, match="wrong shape"):
        T.TruncationModel(wrong, cfg)
    with pytest.raises(T.WeightsError, match="dtype"):
        T.read_safetensors(_write_safetensors(tmp_path / "f16.safetensors", w, dtype="F16"))


def test_config_variants_the_port_does_not_implement_are_refused(tmp_path: Path) -> None:
    c = json.loads(CONFIG.read_text())
    assert T.Config.from_json(CONFIG).hidden == 384
    c["audio_config"]["center"] = True
    (tmp_path / "c.json").write_text(json.dumps(c))
    with pytest.raises(T.WeightsError, match="center"):
        T.Config.from_json(tmp_path / "c.json")


def test_resampled_length_matches_torchaudio_rule() -> None:
    x = np.zeros(SR * 5, np.float32)
    assert len(T.resample(x, SR, 16000)) == 80000
    assert len(T.resample(np.zeros(48000 * 5, np.float32), 48000, 16000)) == 80000
    y = np.sin(np.arange(16000) * 0.01).astype(np.float32)
    assert np.array_equal(T.resample(y, 16000, 16000), y)


# --- equivalence with the audited torch reference -------------------------------------------

torch = pytest.importorskip("torch", reason="the torch reference is the `truncation` extra")
torchaudio = pytest.importorskip("torchaudio")
needs_weights = pytest.mark.skipif(not model_mod.installed(), reason="model weights not installed")


@pytest.mark.parametrize("sr", [44100, 48000, 22050])
def test_resampler_matches_torchaudio(sr: int) -> None:
    x = (np.random.default_rng(sr).standard_normal(sr * 5) * 0.1).astype(np.float32)
    ref = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    assert np.abs(T.resample(x, sr, 16000) - ref).max() < 1e-5


def _windows(n: int) -> list[np.ndarray]:
    out = []
    for i in range(n):
        p = np.concatenate([room(1.0), phrase(4), room(0.3), phrase(3)])
        if i % 2:
            p = p[: -int(0.05 * SR)]                     # ends mid-word
        p = p[-5 * SR:] if len(p) >= 5 * SR else np.pad(p, (5 * SR - len(p), 0))
        out.append(p.astype(np.float32))
    return out


@needs_weights
@pytest.mark.real_models
def test_mel_and_scores_match_the_torch_reference() -> None:
    ref = model_mod.load_reference()
    port = model_mod.load()
    wins = _windows(8)
    mel_ref = ref.get_processor()(audio=wins, sampling_rate=SR)["mel"].numpy()
    # log-mel: torch's STFT is float32, ours float64; near-silent bins differ most in the log (0.1 % power)
    assert np.abs(port.mels(wins, SR) - mel_ref).max() < 1e-3
    s_ref = ref.predict_from_audio(audio=wins, sampling_rate=SR, decision_window_ms=10.0).truncation_score.numpy()
    s_port = port.predict_from_audio(audio=wins, sampling_rate=SR, decision_window_ms=10.0).truncation_score
    assert np.abs(s_port - s_ref).max() < 1e-4
    for ms in (30.0, 100.0):                              # wider decision windows too
        a = ref.predict_from_audio(audio=wins[:2], sampling_rate=SR, decision_window_ms=ms).truncation_score.numpy()
        b = port.predict_from_audio(audio=wins[:2], sampling_rate=SR, decision_window_ms=ms).truncation_score
        assert np.abs(a - b).max() < 1e-4


# --- added after the audit: strict parsing, validation, odd lengths, other rates, DC ---------------


def _raw(tensors: dict[str, np.ndarray], header_edit=None, tail: bytes = b"") -> bytes:  # noqa: ANN001
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if header_edit:
        header = header_edit(header)
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + b"".join(blobs) + tail


@pytest.mark.parametrize("edit, tail", [
    (lambda h: [1, 2], b""),                                                     # header not a dict
    (lambda h: {**h, "a": "x"}, b""),                                            # entry not a dict
    (lambda h: {"a": {**h["a"], "data_offsets": [-8, -4]}, "b": h["b"]}, b""),   # negative offsets
    (lambda h: {"a": {**h["a"], "shape": [2.0]}, "b": h["b"]}, b""),             # float dimension
    (lambda h: {"a": h["a"], "b": {**h["b"], "data_offsets": h["a"]["data_offsets"]}}, b""),  # overlap
    (lambda h: h, b"junk"),                                                      # trailing bytes
    (lambda h: {**h, "__metadata__": [1]}, b""),                                 # bad metadata
])
def test_malformed_weight_files_are_refused(edit, tail) -> None:  # noqa: ANN001
    t = {"a": np.ones(2, np.float32), "b": np.ones(2, np.float32)}
    T.parse_safetensors(_raw(t))                                                  # well-formed: fine
    with pytest.raises(T.WeightsError):
        T.parse_safetensors(_raw(t, edit, tail))


def test_two_tensors_over_the_same_bytes_are_refused() -> None:
    """Overlap on its own: no gap and no trailing bytes to give it away."""
    spec = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    h = json.dumps({"a": spec, "b": spec}).encode()
    with pytest.raises(T.WeightsError, match="overlap"):
        T.parse_safetensors(struct.pack("<Q", len(h)) + h + np.ones(2, "<f4").tobytes())


def test_arguments_the_reference_refuses_are_refused() -> None:
    cfg = _tiny()
    m = T.TruncationModel(_weights(cfg), cfg)
    good = np.zeros(16000, np.float32)
    for bad in (dict(decision_window_ms=0), dict(decision_window_ms=-5), dict(sampling_rate=0),
                dict(sampling_rate=-16000), dict(sampling_rate=44100.5), dict(audio=[np.zeros((2, 100))]),
                dict(audio=[np.zeros(0)])):
        kw = {"audio": [good], "sampling_rate": 16000, "decision_window_ms": 10.0} | bad
        with pytest.raises(ValueError):
            m.predict_from_audio(**kw)


@pytest.mark.parametrize("key, value", [("hidden_act", "gelu"), ("rope_scaling", {"type": "linear", "factor": 2.0}),
                                        ("head_dim", 32)])
def test_unimplemented_config_variants_are_refused(tmp_path: Path, key: str, value) -> None:  # noqa: ANN001
    c = json.loads(CONFIG.read_text())
    c["model_config"]["llama_config"][key] = value
    (tmp_path / "c.json").write_text(json.dumps(c))
    with pytest.raises(T.WeightsError):
        T.Config.from_json(tmp_path / "c.json")


@pytest.mark.parametrize("sr", [11025, 22050, 44100, 88200])
def test_resampler_matches_torchaudio_closely_even_with_dc(sr: int) -> None:
    x = (0.3 + np.random.default_rng(sr).standard_normal(sr * 5) * 0.05).astype(np.float32)
    ref = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    port = T.resample(x, sr, 16000)
    assert port.shape == ref.shape and np.abs(port - ref).max() < 2e-6


def _speechy(sr: int, seconds: float, seed: int, dc: float = 0.0, rumble: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    voiced = sum(np.sin(2 * np.pi * k * (120 + 20 * np.sin(2 * np.pi * 0.7 * t)) * t) / k for k in range(1, 12))
    gate = (np.sin(2 * np.pi * 1.3 * t + seed) > -0.2).astype(float)
    x = 0.08 * voiced * gate + 0.002 * rng.standard_normal(len(t)) + dc + rumble * np.sin(2 * np.pi * 35 * t)
    return x.astype(np.float32)


@needs_weights
@pytest.mark.real_models
@pytest.mark.parametrize("sr, seconds, dc, rumble", [
    (44100, 2.0, 0.0, 0.0), (44100, 7.3, 0.0, 0.0), (44100, 180697 / 44100, 0.0, 0.0),
    (11025, 5.0, 0.0, 0.0), (22050, 5.0, 0.0, 0.0), (44100, 5.0, 0.4, 0.0), (44100, 5.0, 0.0, 0.3)])
def test_scores_match_the_reference_on_odd_lengths_rates_and_low_end(sr, seconds, dc, rumble) -> None:  # noqa: ANN001
    ref, port = model_mod.load_reference(), model_mod.load()
    wins = [_speechy(sr, seconds, s, dc, rumble) for s in range(4)]
    for w in wins:
        w[-int(0.02 * sr):] *= 3.0                               # a loud, abrupt ending
    a = ref.predict_from_audio(audio=wins, sampling_rate=sr, decision_window_ms=10.0).truncation_score.numpy()
    b = port.predict_from_audio(audio=wins, sampling_rate=sr, decision_window_ms=10.0).truncation_score
    assert np.abs(a - b).max() < 1e-4


@needs_weights
@pytest.mark.real_models
def test_decision_window_frames_match_the_reference() -> None:
    ref, port = model_mod.load_reference(), model_mod.load()
    for ms in (1, 5, 9.99, 10, 10.01, 20, 25, 30, 33.3, 100, 5000):
        assert port.decision_frames(ms) == ref._resolve_decision_window_frames(decision_window_ms=ms)


@pytest.mark.parametrize("sr", [11025, 22050, 44100])
def test_resampled_length_matches_torchaudio_at_odd_lengths(sr: int) -> None:
    for n in (sr * 5 + 1, 180697, 241227, sr * 3 + 17):
        x = np.zeros(n, np.float32)
        assert len(T.resample(x, sr, 16000)) == torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).shape[-1]


def test_rms_norm_matches_transformers_even_on_quiet_input() -> None:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    ref = LlamaRMSNorm(8, eps=1e-6)
    with torch.no_grad():
        ref.weight.copy_(torch.linspace(0.5, 1.5, 8))
    for scale in (1.0, 1e-3, 1e-5):
        x = (np.random.default_rng(0).standard_normal((3, 8)) * scale).astype(np.float32)
        a = ref(torch.from_numpy(x)).detach().numpy()
        assert np.allclose(T._rms(x, ref.weight.detach().numpy(), 1e-6), a, rtol=1e-4, atol=1e-7)
