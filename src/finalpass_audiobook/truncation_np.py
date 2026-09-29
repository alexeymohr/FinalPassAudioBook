"""The chopped-word model's inference in numpy — no PyTorch. Numbers only.

A port of the inference path of mythicinfinity/speech-truncation-detection-12M
(Apache-2.0; licence in vendor/speech_truncation). The audited torch code stays
vendored, unmodified, as the reference; tests hold this port to it.

What the reference does, step by step, reproduced here:
1. resample each window to 16 kHz with torchaudio's windowed-sinc resampler
   (Hann window, 6 zero crossings, roll-off 0.99), and keep the last 5 s;
2. mel spectrogram: 400-sample periodic Hamming frames every 160 samples, no
   centring, power 2, 80 HTK-scale triangular bands, no normalisation; then
   (log(max(mel, 1e-5)) + 4) / 4;
3. a linear projection to 384 dims, 8 causal Llama layers (RMSNorm, rotary
   positions with theta 10000, 8 heads of 48, SwiGLU MLP of 888), a final
   RMSNorm and a 2-way classifier;
4. softmax; the score is the largest "truncated" probability over the last
   decision window (ceil(window / 10 ms) frames).

Weights are read straight from the audited model.safetensors (a JSON header and
raw little-endian arrays: no pickle, no code), and every tensor name and shape
must match the architecture exactly.
"""
from __future__ import annotations

import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MEL_LOG_EPS, MEL_MEAN, MEL_STD = 1e-5, -4.0, 4.0     # StyleTTS normalisation used in training


class WeightsError(ValueError):
    """The weights file does not hold exactly the expected tensors."""


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read a float32 safetensors file into numpy arrays (header + raw data; nothing executed)."""
    return parse_safetensors(Path(path).read_bytes())


def _is_int(v) -> bool:         # noqa: ANN001
    return isinstance(v, int) and not isinstance(v, bool)


def parse_safetensors(raw: bytes) -> dict[str, np.ndarray]:
    """Strict parse: a dict header; every tensor F32 with non-negative integer dims and offsets
    0 <= a <= b <= len(data) whose size matches; tensors packed without gaps, overlaps or trailing
    bytes. Anything else is a WeightsError."""
    if len(raw) < 8:
        raise WeightsError("file too short")
    n = struct.unpack("<Q", raw[:8])[0]
    if n > len(raw) - 8 or n > 100_000_000:
        raise WeightsError(f"header length {n} does not fit the file")
    try:
        header = json.loads(raw[8:8 + n])
    except (ValueError, RecursionError) as exc:
        raise WeightsError(f"bad header: {exc}") from exc
    if not isinstance(header, dict):
        raise WeightsError("header is not a JSON object")
    meta = header.pop("__metadata__", None)
    if meta is not None and not (isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values())):
        raise WeightsError("__metadata__ is not a string map")
    data = memoryview(raw)[8 + n:]
    spans, out = [], {}
    for name, spec in header.items():
        if not isinstance(spec, dict):
            raise WeightsError(f"{name}: tensor entry is not an object")
        if spec.get("dtype") != "F32":
            raise WeightsError(f"{name}: dtype {spec.get('dtype')}, expected F32")
        shape, offs = spec.get("shape"), spec.get("data_offsets")
        if not (isinstance(shape, list) and all(_is_int(s) and s >= 0 for s in shape)):
            raise WeightsError(f"{name}: shape must be a list of non-negative integers")
        if not (isinstance(offs, list) and len(offs) == 2 and all(_is_int(o) for o in offs)):
            raise WeightsError(f"{name}: data_offsets must be two integers")
        a, b = offs
        if not 0 <= a <= b <= len(data) or b - a != 4 * math.prod(shape):
            raise WeightsError(f"{name}: offsets {offs} do not fit the data or the shape {shape}")
        spans.append((a, b))
        out[name] = np.frombuffer(data[a:b], dtype="<f4").reshape(tuple(shape))
    end = 0
    for a, b in sorted(spans):
        if a != end:
            raise WeightsError("tensors overlap or leave gaps")
        end = b
    if end != len(data):
        raise WeightsError("trailing bytes after the last tensor")
    return out


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000
    n_fft: int = 400
    hop: int = 160
    n_mels: int = 80
    tail_s: float = 5.0
    hidden: int = 384
    intermediate: int = 888
    layers: int = 8
    heads: int = 8
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    vocab: int = 32
    labels: int = 2

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        c = json.loads(Path(path).read_text(encoding="utf-8"))
        a, m = c["audio_config"], c["model_config"]
        ll = m["llama_config"]
        expected = {"window_fn": "hamming", "center": False, "mel_power": 2.0, "f_min": 0.0, "f_max": None,
                    "use_styletts_mel_normalization": True}
        for k, v in expected.items():
            if a.get(k) != v:
                raise WeightsError(f"config {k}={a.get(k)!r}: this port implements {v!r} only")
        if m.get("disable_causal_attention") or ll.get("attention_bias") or ll.get("mlp_bias"):
            raise WeightsError("config asks for attention/MLP variants this port does not implement")
        if ll["num_key_value_heads"] != ll["num_attention_heads"] or a["win_length"] != a["n_fft"]:
            raise WeightsError("config asks for grouped KV heads or a shorter window: not implemented")
        if ll.get("hidden_act", "silu") != "silu" or ll.get("rope_scaling") or ll.get("rope_parameters") \
                or ll.get("attn_logit_softcapping") \
                or ll.get("head_dim", ll["hidden_size"] // ll["num_attention_heads"]) \
                != ll["hidden_size"] // ll["num_attention_heads"]:
            raise WeightsError("config asks for an activation, rope scaling or head size this port does not implement")
        return cls(sample_rate=a["target_sample_rate"], n_fft=a["n_fft"], hop=a["hop_length"], n_mels=a["n_mels"],
                   tail_s=c["inference"]["tail_seconds"], hidden=ll["hidden_size"],
                   intermediate=ll["intermediate_size"], layers=ll["num_hidden_layers"],
                   heads=ll["num_attention_heads"], rms_eps=ll["rms_norm_eps"], rope_theta=ll["rope_theta"],
                   vocab=ll["vocab_size"], labels=m["num_labels"])

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        h, i = self.hidden, self.intermediate
        out = {"backbone.embed_tokens.weight": (self.vocab, h), "backbone.norm.weight": (h,),
               "input_projection.weight": (h, self.n_mels), "input_projection.bias": (h,),
               "classifier.weight": (self.labels, h), "classifier.bias": (self.labels,)}
        for n in range(self.layers):
            p = f"backbone.layers.{n}."
            out |= {p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,),
                    p + "self_attn.q_proj.weight": (h, h), p + "self_attn.k_proj.weight": (h, h),
                    p + "self_attn.v_proj.weight": (h, h), p + "self_attn.o_proj.weight": (h, h),
                    p + "mlp.gate_proj.weight": (i, h), p + "mlp.up_proj.weight": (i, h),
                    p + "mlp.down_proj.weight": (h, i)}
        return out


# --- preprocessing ----------------------------------------------------------------


def sinc_kernel(orig: int, new: int, zeros: int = 6, rolloff: float = 0.99) -> tuple[np.ndarray, int, int, int]:
    """torchaudio's resampling kernel (sinc_interp_hann), computed the same way: (kernels, width, o, n)."""
    g = math.gcd(orig, new)
    o, n = orig // g, new // g
    base = min(o, n) * rolloff
    width = math.ceil(zeros * o / base)
    # torchaudio.functional.resample passes the waveform's dtype (float32) to the kernel, so
    # every step here is float32 too (its float64 branch is only for dtype=None)
    f32 = np.float32
    idx = np.arange(-width, width + o, dtype=f32)[None, :] / f32(o)
    t = np.arange(0, -n, -1, dtype=f32)[:, None] / f32(n) + idx
    t *= f32(base)
    t = np.clip(t, f32(-zeros), f32(zeros))
    window = np.cos(t * f32(math.pi) / f32(zeros) / f32(2)) ** 2
    t *= f32(math.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        kernels = np.where(t == 0, f32(1.0), np.sin(t) / t).astype(f32)
    kernels *= window * f32(base / o)
    return kernels.astype(f32), width, o, n


def resample(x: np.ndarray, orig: int, new: int) -> np.ndarray:
    """Band-limited resampling identical in method to torchaudio.functional.resample."""
    if orig == new:
        return x.astype(np.float32)
    kernels, width, o, n = sinc_kernel(orig, new)
    xp = np.pad(x.astype(np.float32), (width, width + o))
    frames = sliding_window_view(xp, kernels.shape[1])[::o]
    out = (frames.astype(np.float64) @ kernels.T.astype(np.float64)).reshape(-1)
    # torch takes ceil of the length after rounding it to float32
    return out[:math.ceil(float(np.float32(n * len(x) / o)))].astype(np.float32)


def mel_filterbank(cfg: Config) -> np.ndarray:
    """torchaudio.functional.melscale_fbanks with HTK scale, norm=None, float32: (n_freqs, n_mels)."""
    n_freqs = cfg.n_fft // 2 + 1
    all_freqs = np.linspace(0, cfg.sample_rate // 2, n_freqs, dtype=np.float32)
    m_min, m_max = 0.0, 2595.0 * math.log10(1.0 + (cfg.sample_rate / 2) / 700.0)
    m_pts = np.linspace(m_min, m_max, cfg.n_mels + 2, dtype=np.float32)
    f_pts = (700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)).astype(np.float32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


def log_mel(x16k: np.ndarray, cfg: Config, fb: np.ndarray) -> np.ndarray:
    """(frames, n_mels) normalised log-mel of one 16 kHz window (no centring, periodic Hamming)."""
    win = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(cfg.n_fft) / cfg.n_fft)).astype(np.float32)
    frames = sliding_window_view(x16k, cfg.n_fft)[::cfg.hop]
    spec = np.abs(np.fft.rfft(frames.astype(np.float64) * win, n=cfg.n_fft)) ** 2
    mel = spec.astype(np.float32) @ fb
    return ((np.log(np.maximum(mel, MEL_LOG_EPS)) - MEL_MEAN) / MEL_STD).astype(np.float32)


# --- the network ------------------------------------------------------------------


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return w * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))


def _rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class TruncationModel:
    """Same call as the reference: predict_from_audio(audio=[...], sampling_rate=, decision_window_ms=)."""

    def __init__(self, weights: dict[str, np.ndarray], cfg: Config) -> None:
        expected = cfg.expected_shapes()
        missing = sorted(set(expected) - set(weights))
        extra = sorted(set(weights) - set(expected))
        wrong = sorted(k for k in expected if k in weights and tuple(weights[k].shape) != expected[k])
        if missing or extra or wrong:
            raise WeightsError(f"weights do not match the architecture: missing {missing[:3]}, "
                               f"unexpected {extra[:3]}, wrong shape {wrong[:3]}")
        self.cfg = cfg
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.fb = mel_filterbank(cfg)
        self.head_dim = cfg.hidden // cfg.heads
        self._workers = max(1, min(8, os.cpu_count() or 1))
        self._pool = ThreadPoolExecutor(max_workers=self._workers)

    def _rope(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        d = self.head_dim
        inv = (1.0 / (self.cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / d))
               ).astype(np.float32)
        freqs = np.arange(t, dtype=np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb), np.sin(emb)

    def mels(self, audio: list[np.ndarray], sampling_rate: int) -> np.ndarray:
        n_keep = self.cfg.hop * max(1, round(round(self.cfg.tail_s * self.cfg.sample_rate) / self.cfg.hop))
        out = []
        for wav in audio:
            x = resample(np.asarray(wav, dtype=np.float32), int(sampling_rate), self.cfg.sample_rate)
            x = x[len(x) - n_keep:] if len(x) >= n_keep else np.pad(x, (n_keep - len(x), 0))
            out.append(log_mel(x, self.cfg, self.fb))
        return np.stack(out)

    def logits(self, mel: np.ndarray, keep_last: int | None = None) -> np.ndarray:
        """Classifier logits (batch, frames, labels). With keep_last, only the last frames come out:
        the causal model needs every position up to the last layer, but the last layer's queries and
        MLP only for the frames that are kept (identical maths for those frames)."""
        cfg, w, hd = self.cfg, self.w, self.head_dim
        b, t, _ = mel.shape
        h = mel @ w["input_projection.weight"].T + w["input_projection.bias"]
        cos, sin = self._rope(t)
        causal = np.triu(np.full((t, t), -np.inf, dtype=np.float32), k=1)
        scale = np.float32(1.0 / math.sqrt(hd))
        for n in range(cfg.layers):
            p = f"backbone.layers.{n}."
            last = n == cfg.layers - 1 and keep_last is not None
            r = _rms(h, w[p + "input_layernorm.weight"], cfg.rms_eps)
            q, k, v = (np.ascontiguousarray((r @ w[p + f"self_attn.{x}_proj.weight"].T)
                                            .reshape(b, t, cfg.heads, hd).transpose(0, 2, 1, 3)) for x in "qkv")
            q = (q * cos + _rotate_half(q) * sin) * scale
            k = k * cos + _rotate_half(k) * sin
            mask = causal
            if last:
                q, mask, h = q[:, :, -keep_last:], causal[-keep_last:], h[:, -keep_last:]
            att = np.empty_like(q)
            for i in range(b):              # one window at a time keeps the score matrix small
                s_ = q[i] @ k[i].transpose(0, 2, 1)
                s_ += mask
                s_ -= s_.max(axis=-1, keepdims=True)
                np.exp(s_, out=s_)
                s_ /= s_.sum(axis=-1, keepdims=True)
                att[i] = s_ @ v[i]
            h = h + att.transpose(0, 2, 1, 3).reshape(b, q.shape[2], cfg.hidden) @ w[p + "self_attn.o_proj.weight"].T
            r = _rms(h, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
            g = r @ w[p + "mlp.gate_proj.weight"].T
            u = r @ w[p + "mlp.up_proj.weight"].T
            h = h + ((g / (1.0 + np.exp(-g))) * u) @ w[p + "mlp.down_proj.weight"].T
        h = _rms(h, w["backbone.norm.weight"], cfg.rms_eps)
        return h @ w["classifier.weight"].T + w["classifier.bias"]

    def decision_frames(self, decision_window_ms: float) -> int:
        """Frames in the decision window, computed exactly as the reference does."""
        if not decision_window_ms > 0:
            raise ValueError("decision_window_ms must be > 0")
        frame_s = float(self.cfg.hop) / float(self.cfg.sample_rate)
        return max(1, int(math.ceil((float(decision_window_ms) / 1000.0) / frame_s)))

    def predict_from_audio(self, *, audio: list[np.ndarray], sampling_rate: int, decision_window_ms: float):
        if not (isinstance(sampling_rate, (int, np.integer)) and not isinstance(sampling_rate, bool)
                and sampling_rate > 0):
            raise ValueError(f"sampling_rate must be a positive integer, got {sampling_rate!r}")
        for wav in audio:
            if np.ndim(wav) != 1 or np.size(wav) == 0:
                raise ValueError("each window must be a non-empty mono (1-D) array")
        k = self.decision_frames(decision_window_ms)
        mel = self.mels(audio, int(sampling_rate))
        # numpy's element-wise maths runs on one core but releases the GIL: whole windows in parallel
        chunks = np.array_split(mel, min(len(mel), self._workers)) if len(mel) else []
        z = np.concatenate(list(self._pool.map(lambda c: self.logits(c, keep_last=k), chunks)))
        z = z.astype(np.float64)
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        p1 = (z / z.sum(axis=-1, keepdims=True))[..., 1]
        return Prediction(truncation_score=p1.max(axis=1))


@dataclass(frozen=True)
class Prediction:
    truncation_score: np.ndarray


def build(weights_path: Path, config_path: Path) -> TruncationModel:
    return build_from_bytes(Path(weights_path).read_bytes(), config_path)


def build_from_bytes(raw: bytes, config_path: Path) -> TruncationModel:
    return TruncationModel(parse_safetensors(raw), Config.from_json(config_path))
