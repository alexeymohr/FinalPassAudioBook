"""The breath model in numpy: Respiro-en's network (vendor/respiro/modules.py, MIT) and its features, with no
PyTorch, torchaudio or librosa. One breath probability per 10 ms of 16 kHz audio.

Features, as the authors' `feature_extractor` (librosa 0.10/0.11): a 128-band Slaney mel power spectrogram (400-point
periodic Hann window, hop 160, centred with zero padding) in dB against its own maximum (floor 80 dB down); the
unbiased variance of that across the bands (VMS); the zero-crossing rate over the same 25 ms window (ZCR).
Network, as `DetectionNet` in eval mode: 2 strided 3x3 convolutions (3 -> 1 channel, the time axis padded to even
first), Linear 31 -> 128, torchaudio's Conformer (8 layers, 4 heads, ffn 256, depthwise kernel 31), 2 strided
transposed convolutions back up, a bidirectional LSTM (256 each way) over all of it, Linear 512 -> 1, sigmoid, cut to
the input's length. A chapter is scored in 30 s windows every 20 s, keeping each window's middle 20 s (its own
maximum normalises each window's mel, as the model saw sentences of up to a few tens of seconds).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import firwin, resample_poly

from .truncation_np import WeightsError, read_safetensors

SR = 16000
N_FFT, HOP, N_MELS = 400, 160, 128
WINDOW_S, KEEP_S = 30, 20


@dataclass(frozen=True)
class Config:
    n_mels: int = 128          # mel bands (the downsampling turns them into (n_mels - 3) // 2 + 1 ... = 31)
    hidden: int = 128          # Conformer width
    heads: int = 4
    ffn: int = 256
    layers: int = 8
    kernel: int = 31
    lstm: int = 256

    def reduced_mels(self) -> int:
        f = (self.n_mels - 3) // 2 + 1
        return (f - 3) // 2 + 1

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        h, f, k, L = self.hidden, self.ffn, self.kernel, self.lstm
        s = {"downsampling.conv1.0.weight": (1, 3, 3, 3), "downsampling.conv1.0.bias": (1,),
             "downsampling.conv2.0.weight": (1, 1, 3, 3), "downsampling.conv2.0.bias": (1,),
             "linear.weight": (h, self.reduced_mels()), "linear.bias": (h,),
             "upsampling.deconv.0.weight": (h, h, 3), "upsampling.deconv.0.bias": (h,),
             "upsampling.deconv.2.weight": (h, h, 3), "upsampling.deconv.2.bias": (h,),
             "fc.weight": (1, 2 * L), "fc.bias": (1,)}
        for d in ("", "_reverse"):
            s |= {f"lstm.weight_ih_l0{d}": (4 * L, h), f"lstm.weight_hh_l0{d}": (4 * L, L),
                  f"lstm.bias_ih_l0{d}": (4 * L,), f"lstm.bias_hh_l0{d}": (4 * L,)}
        for i in range(self.layers):
            p = f"conformer.conformer_layers.{i}."
            for ff in ("ffn1", "ffn2"):
                s |= {f"{p}{ff}.sequential.0.weight": (h,), f"{p}{ff}.sequential.0.bias": (h,),
                      f"{p}{ff}.sequential.1.weight": (f, h), f"{p}{ff}.sequential.1.bias": (f,),
                      f"{p}{ff}.sequential.4.weight": (h, f), f"{p}{ff}.sequential.4.bias": (h,)}
            s |= {f"{p}self_attn_layer_norm.weight": (h,), f"{p}self_attn_layer_norm.bias": (h,),
                  f"{p}self_attn.in_proj_weight": (3 * h, h), f"{p}self_attn.in_proj_bias": (3 * h,),
                  f"{p}self_attn.out_proj.weight": (h, h), f"{p}self_attn.out_proj.bias": (h,),
                  f"{p}conv_module.layer_norm.weight": (h,), f"{p}conv_module.layer_norm.bias": (h,),
                  f"{p}conv_module.sequential.0.weight": (2 * h, h, 1), f"{p}conv_module.sequential.0.bias": (2 * h,),
                  f"{p}conv_module.sequential.2.weight": (h, 1, k), f"{p}conv_module.sequential.2.bias": (h,),
                  f"{p}conv_module.sequential.3.weight": (h,), f"{p}conv_module.sequential.3.bias": (h,),
                  f"{p}conv_module.sequential.3.running_mean": (h,), f"{p}conv_module.sequential.3.running_var": (h,),
                  f"{p}conv_module.sequential.5.weight": (h, h, 1), f"{p}conv_module.sequential.5.bias": (h,),
                  f"{p}final_layer_norm.weight": (h,), f"{p}final_layer_norm.bias": (h,)}
        return s


# ---- features -------------------------------------------------------------------------------------------------

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    logstep = np.log(6.4) / 27.0
    m = f / f_sp
    log = f >= min_log_hz
    m[log] = min_log_hz / f_sp + np.log(f[log] / min_log_hz) / logstep
    return m


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    f = f_sp * m
    log = m >= min_log_mel
    f[log] = min_log_hz * np.exp(logstep * (m[log] - min_log_mel))
    return f


def mel_filterbank(n_mels: int = N_MELS, sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, htk=False, norm='slaney'), as float32 (n_mels, 1 + n_fft // 2)."""
    fft = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sr / 2.0]))[0], n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft[None, :]
    w = np.maximum(0.0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


_FB = mel_filterbank()
_WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann


def _frames(y: np.ndarray, n: int, hop: int) -> np.ndarray:
    count = 1 + (len(y) - n) // hop
    return np.lib.stride_tricks.as_strided(y, (count, n), (y.strides[0] * hop, y.strides[0]), writeable=False)


def features(y: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """(3, n_mels, T) float32 for 16 kHz audio: mel dB, VMS, ZCR (T = 1 + len(y) // 160)."""
    y = np.ascontiguousarray(y, np.float32)
    pad = np.pad(y, N_FFT // 2)
    fr = _frames(pad, N_FFT, HOP)
    spec = np.abs(np.fft.rfft(fr * _WIN, axis=1)) ** 2                       # (T, 201)
    fb = _FB if n_mels == N_MELS else mel_filterbank(n_mels)
    mel = (spec @ fb.T).T.astype(np.float32)                                   # (n_mels, T)
    db = 10.0 * np.log10(np.maximum(1e-10, mel))
    db -= 10.0 * np.log10(max(1e-10, float(mel.max())))
    db = np.maximum(db, db.max() - 80.0).astype(np.float32)
    s = np.sign(fr)                                                            # same window and padding
    zcr = (np.abs(s[:, 1:] - s[:, :-1]).sum(axis=1) * 0.5 / N_FFT).astype(np.float32)
    vms = db.var(axis=0, ddof=1).astype(np.float32)
    t = db.shape[1]
    return np.stack([db, np.broadcast_to(vms, (n_mels, t)), np.broadcast_to(zcr, (n_mels, t))]).astype(np.float32)


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Polyphase resampling to 16 kHz (the model's rate) with a long, sharp anti-alias filter (64 zero crossings,
    Kaiser beta 8, cut-off at 0.96 of the new Nyquist). The model was trained on audio librosa resampled with soxr's
    high-quality setting; this filter is 8x closer to that than scipy's default (rms difference 0.0028 against
    0.023 on a test chirp) and changes 3-7x fewer frames' side of 0.5 on real chapters."""
    if sr == SR:
        return np.asarray(x, np.float32)
    g = gcd(SR, int(sr))
    up, down = SR // g, int(sr) // g
    m = max(up, down)
    h = firwin(2 * 64 * m + 1, 0.96 / m, window=("kaiser", 8.0))
    return resample_poly(np.asarray(x, np.float64), up, down, window=h).astype(np.float32)


# ---- network --------------------------------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _conv2d_s2(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Conv2d(kernel 3, stride 2, no padding) on (C, F, T) after padding T to even; ReLU."""
    if x.shape[-1] % 2 == 0:
        x = np.pad(x, ((0, 0), (0, 0), (0, 1)))
    c, f, t = x.shape
    fo, to = (f - 3) // 2 + 1, (t - 3) // 2 + 1
    out = np.full((w.shape[0], fo, to), 0.0, np.float32) + b[:, None, None]
    for i in range(3):
        for j in range(3):
            patch = x[:, i:i + 2 * fo - 1:2, j:j + 2 * to - 1:2]                # (C, fo, to)
            out += np.einsum("oc,cft->oft", w[:, :, i, j], patch, optimize=True)
    return np.maximum(out, 0.0)


def _deconv_s2(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ConvTranspose1d(kernel 3, stride 2) on (C, T) -> (O, 2T + 1); ReLU."""
    c, t = x.shape
    out = np.zeros((w.shape[1], 2 * (t - 1) + 3), np.float32) + b[:, None]
    for k in range(3):
        out[:, k:k + 2 * t - 1:2] += w[:, :, k].T @ x
    return np.maximum(out, 0.0)


def _lstm(x: np.ndarray, wih: np.ndarray, whh: np.ndarray, bih: np.ndarray, bhh: np.ndarray) -> np.ndarray:
    """One direction of torch.nn.LSTM (gates i, f, g, o), zero initial state; (T, I) -> (T, H)."""
    hsz = whh.shape[1]
    gx = x @ wih.T + (bih + bhh)
    whh_t = np.ascontiguousarray(whh.T)
    h = np.zeros(hsz, np.float32)
    c = np.zeros(hsz, np.float32)
    out = np.empty((x.shape[0], hsz), np.float32)
    for t in range(x.shape[0]):
        g = gx[t] + h @ whh_t
        i, f = _sigmoid(g[:hsz]), _sigmoid(g[hsz:2 * hsz])
        gg, o = np.tanh(g[2 * hsz:3 * hsz]), _sigmoid(g[3 * hsz:])
        c = f * c + i * gg
        h = o * np.tanh(c)
        out[t] = h
    return out


class BreathModel:
    def __init__(self, weights: dict[str, np.ndarray], cfg: Config = Config()) -> None:
        want = cfg.expected_shapes()
        missing = sorted(set(want) - set(weights))
        unexpected = sorted(set(weights) - set(want))
        if missing or unexpected:
            raise WeightsError(f"breath model weights: missing {missing[:3]}, unexpected {unexpected[:3]}")
        for k, shape in want.items():
            if tuple(weights[k].shape) != shape:
                raise WeightsError(f"breath model weights: {k} has shape {tuple(weights[k].shape)}, expected {shape}")
        self.w = {k: np.ascontiguousarray(v, np.float32) for k, v in weights.items()}
        self.cfg = cfg

    def _conformer_layer(self, x: np.ndarray, i: int) -> np.ndarray:
        w, cfg = self.w, self.cfg
        p = f"conformer.conformer_layers.{i}."

        def ffn(z, name):
            q = f"{p}{name}.sequential."
            z = _layer_norm(z, w[q + "0.weight"], w[q + "0.bias"])
            z = _silu(z @ w[q + "1.weight"].T + w[q + "1.bias"])
            return z @ w[q + "4.weight"].T + w[q + "4.bias"]

        x = x + 0.5 * ffn(x, "ffn1")
        z = _layer_norm(x, w[p + "self_attn_layer_norm.weight"], w[p + "self_attn_layer_norm.bias"])
        t, h = z.shape
        qkv = z @ w[p + "self_attn.in_proj_weight"].T + w[p + "self_attn.in_proj_bias"]
        d = h // cfg.heads
        q, k, v = (qkv[:, j * h:(j + 1) * h].reshape(t, cfg.heads, d).transpose(1, 0, 2) for j in range(3))
        att = (q / np.sqrt(d).astype(np.float32)) @ k.transpose(0, 2, 1)       # (heads, T, T)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att /= att.sum(axis=-1, keepdims=True)
        z = (att @ v).transpose(1, 0, 2).reshape(t, h)
        x = x + (z @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"])
        q = p + "conv_module."
        z = _layer_norm(x, w[q + "layer_norm.weight"], w[q + "layer_norm.bias"]).T             # (H, T)
        z = w[q + "sequential.0.weight"][:, :, 0] @ z + w[q + "sequential.0.bias"][:, None]
        z = z[:h] * _sigmoid(z[h:])                                                        # GLU over channels
        kk = cfg.kernel
        zp = np.pad(z, ((0, 0), ((kk - 1) // 2, (kk - 1) // 2)))
        dw = w[q + "sequential.2.weight"][:, 0, :]                                          # (H, K)
        z = np.zeros_like(z) + w[q + "sequential.2.bias"][:, None]
        for j in range(kk):
            z += dw[:, j:j + 1] * zp[:, j:j + t]
        z = ((z - w[q + "sequential.3.running_mean"][:, None]) / np.sqrt(w[q + "sequential.3.running_var"][:, None] + 1e-5)
             * w[q + "sequential.3.weight"][:, None] + w[q + "sequential.3.bias"][:, None])
        z = _silu(z)
        z = w[q + "sequential.5.weight"][:, :, 0] @ z + w[q + "sequential.5.bias"][:, None]
        x = x + z.T
        x = x + 0.5 * ffn(x, "ffn2")
        return _layer_norm(x, w[p + "final_layer_norm.weight"], w[p + "final_layer_norm.bias"])

    def probs_from_features(self, feat: np.ndarray) -> np.ndarray:
        """(3, n_mels, T) -> (T,) breath probabilities."""
        w = self.w
        t_in = feat.shape[-1]
        x = _conv2d_s2(feat, w["downsampling.conv1.0.weight"], w["downsampling.conv1.0.bias"])
        x = _conv2d_s2(x, w["downsampling.conv2.0.weight"], w["downsampling.conv2.0.bias"])
        x = x[0].T @ w["linear.weight"].T + w["linear.bias"]                               # (S, H)
        for i in range(self.cfg.layers):
            x = self._conformer_layer(x, i)
        x = _deconv_s2(x.T, w["upsampling.deconv.0.weight"], w["upsampling.deconv.0.bias"])
        x = _deconv_s2(x, w["upsampling.deconv.2.weight"], w["upsampling.deconv.2.bias"]).T   # (L, H)
        fwd = _lstm(x, w["lstm.weight_ih_l0"], w["lstm.weight_hh_l0"], w["lstm.bias_ih_l0"], w["lstm.bias_hh_l0"])
        bwd = _lstm(x[::-1], w["lstm.weight_ih_l0_reverse"], w["lstm.weight_hh_l0_reverse"],
                    w["lstm.bias_ih_l0_reverse"], w["lstm.bias_hh_l0_reverse"])[::-1]
        logit = np.concatenate([fwd, bwd], axis=1) @ w["fc.weight"][0] + w["fc.bias"][0]
        return _sigmoid(logit)[:t_in].astype(np.float32)

    def probs(self, y16: np.ndarray) -> np.ndarray:
        """Breath probability per 10 ms frame (frame i is centred on i * 10 ms) for 16 kHz audio of any length."""
        y16 = np.asarray(y16, np.float32)
        n = 1 + len(y16) // HOP
        out = np.zeros(n, np.float32)
        step, margin = KEEP_S * SR, (WINDOW_S - KEEP_S) // 2 * SR
        start = 0
        while True:
            a, b = max(0, start - margin), min(len(y16), start + step + margin)
            p = self.probs_from_features(features(y16[a:b], self.cfg.n_mels))
            f0, k0 = a // HOP, start // HOP
            last = start + step >= len(y16)
            k1 = n if last else (start + step) // HOP
            out[k0:k1] = p[k0 - f0:k1 - f0]
            if last:
                return out
            start += step


def build(weights_path: Path) -> BreathModel:
    return BreathModel(read_safetensors(weights_path))
