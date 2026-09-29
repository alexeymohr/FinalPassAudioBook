"""Noisy section: broadband noise floor tracked through speech. Numbers only.

Minimum statistics (Martin, 2001): in each ⅓-octave band, the quietest moments
within a few seconds are the noise, because speech keeps stopping — between
words, between phrases — while steady noise does not. Band minima are corrected
for the bias of taking a minimum, summed into one broadband floor, and a stretch
where that floor stays above the threshold is reported. Works with or without
pauses in the stretch, so noise under continuous narration is caught too. A
stretch must stay raised for about 5.5 s (3 s of the 2.5 s-minimum track, as
calibrated); its reported start and end are the raised floor's true extent.

Levels are relative to the speech, because voices and masters differ. A stretch
is listed when its floor comes within 41 dB of the chapter's narration (-60 dBFS
at the usual -19 dBFS narration; no published threshold exists). Its severity is
the gap between the speech around it (±10 s) and its floor: 3 under 25 dB, 2
under 32 dB, else 1 — gentle on purpose (the operator judged most such QC notes
on the calibration title not to be real problems).

Calibration on one audiobook (12 chapters) whose QC notes marked 7 noisy
blocks: the check finds 6 of the 7. By speech-to-floor gap, stretches QC noted:
all of those under 25 dB, 2 of 5 at 25-32 dB, a few of the rest. The missed
block's floor was the chapter's normal floor — a room character, not louder
noise (a reverb/roominess question). The narration level was nearly constant
across that title, so there relative and absolute thresholds agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from ..chapter import Chapter, dbfs
from ..findings import Finding

FRAME_S, HOP_S = 0.020, 0.010
SMOOTH_FRAMES = 3
BAND_CENTRES_HZ = tuple(1000.0 * 2 ** (k / 3) for k in range(-12, 13))   # 63 Hz - 15.9 kHz
BLOCK_FRAMES = 4096
# Bias of a band minimum, as a function of how many FFT bins the band holds:
# multiply the minimum by this to get the band's mean. Measured on 4 x 60 s of
# stationary white noise with exactly this framing and smoothing (44.1 kHz);
# indexing by bin count carries it to other sample rates. Verified in the tests
# on white and pink noise at 44.1 and 48 kHz.
BIAS_BINS = (1, 2, 3, 4, 6, 9, 11, 13, 17, 22, 27, 34, 43, 54, 68, 87, 108, 136, 173)
BIAS_FACTOR = (10.99, 8.50, 6.77, 5.51, 4.61, 3.60, 3.08, 2.88, 2.51, 2.21, 2.06, 1.91, 1.77,
               1.66, 1.60, 1.48, 1.42, 1.36, 1.30)


def bias_for(bins: int) -> float:
    return float(np.exp(np.interp(np.log(bins), np.log(BIAS_BINS), np.log(BIAS_FACTOR))))


@dataclass(frozen=True)
class NoiseTunables:
    max_gap_db: float = 41.0           # listed when the floor is within this of the narration
    fallback_threshold_dbfs: float = -60.0   # when the file has no measurable narration
    sev2_under_gap_db: float = 32.0
    sev3_under_gap_db: float = 25.0
    fallback_sev3_dbfs: float = -45.0  # files with no narration: graded on the floor itself
    fallback_sev2_dbfs: float = -51.0
    min_window_s: float = 2.5
    min_stretch_s: float = 3.0
    merge_gap_s: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


def floor_track(x: np.ndarray, sr: int, t: NoiseTunables = NoiseTunables(),
                exclude: list[tuple[list[float], float, float]] | None = None) -> tuple[np.ndarray, float]:
    """Broadband noise floor (dBFS) every HOP_S, and the hop in seconds.

    Each frame's DC is removed first (an offset would leak into the lowest band). A band's floor is
    its bias-corrected minimum, but never above its running mean: a steady tone's minimum already
    is its mean, and correcting it again would inflate it. `exclude` = (frequencies, start s, end s):
    bands holding a listed hum are left out while it sounds (the hum check reports it)."""
    n, hop = int(round(FRAME_S * sr)), int(round(HOP_S * sr))
    if len(x) < n:
        return np.array([]), HOP_S
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    w = np.hanning(n)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    edges = [(c * 2 ** (-1 / 6), c * 2 ** (1 / 6)) for c in BAND_CENTRES_HZ if c * 2 ** (1 / 6) < sr / 2]
    idx = [np.flatnonzero((f >= lo) & (f < hi)) for lo, hi in edges]
    idx = [i for i in idx if i.size]
    scale = 2.0 / (nfft * float((w * w).sum()))
    view = sliding_window_view(x, n)[::hop]
    bands = np.empty((len(view), len(idx)))
    for b0 in range(0, len(view), BLOCK_FRAMES):
        frames = view[b0:b0 + BLOCK_FRAMES]
        frames = frames - frames.mean(axis=1, keepdims=True)
        p = np.abs(np.fft.rfft(frames * w, nfft, axis=1)) ** 2 * scale
        bands[b0:b0 + len(p)] = np.stack([p[:, i].sum(axis=1) for i in idx], axis=1)
    smooth = uniform_filter1d(bands, SMOOTH_FRAMES, axis=0, mode="nearest")
    size = max(1, int(round(t.min_window_s / HOP_S)))
    minima = minimum_filter1d(smooth, size, axis=0, mode="nearest")
    bias = np.array([bias_for(i.size) for i in idx])
    floor = np.minimum(minima * bias, uniform_filter1d(smooth, size, axis=0, mode="nearest"))
    kept_edges = [e for e, i in zip(edges, [np.flatnonzero((f >= lo) & (f < hi)) for lo, hi in edges]) if i.size]
    # a tone spreads over its main lobe and first sidelobes (about four FFT bins either side in these
    # 20 ms frames), and the few-bin low bands' bias correction lifts that leakage by up to 10 dB
    leak = 4.0 * sr / nfft
    for freqs, a_s, b_s in exclude or []:
        cols = [j for j, (lo, hi) in enumerate(kept_edges) if any(lo < fr + leak and fr - leak < hi for fr in freqs)]
        r0, r1 = max(0, int(a_s / HOP_S)), min(len(floor), int(b_s / HOP_S) + 1)
        if cols and r1 > r0:
            floor[r0:r1, cols] = 0.0
    return dbfs(floor.sum(axis=1)), HOP_S


def _stretches(above: np.ndarray, hop_s: float, t: NoiseTunables) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * hop_s <= t.merge_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if (b - a) * hop_s >= t.min_stretch_s]


def noise_findings(ch: Chapter, t: NoiseTunables = NoiseTunables(),
                   exclude: list[tuple[list[float], float, float]] | None = None) -> tuple[list[Finding], float | None]:
    track, hop_s = floor_track(ch.x, ch.sr, t, exclude)
    if track.size == 0:
        return [], None
    audible = track[track > -120]
    median_floor = float(np.median(audible)) if audible.size else None
    narration = ch.narration_dbfs
    threshold = narration - t.max_gap_db if np.isfinite(narration) else t.fallback_threshold_dbfs
    above = track > threshold
    # The minimum over min_window_s (2.5 s) trims each raised stretch by half a window at both ends,
    # so min_stretch_s (3 s) of track means about 5.5 s of raised floor. That is the length the QC
    # calibration was done with, so it stays; only the reported times are widened back out.
    half = int(round(t.min_window_s / hop_s)) // 2
    out = []
    for a0, b0 in _stretches(above, hop_s, t):
        seg = track[a0:b0]
        a, b = max(0, a0 - half), min(len(track), b0 + half)
        s = int(a * hop_s * ch.sr)
        e = min(len(ch.x), int((b * hop_s + FRAME_S) * ch.sr))
        level = round(float(np.median(seg)), 1)
        speech = ch.local_speech_dbfs(s, e) if np.isfinite(narration) else float("nan")
        if np.isfinite(speech):
            gap = round(speech - level, 1)
            severity = 3 if gap < t.sev3_under_gap_db else 2 if gap < t.sev2_under_gap_db else 1
            text = f"noisy section: noise floor about {level:.1f} dBFS, {gap:.1f} dB under the speech"
        else:                           # no narration to compare with: the dBFS equivalents of the ladder
            gap = None
            severity = 3 if level >= t.fallback_sev3_dbfs else 2 if level >= t.fallback_sev2_dbfs else 1
            text = f"noisy section: noise floor about {level:.1f} dBFS (no narration measured in this file)"
        out.append(Finding(
            file=ch.name, check="noise", start_sample=s, end_sample=e,
            start_time=ch.clock(s), end_time=ch.clock(e), severity=severity, problem=text,
            measures={"floor_median_dbfs": level, "floor_max_dbfs": round(float(seg.max()), 1),
                      "speech_dbfs": round(speech, 1) if np.isfinite(speech) else "n/a",
                      "floor_under_speech_db": gap if gap is not None else "n/a",
                      "duration_s": round((e - s) / ch.sr, 1)},
        ))
    return out, median_floor
