"""Hard plosive ("p-pop"): a short, strong low-frequency burst. Numbers only.

A plosive pop is the burst of air from a "p" or "b" hitting the microphone: a
thump concentrated at the very bottom, where the voice itself has almost nothing.
Long-term spectrum, 12 chapters of one AI narration: 40-65 Hz holds -27 to -28 dB
of all energy, 20-40 Hz -34 to -36 dB, 65-100 Hz -11 dB. So the check watches the
20-65 Hz band (steep edges, so the voice's low pitch does not leak in) and reports a
burst that rises fast, stays short, stands well above the chapter's own typical
low-band level during speech, and carries a large share of all the energy at
that instant — a sharp word onset also splashes into the low band, but only as
a tiny share of its energy. It does not check that the burst sits on
a "p" or "b" (it cannot see the words); any such thump is worth a listen.
Bursts inside a breath (a T-inhale's thump) belong to the breath check and are
skipped.

Selective on level (operator: a pop has to be loud to matter). Short swells in
that band are common — in the 12 chapters, 4.7 a minute at -30 dBFS or louder,
0.9 at -22 — so only a burst at least 3 dB louder than the speech around it is
listed. Severity by that margin: 3 from +9 dB, 2 from +6 dB, else 1. Built
check, same 12 chapters (163 min): 14 listed, 0-5 per chapter (1 / 3 / 10 at
severity 3 / 2 / 1). The first sound after digital black (an edit onset) is
skipped. The band filter is steep, so it rings a little before a pop: the
loudest peak within 0.1 s is the one reported. To be checked against the
operator's labelled pops.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..chapter import Chapter
from ..findings import Finding, ladder


@dataclass(frozen=True)
class PlosiveTunables:
    band_hz: tuple[float, float] = (20.0, 65.0)
    filter_order: int = 8
    block_ms: float = 5.0
    above_speech_db: float = 18.0     # over the chapter's median low-band level while speaking
    min_vs_speech_db: float = 3.0     # listed: the thump is louder than the speech around it
    sev2_vs_speech_db: float = 6.0
    sev3_vs_speech_db: float = 9.0
    min_low_share_db: float = -10.0   # the low band carries at least this share of the moment's energy
    rise_db: float = 12.0             # within rise_ms before the peak
    rise_ms: float = 20.0
    max_width_ms: float = 80.0        # time within 10 dB of the peak
    breath_margin_ms: float = 30.0
    after_black_ms: float = 80.0      # the first sound after digital black is an edit onset, not a pop

    def as_dict(self) -> dict:
        return asdict(self)


def _black_run_ends(x: np.ndarray, min_run: int = 50) -> np.ndarray:
    """Sample where each stretch of digital black (>= min_run exact zeros) ends."""
    d = np.diff(np.concatenate(([0], (x == 0.0).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends[(ends - starts) >= min_run].astype(float)


def _block_db(y: np.ndarray, n: int) -> np.ndarray:
    k = len(y) // n
    ms = np.mean(y[:k * n].reshape(k, n) ** 2, axis=1) if k else np.array([])
    return 10 * np.log10(np.maximum(ms, 1e-20))


def plosive_findings(ch: Chapter, breath_spans: tuple[tuple[int, int], ...],
                     t: PlosiveTunables = PlosiveTunables()) -> list[Finding]:
    y, fs = ch.low
    if len(y) < int(fs):
        return []
    lf = sosfiltfilt(butter(t.filter_order, list(t.band_hz), btype="band", fs=fs, output="sos"), y)
    n = max(1, int(round(t.block_ms * fs / 1000.0)))
    env = _block_db(lf, n)
    full = _block_db(y, n)
    block_s = n / fs

    speech = ch.frames.voicing > 0.7
    speech &= ch.frames.rms_db > ch.narration_dbfs - 25.0
    frame_s = ch.frames.hop / ch.sr
    speech_blocks = np.clip((np.flatnonzero(speech) * frame_s / block_s).astype(int), 0, len(env) - 1)
    if speech_blocks.size == 0:
        return []
    ref = float(np.median(env[speech_blocks]))
    gate = max(ch.narration_dbfs + t.min_vs_speech_db - 10.0, ref + t.above_speech_db)   # loose; exact per burst
    black_ends = _black_run_ends(ch.x) / ch.sr

    rise_blocks = max(1, int(round(t.rise_ms / t.block_ms)))
    breaths = [(s / ch.sr - t.breath_margin_ms / 1000, e / ch.sr + t.breath_margin_ms / 1000)
               for s, e in breath_spans]
    out, taken = [], []
    cand = np.flatnonzero((env[1:-1] >= gate) & (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:])) + 1
    for i in cand[np.argsort(-env[cand], kind="stable")]:      # loudest first: a steep filter rings before a pop
        t_s = i * block_s
        if any(abs(t_s - u) < 0.1 for u in taken) or any(a <= t_s <= b for a, b in breaths):
            continue
        k = np.searchsorted(black_ends, t_s)
        if k > 0 and t_s - black_ends[k - 1] <= t.after_black_ms / 1000.0:
            continue
        pk = float(env[i])
        if pk - float(full[i]) < t.min_low_share_db:
            continue                  # a word onset's low-frequency splash, not a pop
        if pk - float(env[max(0, i - rise_blocks):i].min(initial=pk)) < t.rise_db:
            continue
        lo = i
        while lo > 0 and env[lo - 1] >= pk - 10:
            lo -= 1
        hi = i
        while hi < len(env) - 1 and env[hi + 1] >= pk - 10:
            hi += 1
        if (hi - lo + 1) * t.block_ms > t.max_width_ms:
            continue
        s = int(lo * block_s * ch.sr)
        e = int((hi + 1) * block_s * ch.sr)
        rel = round(pk - ch.local_speech_dbfs(s, e), 1)     # graded on the value the text shows
        if rel < t.min_vs_speech_db:
            continue
        out.append(Finding(
            file=ch.name, check="plosive", start_sample=s, end_sample=e,
            start_time=ch.clock(s), end_time=ch.clock(e),
            severity=ladder(rel, (t.min_vs_speech_db, t.sev2_vs_speech_db, t.sev3_vs_speech_db)),
            problem=f"plosive pop: burst below 65 Hz, {rel:+.1f} dB against the speech",
            measures={"burst_dbfs": round(pk, 1), "burst_vs_speech_db": round(rel, 1),
                      "speech_low_band_dbfs": round(ref, 1), "width_ms": int((hi - lo + 1) * t.block_ms)},
        ))
        taken.append(t_s)
    return sorted(out, key=lambda f: f.start_sample)
