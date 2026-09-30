"""Click in a pause: a tick or click in the silence between words. Numbers only.

Inside speech a click is not a defect: every t, k, p and ch is one, and the
operator heard consonants in almost every candidate a speech-wide search listed.
A click in a pause is different — nothing masks it — so only those are listed,
and only when they stand clear of everything a voice does there:

* brief: in several half-octave bands (1-16 kHz) it rises at least 8 dB over
  the same band's loudest level within 15 ms on both sides (the idea of Paul L's
  Audacity De-Clicker, reimplemented; a sustained sound such as an "s" cannot);
* loud: sample peak at least -45 dBFS (operator's floor);
* clear: at least 100 ms from the words on both sides — a word's final release
  and a word's onset sit within that — and at least 100 ms from any breath,
  whose mouth clicks the breath check reports.

Severity 3: in a pause QC calls it. Evidence (one title, 12 chapters): the
operator heard 64 candidates from looser rules and confirmed 2 ticks; this rule
lists exactly those 2 and nothing else in the 12 chapters. The limits were set on
the same chapters, so they need a second title to confirm them.

The same rule covers the room tone before the first word and after the last one
(operator: as audible there as between words), where only the word side needs the
100 ms clearance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..chapter import Chapter
from ..findings import Finding

CLICK_SEVERITY = 3


@dataclass(frozen=True)
class ClickTunables:
    window_samples_44k: int = 128      # STFT window at 44.1 kHz (samples; scaled with the rate), hop 1/4
    band_lo_hz: float = 1000.0
    band_hi_hz: float = 16000.0        # half-octave bands between these
    left_guard_ms: float = 3.0         # window smear before the peak
    right_guard_ms: float = 8.0        # smear and a few ms of decay after it
    side_ms: float = 15.0              # the band's own maximum over this long, each side
    top_bands: int = 4                 # rise = mean of the best this-many bands
    min_rise_db: float = 8.0
    min_peak_dbfs: float = -45.0
    clear_ms: float = 100.0            # from words and breaths, both sides
    one_per_ms: float = 20.0           # one click per this long

    def as_dict(self) -> dict:
        return asdict(self)


def _band_db(x: np.ndarray, sr: int, t: ClickTunables) -> tuple[np.ndarray, int, int]:
    n = int(2 ** round(np.log2(t.window_samples_44k * sr / 44100.0)))
    hop = n // 4
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges, lo = [], t.band_lo_hz
    while lo * np.sqrt(2) <= min(t.band_hi_hz, 0.95 * sr / 2) + 1e-6:
        edges.append((lo, lo * np.sqrt(2)))
        lo *= np.sqrt(2)
    nfr = (len(x) - n) // hop + 1 if len(x) >= n else 0
    if not edges or nfr <= 0:
        return np.zeros((0, max(1, len(edges))), np.float32), n, hop
    masks = np.stack([(freqs >= a) & (freqs < b) for a, b in edges], axis=1).astype(np.float32)
    win = np.hanning(n).astype(np.float32)
    xs = x.astype(np.float32)
    out = np.empty((nfr, len(edges)), np.float32)
    chunk = 50_000
    for f0 in range(0, nfr, chunk):
        f1 = min(nfr, f0 + chunk)
        idx = (np.arange(f0, f1) * hop)[:, None] + np.arange(n)[None, :]
        out[f0:f1] = (np.abs(np.fft.rfft(xs[idx] * win, axis=1)) ** 2) @ masks
    return 10 * np.log10(np.maximum(out, 1e-20)), n, hop


def rise_db(x: np.ndarray, sr: int, t: ClickTunables = ClickTunables()) -> tuple[np.ndarray, int, int]:
    """Per STFT frame: how far the best bands rise over their own level on both sides."""
    db, n, hop = _band_db(x, sr, t)
    if db.shape[0] == 0:
        return np.zeros(0), n, hop
    per_ms = sr / 1000.0 / hop
    gl, gr = int(round(t.left_guard_ms * per_ms)), int(round(t.right_guard_ms * per_ms))
    w = int(round(t.side_ms * per_ms)) | 1
    m = maximum_filter1d(db, size=w, axis=0, mode="nearest")
    idx = np.arange(db.shape[0])
    last = db.shape[0] - 1
    around = np.maximum(m[np.clip(idx - gl - w // 2 - 1, 0, last)], m[np.clip(idx + gr + w // 2 + 1, 0, last)])
    k = min(t.top_bands, db.shape[1])
    return np.sort(db - around, axis=1)[:, -k:].mean(axis=1), n, hop


def click_findings(ch: Chapter, pauses, breath_spans, t: ClickTunables = ClickTunables(),
                   rise: tuple[np.ndarray, int, int] | None = None,
                   first_sound: int | None = None, last_sound: int | None = None) -> list[Finding]:
    """Clicks in the silence: inside pauses (word to word), and in the room tone before the first
    word and after the last one (`first_sound`, `last_sound`), clear of words and breaths.
    `rise`: rise_db(), if computed."""
    # (start, end, a word on the left, a word on the right, text)
    regions = [(a, b, True, True, "click in a pause") for a, b in pauses]
    if first_sound is not None and last_sound is not None:
        regions = ([(0, first_sound, False, True, "click before the first word")] + regions
                   + [(last_sound, len(ch.x), True, False, "click after the last word")])
    if not regions:
        return []
    rise, n, hop = rise if rise is not None else rise_db(ch.x, ch.sr, t)
    if rise.size == 0:
        return []
    sr = ch.sr
    near = int(round(t.one_per_ms * sr / 1000.0 / hop))
    peaks = np.flatnonzero((rise >= t.min_rise_db) & (rise == maximum_filter1d(rise, size=2 * near + 1, mode="nearest")))
    clear = int(t.clear_ms * sr / 1000)
    starts = np.array([r[0] for r in regions])
    spans = sorted(breath_spans)
    k2 = int(0.002 * sr)
    out: list[Finding] = []
    for f in peaks:
        s = int(f * hop + n // 2)
        i = int(np.searchsorted(starts, s, side="right")) - 1
        if i < 0:
            continue
        a, b, word_left, word_right, text = regions[i]
        if not (a + (clear if word_left else 0) <= s < b - (clear if word_right else 0)):
            continue
        if any(bs - clear <= s < be + clear for bs, be in spans):
            continue
        seg = ch.x[max(0, s - k2):s + k2]
        peak = 20 * np.log10(max(float(np.max(np.abs(seg))) if seg.size else 0.0, 1e-10))
        if peak < t.min_peak_dbfs:
            continue
        at = max(0, s - k2) + int(np.argmax(np.abs(seg)))
        out.append(Finding(
            file=ch.name, check="clicks", start_sample=at, end_sample=at + 1, start_time=ch.clock(at),
            end_time=ch.clock(at + 1), severity=CLICK_SEVERITY, problem=text,
            measures={"peak_dbfs": round(peak, 1), "rise_db": round(float(rise[f]), 1),
                      "ms_after_word": round((s - a) * 1000 / sr) if word_left else "",
                      "ms_before_word": round((b - s) * 1000 / sr) if word_right else ""},
        ))
    return out
