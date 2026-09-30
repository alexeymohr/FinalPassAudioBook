"""Hum: a narrow tone that holds one frequency for seconds. Numbers only.

Long analysis windows (2 s, ~0.14 Hz bins after zero-padding) make a steady tone
a sharp spectral line while speech smears: voice harmonics move with pitch. A
line that stands well above its ±10 Hz neighbourhood is tracked from window to
window; it becomes a hum only if it stays within a fraction of a hertz for at
least MIN_PERSIST_S. A word held for a few hundred milliseconds cannot pass, and
neither can a voice harmonic gliding with intonation. The level at the start and
end shows a slow build-up.

Any steady hum is listed, whatever its level; there is no level floor. Only the
40 Hz lower limit stays: very low lines at these levels are inaudible.

Severity (operator, after auditioning every edge of one title's 12 hums): most
hums are low-level and hardly noticeable, 1; a strong hum (its loudest line at
least -55 dBFS) is 2; strong and starting or cutting off abruptly, 3. The two
hums the operator called strong measured -53.8 and -54.3 dBFS; the loudest of
the rest -55.5.

A tracked line must also be heard in the pauses. On one title's 12 chapters,
tracking alone found 8 lines; the operator auditioned all 8 and only one was hum
(part of a two-tone stack) — and only that one showed its own line in a pause.
So a tracked line is listed only if it is a steady line in a pause inside the
hum or within 2 s of it — unless there are no pauses around it at all, which a
hum loud enough to fill every pause causes. The price: a quieter hum under
narration with no pause >= 0.4 s nearby is not listed.

Harmonics: once a hum is found, every whole multiple of its frequency is
measured over the same windows and reported if it stands at least 6 dB over its
neighbourhood (median over the hum).

Heard in the pauses (PLAN §8). Speech hides lines in the voice's range. So every
hum is also described from its pauses (stretches >= 0.4 s at least 30 dB under
the narration): each steady line >= 12 dB over its ±10 Hz neighbourhood and
within 30 dB of the hum, present in at least half of them, is listed — lines
about as loud as the hum in the headline, whole multiples as harmonics, the rest
as "other steady lines". On the operator's confirmed two-tone example this lists
both tones, their harmonics, the sum and difference tones, and the cut. The event
text names only the tones, level and edges; harmonics and other lines are measures.

A cut that lands on the next word is named when a listed line falls >= 10 dB
within 0.2 s from its full level and sits >= 20 dB lower in the next pause. For
a hum known only from its pauses, that cut is looked for between its last pause
and the next pause that no longer carries the line (and the start likewise).

Hums speech always hides: a line at one frequency (±0.5 Hz) in >= 2 pauses
spread over >= 3 s (gaps <= 10 s), at least -70 dBFS (operator: quieter lines
are not worth a finding; without a floor, inaudible lines far below it were
listed). On one title's 12 chapters this found 11 hums; the operator auditioned
all 11 and all were real.

Leakage: with a very clean background, the analysis window's own sidelobes
beside a strong hum (±1-3 Hz, 30-45 dB down) look like lines; a line 25 dB or
more under a louder one within ±10 Hz is dropped as leakage.

Abrupt start or end: the 2 s windows overlap, so a hard cut looks like a slope
there. Instead the tone's own level is followed in 0.1 s steps through a narrow
band (±3 Hz); a change of 20 dB or more within 0.5 s at either end, from or to
the hum's full level (within 6 dB of its median), is reported ("cuts off
abruptly", "starts abruptly"). A fade does not qualify: its last 20 dB happen
well below full level. Heard blind on one title's 24 hum edges: 6 of the 8
claimed abrupt were, 2 of the other 16 were abrupt too (7 could not be judged).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from ..chapter import Chapter
from ..findings import Finding



@dataclass(frozen=True)
class HumTunables:
    window_s: float = 2.0
    hop_s: float = 0.5
    f_lo_hz: float = 40.0              # below this a tone at these levels is inaudible
    f_hi_hz: float = 1000.0
    neighbourhood_hz: float = 10.0
    prominence_db: float = 10.0
    min_level_dbfs: float | None = None   # no level floor: any steady hum is listed
    freq_tolerance_hz: float = 0.6     # window to window
    max_spread_hz: float = 1.0         # over the whole hum: "fixed frequency"
    max_missed: int = 2
    min_persist_s: float = 3.0
    rise_db: float = 6.0
    harmonic_prominence_db: float = 6.0
    harmonic_tolerance_hz: float = 0.02   # per multiple: how far n * f0 may drift from the measured f0
    sidelobe_db: float = 25.0          # a line this far under a louder one within ±10 Hz is its leakage
    strong_dbfs: float = -55.0         # a hum this loud (loudest line) is at least severity 2
    edge_band_hz: float = 3.0
    edge_block_s: float = 0.1
    edge_within_s: float = 0.5
    edge_step_db: float = 20.0
    edge_full_level_db: float = 6.0    # the step must leave from (or arrive at) the hum's full level
    # --- the hum as heard in the pauses (PLAN §8) ---
    pause_below_narration_db: float = 30.0
    pause_min_s: float = 0.4
    pause_trim_s: float = 0.05         # keep word tails and breaths out of the pause spectrum
    pause_line_prominence_db: float = 12.0
    pause_line_within_db: float = 30.0 # a pause line counts toward a hum if within this of its level
    pause_line_tol_hz: float = 1.0
    headline_within_db: float = 6.0    # a line this close to the hum's level is named in the headline
    word_cut_fall_db: float = 10.0     # a cut that lands on a word: this fall within word_cut_within_s,
    word_cut_within_s: float = 0.2
    word_cut_after_db: float = 20.0    # and in the next pause the line sits this far under its level
    word_cut_search_s: float = 10.0
    pause_hums: bool = True            # find hums that speech always hides (§8 C)
    pause_hum_tol_hz: float = 0.5
    pause_hum_min_pauses: int = 2
    pause_hum_min_span_s: float = 3.0
    pause_hum_max_gap_s: float = 10.0
    pause_hum_min_dbfs: float = -70.0  # operator: lines quieter than this are not worth a hum finding
    confirm_in_pauses: bool = True     # a tracked hum is listed only if its line is heard in its pauses
    confirm_margin_s: float = 2.0      # pauses this close to the hum also count
    join_gap_s: float = 3.0            # same-frequency pieces this close are one hum (speech breaks tracks)
    span_within_db: float = 10.0       # a hum's start/end: where its own tone comes within this of full level

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Track:
    first: int
    last: int
    freqs: list[float]
    levels: list[float]
    proms: list[float]
    missed: int = 0


BLOCK_FRAMES = 128


def _spectra(y: np.ndarray, fs: float, t: HumTunables, rows: slice | None = None):
    """Per 2 s window: line level (dBFS) and prominence over the ±10 Hz neighbourhood, 40 Hz-1 kHz.

    Yields (first window index, frequencies, level rows, prominence rows) in blocks."""
    win = int(round(t.window_s * fs))
    hop = int(round(t.hop_s * fs))
    if len(y) < win:
        return
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    w = np.hanning(win)
    to_rms = 2.0 / w.sum() / np.sqrt(2.0)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    # The neighbourhood stays inside 40 Hz-1 kHz (rumble below the audible limit must not hide a line
    # above it) and is mirrored at the edges: repeating the edge bin would hide a tone sitting on it.
    df = f[1]
    band = np.flatnonzero((f >= t.f_lo_hz - 2 * df) & (f <= t.f_hi_hz + 2 * df))
    size = 2 * int(round(t.neighbourhood_hz / df)) + 1
    view = sliding_window_view(y, win)[::hop]
    start, stop = (0, len(view)) if rows is None else (max(0, rows.start), min(len(view), rows.stop))
    for b0 in range(start, stop, BLOCK_FRAMES):
        block = view[b0:min(stop, b0 + BLOCK_FRAMES)] * w
        mag = np.abs(np.fft.rfft(block, nfft, axis=1))[:, band]
        db = 20 * np.log10(np.maximum(mag * to_rms, 1e-12))
        yield b0, f[band], db, db - median_filter(db, size=(1, size), mode="mirror")


def _peaks(y: np.ndarray, fs: float, t: HumTunables):
    """Yield, per analysis window, the sharp lines: (freq, tone level dBFS, prominence dB)."""
    for b0, f, db, prom in _spectra(y, fs, t):
        df = f[1] - f[0]
        is_peak = np.zeros_like(db, dtype=bool)
        is_peak[:, 1:-1] = (db[:, 1:-1] >= db[:, :-2]) & (db[:, 1:-1] > db[:, 2:])
        # one bin of slack: a tone exactly at a band edge can peak in the bin just outside it
        keep = is_peak & (prom >= t.prominence_db) & (f >= t.f_lo_hz - df) & (f <= t.f_hi_hz + df)
        if t.min_level_dbfs is not None:
            keep &= db >= t.min_level_dbfs
        for r in range(len(db)):
            lines = []
            for k in np.flatnonzero(keep[r]):
                a, b, c = db[r, k - 1], db[r, k], db[r, k + 1]
                shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
                lines.append((float(f[k] + shift * df), float(b), float(prom[r, k])))
            yield b0 + r, lines


def _tracks(y: np.ndarray, fs: float, t: HumTunables) -> list[_Track]:
    open_: list[_Track] = []
    done: list[_Track] = []
    for i, lines in _peaks(y, fs, t):
        used: set[int] = set()
        for fr, lv, pr in sorted(lines, key=lambda z: -z[1]):
            best = None
            for j, tr in enumerate(open_):
                if j not in used and abs(fr - tr.freqs[-1]) <= t.freq_tolerance_hz:
                    if best is None or abs(fr - tr.freqs[-1]) < abs(fr - open_[best].freqs[-1]):
                        best = j
            if best is None:
                open_.append(_Track(i, i, [fr], [lv], [pr]))
                used.add(len(open_) - 1)
            else:
                tr = open_[best]
                tr.last, tr.missed = i, 0
                tr.freqs.append(fr), tr.levels.append(lv), tr.proms.append(pr)
                used.add(best)
        still = []
        for j, tr in enumerate(open_):
            if j not in used:
                tr.missed += 1
            (done if tr.missed > t.max_missed else still).append(tr)
        open_ = still
    done += open_
    # Only pieces that are hums on their own are joined: fragments must not add up to one.
    pieces = [tr for tr in done if (tr.last - tr.first) * t.hop_s >= t.min_persist_s]
    return [tr for tr in _join(pieces, t) if max(tr.freqs) - min(tr.freqs) <= t.max_spread_hz]


def _join(tracks: list[_Track], t: HumTunables) -> list[_Track]:
    """Join pieces of one line that speech broke apart (same frequency, gap <= join_gap_s)."""
    gap = int(round(t.join_gap_s / t.hop_s))
    out: list[_Track] = []
    for tr in sorted(tracks, key=lambda z: (round(float(np.median(z.freqs)) / t.freq_tolerance_hz), z.first)):
        prev = next((o for o in reversed(out)
                     if abs(float(np.median(o.freqs)) - float(np.median(tr.freqs))) <= t.freq_tolerance_hz
                     and 0 <= tr.first - o.last <= gap), None)
        if prev is None:
            out.append(tr)
        else:
            prev.last = tr.last
            prev.freqs += tr.freqs
            prev.levels += tr.levels
            prev.proms += tr.proms
    return out


def _drop_sidelobes(tracks: list[_Track], t: HumTunables) -> list[_Track]:
    """Remove lines that are only the analysis window's leakage beside a much louder line."""
    def masked(tr: _Track) -> bool:
        f, lv = float(np.median(tr.freqs)), float(np.median(tr.levels))
        return any(o is not tr and abs(float(np.median(o.freqs)) - f) <= t.neighbourhood_hz
                   and min(tr.last, o.last) >= max(tr.first, o.first)
                   and float(np.median(o.levels)) - lv >= t.sidelobe_db for o in tracks)
    return [tr for tr in tracks if not masked(tr)]


def _group(tracks: list[_Track]) -> list[tuple[_Track, list[_Track]]]:
    groups: list[tuple[_Track, list[_Track]]] = []
    for tr in sorted(tracks, key=lambda z: float(np.median(z.freqs))):
        f = float(np.median(tr.freqs))
        for base, harmonics in groups:
            f0 = float(np.median(base.freqs))
            n = round(f / f0)
            overlap = min(tr.last, base.last) - max(tr.first, base.first)
            if n >= 2 and abs(f - n * f0) <= 0.01 * f and overlap > 0:
                harmonics.append(tr)
                break
        else:
            groups.append((tr, []))
    return groups


def _measured_harmonics(y: np.ndarray, fs: float, t: HumTunables, base: _Track, f0: float) -> list[int]:
    """Whole multiples of f0 standing >= harmonic_prominence_db over their neighbourhood, over the hum."""
    n_max = int(t.f_hi_hz // f0)
    if n_max < 2:
        return []
    proms: list[np.ndarray] = []
    freqs = None
    for _, f, _db, prom in _spectra(y, fs, t, slice(base.first, base.last + 1)):
        freqs = f
        proms.append(prom)
    if freqs is None:
        return []
    prom = np.concatenate(proms)
    found = []
    df = freqs[1] - freqs[0]
    for n in range(2, n_max + 1):
        tol = max(1.5 * df, n * t.harmonic_tolerance_hz)
        cols = np.flatnonzero(np.abs(freqs - n * f0) <= tol)
        if cols.size and float(np.median(prom[:, cols].max(axis=1))) >= t.harmonic_prominence_db:
            found.append(int(round(n * f0)))
    return found


def _abrupt_edges(y: np.ndarray, fs: float, t: HumTunables, f0: float,
                  start_s: float, end_s: float) -> tuple[float, float]:
    """Largest rise at the hum's start and largest drop at its end, dB within `edge_within_s`."""
    lo_hz, hi_hz = f0 - t.edge_band_hz, f0 + t.edge_band_hz
    if lo_hz <= 0 or hi_hz >= fs / 2:
        return 0.0, 0.0
    a = max(0, int((start_s - t.window_s) * fs))
    b = min(len(y), int((end_s + t.window_s) * fs))
    z = sosfiltfilt(butter(2, [lo_hz, hi_hz], btype="band", fs=fs, output="sos"), y[a:b])
    n = max(1, int(round(t.edge_block_s * fs)))
    k = len(z) // n
    if k == 0:
        return 0.0, 0.0
    env = 10 * np.log10(np.maximum(np.mean(z[:k * n].reshape(k, n) ** 2, axis=1), 1e-20))
    s = max(1, int(round(t.edge_within_s / t.edge_block_s)))
    if k <= s:
        return 0.0, 0.0
    t_blk = a / fs + (np.arange(k) + 0.5) * t.edge_block_s
    inside = (t_blk >= start_s + t.window_s / 2) & (t_blk <= end_s - t.window_s / 2)
    full = float(np.median(env[inside])) if inside.any() else float(np.max(env))
    change = env[s:] - env[:-s]                     # rise from block i to block i + s
    at = t_blk[:-s]
    near_start = (np.abs(at - start_s) <= t.window_s) & (env[s:] >= full - t.edge_full_level_db)
    near_end = (np.abs(at - (end_s - t.window_s / 2)) <= t.window_s) & (env[:-s] >= full - t.edge_full_level_db)
    rise = float(change[near_start].max()) if near_start.any() else 0.0
    drop = float(-change[near_end].min()) if near_end.any() else 0.0
    return rise, drop


def _pauses(ch: Chapter, t: HumTunables) -> list[tuple[float, float]]:
    """Quiet stretches (seconds): full-band 50 ms level at least 30 dB under the narration."""
    narration = ch.narration_dbfs
    limit = narration - t.pause_below_narration_db if np.isfinite(narration) else -50.0
    win = max(1, int(round(50.0 / 1.0)))                  # 50 one-millisecond bins
    quiet = ch.envelope_db(win) <= limit
    d = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) + win - 1
    ms = ch.bin_samples / ch.sr
    return [(a * ms, min(b * ms, ch.duration_s)) for a, b in zip(starts, ends) if (b - a) * ms >= t.pause_min_s]


def _pause_lines(y: np.ndarray, fs: float, a_s: float, b_s: float, t: HumTunables) -> list[tuple[float, float]]:
    """Steady lines in one pause: (frequency Hz, tone level dBFS RMS), 40 Hz-1 kHz, >= 12 dB proud."""
    x = y[int((a_s + t.pause_trim_s) * fs):int((b_s - t.pause_trim_s) * fs)]
    if len(x) < int(0.2 * fs):
        return []
    w = np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(8 * len(x))))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = (f >= t.f_lo_hz - t.neighbourhood_hz) & (f <= t.f_hi_hz + t.neighbourhood_hz)
    f = f[keep]
    m = 20 * np.log10(np.abs(np.fft.rfft(x * w, nfft))[keep] * 2.0 / w.sum() / np.sqrt(2.0) + 1e-12)
    prom = m - median_filter(m, size=2 * int(round(t.neighbourhood_hz / (f[1] - f[0]))) + 1, mode="nearest")
    ok = np.zeros_like(m, dtype=bool)
    ok[1:-1] = (m[1:-1] >= m[:-2]) & (m[1:-1] > m[2:])
    df = f[1] - f[0]                        # one bin of slack at the band edges, as for tracking
    ok &= (prom >= t.pause_line_prominence_db) & (f >= t.f_lo_hz - df) & (f <= t.f_hi_hz + df)
    lines = [(float(f[k]), float(m[k])) for k in np.flatnonzero(ok)]
    reach = 6.0 * fs / len(x)                 # Hann main lobe and first sidelobes of this short pause
    return [(g, lv) for g, lv in lines        # drop a louder line's leakage, as for the tracked lines
            if not any(abs(g - h) <= reach and hl - lv >= t.sidelobe_db for h, hl in lines)]


def _cluster(per_pause: list[list[tuple[float, float]]], tol: float) -> list[tuple[float, float, list[int]]]:
    """Group lines across pauses by frequency: (median Hz, median dBFS, pause indices)."""
    pts = sorted((f, lv, i) for i, lines in enumerate(per_pause) for f, lv in lines)
    groups: list[list[tuple[float, float, int]]] = []
    for p in pts:
        if groups and p[0] - groups[-1][-1][0] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [(float(np.median([g[0] for g in grp])), float(np.median([g[1] for g in grp])),
             sorted({g[2] for g in grp})) for grp in groups]


def _is_multiple(f: float, of: float) -> bool:
    n = round(f / of)
    return n >= 2 and abs(f - n * of) <= max(1.0, 0.01 * f)


def _level_at(y: np.ndarray, fs: float, a_s: float, b_s: float, f0: float, t: HumTunables) -> float | None:
    """Tone level (dBFS RMS) at f0 within one pause: the strongest bin within ±tol."""
    x = y[int((a_s + t.pause_trim_s) * fs):int((b_s - t.pause_trim_s) * fs)]
    if len(x) < int(0.2 * fs):
        return None
    w = np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(8 * len(x))))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    near = np.abs(f - f0) <= t.pause_line_tol_hz
    m = np.abs(np.fft.rfft(x * w, nfft))[near] * 2.0 / w.sum() / np.sqrt(2.0)
    return float(20 * np.log10(m.max() + 1e-12)) if m.size else None


def _word_cut(y: np.ndarray, fs: float, t: HumTunables, f0: float, full_db: float,
              start_s: float, end_s: float, pauses: list[tuple[float, float]]) -> tuple[bool, bool]:
    """A start or end that lands on a word: a fast 10 dB step from full level, confirmed in the
    neighbouring pause (the line 20 dB under its level there)."""
    lo_hz, hi_hz = f0 - t.edge_band_hz, f0 + t.edge_band_hz
    if lo_hz <= 0 or hi_hz >= fs / 2:
        return False, False
    a = max(0, int((start_s - t.window_s) * fs))
    b = min(len(y), int((end_s + t.window_s) * fs))
    z = sosfiltfilt(butter(2, [lo_hz, hi_hz], btype="band", fs=fs, output="sos"), y[a:b])
    n = max(1, int(round(0.05 * fs)))
    k = len(z) // n
    s = max(1, int(round(t.word_cut_within_s / 0.05)))
    if k <= s:
        return False, False
    env = 10 * np.log10(np.maximum(np.mean(z[:k * n].reshape(k, n) ** 2, axis=1), 1e-20))
    at = a / fs + (np.arange(k - s) + 0.5) * 0.05
    change = env[s:] - env[:-s]

    def gone_in(pause: tuple[float, float] | None) -> bool:
        lv = _level_at(y, fs, *pause, f0, t) if pause else None
        return lv is not None and lv <= full_db - t.word_cut_after_db

    ends = (np.abs(at - (end_s - t.window_s / 2)) <= t.window_s) & (env[:-s] >= full_db - t.edge_full_level_db) \
        & (change <= -t.word_cut_fall_db)
    starts = (np.abs(at - start_s) <= t.window_s) & (env[s:] >= full_db - t.edge_full_level_db) \
        & (change >= t.word_cut_fall_db)
    cut_end = cut_start = False
    if ends.any():
        when = float(at[np.flatnonzero(ends)[0]])
        after = [p for p in pauses if when < p[0] <= when + t.word_cut_search_s]
        cut_end = gone_in(after[0] if after else None)
    if starts.any():
        when = float(at[np.flatnonzero(starts)[-1]])
        before = [p for p in pauses if when - t.word_cut_search_s <= p[1] < when]
        cut_start = gone_in(before[-1] if before else None)
    return cut_start, cut_end


def _describe(y: np.ndarray, fs: float, t: HumTunables, f0: float, level: float, start_s: float, end_s: float,
              pauses: list[tuple[float, float]], harmonics: set[int]) -> dict:
    """Every steady line heard in the hum's pauses, sorted into headline lines, harmonics and others."""
    inside = [p for p in pauses if p[0] >= start_s and p[1] <= end_s]
    per_pause = [[(f, lv) for f, lv in _pause_lines(y, fs, a, b, t) if lv >= level - t.pause_line_within_db]
                 for a, b in inside]
    need = 1 if len(inside) <= 2 else int(np.ceil(len(inside) / 2))
    found = [(f, lv) for f, lv, idx in _cluster(per_pause, t.pause_line_tol_hz) if len(idx) >= max(1, need)]
    base_lv = next((lv for f, lv in found if abs(f - f0) <= t.pause_line_tol_hz), level)
    heads, harm, other = [(f0, base_lv)], set(harmonics), []
    for f, lv in sorted(found, key=lambda z: -z[1]):
        if any(abs(f - h) <= t.pause_line_tol_hz for h, _ in heads):
            continue
        if any(_is_multiple(f, h) for h, _ in heads):
            harm.add(int(round(f)))
        elif lv >= base_lv - t.headline_within_db:
            heads.append((f, lv))
        else:
            other.append((f, lv))
    harm = {h for h in harm if not any(abs(h - f) <= t.pause_line_tol_hz for f, _ in heads)}
    merged: list[int] = []
    for h in sorted(harm):                    # the same harmonic seen by tracking and by measurement
        if not merged or h - merged[-1] > max(1, round(0.005 * h)):
            merged.append(h)
    harm = set(merged)
    return {"heads": heads, "harmonics": sorted(harm), "other": sorted(other), "pauses": len(inside),
            "lines": sorted(found)}


def _cut_between_pauses(y: np.ndarray, fs: float, t: HumTunables, f0: float, full_db: float,
                        a_s: float, b_s: float, falling: bool) -> float | None:
    """For a hum known only from its pauses: its line is present at one end of the stretch [a_s, b_s]
    between two pauses and gone (>= 20 dB under its level) in the pause at the other end. If it
    changes by >= 10 dB within 0.2 s from full level somewhere in between, that is an abrupt cut
    (or start); returns its time."""
    lo_hz, hi_hz = f0 - t.edge_band_hz, f0 + t.edge_band_hz
    if lo_hz <= 0 or hi_hz >= fs / 2 or b_s - a_s < 0.3:
        return None
    a, b = max(0, int((a_s - 0.3) * fs)), min(len(y), int((b_s + 0.3) * fs))
    z = sosfiltfilt(butter(2, [lo_hz, hi_hz], btype="band", fs=fs, output="sos"), y[a:b])
    n = max(1, int(round(0.05 * fs)))
    k = len(z) // n
    s = max(1, int(round(t.word_cut_within_s / 0.05)))
    if k <= s:
        return None
    env = 10 * np.log10(np.maximum(np.mean(z[:k * n].reshape(k, n) ** 2, axis=1), 1e-20))
    at = a / fs + (np.arange(k - s) + 0.5) * 0.05
    inside = (at >= a_s - 0.1) & (at <= b_s)
    if falling:
        hit = np.flatnonzero(inside & (env[:-s] >= full_db - t.edge_full_level_db)
                             & (env[:-s] - env[s:] >= t.word_cut_fall_db))
        return float(at[hit[0]] + t.word_cut_within_s) if hit.size else None
    hit = np.flatnonzero(inside & (env[s:] >= full_db - t.edge_full_level_db) & (env[s:] - env[:-s] >= t.word_cut_fall_db))
    return float(at[hit[-1]]) if hit.size else None


def _pause_found_edges(y: np.ndarray, fs: float, t: HumTunables, heads: list[tuple[float, float]],
                       carrying: list[tuple[float, float]], all_pauses: list[tuple[float, float]]
                       ) -> tuple[float | None, float | None]:
    """Abrupt start / cut of a pause-found hum, looked for between its outer carrying pauses and the
    neighbouring pauses that do not carry it."""
    first, last = carrying[0], carrying[-1]
    prev = [p for p in all_pauses if p[1] <= first[0] and first[0] - p[1] <= t.word_cut_search_s]
    nxt = [p for p in all_pauses if p[0] >= last[1] and p[0] - last[1] <= t.word_cut_search_s]
    start = end = None
    for f, lv in sorted(heads, key=lambda z: -z[1]):
        if end is None and nxt:
            gone = _level_at(y, fs, *nxt[0], f, t)
            if gone is not None and gone <= lv - t.word_cut_after_db:
                end = _cut_between_pauses(y, fs, t, f, lv, last[1], nxt[0][0], falling=True)
        if start is None and prev:
            gone = _level_at(y, fs, *prev[-1], f, t)
            if gone is not None and gone <= lv - t.word_cut_after_db:
                start = _cut_between_pauses(y, fs, t, f, lv, prev[-1][1], first[0], falling=False)
    return start, end


def _finding(ch: Chapter, t: HumTunables, y: np.ndarray, fs: float, f0: float, level: float, start_s: float,
             end_s: float, pauses: list[tuple[float, float]], shape: str | None, harmonics: set[int],
             measures: dict, heard_in_pauses_only: bool = False,
             all_pauses: list[tuple[float, float]] | None = None) -> Finding:
    d = _describe(y, fs, t, f0, level, start_s, end_s, pauses, harmonics)
    heads = d["heads"]
    rise = drop = 0.0
    cut_start = cut_end = False
    if heard_in_pauses_only and all_pauses is not None and pauses:
        # its span starts and ends at pauses, where speech starts: look for a cut only between those
        # pauses and the neighbouring ones that no longer carry the line
        t_start, t_end = _pause_found_edges(y, fs, t, heads, pauses, all_pauses)
        if t_start is not None:
            cut_start, start_s = True, min(start_s, t_start)
        if t_end is not None:
            cut_end, end_s = True, max(end_s, t_end)
    if not heard_in_pauses_only:
        rise, drop = _abrupt_edges(y, fs, t, f0, start_s, end_s)
        cut_start, cut_end = rise >= t.edge_step_db, drop >= t.edge_step_db
        for f, lv in sorted(heads, key=lambda z: -z[1]):
            if cut_start and cut_end:
                break
            ws, we = _word_cut(y, fs, t, f, lv, start_s, end_s, pauses)
            cut_start, cut_end = cut_start or ws, cut_end or we
    # The event text stays short (it sets a spreadsheet column's width): tone(s), level, edges.
    # Harmonics, the other lines heard in the pauses and how it was found are measures.
    text = "hum " + " + ".join(f"{f:.1f}" for f, _ in heads) + " Hz"
    loudest = max(lv for _, lv in heads)
    severity = hum_severity(loudest, cut_start or cut_end, t)
    parts = [text, shape if shape and len(heads) == 1 else f"{loudest:.0f} dBFS"]
    parts += [w for w, on in (("starts abruptly", cut_start), ("cuts off abruptly", cut_end)) if on]
    s, e = int(start_s * ch.sr), int(end_s * ch.sr)
    return Finding(
        file=ch.name, check="hum", start_sample=s, end_sample=e, start_time=ch.clock(s), end_time=ch.clock(e),
        severity=severity, problem=", ".join(parts),
        measures={**measures, "duration_s": round(end_s - start_s, 1),
                  "harmonics_hz": ",".join(str(h) for h in d["harmonics"]),
                  "other_lines_hz": ",".join(str(round(f)) for f, _ in d["other"]),
                  "heard": "in the pauses only (speech covers it)" if heard_in_pauses_only else "throughout",
                  "pause_lines_hz": ",".join(f"{f:.1f}" for f, _ in d["lines"]),
                  "pause_lines_dbfs": ",".join(f"{lv:.1f}" for _, lv in d["lines"]),
                  "pauses_measured": d["pauses"], "start_rise_db": round(rise, 1), "end_drop_db": round(drop, 1),
                  "starts_abruptly": "yes" if cut_start else "no", "cuts_off_abruptly": "yes" if cut_end else "no"},
    )


def hum_severity(loudest_dbfs: float, abrupt: bool, t: HumTunables = HumTunables()) -> int:
    """1 for a low-level hum; 2 when strong; 3 when strong and it starts or cuts off abruptly."""
    if loudest_dbfs < t.strong_dbfs:
        return 1
    return 3 if abrupt else 2


def _hidden_hums(y: np.ndarray, fs: float, t: HumTunables, pauses: list[tuple[float, float]],
                 known: list[tuple[float, float, list[float]]]
                 ) -> list[tuple[float, float, float, float, list[tuple[float, float]]]]:
    """Lines that recur at one frequency across pauses spread over >= 3 s, not already part of a
    found hum: (Hz, dBFS, start s, end s, the pauses that carry it)."""
    per_pause = [[(f, lv) for f, lv in _pause_lines(y, fs, a, b, t) if lv >= t.pause_hum_min_dbfs]
                 for a, b in pauses]
    out = []
    for f, _lv, idx in _cluster(per_pause, t.pause_hum_tol_hz):
        runs: list[list[int]] = []
        for i in idx:
            if runs and pauses[i][0] - pauses[runs[-1][-1]][1] <= t.pause_hum_max_gap_s:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            a_s, b_s = pauses[run[0]][0], pauses[run[-1]][1]
            if len(run) < t.pause_hum_min_pauses or b_s - a_s < t.pause_hum_min_span_s:
                continue
            same = lambda g: abs(f - g) <= t.pause_line_tol_hz or _is_multiple(f, g)  # noqa: E731
            # A known hum's own tone (or a multiple) overlapping it in time is that hum: a pause run
            # covers whole pauses, so it can run past the hum's end. Its other listed lines only
            # when the run lies within the hum.
            if any((_overlaps(a_s, b_s, ks, ke, t.window_s) and same(lines[0]))
                   or (_within(a_s, b_s, ks, ke, t.window_s) and any(same(g) for g in lines))
                   for ks, ke, lines in known):
                continue
            near = [lv for i in run for g, lv in per_pause[i] if abs(g - f) <= 2 * t.pause_hum_tol_hz]
            lv = float(np.median(near)) if near else t.pause_hum_min_dbfs
            out.append((f, lv, a_s, b_s, [pauses[i] for i in run]))
    return out


def _refine_span(y: np.ndarray, fs: float, t: HumTunables, f0: float, start_s: float,
                 end_s: float) -> tuple[float, float]:
    """Start and end from the tone's own level (±3 Hz, 0.1 s steps) rather than the 2 s windows,
    which place them up to a window early or late. Searched only within one window of each."""
    lo_hz, hi_hz = f0 - t.edge_band_hz, f0 + t.edge_band_hz
    if lo_hz <= 0 or hi_hz >= fs / 2:
        return start_s, end_s
    a = max(0, int((start_s - t.window_s) * fs))
    b = min(len(y), int((end_s + t.window_s) * fs))
    z = sosfiltfilt(butter(2, [lo_hz, hi_hz], btype="band", fs=fs, output="sos"), y[a:b])
    n = max(1, int(round(t.edge_block_s * fs)))
    k = len(z) // n
    if k < 3:
        return start_s, end_s
    env = 10 * np.log10(np.maximum(np.mean(z[:k * n].reshape(k, n) ** 2, axis=1), 1e-20))
    tb = a / fs + (np.arange(k) + 0.5) * t.edge_block_s
    inside = (tb >= start_s + t.window_s) & (tb <= end_s - t.window_s)
    on = env >= (float(np.median(env[inside])) if inside.any() else float(env.max())) - t.span_within_db
    first = np.flatnonzero(on & (tb >= start_s) & (tb <= start_s + t.window_s))
    last = np.flatnonzero(on & (tb >= end_s - t.window_s) & (tb <= end_s))
    s = float(tb[first[0]] - t.edge_block_s / 2) if first.size else start_s
    e = float(tb[last[-1]] + t.edge_block_s / 2) if last.size else end_s
    return (s, e) if e > s else (start_s, end_s)


def _within(start_s: float, end_s: float, ks: float, ke: float, slack: float) -> bool:
    return ks - slack <= start_s and end_s <= ke + slack


def _overlaps(start_s: float, end_s: float, ks: float, ke: float, slack: float) -> bool:
    """A pause run's span covers whole pauses, so it can run past the hum it belongs to."""
    return start_s < ke + slack and end_s > ks - slack


def _heard_in_pauses(y: np.ndarray, fs: float, t: HumTunables, f0: float, start_s: float, end_s: float,
                     pauses: list[tuple[float, float]]) -> bool:
    """Is the tracked line itself a steady line in at least one pause in or next to the hum?"""
    near = [p for p in pauses if p[1] > start_s - t.confirm_margin_s and p[0] < end_s + t.confirm_margin_s]
    return any(abs(f - f0) <= t.pause_line_tol_hz for a, b in near for f, _ in _pause_lines(y, fs, a, b, t))


def hum_findings(ch: Chapter, t: HumTunables = HumTunables()) -> list[Finding]:
    y, fs = ch.low
    pauses = _pauses(ch, t)
    narration = ch.narration_dbfs
    pause_limit = narration - t.pause_below_narration_db if np.isfinite(narration) else -50.0
    out: list[Finding] = []
    known: list[tuple[float, float, list[float]]] = []
    for base, harmonics in _group(_drop_sidelobes(_tracks(y, fs, t), t)):
        f0 = float(np.median(base.freqs))
        start_s = max(0.0, base.first * t.hop_s)
        end_s = min(ch.duration_s, base.last * t.hop_s + t.window_s)
        start_s, end_s = _refine_span(y, fs, t, f0, start_s, end_s)
        if any(_within(start_s, end_s, ks, ke, t.window_s) and any(abs(f0 - g) <= t.pause_line_tol_hz for g in lines)
               for ks, ke, lines in known):
            continue                  # already named as part of another hum heard over the same stretch
        lv0, lv1 = float(np.median(base.levels[:3])), float(np.median(base.levels[-3:]))
        level = float(np.median(base.levels))
        # Confirm in the pauses — unless there are none around it: a hum loud enough to fill every
        # pause (above the pause limit) leaves nothing to confirm it in.
        near = [p for p in pauses if p[1] > start_s - t.confirm_margin_s and p[0] < end_s + t.confirm_margin_s]
        if t.confirm_in_pauses and (near or level < pause_limit) \
                and not _heard_in_pauses(y, fs, t, f0, start_s, end_s, pauses):
            continue                  # steady only inside the speech windows: not a hum (operator's audition)
        shape = (f"building {lv0:.0f} → {lv1:.0f} dBFS" if lv1 - lv0 >= t.rise_db else f"{level:.0f} dBFS")
        harm = ({int(round(float(np.median(h.freqs)))) for h in harmonics}
                | set(_measured_harmonics(y, fs, t, base, f0)))
        f = _finding(ch, t, y, fs, f0, level, start_s, end_s, pauses, shape, harm,
                     {"frequency_hz": round(f0, 2), "level_start_dbfs": round(lv0, 1),
                      "level_end_dbfs": round(lv1, 1), "level_max_dbfs": round(float(max(base.levels)), 1),
                      "prominence_db": round(float(np.median(base.proms)), 1), "found_by": "tracking"})
        out.append(f)
        lines = [f0] + [float(v) for v in f.measures["pause_lines_hz"].split(",") if v]
        known.append((start_s, end_s, lines))
    if t.pause_hums:
        for f0, level, start_s, end_s, carrying in _hidden_hums(y, fs, t, pauses, known):
            out.append(_finding(ch, t, y, fs, f0, level, start_s, end_s, carrying, None, set(),
                                {"frequency_hz": round(f0, 2), "level_max_dbfs": round(level, 1),
                                 "pauses_with_line": len(carrying), "found_by": "pauses"},
                                heard_in_pauses_only=True, all_pauses=pauses))
            known.append((start_s, end_s, [f0]))
    return sorted(out, key=lambda f: f.start_sample)
