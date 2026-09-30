"""Possible chopped word where a generated clip ends — a local model's call. Numbers only.

`speech-truncation-detection-12M` asks one question of a 5 s window: was speech still active when the
buffer ended? It was evaluated (2026-08-13/14) on exactly one kind of point, and is used only there:
the end of a generated clip, where the rendered audio stops and digital black begins — the first
sample of a run of at least 50 ms of exact zeros after sound. The window ends on that sample (the
black is not included). Reported when the model is confident (score >= 0.979795 over the last
10 ms) AND the final 30 ms peak is at least -23.5 dBFS (a cut you can hear).

Evidence: on the evaluation title this finds the same 118 clip ends and lists the same 10 of them —
all 9 audited real chops and 1 clean ending — and on a second title 0 of 27 (it has no chop). fpab
had instead scored every pause, with the window ending where the level first fell below the pause
threshold: blind, 64 of those listings from two titles held no real chop (mostly a word running
straight into an inhale). The engine was not at fault: the numpy port matches the original code to
1e-7 on those windows; the points it was asked about were.

Severity 1, listed as just "word ends abruptly" (operator): on two further titles it listed 8 clip
ends, all very hard, abrupt word endings and none missing part of the word; only one would draw a
QC note, and for a click just before the ending. An abrupt ending is what it reliably finds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..chapter import Chapter
from ..findings import Finding

TRUNCATION_SEVERITY = 1

@dataclass(frozen=True)
class TruncationTunables:
    threshold: float = 0.979795
    decision_window_ms: float = 10.0
    peak_gate_dbfs: float = -23.5
    peak_window_ms: float = 30.0
    context_s: float = 5.0
    batch: int = 32
    black_ms: float = 50.0               # a clip end: sound followed by at least this much exact zero

    def as_dict(self) -> dict:
        return asdict(self)


def _peak_dbfs(x: np.ndarray, end: int, n: int) -> float:
    seg = x[max(0, end - n):end]
    return float(20 * np.log10(max(float(np.max(np.abs(seg))) if seg.size else 0.0, 1e-12)))


def _context(x: np.ndarray, end: int, n: int) -> np.ndarray:
    seg = x[max(0, end - n):end].astype(np.float32)
    return np.concatenate([np.zeros(n - len(seg), np.float32), seg]) if len(seg) < n else seg


def clip_ends(ch: Chapter, t: TruncationTunables = TruncationTunables()) -> list[int]:
    """Where a generated clip ends into digital black: the first sample of a run of exact zeros at
    least `black_ms` long that follows sound. The model was evaluated on exactly these points."""
    x = ch.x
    need = int(round(t.black_ms / 1000 * ch.sr))
    d = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [int(a) for a, b in zip(starts, ends) if b - a >= need and a > 0]


def score_phrase_ends(ch: Chapter, ends: list[int], model,
                      t: TruncationTunables = TruncationTunables()) -> list[dict]:
    """Score every phrase end; returns one record per end (all of them, flagged or not)."""
    n = int(round(t.context_s * ch.sr))
    w30 = int(round(t.peak_window_ms * ch.sr / 1000.0))
    records = []
    for b0 in range(0, len(ends), t.batch):
        batch = ends[b0:b0 + t.batch]
        pred = model.predict_from_audio(audio=[_context(ch.x, e, n) for e in batch], sampling_rate=ch.sr,
                                        decision_window_ms=t.decision_window_ms)
        s = pred.truncation_score
        scores = s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
        for e, s in zip(batch, scores):
            pk = _peak_dbfs(ch.x, e, w30)
            records.append({"phrase_end_sample": int(e), "time": ch.clock(e), "score": round(float(s), 5),
                            "peak_final_30ms_dbfs": round(pk, 1),
                            "flagged": bool(s >= t.threshold and pk >= t.peak_gate_dbfs)})
    return records


def truncation_findings(ch: Chapter, records: list[dict]) -> list[Finding]:
    out = []
    for r in records:
        if not r["flagged"]:
            continue
        e = r["phrase_end_sample"]
        out.append(Finding(
            file=ch.name, check="truncation", start_sample=e, end_sample=e,
            start_time=ch.clock(e), end_time=ch.clock(e), severity=TRUNCATION_SEVERITY,
            problem="word ends abruptly",
            measures={"model_score": r["score"], "peak_final_30ms_dbfs": r["peak_final_30ms_dbfs"]},
        ))
    return out
