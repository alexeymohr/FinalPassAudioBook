"""Possible chopped word at the end of a phrase — a local model's call. Numbers only.

Every phrase end (the start of each pause, and the chapter's last sound) is
scored by `speech-truncation-detection-12M` on the 5 s that end exactly there.
A phrase end is reported only when the model is confident AND the final 30 ms
is loud enough for a cut to be audible — the combination that, on the one
title where it was checked by ear, flagged 10 phrase ends of which 9 were
real, including every known chopped word (2026-08-13 evaluation).

It is a triage signal, not a verdict: validated on one title, on clip ends into
digital black; phrase ends into room tone and recall below the threshold have
not been measured.

Severity 3 when flagged (operator): if real, part of the word is missing, which
cannot be repaired in the mix. The model's gate is left exactly as evaluated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..chapter import Chapter
from ..findings import Finding

TRUNCATION_SEVERITY = 3

@dataclass(frozen=True)
class TruncationTunables:
    threshold: float = 0.979795
    decision_window_ms: float = 10.0
    peak_gate_dbfs: float = -23.5
    peak_window_ms: float = 30.0
    context_s: float = 5.0
    batch: int = 32

    def as_dict(self) -> dict:
        return asdict(self)


def _peak_dbfs(x: np.ndarray, end: int, n: int) -> float:
    seg = x[max(0, end - n):end]
    return float(20 * np.log10(max(float(np.max(np.abs(seg))) if seg.size else 0.0, 1e-12)))


def _context(x: np.ndarray, end: int, n: int) -> np.ndarray:
    seg = x[max(0, end - n):end].astype(np.float32)
    return np.concatenate([np.zeros(n - len(seg), np.float32), seg]) if len(seg) < n else seg


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
            problem="possible chopped word at the end of this phrase — audition",
            measures={"model_score": r["score"], "peak_final_30ms_dbfs": r["peak_final_30ms_dbfs"]},
        ))
    return out
