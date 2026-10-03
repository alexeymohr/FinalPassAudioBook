"""What a run reports: findings, pauses, and the run record. Numbers only."""
from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

SCHEMA_VERSION = 5
SEVERITIES = (3, 2, 1)   # 3 worst, 1 lowest; nothing here is a rejection


def ladder(value: float, cuts: Sequence[float]) -> int:
    """Severity from a rising measure: at or above cuts[0] -> 1, cuts[1] -> 2, cuts[2] -> 3."""
    return max(1, sum(value >= c for c in cuts))


class Finding(BaseModel):
    file: str
    check: str
    start_sample: int
    end_sample: int
    start_time: str
    end_time: str
    problem: str = Field(description="plain language; never overstates what was measured")
    severity: int = Field(ge=0, le=3, description="3 worst, 1 lowest; 0 = informational (listed, not a problem)")
    measures: dict[str, float | int | str] = {}


class Pause(BaseModel):
    start_sample: int
    end_sample: int
    start_time: str
    duration_ms: int
    kind: str = Field(description="head, tail, or internal")
    guess: str = Field(description="what the rule set suggests this pause is — a guess")


class FileResult(BaseModel):
    file: str
    path: str
    sample_rate: int
    duration_seconds: float
    audio_format: str = ""          # e.g. "WAV (Microsoft), Signed 24 bit PCM"
    channels: int = 0
    narration_dbfs: float | None
    noise_floor_dbfs: float | None
    findings: list[Finding]
    informational: list[Finding] = []    # severity 0: every event worth knowing that is not a problem (quiet breaths)
    pauses: list[Pause]
    truncation_candidates: list[dict] = []
    counts: dict[str, int] = {}
    breath_model: str = ""          # "on", or "off" and why: whether the breath model confirmed this file's breaths
    notes: list[str] = []


class RunReport(BaseModel):
    tool: str = "finalpass-audiobook"
    version: str
    finalpass_version: str
    schema_version: int = SCHEMA_VERSION
    run_id: str
    run_started_at: str
    rules: str
    tunables: dict
    network_attempts: int
    files: list[FileResult]
    notes: list[str] = []
