"""Run every check over a set of chapter files, inside the network guard."""
from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import finalpass
from finalpass.breath_check import BreathTunables
from finalpass.errors import FinalPassError

from . import __version__
from .activity import measure
from .chapter import Chapter, ChapterError
from .checks.breaths import BreathSeverity, breath_findings
from .checks.clicks import ClickTunables, click_findings, rise_db
from .checks.ticks import TickTunables, tick_findings
from .checks.dropouts import DropoutTunables, dropout_findings
from .checks.hum import HumTunables, hum_findings
from .checks.noise import NoiseTunables, noise_findings
from .checks.pauses import pause_map
from .checks.plosives import PlosiveTunables, plosive_findings
from .checks.truncation import TruncationTunables, score_phrase_ends, truncation_findings
from .findings import FileResult, Finding, RunReport
from .model import ModelError, load
from .netguard import NetworkGuard
from .rules import RULE_SETS, RuleSet

STAGES = ("loading", "breaths", "pauses", "hum", "noise", "dropouts", "plosives", "clicks", "ticks", "chopped words")


@dataclass(frozen=True)
class RunOptions:
    rules: RuleSet = RULE_SETS["standard"]
    truncation: bool = True
    breaths: BreathTunables = field(default_factory=BreathTunables)
    breath_severity: BreathSeverity = field(default_factory=BreathSeverity)
    hum: HumTunables = field(default_factory=HumTunables)
    noise: NoiseTunables = field(default_factory=NoiseTunables)
    dropouts: DropoutTunables = field(default_factory=DropoutTunables)
    plosives: PlosiveTunables = field(default_factory=PlosiveTunables)
    clicks: ClickTunables = field(default_factory=ClickTunables)
    ticks: TickTunables = field(default_factory=TickTunables)
    truncation_tunables: TruncationTunables = field(default_factory=TruncationTunables)

    def tunables(self) -> dict:
        return {"breaths": self.breaths.as_dict(), "breath_severity": self.breath_severity.as_dict(),
                "hum": self.hum.as_dict(), "noise": self.noise.as_dict(),
                "dropouts": self.dropouts.as_dict(), "plosives": self.plosives.as_dict(),
                "clicks": self.clicks.as_dict(), "ticks": self.ticks.as_dict(),
                "truncation": self.truncation_tunables.as_dict() if self.truncation else "off"}


def analyze_file(path: Path, opts: RunOptions, model=None, stage: Callable[[str], None] | None = None) -> FileResult:
    say = stage or (lambda name: None)
    say("loading")
    ch = Chapter.load(path)
    findings: list[Finding] = []
    if ch.invalid_samples.size:
        s = int(ch.invalid_samples[0])
        findings.append(Finding(
            file=ch.name, check="file", start_sample=s, end_sample=int(ch.invalid_samples[-1]),
            start_time=ch.clock(s), end_time=ch.clock(int(ch.invalid_samples[-1])), severity=3,
            problem=f"file contains {ch.invalid_samples.size} invalid (NaN/Inf) samples — corrupt audio",
            measures={"invalid_samples": int(ch.invalid_samples.size)}))
    say("breaths")
    rise = rise_db(ch.x, ch.sr, opts.clicks)
    breath_list, breaths = breath_findings(ch, opts.breaths, opts.breath_severity, rise)
    findings += breath_list
    say("pauses")
    act = measure(ch)
    pauses = pause_map(ch, act, opts.rules)
    say("hum")
    hums = hum_findings(ch, opts.hum)
    findings += hums
    say("noise")
    tones = [([float(h.measures["frequency_hz"])]
              + [float(v) for k in ("harmonics_hz", "pause_lines_hz") for v in str(h.measures.get(k, "")).split(",") if v],
              h.start_sample / ch.sr, h.end_sample / ch.sr) for h in hums]
    noise_list, floor = noise_findings(ch, opts.noise, exclude=tones)
    findings += noise_list
    say("dropouts")
    findings += dropout_findings(ch, act.floor_dbfs, opts.dropouts)
    say("plosives")
    spans = act.breath_spans + tuple((e.start_sample, e.end_sample) for e in breaths.breaths)
    findings += plosive_findings(ch, spans, opts.plosives)
    say("clicks")
    clicks = click_findings(ch, act.pauses, spans, opts.clicks, rise)
    say("ticks")
    ticks = tick_findings(ch, opts.ticks)
    near = int(0.003 * ch.sr)                  # a digital tick in a pause: listed once, as the tick
    findings += [c for c in clicks if all(abs(c.start_sample - t.start_sample) > near for t in ticks)] + ticks
    records: list[dict] = []
    if model is not None:
        say("chopped words")
        ends = [p0 for p0, _ in act.pauses] + ([act.last_sound] if act.last_sound else [])
        records = score_phrase_ends(ch, ends, model, opts.truncation_tunables)
        findings += truncation_findings(ch, records)
    findings.sort(key=lambda f: (f.start_sample, -f.severity))
    counts = {"breaths": breaths.counts.breaths, "mouth_click_inhales": sum(f.measures.get("mouth_click") == "yes" for f in breath_list),
              "loud_breaths": breaths.counts.grade_3, "pauses": len(act.pauses),
              "phrase_ends_scored": len(records)}
    for f in findings:
        counts[f"{f.check}_findings"] = counts.get(f"{f.check}_findings", 0) + 1
        counts[f"sev_{f.severity}"] = counts.get(f"sev_{f.severity}", 0) + 1
    return FileResult(
        file=ch.name, path=str(path), sample_rate=ch.sr, duration_seconds=round(ch.duration_s, 3),
        narration_dbfs=round(ch.narration_dbfs, 2) if ch.narration_dbfs == ch.narration_dbfs else None,
        noise_floor_dbfs=round(floor, 1) if floor is not None else None,
        findings=findings, pauses=pauses, truncation_candidates=records, counts=counts,
        notes=ch.notes + [n for n in breaths.notes if n not in ch.notes],
    )


def run(paths: list[Path], opts: RunOptions = RunOptions(), progress=None,
        stage: Callable[[int, str], None] | None = None,
        file_done: Callable[[int, FileResult], None] | None = None) -> RunReport:
    """Check every file. `progress(i, n, path)` before each file, `stage(i, name)` as each check
    starts (i = -1 while the model loads), `file_done(i, result)` as each file finishes."""
    started = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = started.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + secrets.token_hex(3)
    notes: list[str] = []
    files: list[FileResult] = []
    with NetworkGuard() as guard:
        model = None
        if opts.truncation:
            if stage:
                stage(-1, "loading model")
            try:
                model = load()
            except ModelError as exc:
                notes.append(f"truncation check skipped: {exc}")
        for i, path in enumerate(paths):
            if progress:
                progress(i, len(paths), path)
            try:
                files.append(analyze_file(path, opts, model, (lambda name, i=i: stage(i, name)) if stage else None))
            except Exception as exc:        # one bad file is skipped with a note, never the whole batch
                why = str(exc) if isinstance(exc, (FinalPassError, ChapterError)) else f"{type(exc).__name__}: {exc}"
                files.append(FileResult(file=path.name, path=str(path), sample_rate=0, duration_seconds=0.0,
                                        narration_dbfs=None, noise_floor_dbfs=None, findings=[], pauses=[],
                                        notes=[f"skipped: {why}"]))
            if file_done:
                file_done(i, files[-1])
    return RunReport(version=__version__, finalpass_version=finalpass.__version__, run_id=run_id,
                     run_started_at=started.isoformat().replace("+00:00", "Z"), rules=opts.rules.name,
                     tunables=opts.tunables(), network_attempts=len(guard.attempts), files=files, notes=notes)
