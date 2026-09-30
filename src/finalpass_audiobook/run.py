"""Run every check over a set of chapter files, inside the network guard."""
from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import finalpass
import numpy as np
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
from .checks.truncation import TruncationTunables, clip_ends, score_phrase_ends, truncation_findings
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


MIN_DURATION_S = 1.0                     # shorter than this cannot be a chapter
_DATA_SIZE = re.compile(r"^data\s*:\s*(\d+)\s*\(should be (\d+)\)", re.MULTILINE)
_BLOCK_ALIGN = re.compile(r"Block Align\s*:\s*(\d+)")
SHORT_EVENTS = ("dropout", "ticks", "clicks", "plosive", "truncation")   # what a zeroed NaN run can cause


def _audio_info(path: Path) -> tuple[str, str]:
    """(format line, libsndfile's header log) — both courtesies: never fail a file over them."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return f"{info.format_info}, {info.subtype_info}", info.extra_info or ""
    except Exception:
        return "", ""


def _file_event(ch: Chapter, start: int, end: int, problem: str, **measures) -> Finding:
    return Finding(file=ch.name, check="file", start_sample=start, end_sample=end, start_time=ch.clock(start),
                   end_time=ch.clock(end), severity=3, problem=problem, measures=measures)


def _file_findings(ch: Chapter, header_log: str) -> list[Finding]:
    """Whole-file problems, each severity 3: corrupt samples, a file cut short, no narration, too short."""
    out, n = [], len(ch.x)
    if ch.invalid_samples.size:
        a, b = int(ch.invalid_samples[0]), int(ch.invalid_samples[-1])
        out.append(_file_event(ch, a, b, f"file contains {ch.invalid_samples.size} invalid (NaN/Inf) samples — "
                                         "corrupt audio", invalid_samples=int(ch.invalid_samples.size)))
    m = _DATA_SIZE.search(header_log)
    frame = int(b.group(1)) if (b := _BLOCK_ALIGN.search(header_log)) else 1
    # short by at least one whole frame (less loses no audio); 0 / 0xFFFFFFFF: a header never finished
    if m and int(m.group(2)) + max(1, frame) <= int(m.group(1)) < 0x7FFFFFFF:
        out.append(_file_event(ch, n, n, "file is cut short: its audio ends before its header says it should",
                               data_bytes=int(m.group(2)), header_bytes=int(m.group(1))))
    if ch.duration_s < MIN_DURATION_S:
        out.append(_file_event(ch, 0, n, f"file is only {ch.duration_s * 1000:.0f} ms long",
                               duration_ms=round(ch.duration_s * 1000)))
    elif not np.isfinite(ch.narration_dbfs):
        out.append(_file_event(ch, 0, n, "no narration found in the file"))
    return out


def _clear_of_invalid(findings: list[Finding], ch: Chapter, pad_s: float = 0.005) -> list[Finding]:
    """Drop the short events the zeroed NaN/Inf samples themselves caused (a hole, its edges): the file
    event covers them. Long events (a hum, a noisy section, a breath) stand."""
    bad, pad = ch.invalid_samples, int(pad_s * ch.sr)
    if not bad.size:
        return findings
    def near(f: Finding) -> bool:            # noqa: E306
        k = np.searchsorted(bad, f.start_sample - pad)
        return k < bad.size and bad[k] <= f.end_sample + pad
    return [f for f in findings if f.check not in SHORT_EVENTS or not near(f)]


def analyze_file(path: Path, opts: RunOptions, model=None, stage: Callable[[str], None] | None = None) -> FileResult:
    say = stage or (lambda name: None)
    say("loading")
    ch = Chapter.load(path)
    audio_format, header_log = _audio_info(path)
    findings: list[Finding] = _file_findings(ch, header_log)
    notes: list[str] = []
    say("breaths")
    rise = rise_db(ch.x, ch.sr, opts.clicks)
    breath_list, quiet_breaths, breaths = breath_findings(ch, opts.breaths, opts.breath_severity, rise)
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
    dropouts: list[Finding] = []
    if 0 < ch.audio.bit_depth <= 8:              # 8-bit steps round quiet audio to exact zero: no dropout test
        notes.append("dropout check skipped: 8-bit audio")
    else:
        dropouts = dropout_findings(ch, opts.dropouts)
    findings += dropouts
    say("plosives")
    spans = act.breath_spans + tuple((e.start_sample, e.end_sample) for e in breaths.breaths)
    mouth_clicks = tuple((f.start_sample, f.end_sample) for f in breath_list if f.measures.get("mouth_click") == "yes")
    findings += plosive_findings(ch, mouth_clicks, opts.plosives)
    say("clicks")
    clicks = click_findings(ch, act.pauses, spans, opts.clicks, rise, act.first_sound, act.last_sound)
    say("ticks")
    near = int(0.003 * ch.sr)
    # The step into and out of a dropout is the dropout's own edge, not a separate tick; a digital
    # tick in a pause is listed once, as the tick.
    edges = [e for d in dropouts for e in (d.start_sample, d.end_sample)]
    ticks = [k for k in tick_findings(ch, opts.ticks) if all(abs(k.start_sample - e) > near for e in edges)]
    findings += [c for c in clicks if all(abs(c.start_sample - k.start_sample) > near for k in ticks)] + ticks
    records: list[dict] = []
    if model is not None:
        say("chopped words")
        records = score_phrase_ends(ch, clip_ends(ch, opts.truncation_tunables), model, opts.truncation_tunables)
        # a word that ends in a digital tick is listed once, as the tick (the worse of the two)
        findings += [f for f in truncation_findings(ch, records)
                     if all(abs(f.start_sample - k.start_sample) > near for k in ticks)]
    findings = _clear_of_invalid(findings, ch)
    findings.sort(key=lambda f: (f.start_sample, -f.severity))
    counts = {"breaths": breaths.counts.breaths,
              "mouth_click_inhales": sum(f.measures.get("mouth_click") == "yes" for f in breath_list),
              "breaths_listed": len(breath_list), "quiet_breaths": len(quiet_breaths), "pauses": len(act.pauses),
              "phrase_ends_scored": len(records)}
    for f in findings:
        counts[f"{f.check}_findings"] = counts.get(f"{f.check}_findings", 0) + 1
        counts[f"sev_{f.severity}"] = counts.get(f"sev_{f.severity}", 0) + 1
    return FileResult(
        file=ch.name, path=str(path), sample_rate=ch.sr, duration_seconds=round(ch.duration_s, 3),
        audio_format=audio_format, channels=ch.audio.channel_count,
        narration_dbfs=round(ch.narration_dbfs, 2) if ch.narration_dbfs == ch.narration_dbfs else None,
        noise_floor_dbfs=round(floor, 1) if floor is not None else None,
        findings=findings, informational=sorted(quiet_breaths, key=lambda f: f.start_sample), pauses=pauses, truncation_candidates=records, counts=counts,
        notes=ch.notes + [n for n in breaths.notes if n not in ch.notes] + notes,
    )


def run(paths: list[Path], opts: RunOptions = RunOptions(), progress=None,
        stage: Callable[[int, str], None] | None = None,
        file_done: Callable[[int, FileResult], None] | None = None,
        model_loaded: Callable[[bool], None] | None = None) -> RunReport:
    """Check every file. `progress(i, n, path)` before each file, `stage(i, name)` as each check
    starts (i = -1 while the model loads), `file_done(i, result)` as each file finishes, and
    `model_loaded(ok)` once, before the first file, when the chopped-word check is on."""
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
            if model_loaded:
                model_loaded(model is not None)
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
