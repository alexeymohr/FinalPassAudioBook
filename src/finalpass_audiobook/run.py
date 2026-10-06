"""Run every check over a set of chapter files, inside the network guard."""
from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import finalpass
import numpy as np
from finalpass.breath_check import BreathTunables
from finalpass.errors import FinalPassError

from . import __version__
from .activity import measure
from .activity import tunables as pause_tunables
from . import breath_model as breath_weights
from .breath_np import to_16k
from .chapter import Chapter, ChapterError
from .checks.breaths import BreathConfirm, BreathCutTunables, BreathSeverity, breath_cut_findings, breath_findings
from .checks.clicks import ClickTunables, click_findings, rise_db
from .checks.ticks import TickTunables, tick_findings
from .checks.dropouts import DropoutTunables, dropout_findings
from .checks.hum import HumTunables, hum_findings
from .checks.noise import NoiseTunables, noise_findings
from .checks.pauses import pause_map
from .checks.plosives import PlosiveTunables, plosive_findings
from .checks.truncation import TruncationTunables, clip_ends, score_phrase_ends, truncation_findings
from .findings import FileResult, Finding, RunReport
from .model import ModelError, load, weights_path
from .netguard import NetworkGuard
from .rules import RULE_SETS, RuleSet

STAGES = ("loading", "breaths", "pauses", "hum", "noise", "dropouts", "plosives", "clicks", "ticks", "chopped words")


@dataclass(frozen=True)
class RunOptions:
    rules: RuleSet = RULE_SETS["standard"]
    truncation: bool = True
    breath_model: bool = True
    breath_confirm: BreathConfirm = field(default_factory=BreathConfirm)
    breath_cut: BreathCutTunables = field(default_factory=BreathCutTunables)
    breaths: BreathTunables = field(default_factory=BreathTunables)
    breath_severity: BreathSeverity = field(default_factory=BreathSeverity)
    hum: HumTunables = field(default_factory=HumTunables)
    noise: NoiseTunables = field(default_factory=NoiseTunables)
    dropouts: DropoutTunables = field(default_factory=DropoutTunables)
    plosives: PlosiveTunables = field(default_factory=PlosiveTunables)
    clicks: ClickTunables = field(default_factory=ClickTunables)
    ticks: TickTunables = field(default_factory=TickTunables)
    truncation_tunables: TruncationTunables = field(default_factory=TruncationTunables)

    def tunables(self, breath_state: str | None = None) -> dict:
        """Every tunable, as the run used it. `breath_state`: the run's breath model ("on" or "off …"); where the
        model ran, FinalPass's speech-loud rule was off (its confirmation replaces it)."""
        state = breath_state if breath_state is not None else ("on" if self.breath_model else "off")
        breaths = self.breaths.as_dict()
        if state == "on":
            breaths["speech_loud_rule"] = "off where the breath model ran"
        return {"breaths": breaths, "breath_severity": self.breath_severity.as_dict(),
                "breath_model": self.breath_confirm.as_dict() if state == "on" else state,
                "breath_cut_off": self.breath_cut.as_dict() if state == "on" else "off (needs the breath model)",
                "hum": self.hum.as_dict(), "noise": self.noise.as_dict(),
                "dropouts": self.dropouts.as_dict(), "plosives": self.plosives.as_dict(),
                "clicks": self.clicks.as_dict(), "ticks": self.ticks.as_dict(), "pauses": pause_tunables(),
                "truncation": self.truncation_tunables.as_dict() if self.truncation else "off"}


MIN_DURATION_S = 1.0                     # shorter than this cannot be a chapter
MIN_BREATH_RATE = 16000
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


INVALID_GAP_S = 1.0                      # invalid samples this far apart are separate stretches
MAX_INVALID_EVENTS = 20


def _invalid_events(ch: Chapter) -> list[Finding]:
    """One event per stretch of invalid samples (closer than INVALID_GAP_S joined), so a few bad samples at both ends
    of a file are not shown as one event over the whole file; past MAX_INVALID_EVENTS the rest share one event."""
    bad = ch.invalid_samples
    cuts = np.flatnonzero(np.diff(bad) > int(INVALID_GAP_S * ch.sr)) + 1
    runs = np.split(bad, cuts)
    if len(runs) > MAX_INVALID_EVENTS:
        runs = runs[:MAX_INVALID_EVENTS - 1] + [np.concatenate(runs[MAX_INVALID_EVENTS - 1:])]
    out = []
    for r in runs:
        a, b = int(r[0]), int(r[-1])
        out.append(_file_event(ch, a, b + 1, f"file contains {r.size} invalid (NaN, Inf or out-of-range) "
                                             f"sample{'' if r.size == 1 else 's'} — corrupt audio",
                               invalid_samples=int(r.size)))
    return out


def _file_findings(ch: Chapter, header_log: str) -> list[Finding]:
    """Whole-file problems, each severity 3: corrupt samples, a file cut short, no narration, too short."""
    out, n = [], len(ch.x)
    if ch.invalid_samples.size:
        a, b = int(ch.invalid_samples[0]), int(ch.invalid_samples[-1])
        out += _invalid_events(ch)
    m = _DATA_SIZE.search(header_log)
    frame = int(b.group(1)) if (b := _BLOCK_ALIGN.search(header_log)) else 1
    # short by at least one whole frame (less loses no audio); 0xFFFFFFFF / 0x7FFFFFFF: a header that
    # was never finished (a 0 size is never "more than the data")
    if m and int(m.group(2)) + max(1, frame) <= int(m.group(1)) and int(m.group(1)) not in (0xFFFFFFFF, 0x7FFFFFFF):
        out.append(_file_event(ch, n, n, "file is cut short: its audio ends before its header says it should",
                               data_bytes=int(m.group(2)), header_bytes=int(m.group(1))))
    if ch.duration_s < MIN_DURATION_S:
        out.append(_file_event(ch, 0, n, f"file is only {int(ch.duration_s * 1000)} ms long",
                               duration_ms=int(ch.duration_s * 1000)))
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


def analyze_file(path: Path, opts: RunOptions, model=None, stage: Callable[[str], None] | None = None,
                 breath_model=None, breath_state: str | None = None) -> FileResult:
    """One file's findings. `breath_model`: the loaded breath model or None; `breath_state`: why there is none
    ("off", "off (model not installed)", …), shown in the reports."""
    say = stage or (lambda name: None)
    say("loading")
    ch = Chapter.load(path)
    audio_format, header_log = _audio_info(path)
    findings: list[Finding] = _file_findings(ch, header_log)
    notes: list[str] = []
    say("breaths")
    rise = rise_db(ch.x, ch.sr, opts.clicks)
    confirm = None
    # "on" only where the model really ran on this file; otherwise the run's reason, or plain "off"
    state = breath_state if breath_state not in (None, "on") else "off"
    if breath_model is not None:
        if ch.sr < MIN_BREATH_RATE:
            state = f"off (not run: {ch.sr / 1000:g} kHz audio)"
        else:
            try:
                confirm = breath_model.probs(to_16k(ch.x, ch.sr))
                state = "on"
            except Exception as exc:        # the model is a refinement: a failure costs the file its confirmation only
                state = f"off (failed on this file: {type(exc).__name__}: {exc})"
                notes.append(f"breath model failed on this file, breaths listed without it: {type(exc).__name__}: {exc}")
    # with the model, its confirmation replaces FinalPass's "as loud as speech" rule, which there only loses short
    # loud breaths the model keeps (it rejects every consonant the rule removes)
    breath_tunables = replace(opts.breaths, speech_loud_rule=False) if confirm is not None else opts.breaths
    breath_list, quiet_breaths, breaths = breath_findings(ch, breath_tunables, opts.breath_severity, rise,
                                                          confirm, opts.breath_confirm)
    breath_check = ch.sr >= MIN_BREATH_RATE
    if not breath_check:                     # breath features need the band above 5 kHz: room tone reads as breath
        breath_list, quiet_breaths = [], []
        notes.append(f"breath check skipped: {ch.sr / 1000:g} kHz audio (needs {MIN_BREATH_RATE / 1000:g} kHz or more)")
    # a breath cut off into a short hole of digital silence (the model finds the short ones FinalPass skips)
    cut_off = breath_cut_findings(ch, confirm, [(f.start_sample, f.end_sample) for f in breath_list], opts.breath_cut) \
        if confirm is not None and breath_check else []
    kept = [q for q in quiet_breaths
            if not any(q.start_sample < f.end_sample and q.end_sample > f.start_sample for f in cut_off)]
    promoted, quiet_breaths = len(quiet_breaths) - len(kept), kept      # a quiet breath cut off is reported as that
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
              h.start_sample / ch.sr, h.end_sample / ch.sr, float(h.measures["level_dbfs"])) for h in hums]
    noise_list, floor = noise_findings(ch, opts.noise, exclude=tones)
    findings += noise_list
    say("dropouts")
    dropouts: list[Finding] = []
    eight_bit = 0 < ch.audio.bit_depth <= 8     # 8-bit steps round quiet audio to exact zero: no dropout or
    if eight_bit:                               # clip-end test (both read exact zeros)
        notes.append("dropout and chopped-word checks skipped: 8-bit audio")
    else:
        dropouts = dropout_findings(ch, opts.dropouts)
    findings += dropouts
    if opts.plosives.high_band_hz[1] >= ch.sr / 2:
        notes.append(f"plosive check skipped: {ch.sr / 1000:g} kHz audio (needs more than "
                     f"{2 * opts.plosives.high_band_hz[1] / 1000:g} kHz)")
    if opts.ticks.above_hz >= 0.95 * ch.sr / 2:
        notes.append(f"digital-tick check skipped: {ch.sr / 1000:g} kHz audio (needs more than "
                     f"{2 * opts.ticks.above_hz / 0.95 / 1000:.1f} kHz)")
    say("plosives")
    # what FinalPass calls a breath keeps clicks out of it — unless the breath check is skipped (below 16 kHz, where
    # room tone reads as breath)
    spans = act.breath_spans + (tuple((e.start_sample, e.end_sample) for e in breaths.breaths) if breath_check else ())
    mouth_clicks = tuple((f.start_sample, f.end_sample) for f in breath_list if f.measures.get("mouth_click") == "yes")
    findings += plosive_findings(ch, mouth_clicks, opts.plosives)
    say("clicks")
    clicks = click_findings(ch, act.pauses, spans, opts.clicks, rise, act.first_sound, act.last_sound, floor)
    say("ticks")
    near = int(0.003 * ch.sr)
    # The step into and out of a dropout is the dropout's own edge, not a separate tick; a digital
    # tick in a pause is listed once, as the tick.
    edges = [e for d in dropouts for e in (d.start_sample, d.end_sample)]
    ticks = [k for k in tick_findings(ch, opts.ticks) if all(abs(k.start_sample - e) > near for e in edges)]
    findings += [c for c in clicks if all(abs(c.start_sample - k.start_sample) > near for k in ticks)] + ticks
    records: list[dict] = []
    if model is not None and not eight_bit:
        say("chopped words")
        records = score_phrase_ends(ch, clip_ends(ch, opts.truncation_tunables), model, opts.truncation_tunables)
        # a word that ends in a digital tick is listed once, as the tick (the worse of the two)
        findings += [f for f in truncation_findings(ch, records)
                     if all(abs(f.start_sample - k.start_sample) > near for k in ticks)]
    findings = _clear_of_invalid(findings, ch)
    findings.sort(key=lambda f: (f.start_sample, -f.severity))
    counts = {"breaths": (len(breath_list) + len(quiet_breaths) + len(cut_off)) if breath_check else 0,
              "breaths_cut_off": len(cut_off),
              "breaths_detected": len(breaths.breaths),
              "mouth_click_inhales": sum(f.measures.get("mouth_click") == "yes" for f in breath_list),
              "breaths_listed": len(breath_list), "quiet_breaths": len(quiet_breaths), "pauses": len(act.pauses),
              "breaths_not_confirmed": (len(breaths.breaths) - len(breath_list) - len(quiet_breaths) - promoted
                                        if confirm is not None else 0),
              "phrase_ends_scored": len(records)}
    for f in findings:
        counts[f"{f.check}_findings"] = counts.get(f"{f.check}_findings", 0) + 1
        counts[f"sev_{f.severity}"] = counts.get(f"sev_{f.severity}", 0) + 1
    return FileResult(
        file=ch.name, path=str(path), sample_rate=ch.sr, duration_seconds=round(ch.duration_s, 3),
        audio_format=audio_format, channels=ch.audio.channel_count,
        narration_dbfs=round(ch.narration_dbfs, 2) if ch.narration_dbfs == ch.narration_dbfs else None,
        noise_floor_dbfs=round(floor, 1) if floor is not None else None,
        findings=findings, informational=sorted(quiet_breaths + cut_off, key=lambda f: f.start_sample), pauses=pauses,
        truncation_candidates=records, counts=counts, breath_model=state,
        notes=ch.notes + [n for n in breaths.notes if n not in ch.notes] + notes,
    )


def run(paths: list[Path], opts: RunOptions = RunOptions(), progress=None,
        stage: Callable[[int, str], None] | None = None,
        file_done: Callable[[int, FileResult], None] | None = None,
        model_loaded: Callable[[bool, str], None] | None = None) -> RunReport:
    """Check every file. `progress(i, n, path)` before each file, `stage(i, name)` as each check
    starts (i = -1 while the models load), `file_done(i, result)` as each file finishes, and
    `model_loaded(ok, why)` once, before the first file, when the chopped-word check is on. Whether the breath
    model ran is in every file's result (`breath_model`)."""
    started = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = started.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + secrets.token_hex(3)
    notes: list[str] = []
    files: list[FileResult] = []
    with NetworkGuard() as guard:
        model = None
        if opts.truncation:
            if stage:
                stage(-1, "loading model")
            why = ""
            try:
                model = load()
            except ModelError as exc:
                notes.append(f"truncation check skipped: {exc}")
                why = str(exc).removeprefix(f"{weights_path()}: ")      # the reason, without the long path
            if model_loaded:
                model_loaded(model is not None, why)
        bmodel, bstate = None, "off"
        if opts.breath_model:
            try:
                bmodel = breath_weights.load()
                bstate = "on"
            except ModelError as exc:
                notes.append(f"breath model off: {exc}")
                why = str(exc).removeprefix(f"{breath_weights.weights_path()}: ")
                bstate = ("off (model not installed)" if not breath_weights.installed()
                          else f"off (model could not be loaded: {why})")
        for i, path in enumerate(paths):
            if progress:
                progress(i, len(paths), path)
            try:
                files.append(analyze_file(path, opts, model, (lambda name, i=i: stage(i, name)) if stage else None,
                                          bmodel, bstate))
            except Exception as exc:        # one bad file is skipped with a note, never the whole batch
                why = str(exc) if isinstance(exc, (FinalPassError, ChapterError)) else f"{type(exc).__name__}: {exc}"
                files.append(FileResult(file=path.name, path=str(path), sample_rate=0, duration_seconds=0.0,
                                        narration_dbfs=None, noise_floor_dbfs=None, findings=[], pauses=[],
                                        notes=[f"skipped: {why}"]))
            if file_done:
                file_done(i, files[-1])
    return RunReport(version=__version__, finalpass_version=finalpass.__version__, run_id=run_id,
                     run_started_at=started.isoformat().replace("+00:00", "Z"), rules=opts.rules.name,
                     tunables=opts.tunables(bstate), network_attempts=len(guard.attempts), files=files, notes=notes)
