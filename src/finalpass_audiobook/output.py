"""Write a run: issues and pauses as text and CSV, everything as JSON."""
from __future__ import annotations

import csv
from pathlib import Path

from finalpass.timecode import samples_to_clock

from .checks.pauses import listed
from .findings import SEVERITIES, FileResult, Finding, RunReport

ISSUES_TXT, ISSUES_CSV = "issues.txt", "issues.csv"
PAUSES_TXT, PAUSES_CSV = "pauses.txt", "pauses.csv"
REPORT_JSON = "report.json"
ISSUE_COLUMNS = ["file", "start_time", "end_time", "event", "severity", "check", "measures"]
REPORT_MARKER = "FinalPass AudioBook report"       # first cell of a per-file CSV: how "ours" is recognised
LEGACY_HEADER = "file,time,problem,severity,end_time,check,measures"   # per-file CSVs before the summary
# Section titles stay short (spreadsheets size a column to its longest cell); the explanation
# sits in the event column, which is wide anyway.
PROBLEMS_LABEL = "PROBLEM EVENTS"
PROBLEMS_NOTE = "severity 3 = worst, 2 = likely to draw a note, 1 = worth a listen"
INFO_LABEL = "INFORMATIONAL EVENTS"
INFO_NOTE = "quiet breaths and, when asked for, the pause map; no severity"
# An empty row written as empty cells (",,,,,,"): a truly blank line is dropped by some spreadsheet
# apps when they open a CSV, which would lose the spacing.
SPACER = [""] * len(ISSUE_COLUMNS)

LEGEND = [
    "Severity 3 = worst, 2 = likely to draw a note, 1 = worth a listen.",
    "Nothing here fails a delivery: it tells you where to listen first.",
    "Times are from the start of each file (H:MM:SS.mmm).",
]


def shown(findings: list[Finding], min_sev: int = 1) -> list[Finding]:
    return [f for f in findings if f.severity >= min_sev]


def tally(findings: list[Finding]) -> str:
    return ", ".join(f"{sum(f.severity == s for f in findings)} × sev {s}" for s in SEVERITIES)


# The measures a CSV shows, per check: the few a mixer uses (the rest repeat the event text or are
# internal). report.json keeps every measure. A check not listed here shows all of its measures.
CSV_MEASURES = {
    "breaths": ("loudness_db", "duration_ms", "ms_since_word", "click_db_vs_narration"),
    "hum": ("level_max_dbfs", "harmonics_hz", "other_lines_hz", "found_by"),
    "noise": ("floor_under_speech_db", "floor_median_dbfs", "duration_s"),
    "dropout": ("level_before_dbfs", "silence_ms"),
    "plosive": ("low_dbfs", "low_over_high_db", "ms_to_word"),
    "truncation": ("model_score", "peak_final_30ms_dbfs"),
    "clicks": ("peak_dbfs", "ms_after_word", "ms_before_word"),
    "ticks": ("above_16k_dbfs", "width_samples", "peak_dbfs"),
    "file": ("invalid_samples",),
}


def _measures(f: Finding) -> str:
    keys = CSV_MEASURES.get(f.check, tuple(f.measures))
    return "; ".join(f"{k}={f.measures[k]}" for k in keys if f.measures.get(k, "") != "")


def issues_text(report: RunReport, min_sev: int = 1) -> str:
    lines = [f"FinalPassAudioBook {report.version} — run {report.run_id} — rules: {report.rules}", *LEGEND]
    if min_sev > 1:
        lines.append(f"Listing severity {min_sev} and above only.")
    lines += [f"note: {n}" for n in report.notes]
    for fr in report.files:
        rows = shown(fr.findings, min_sev)
        lines += ["", f"== {fr.file} — {tally(rows)}"]
        lines += [f"   note: {n}" for n in fr.notes]
        lines += [f"   {f.start_time}  sev {f.severity}  {f.problem}" for f in rows]
    return "\n".join(lines) + "\n"


def pauses_text(report: RunReport) -> str:
    lines = [f"FinalPassAudioBook {report.version} — pause map — rules: {report.rules}",
             "Informational. The kind of each pause is a GUESS from its length alone;",
             "the tool cannot see headings, sections or paragraphs."]
    for fr in report.files:
        lines += ["", f"== {fr.file}"]
        for p in listed(fr.pauses):
            lines.append(f"   {p.start_time}  {p.duration_ms / 1000:6.2f} s  {p.guess}")
    return "\n".join(lines) + "\n"


def write(report: RunReport, out_dir: Path, min_sev: int = 1) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / n for n in (ISSUES_TXT, ISSUES_CSV, PAUSES_TXT, PAUSES_CSV, REPORT_JSON)]
    paths[0].write_text(issues_text(report, min_sev), encoding="utf-8")
    with open(paths[1], "w", newline="", encoding=CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(ISSUE_COLUMNS)
        for fr in report.files:
            if not fr.sample_rate:                      # skipped files are listed too, with the reason
                w.writerow([_cell(fr.file), "", "", "; ".join(fr.notes) or "skipped", "", "file", ""])
            for f in shown(fr.findings, min_sev):
                w.writerow(_row(f))
    paths[2].write_text(pauses_text(report), encoding="utf-8")
    with open(paths[3], "w", newline="", encoding=CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(["file", "start_time", "duration_s", "guess", "kind"])
        for fr in report.files:
            for p in listed(fr.pauses):
                w.writerow([_cell(fr.file), p.start_time, f"{p.duration_ms / 1000:.3f}", p.guess, p.kind])
    paths[4].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return paths


def csv_name(wav: Path) -> str:
    """One CSV per WAV: same name, .csv (a sandboxed app may create exactly that beside the WAV)."""
    return wav.with_suffix(".csv").name


CSV_ENCODING = "utf-8-sig"      # with a BOM, so Excel shows "—" and "→" correctly


def is_ours(path: Path) -> bool:
    """A CSV this tool wrote (its first line), as opposed to someone else's file of that name."""
    try:
        with open(path, encoding=CSV_ENCODING) as fh:
            first = fh.readline().rstrip("\r\n")
    except (OSError, UnicodeDecodeError):
        return False
    return first.split(",", 1)[0] == REPORT_MARKER or first == LEGACY_HEADER


def _cell(value) -> str:        # noqa: ANN001
    """File names are client text: never let a spreadsheet read one as a formula."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _row(f: Finding) -> list:
    return [_cell(f.file), f.start_time, f.end_time, f.problem, f.severity or "", f.check, _measures(f)]


def _level(dbfs: float | None) -> str:
    return "n/a" if dbfs is None else f"{dbfs:.1f} dBFS".replace("-", "\u2212")   # a true minus: never a formula


def summary_rows(fr: FileResult, pauses: int | None, context: dict | None = None) -> list[list]:
    """The top of a per-file CSV: a label, then one short fact per cell (so no column is widened by
    it; counts are numbers). Free text (notes) goes in the event column."""
    context = context or {}
    sr, c = fr.sample_rate, fr.counts
    fmt = [x.strip() for x in fr.audio_format.split(",", 1)] if fr.audio_format else []
    fmt += [f"{sr / 1000:g} kHz"] if sr else []
    fmt += [f"{fr.channels} channel{'' if fr.channels == 1 else 's'}"] if fr.channels else []
    quiet = len(fr.informational)
    rows = [[REPORT_MARKER, context.get("version", "")],
            ["file", _cell(fr.file)],
            ["format", *fmt],
            ["duration", samples_to_clock(round(fr.duration_seconds * sr), sr) if sr else ""],
            ["problem events", len(fr.findings), *(f"{sum(f.severity == k for f in fr.findings)} × sev {k}"
                                                   for k in SEVERITIES)],
            ["informational events", quiet + (pauses or 0), f"{quiet} quiet breaths",
             f"{pauses} pauses" if pauses is not None else "pause rows off"]]
    if "breaths" in c:
        rows.append(["breaths", c["breaths"], f"{c.get('breaths_listed', 0)} problems",
                     f"{c.get('mouth_click_inhales', 0)} mouth-click inhales", f"{c.get('quiet_breaths', 0)} quiet"])
    rows += [["narration level", _level(fr.narration_dbfs)], ["noise floor", _level(fr.noise_floor_dbfs)]]
    rows += [[k, _cell(context[k])] for k in ("rules", "chopped-word check", "analysed") if context.get(k)]
    if fr.notes:
        rows.append(["notes", "", "", _cell("; ".join(fr.notes))])
    return rows


def file_csv(fr: FileResult, path: Path, with_pauses: bool = False, context: dict | None = None) -> Path:
    """One file's report: a summary, then every problem event in time order, then (if asked) the
    informational pause map, a few empty rows apart."""
    problems = [_row(f) for f in sorted(fr.findings, key=lambda f: f.start_sample)]
    pauses = None
    if with_pauses and fr.sample_rate:
        pauses = [(p.start_sample, [_cell(fr.file), p.start_time, samples_to_clock(p.end_sample, fr.sample_rate),
                                    f"pause {p.duration_ms / 1000:.2f} s — {p.guess}", "", "pause",
                                    f"kind={p.kind}; duration_s={p.duration_ms / 1000:.3f}"]) for p in listed(fr.pauses)]
    info = [r for _, r in sorted([(f.start_sample, _row(f)) for f in fr.informational] + (pauses or []),
                                 key=lambda z: z[0])]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerows(summary_rows(fr, None if pauses is None else len(pauses), context))
        w.writerow(SPACER)
        w.writerow([PROBLEMS_LABEL, "", "", PROBLEMS_NOTE])       # the key, standing on its own
        w.writerow(SPACER)
        w.writerow(ISSUE_COLUMNS)
        w.writerows(problems)
        w.writerows([SPACER] * 2)
        w.writerow([INFO_LABEL, "", "", INFO_NOTE])
        w.writerow(SPACER)
        w.writerow(ISSUE_COLUMNS)
        w.writerows(info)
        if pauses is None:
            w.writerow(["", "", "", "pause map not included (pause rows are off)"])
    return path
