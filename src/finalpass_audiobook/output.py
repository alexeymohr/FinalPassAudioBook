"""Write a run: issues and pauses as text and CSV, everything as JSON."""
from __future__ import annotations

import csv
import itertools
import os
import unicodedata
from pathlib import Path

from finalpass.timecode import samples_to_clock

from .checks.pauses import listed
from .findings import SEVERITIES, FileResult, Finding, RunReport

ISSUES_TXT, ISSUES_CSV = "issues.txt", "issues.csv"
PAUSES_TXT, PAUSES_CSV = "pauses.txt", "pauses.csv"
REPORT_JSON = "report.json"
RUN_FILES = (ISSUES_TXT, ISSUES_CSV, PAUSES_TXT, PAUSES_CSV, REPORT_JSON)
ISSUE_COLUMNS = ["file", "start_time", "end_time", "event", "severity", "check", "measures"]
PAUSE_COLUMNS = ["file", "start_time", "duration_s", "guess", "kind"]
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
    "hum": ("level_dbfs", "harmonics_hz", "other_lines_hz", "found_by"),
    "noise": ("floor_under_speech_db", "floor_median_dbfs", "duration_s"),
    "dropout": ("silence_ms", "level_before_dbfs", "level_after_dbfs"),
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


def _open_write(path: Path, encoding: str = "utf-8"):
    """Open for writing without following a symlink planted at `path`: a report never lands elsewhere."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    return open(fd, "w", newline="", encoding=encoding)


def run_file_is_ours(path: Path) -> bool:
    """A run-report file (issues/pauses text and CSV, report.json) this tool wrote."""
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8-sig", errors="replace")
    except OSError:
        return False
    first = head.split("\n", 1)[0].rstrip("\r")
    if path.name in (ISSUES_TXT, PAUSES_TXT):
        return first.startswith("FinalPassAudioBook ")
    if path.name == REPORT_JSON:
        return '"tool": "finalpass-audiobook"' in head
    return first in {ISSUES_CSV: {",".join(ISSUE_COLUMNS), LEGACY_HEADER},
                     PAUSES_CSV: {",".join(PAUSE_COLUMNS), "file,time,duration_s,guess,kind"}}.get(path.name, set())


def run_report_clashes(out_dir: Path) -> list[Path]:
    """Files in `out_dir` the run report would replace although this tool did not write them."""
    return [p for p in (out_dir / n for n in RUN_FILES) if (p.exists() or p.is_symlink()) and not run_file_is_ours(p)]


def write(report: RunReport, out_dir: Path, min_sev: int = 1) -> list[Path]:
    """The run report. The caller checks `run_report_clashes` first; a file planted here since is
    still never followed through a symlink."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / n for n in RUN_FILES]
    with _open_write(paths[0]) as fh:
        fh.write(issues_text(report, min_sev))
    with _open_write(paths[1], CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(ISSUE_COLUMNS)
        for fr in report.files:
            if not fr.sample_rate:                      # skipped files are listed too, with the reason
                w.writerow([_cell(fr.file), "", "", _cell("; ".join(fr.notes) or "skipped"), "", "file", ""])
            for f in shown(fr.findings, min_sev):
                w.writerow(_row(f))
    with _open_write(paths[2]) as fh:
        fh.write(pauses_text(report))
    with _open_write(paths[3], CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(PAUSE_COLUMNS)
        for fr in report.files:
            for p in listed(fr.pauses):
                w.writerow([_cell(fr.file), p.start_time, f"{p.duration_ms / 1000:.3f}", p.guess, p.kind])
    with _open_write(paths[4]) as fh:
        fh.write(report.model_dump_json(indent=2))
    return paths


def csv_name(wav: Path) -> str:
    """One CSV per WAV: same name, .csv (a sandboxed app may create exactly that beside the WAV)."""
    return wav.with_suffix(".csv").name


CSV_ENCODING = "utf-8-sig"      # with a BOM, so Excel shows "—" and "→" correctly


def name_key(text: str | Path) -> str:
    """How the file system compares names: APFS ignores case and Unicode normalisation."""
    return unicodedata.normalize("NFC", str(text)).casefold()


def _real(p: Path) -> Path:
    """`p` with its folder's symlinks resolved (the file itself need not exist)."""
    try:
        return p.parent.resolve() / p.name
    except OSError:
        return p.absolute()


def report_source(path: Path) -> tuple[str, str] | None:
    """(file name, folder) of the WAV a per-file CSV this tool wrote belongs to; None when the file
    is not one of ours. Older reports recorded no folder ("") and, before the summary, no file ("")."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with open(path, newline="", encoding=CSV_ENCODING) as fh:
            rows = list(itertools.islice(csv.reader(fh), 40))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if not rows or not rows[0]:
        return None
    if rows[0] == LEGACY_HEADER.split(","):
        return "", ""
    if rows[0][0] != REPORT_MARKER:
        return None
    facts = {}
    for r in rows[1:]:
        if not any(r):
            break                                       # the summary ends at the first empty row
        facts.setdefault(r[0], r[1] if len(r) > 1 else "")
    return _uncell(facts.get("file", "")), _uncell(facts.get("folder", ""))


def is_ours(path: Path) -> bool:
    """A per-file CSV this tool wrote, as opposed to someone else's file of that name."""
    return report_source(path) is not None


def replaceable(target: Path, wav: Path) -> bool:
    """May `wav`'s report be written at `target`? Over nothing, or over a report this tool wrote for
    that same WAV: the same file name and — when the CSV sits in another folder — the same folder, so
    two books' "Chapter 01" never share a report. Never through a symlink, never over a folder."""
    if target.is_symlink():
        return False
    if not target.exists():
        return True
    source = report_source(target)
    if source is None:
        return False
    name, folder = source
    if name and name_key(name) != name_key(wav.name):
        return False
    here = _real(wav).parent
    if name_key(_real(target).parent) != name_key(here) and name_key(folder) != name_key(here):
        return False                  # away from the WAV, only a report that names this WAV's folder
    return True


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r", "'")   # "'" too, so the guard always reads back


def _cell(value) -> str:        # noqa: ANN001
    """File names are client text: never let a spreadsheet read one as a formula."""
    text = str(value)
    return "'" + text if text[:1] in _FORMULA_START else text


def _uncell(text: str) -> str:
    return text[1:] if text[:1] == "'" and text[1:2] in _FORMULA_START else text


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
            ["folder", _cell(_real(Path(fr.path)).parent) if fr.path else ""],
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
    with _open_write(path, CSV_ENCODING) as fh:
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
