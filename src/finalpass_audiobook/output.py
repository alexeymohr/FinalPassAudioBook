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
ISSUE_COLUMNS = ["file", "time", "problem", "severity", "end_time", "check", "measures"]

LEGEND = [
    "Severity 3 = worst, 2 = likely to draw a note, 1 = worth a listen.",
    "Nothing here fails a delivery: it tells you where to listen first.",
    "Times are from the start of each file (H:MM:SS.mmm).",
]


def shown(findings: list[Finding], min_sev: int = 1) -> list[Finding]:
    return [f for f in findings if f.severity >= min_sev]


def tally(findings: list[Finding]) -> str:
    return ", ".join(f"{sum(f.severity == s for f in findings)} × sev {s}" for s in SEVERITIES)


def _measures(f: Finding) -> str:
    return "; ".join(f"{k}={v}" for k, v in f.measures.items())


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
                w.writerow([_cell(fr.file), "", "; ".join(fr.notes) or "skipped", "", "", "file", ""])
            for f in shown(fr.findings, min_sev):
                w.writerow(_row(f))
    paths[2].write_text(pauses_text(report), encoding="utf-8")
    with open(paths[3], "w", newline="", encoding=CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(["file", "time", "duration_s", "guess", "kind"])
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
    """A CSV this tool wrote (its exact header), as opposed to someone else's file of that name."""
    try:
        with open(path, encoding=CSV_ENCODING) as fh:
            return fh.readline().rstrip("\r\n") == ",".join(ISSUE_COLUMNS)
    except (OSError, UnicodeDecodeError):
        return False


def _cell(value) -> str:        # noqa: ANN001
    """File names are client text: never let a spreadsheet read one as a formula."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _row(f: Finding) -> list:
    return [_cell(f.file), f.start_time, f.problem, f.severity, f.end_time, f.check, _measures(f)]


def file_csv(fr: FileResult, path: Path, with_pauses: bool = False) -> Path:
    """Every finding of one file (and, if asked, its pause map) as one CSV, in time order."""
    rows = [(f.start_sample, _row(f)) for f in fr.findings]
    if with_pauses and fr.sample_rate:
        rows += [(p.start_sample, [_cell(fr.file), p.start_time, f"pause {p.duration_ms / 1000:.2f} s — {p.guess}",
                                   "", samples_to_clock(p.end_sample, fr.sample_rate), "pause",
                                   f"kind={p.kind}; duration_s={p.duration_ms / 1000:.3f}"])
                 for p in listed(fr.pauses)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=CSV_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(ISSUE_COLUMNS)
        w.writerows(r for _, r in sorted(rows, key=lambda z: z[0]))
    return path
