"""`fpab`: command line. Parses, runs, prints; the work happens in `run`."""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape

from . import __version__
from .rules import RULE_SETS

out = Console()
err = Console(stderr=True)
AUDIO_SUFFIXES = {".wav", ".bwf"}       # FinalPass's reader refuses ".wave"


def _expand(paths: tuple[Path, ...]) -> list[Path]:
    """Files, and the audio files in folders (not searched recursively); each file once (the same file
    by any path; a file that cannot be looked at, by its resolved path)."""
    from .output import _real

    files: list[Path] = []
    seen: set = set()
    for p in paths:
        try:
            found = (sorted(q for q in p.iterdir() if q.suffix.lower() in AUDIO_SUFFIXES and not q.name.startswith("."))
                     if p.is_dir() else [p])
        except OSError:
            found = [p]                         # a folder that cannot be read: skipped with a note, like a bad file
        for q in found:
            try:
                st = q.stat()
                key = (st.st_dev, st.st_ino)
            except OSError:
                key = str(_real(q))
            if key not in seen:
                seen.add(key)
                files.append(q)
    return files


@click.group()
@click.version_option(__version__, prog_name="fpab")
def main() -> None:
    """FinalPassAudioBook: local, offline QC for audiobook chapters."""


MAX_NUMBERED = 10_000


def _pick(first: Path, seen: set[str], wav: Path | None = None) -> Path:
    """`first`, or the first of `first (2)`, `(3)`, … where nothing exists yet (an earlier report is
    never replaced), not taken in this run (`seen`) and not another audio file's own CSV name there
    (`wav`: the file this CSV is for, whose own name it may take). OSError when no name can be had
    (a name too long for the file system, a folder that cannot be searched)."""
    from .output import _real, is_free, name_key

    def audio_named(target: Path) -> bool:
        others = [target.with_suffix(s) for s in (*AUDIO_SUFFIXES, *(x.upper() for x in AUDIO_SUFFIXES))]
        return any(o.exists() and (wav is None or not o.samefile(wav)) for o in others)

    for k in range(1, MAX_NUMBERED):
        target = first if k == 1 else first.with_name(f"{first.stem} ({k}){first.suffix}")
        if len(target.name.encode()) > 255:
            raise OSError(f"no CSV name for {first.stem!r}: the numbered name would be too long")
        if name_key(_real(target)) in seen or not is_free(target) or audio_named(target):
            continue
        return target
    raise OSError(f"no free CSV name for {first.stem!r} after {MAX_NUMBERED} tries")


def _csv_targets(files: list[Path], csv_dir: Path | None,
                 reserved: tuple[Path, ...] = ()) -> list[Path | OSError]:
    """Where each file's CSV goes: beside the WAV, or into csv_dir (see `_pick`) — always a new file.
    `reserved`: the run report's files. A file whose CSV cannot be named gets the error instead (a
    note on that file when it is done; the batch goes on)."""
    from .output import _real, csv_name, name_key

    out: list[Path | OSError] = []
    seen = {name_key(_real(p)) for p in reserved}
    for f in files:
        try:
            target = _pick((csv_dir or f.parent) / csv_name(f), seen, f)
        except OSError as exc:
            out.append(exc)
            continue
        seen.add(name_key(_real(target)))
        out.append(target)
    return out


def _usable_out_dir(out_dir: Path) -> str | None:
    """Why the run report cannot go into `out_dir` (checked before the analysis), or None."""
    import tempfile

    from .output import run_report_clashes

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=out_dir):
            pass
    except OSError as exc:
        return f"cannot write the run report into {out_dir}: {exc}"
    clash = run_report_clashes(out_dir)
    if clash:
        return (f"{out_dir} holds {', '.join(p.name for p in clash)}, which this tool did not write; "
                "choose another --out folder")
    return None


@main.command("check")
@click.argument("paths", nargs=-1, required=True, type=click.Path(readable=False, path_type=Path))
@click.option("--rules", "rules_name", type=click.Choice(sorted(RULE_SETS)), default="standard", show_default=True,
              help="Pause rule set used to guess what each pause is.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Folder for the run report (issues, pause map, report.json). Default ./fpab-report, "
                   "or none when writing one CSV per file.")
@click.option("--min-sev", "min_sev", type=click.IntRange(1, 3), default=1, show_default=True,
              help="List findings of this severity and above in issues.txt/csv (3 worst). report.json keeps all.")
@click.option("--no-truncation", is_flag=True, help="Skip the truncated-word model.")
@click.option("--no-breath-model", is_flag=True,
              help="List breaths without the breath model's confirmation (consonants may be listed as breaths).")
@click.option("--csv-per-file", is_flag=True, help="Write one CSV per WAV (<name>.csv) with every finding.")
@click.option("--csv-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Put the per-file CSVs here instead of beside each WAV (implies --csv-per-file).")
@click.option("--with-pauses", is_flag=True, help="Add the pause map's rows to each per-file CSV.")
@click.option("--progress", "progress_mode", type=click.Choice(["text", "jsonl"]), default="text", show_default=True,
              help="jsonl: one JSON object per line on stdout (for the macOS app); messages go to stderr.")
def check_cmd(paths: tuple[Path, ...], rules_name: str, out_dir: Path | None, min_sev: int, no_truncation: bool,
              no_breath_model: bool, csv_per_file: bool, csv_dir: Path | None, with_pauses: bool,
              progress_mode: str) -> None:
    """Check chapter files (or folders of them) and write the issue list and pause map."""
    import json

    from .model import installed
    from .output import (RUN_FILES, _real, breath_model_state, csv_name, file_csv, name_key, printable,
                         run_report_clashes, tally, write)
    from .run import STAGES, RunOptions, run

    files = _expand(paths)
    if not files:
        err.print("[red]error:[/red] no audio files found.")
        sys.exit(2)
    per_file = csv_per_file or csv_dir is not None
    if out_dir is None and not per_file:
        out_dir = Path("./fpab-report")
    if out_dir is not None and (why := _usable_out_dir(out_dir)):
        err.print(f"[red]error:[/red] {escape(why)}", soft_wrap=True)
        sys.exit(2)
    reserved = tuple(out_dir / n for n in RUN_FILES) if out_dir is not None else ()
    targets = _csv_targets(files, csv_dir, reserved) if per_file else []
    planned = {name_key(_real(p)) for p in (*reserved, *targets) if isinstance(p, Path)}
    opts = RunOptions(rules=RULE_SETS[rules_name], truncation=not no_truncation, breath_model=not no_breath_model)
    jsonl = progress_mode == "jsonl"
    say = err if jsonl else out

    def emit(**event) -> None:
        print(json.dumps(event), flush=True)

    written_csv: dict[int, str] = {}
    from datetime import datetime

    from . import __version__
    csv_context = {"version": __version__, "rules": rules_name,
                   "chopped-word check": "off" if not opts.truncation else "off (model not installed)",
                   "analysed": datetime.now().strftime("%Y-%m-%d %H:%M")}
    failed_csv: list[int] = []

    def model_loaded(ok: bool, why: str = "") -> None:   # what every CSV says: whether the model really ran
        if opts.truncation:
            csv_context["chopped-word check"] = "on" if ok else (
                "off (model not installed)" if not installed() else f"off (model could not be loaded: {why})")
        if not ok and "chopped words" in stages:        # the progress no longer counts a step that never comes
            stages.remove("chopped words")

    def file_done(i: int, fr) -> None:
        csv_path = None
        if per_file and fr.sample_rate:
            try:
                if isinstance(targets[i], OSError):
                    raise targets[i]
                for _ in range(100):            # a file that appeared there during the run: the next name
                    try:
                        csv_path = written_csv[i] = str(file_csv(fr, targets[i], with_pauses, csv_context))
                        break
                    except FileExistsError:
                        targets[i] = _pick((csv_dir or files[i].parent) / csv_name(files[i]), planned, files[i])
                        planned.add(name_key(_real(targets[i])))
                else:
                    raise OSError("every name tried was taken while it was being written")
            except Exception as exc:        # an unwritable CSV is a note on this file, not the end of the run
                fr.notes.append(f"could not write the CSV: {exc}")
                failed_csv.append(i)
        if jsonl:
            emit(event="file_done", index=i, path=fr.path, csv=csv_path, notes=fr.notes,
                 **{f"sev{s}": sum(f.severity == s for f in fr.findings) for s in (3, 2, 1)})

    stages = [s for s in STAGES if s != "chopped words" or (opts.truncation and installed())]
    if jsonl:
        emit(event="start", files=len(files), stages=stages, paths=[str(f) for f in files])
        report = run(files, opts,
                     progress=lambda i, n, p: emit(event="file", index=i, total=n, path=str(p)),
                     stage=lambda i, name: emit(event="stage", index=i, stage=name,
                                                step=stages.index(name) if name in stages else -1,
                                                steps=len(stages)),
                     file_done=file_done, model_loaded=model_loaded)
    else:
        with err.status("Checking...") as status:
            report = run(files, opts, progress=lambda i, n, p: status.update(f"Checking {i + 1}/{n}: {p.name}"),
                         file_done=file_done, model_loaded=model_loaded)
    written, report_error = [], None
    if out_dir is not None and (clash := run_report_clashes(out_dir)):          # appeared during the run
        report_error = (f"did not write the run report: {out_dir} now holds "
                        f"{', '.join(p.name for p in clash)}, which this tool did not write")
    elif out_dir is not None:
        try:
            written = write(report, out_dir, min_sev)
        except Exception as exc:            # the files are analysed: still print what was found
            report_error = f"could not write the run report into {out_dir}: {exc}"
    for note in report.notes:
        say.print(f"[dim]note: {escape(note)}[/dim]")
    for fr in report.files:
        worst = max((f.severity for f in fr.findings), default=0)
        colour = "red" if not fr.sample_rate else {3: "red", 2: "yellow", 1: "cyan"}.get(worst, "green")
        say.print(f"[{colour}]{escape(printable(fr.file))}[/{colour}]  {tally(fr.findings)}"
                  + "".join(f"  [dim]({escape(n)})[/dim]" for n in fr.notes), soft_wrap=True)
    say.print(escape(f"breath model: {breath_model_state(report)}"), soft_wrap=True)
    say.print(f"network attempts: {report.network_attempts}")
    for p in [*written, *written_csv.values()]:
        say.print(escape(f"Wrote {p}"), soft_wrap=True)
    if report_error:
        err.print(f"[red]error:[/red] {escape(report_error)}", soft_wrap=True)
    if jsonl:
        emit(event="done", network_attempts=report.network_attempts, notes=report.notes)
    elif report_error or failed_csv or any(not fr.sample_rate for fr in report.files):
        sys.exit(1)                         # something was skipped or not written: say so to scripts


@main.command("setup-model")
@click.option("--from-file", "from_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Install an already-downloaded truncated-word model.safetensors instead of downloading it.")
@click.option("--breath-from-file", "breath_from_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Install an already-downloaded breath model (respiro-en-fpab-v2.safetensors) instead of downloading it.")
def setup_model_cmd(from_file: Path | None, breath_from_file: Path | None) -> None:
    """Install both models' weights: the truncated-word model and the breath model (each verified against its
    recorded size and SHA-256). A model already installed and verified is left as it is (no download), so
    `--from-file` alone stays offline once the breath model is in place."""
    import http.client

    from . import breath_model, model

    failed = False
    for name, mod, src in (("truncated-word model", model, from_file), ("breath model", breath_model, breath_from_file)):
        try:
            if src:
                dest = mod.install_from_file(src)
            elif mod.installed():
                mod._verify(mod.weights_path())
                out.print(escape(f"The {name} is already installed and verified: {mod.weights_path()}"), soft_wrap=True)
                continue
            else:
                dest = mod.download()
        except (model.ModelError, OSError, ValueError, http.client.HTTPException) as exc:
            err.print(f"[red]error:[/red] {escape(f'{name}: {type(exc).__name__}: {exc}')}", soft_wrap=True)
            failed = True
            continue
        out.print(escape(f"Installed and verified the {name}: {dest}" + ("" if src else f" (from {mod.WEIGHTS_URL})")),
                  soft_wrap=True)
    if failed:
        sys.exit(2)


@main.command("rules")
def rules_cmd() -> None:
    """List the pause rule sets."""
    for r in RULE_SETS.values():
        para = f"{r.paragraph_s[0]:g}-{r.paragraph_s[1]:g} s" if r.paragraph_s else "no rule"
        out.print(f"[bold]{r.name}[/bold] — {r.title}: chapter start {r.chapter_head_s:g} s, end {r.chapter_tail_s:g} s, "
                  f"after heading {r.heading_s:g} s, section break {r.section_s:g} s, paragraph {para}")
        for n in r.notes:
            out.print(f"   {n}")
