"""The list of WAVs, the two settings, and one run of the engine at a time.

Toolkit-agnostic state machine (port of the macOS app's `RunModel.swift`). The frontend
supplies `on_change` (called from any thread when state changes) and `confirm` (asks
before unsaved reports are thrown away); nothing here imports a GUI toolkit.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import config, placement
from .engine import AUDIO_EXTENSIONS, EngineEvent, build_command


@dataclass
class Item:
    path: Path
    state: str = "waiting"          # waiting|running|done|unsaved|failed|cancelled
    csv: Path | None = None
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    why: str = ""                   # reason, for unsaved/failed


def _identity(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_dev}:{st.st_ino}"
    except OSError:
        try:
            return os.path.realpath(path)
        except OSError:
            return str(path)


class RunModel:
    def __init__(self, engine_argv: list[str], settings: config.Settings | None = None,
                 on_change=None, confirm=None) -> None:
        self.engine_argv = engine_argv
        self.settings = settings or config.Settings.load()
        self.on_change = on_change or (lambda: None)
        self.confirm = confirm or (lambda count, action: False)
        self.items: list[Item] = []
        self.running = False
        self.progress = 0.0
        self.status = "Drop WAV files to begin."
        self.error_text: str | None = None
        self.run_notes: list[str] = []
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._workdir: Path | None = None
        self._stderr_path: Path | None = None
        self._cancelled = False
        self._file_index = 0
        self._taken: set[str] = set()
        config.sweep_old_runs()

    # MARK: derived

    @property
    def unsaved_count(self) -> int:
        return sum(1 for item in self.items if item.state == "unsaved")

    @property
    def can_go(self) -> bool:
        return (not self.running and bool(self.items)
                and (self.settings.output == "beside" or bool(self.settings.folder)))

    def _changed(self) -> None:
        self.on_change()

    # MARK: settings

    def set_output(self, output: str) -> None:
        if output in ("beside", "folder") and self.settings.output != output:
            self.settings.output = output
            self.settings.save()
            self._changed()

    def set_folder(self, folder: Path | None) -> None:
        self.settings.folder = str(folder) if folder else None
        if folder:
            self.settings.output = "folder"
        self.settings.save()
        self._changed()

    def set_with_pauses(self, value: bool) -> None:
        if self.settings.with_pauses != value:
            self.settings.with_pauses = value
            self.settings.save()

    # MARK: files

    def add(self, paths) -> None:
        if self.running:
            return
        found: list[Path] = []
        empty: list[str] = []
        links = 0
        for raw in paths:
            path = Path(raw)
            if path.is_symlink():
                links += 1
                continue
            if path.is_dir():
                try:
                    inside = sorted(p for p in path.iterdir() if not p.name.startswith("."))
                except OSError:
                    inside = []
                wavs = []
                for child in inside:
                    if child.suffix.lower().lstrip(".") not in AUDIO_EXTENSIONS:
                        continue
                    if child.is_symlink():
                        links += 1
                    elif child.is_file():
                        wavs.append(child)
                if not wavs:
                    empty.append(path.name)
                found.extend(wavs)
            elif path.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS and path.is_file():
                found.append(path)

        known = {_identity(item.path) for item in self.items}
        for path in found:
            key = _identity(path)
            if key in known:
                continue
            known.add(key)
            self.items.append(Item(path=path))

        if links:
            self.status = f"{links} dropped item{'' if links == 1 else 's'} {'is a link' if links == 1 else 'are links'}; drop the original files instead."
        elif empty:
            self.status = f"No WAV files found in “{'”, “'.join(empty)}” (sub-folders are not searched)."
        elif self.items:
            self.status = f"{len(self.items)} file{'' if len(self.items) == 1 else 's'} ready."
        self._changed()

    def remove(self, index: int) -> None:
        if self.running or not (0 <= index < len(self.items)):
            return
        item = self.items[index]
        if item.state == "unsaved":
            if not self.confirm(1, "Remove"):
                return
            self._discard(item)
        del self.items[index]
        self._changed()

    def clear(self) -> None:
        if self.running or (self.unsaved_count and not self.confirm(self.unsaved_count, "Clear")):
            return
        for item in self.items:
            self._discard(item)
        self.items.clear()
        self.progress = 0.0
        self.status = "Drop WAV files to begin."
        self.error_text = None
        self.run_notes = []
        self._changed()

    def _discard(self, item: Item) -> None:
        if item.state == "unsaved" and item.csv:
            config.discard_kept(item.csv)

    # MARK: run

    def go(self) -> None:
        if not self.can_go or (self.unsaved_count and not self.confirm(self.unsaved_count, "Go")):
            return
        self.error_text = None
        self.run_notes = []
        self._taken = set()
        try:
            work = Path(tempfile.mkdtemp(prefix=config.RUN_PREFIX))
            (work / config.OWNER_FILE).write_text(str(os.getpid()), encoding="utf-8")
            err = work / "engine-stderr.txt"
            err.write_bytes(b"")
        except OSError as exc:
            self.error_text = f"Could not create a working folder: {exc}"
            self._changed()
            return

        previous = list(self.items)
        for item in self.items:
            item.state = "waiting"
            item.csv = None
            item.summary = ""
            item.notes = []
            item.why = ""
        self._workdir = work
        self._stderr_path = err
        self._cancelled = False
        self._file_index = 0
        self.progress = 0.0
        self.status = "Starting…"

        argv = build_command([item.path for item in self.items], work,
                             self.settings.with_pauses, self.engine_argv)
        stderr = open(err, "wb")                     # noqa: SIM115 — closed by the reader thread
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=stderr, cwd=str(work), text=True, encoding="utf-8")
        except OSError as exc:
            stderr.close()
            self.error_text = f"Could not start the engine: {exc}"
            self.items = previous
            self.status = "Not started."
            shutil.rmtree(work, ignore_errors=True)
            self._workdir = None
            self._changed()
            return

        for item in previous:                        # agreed above, and the run has started
            self._discard(item)
        self.running = True
        self._proc = proc
        self._changed()
        threading.Thread(target=self._reader, args=(proc, stderr), daemon=True).start()

    def _reader(self, proc: subprocess.Popen, stderr) -> None:
        code = 0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                event = EngineEvent.parse(line)
                if event is None:
                    continue
                with self._lock:
                    self._handle(event)
                self._changed()
            proc.stdout.close()
            code = proc.wait()
        except Exception as exc:                     # pragma: no cover — defensive
            self.error_text = f"The engine stream failed: {exc}"
            code = proc.wait()
        finally:
            stderr.close()
        with self._lock:
            self._finished(code)
        self._changed()

    def _handle(self, event: EngineEvent) -> None:
        total = max(len(self.items), 1)
        if event.event == "start":
            if event.paths is not None and tuple(event.paths) != tuple(str(i.path) for i in self.items):
                self.error_text = "The engine received a different file list than the app shows; stopped."
                self.cancel()
        elif event.event == "file":
            if event.index is not None:
                self._file_index = event.index
            if not (0 <= self._file_index < len(self.items)):
                return
            if event.path is not None and event.path != str(self.items[self._file_index].path):
                return
            self.items[self._file_index].state = "running"
            self.status = f"File {self._file_index + 1} of {total}: {self.items[self._file_index].path.name}"
        elif event.event == "stage":
            if (event.index or 0) < 0:
                self.status = "Loading the models…"
            elif event.step is not None and event.steps and event.step >= 0:
                self.progress = min(1.0, (self._file_index + (event.step + 1) / event.steps) / total)
                name = self.items[self._file_index].path.name if 0 <= self._file_index < len(self.items) else ""
                self.status = f"File {self._file_index + 1} of {total}: {name} — {event.stage or ''}"
        elif event.event == "file_done":
            self._file_done(event, total)
        elif event.event == "done":
            self.run_notes = list(event.notes or [])
            self.status = f"Done. Network attempts: {event.network_attempts or 0}"

    def _file_done(self, event: EngineEvent, total: int) -> None:
        index = event.index
        if index is None or not (0 <= index < len(self.items)):
            return
        item = self.items[index]
        self.progress = min(1.0, (index + 1) / total)
        item.notes = list(event.notes or [])
        item.summary = f"{event.sev3 or 0} × sev 3 · {event.sev2 or 0} × sev 2 · {event.sev1 or 0} × sev 1"
        if event.path is not None and event.path != str(item.path):
            item.state = "failed"
            item.why = "the result did not match this file"
            return
        if not event.csv:
            item.state = "failed"
            item.why = "; ".join(item.notes) or "skipped (no reason given)"
            return
        try:
            item.csv = self._place(Path(event.csv), item.path)
            item.state = "done"
        except OSError as exc:
            kept = config.keep_unsaved(Path(event.csv))
            if kept:
                item.csv = kept
                item.state = "unsaved"
                item.why = str(exc)
            else:
                item.state = "failed"
                item.why = f"not saved, and the report could not be kept: {exc}"

    def _place(self, csv: Path, wav: Path) -> Path:
        if self.settings.output == "folder" and self.settings.folder:
            return placement.into_folder(csv, Path(self.settings.folder), wav.stem, self._taken)
        return placement.beside_or_numbered(csv, wav, self._taken)

    def cancel(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._cancelled = True
        proc.terminate()

    def stop_for_quit(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self.cancel()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def _finished(self, code: int) -> None:
        self.running = False
        self._proc = None
        for item in self.items:
            if item.state in ("running", "waiting"):
                item.state = "cancelled" if self._cancelled else "failed"
                if item.state == "failed":
                    item.why = item.why or "not checked"
        if self._cancelled:
            self.status = "Cancelled."
        elif code < 0:
            self.status = "The engine stopped unexpectedly."
            self.error_text = f"The engine was stopped by signal {-code}."
        elif code != 0:
            self.status = "The engine stopped with an error."
            tail = ""
            if self._stderr_path is not None:
                try:
                    tail = self._stderr_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    tail = ""
            lines = "\n".join(tail.splitlines()[-8:])
            self.error_text = lines or f"Exit code {code}."
        else:
            self.progress = 1.0
        if self.unsaved_count and not self._cancelled:
            self.status += f" {self.unsaved_count} report{'' if self.unsaved_count == 1 else 's'} could not be saved — use “Save Unsaved CSVs…”."
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    # MARK: unsaved reports

    def save_unsaved(self, dest: Path) -> None:
        used = set(self._taken) | {placement.fold(str(i.csv)) for i in self.items if i.csv}
        for item in self.items:
            if item.state != "unsaved" or not item.csv:
                continue
            try:
                placed = placement.into_folder(item.csv, dest, item.path.stem, used)
            except OSError:
                continue
            config.discard_kept(item.csv)
            item.csv = placed
            item.state = "done"
            item.why = ""
        if self.unsaved_count == 0:
            self.status = "All reports saved."
        self._changed()

    # MARK: reports an earlier session could not place

    def save_leftovers(self, dest: Path) -> tuple[int, int]:
        saved = failed = 0
        used: set[str] = set()
        for entry, csvs in config.leftover_reports():
            placed_all = True
            for csv in csvs:
                try:
                    placement.into_folder(csv, dest, csv.stem, used)
                    saved += 1
                except OSError:
                    placed_all = False
                    failed += 1
            if placed_all:
                shutil.rmtree(entry, ignore_errors=True)
        self.status = (f"Saved {saved} earlier report{'' if saved == 1 else 's'}."
                       if failed == 0 else
                       f"{failed} earlier report{'' if failed == 1 else 's'} could not be saved; offered again next time.")
        self._changed()
        return saved, failed

    @staticmethod
    def discard_leftovers() -> None:
        for entry, _ in config.leftover_reports():
            shutil.rmtree(entry, ignore_errors=True)

    @staticmethod
    def has_leftovers() -> bool:
        return bool(config.leftover_reports())
