"""Settings and the app's own storage, per-user.

Linux/macOS use the XDG base directories; Windows uses `%APPDATA%` (settings) and
`%LOCALAPPDATA%` (kept reports). Reports that could not be placed are kept and offered
again at the next launch; a working folder from a crash is swept at startup. Port of the
macOS app's `UserDefaults` + "Unsaved Reports" storage, without the sandbox bookmarks.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

APP = "fpab-gui"
OWNER_FILE = "owner.pid"
RUN_PREFIX = "fpab-gui-run-"

if sys.platform == "win32":
    import ctypes
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _kernel32 = ctypes.windll.kernel32


def _config_base() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _data_base() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def config_dir() -> Path:
    return _config_base() / APP


def data_dir() -> Path:
    return _data_base() / APP


def settings_path() -> Path:
    return config_dir() / "settings.json"


def unsaved_root() -> Path:
    return data_dir() / "unsaved"


@dataclass
class Settings:
    output: str = "beside"          # "beside" (next to each WAV) or "folder"
    folder: str | None = None
    with_pauses: bool = False

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        output = raw.get("output")
        return cls(
            output=output if output in ("beside", "folder") else "beside",
            folder=raw.get("folder") if isinstance(raw.get("folder"), str) else None,
            with_pauses=bool(raw.get("with_pauses", False)),
        )

    def save(self) -> None:
        try:
            config_dir().mkdir(parents=True, exist_ok=True)
            settings_path().write_text(
                json.dumps({"output": self.output, "folder": self.folder,
                            "with_pauses": self.with_pauses}, indent=2) + "\n",
                encoding="utf-8")
        except OSError:
            pass                                     # a lost preference is not worth failing over


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        _kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)                              # POSIX only: signal 0 does not deliver anything
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner_pid(entry: Path) -> int | None:
    try:
        return int((entry / OWNER_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def sweep_old_runs() -> None:
    """Remove working folders a crash left in the temp dir (only ours, only not in use)."""
    base = Path(tempfile.gettempdir())
    try:
        entries = list(base.glob(f"{RUN_PREFIX}*"))
    except OSError:
        return
    for entry in entries:
        pid = _owner_pid(entry)
        if pid is not None and pid != os.getpid() and _alive(pid):
            continue
        shutil.rmtree(entry, ignore_errors=True)


def keep_unsaved(csv: Path) -> Path | None:
    """Keep a CSV the app could not place, in its own folder, never discarding it."""
    folder = unsaved_root() / uuid.uuid4().hex
    try:
        folder.mkdir(parents=True)
        (folder / OWNER_FILE).write_text(str(os.getpid()), encoding="utf-8")
        dest = folder / csv.name
        shutil.move(str(csv), dest)
    except OSError:
        shutil.rmtree(folder, ignore_errors=True)
        return None
    return dest


def discard_kept(csv: Path) -> None:
    """Remove a kept report (its folder) once the operator agreed to let it go."""
    root = unsaved_root()
    if root in csv.parents:
        shutil.rmtree(csv.parent, ignore_errors=True)


def leftover_reports() -> list[tuple[Path, list[Path]]]:
    """Reports an earlier session could not place: (entry folder, its CSVs). Skips a
    running copy's."""
    root = unsaved_root()
    if not root.is_dir():
        return []
    out: list[tuple[Path, list[Path]]] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        pid = _owner_pid(entry)
        if pid is not None and pid != os.getpid() and _alive(pid):
            continue
        csvs = sorted(p for p in entry.iterdir() if p.suffix.lower() == ".csv")
        out.append((entry, csvs))
    return out
