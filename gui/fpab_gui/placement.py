"""Put a finished CSV where the operator asked, never replacing a file.

Port of the macOS app's `Placement.swift`, POSIX only: Linux has no App Sandbox,
so the file-coordinator and security-scope machinery is unnecessary and the
numbered-name dance is always allowed (the difference the Mac had to ask a folder
grant for). An existing file is never replaced — a new report gets `<name> (2).csv`,
`(3)`, … — and a name that belongs to another audio file (`a (2).wav`) is skipped.
"""
from __future__ import annotations

import errno
import itertools
import os
import unicodedata
from pathlib import Path

AUDIO_SUFFIXES = (".wav", ".bwf", ".WAV", ".BWF")
MAX_NUMBERED = 10_000


def fold(name: str) -> str:
    """How the file system compares names (canonical form, case-folded)."""
    return unicodedata.normalize("NFC", name).lower()


def beside_target(wav: Path) -> Path:
    """`<name>.csv` beside `<name>.wav`."""
    return wav.with_suffix(".csv")


def exists(path: Path) -> bool:
    """Anything at `path`, a dangling link included."""
    return os.path.lexists(path)


def _audio_named(target: Path) -> bool:
    """An audio file (`stem.wav`, `stem.bwf`, …) owns this CSV name here."""
    return any(target.with_suffix(s).exists() for s in AUDIO_SUFFIXES)


def _write_new(data: bytes, target: Path) -> bool:
    """Write `data` as a new file at `target`; False if something is already there.

    Never over a file, never through a symlink (`O_EXCL | O_NOFOLLOW`). A file this
    call created but could not finish is removed — only that file, by device and inode.
    """
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.ELOOP:                 # a symlink sits at the name: leave it
            return False
        raise
    mine = os.fstat(fd)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except OSError:
        if exists(target):
            now = os.lstat(target)
            if (now.st_dev, now.st_ino) == (mine.st_dev, mine.st_ino):
                try:
                    os.unlink(target)
                except OSError:
                    pass
        raise
    return True


def into_folder(csv: Path, folder: Path, stem: str, taken: set[str]) -> Path:
    """The first of `<stem>.csv`, `<stem> (2).csv`, … that is free, not already taken
    in this run (`taken`, folded paths), and not another audio file's own CSV name.

    `taken` is updated with the placed path. Raises OSError when nothing can be placed.
    """
    data = csv.read_bytes()
    for k in itertools.islice(itertools.count(1), MAX_NUMBERED):
        numbered = stem if k == 1 else f"{stem} ({k})"
        target = folder / f"{numbered}.csv"
        if k > 1 and _audio_named(target):
            continue
        key = fold(str(target))
        if key in taken or exists(target):
            continue
        if not _write_new(data, target):
            continue
        taken.add(key)
        return target
    raise OSError(f"no free CSV name for {stem!r} in {folder}")


def beside_or_numbered(csv: Path, wav: Path, taken: set[str]) -> Path:
    """Place `csv` in the WAV's own folder as `<stem>.csv` (or the next free numbered name)."""
    return into_folder(csv, wav.parent, wav.stem, taken)
