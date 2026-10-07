"""Run `fpab check --progress jsonl` and turn its stdout into engine events.

Mirror of the macOS app's `Engine.swift` + `EngineEvent`: the engine is the same
Python CLI, spawned as a subprocess and spoken to one JSON object per line. No GUI
imports here, so this module is reusable and testable without a display.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = ("wav", "bwf")


class EngineError(Exception):
    """The engine cannot be found or started."""


@dataclass(frozen=True)
class EngineEvent:
    """One line of `fpab check --progress jsonl` (unknown fields ignored)."""

    event: str
    index: int | None = None
    total: int | None = None
    files: int | None = None
    stages: tuple[str, ...] | None = None
    stage: str | None = None
    step: int | None = None
    steps: int | None = None
    path: str | None = None
    paths: tuple[str, ...] | None = None
    csv: str | None = None
    notes: tuple[str, ...] | None = None
    sev3: int | None = None
    sev2: int | None = None
    sev1: int | None = None
    network_attempts: int | None = None

    @classmethod
    def parse(cls, line: str) -> "EngineEvent | None":
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(raw, dict) or not isinstance(raw.get("event"), str):
            return None
        known: dict[str, object] = {}
        for field in cls.__dataclass_fields__:
            if field == "event" or field not in raw:
                continue
            value = raw[field]
            if field in ("stages", "paths", "notes") and value is not None:
                if not isinstance(value, (list, tuple)):
                    return None
                value = tuple(value)
            known[field] = value
        return cls(event=raw["event"], **known)  # type: ignore[arg-type]


def repo_root() -> Path:
    """The checked-out project this GUI lives in (`gui/fpab_gui/` -> repo)."""
    return Path(__file__).resolve().parents[2]


def find_engine(explicit: str | None = None) -> list[str]:
    """The argv prefix that runs the engine, most specific first.

    Order: an explicit path, `$FPAB_ENGINE`, the project's `.venv/bin/fpab`,
    `uv run` in the repo, then `fpab` on `$PATH`.
    """
    if explicit:
        return [explicit]
    from_env = os.environ.get("FPAB_ENGINE")
    if from_env:
        return [from_env]
    repo = repo_root()
    venv = repo / ".venv" / "bin" / "fpab"
    if venv.is_file() and os.access(venv, os.X_OK):
        return [str(venv)]
    if (repo / "pyproject.toml").is_file() and shutil.which("uv"):
        return ["uv", "run", "--project", str(repo), "fpab"]
    found = shutil.which("fpab")
    if found:
        return [found]
    raise EngineError(
        "could not find the fpab engine: run `uv sync` in the project, put `fpab` on "
        "PATH, or pass --engine / set FPAB_ENGINE"
    )


def build_command(files: list[Path], csv_dir: Path, with_pauses: bool,
                  engine_argv: list[str]) -> list[str]:
    """`fpab check` writing one CSV per file into `csv_dir`, as JSON-lines progress."""
    argv = [*engine_argv, "check", "--csv-dir", str(csv_dir), "--progress", "jsonl"]
    if with_pauses:
        argv.append("--with-pauses")
    return [*argv, "--", *(str(f) for f in files)]
