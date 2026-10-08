"""Shared command line: choose the frontend toolkit and locate the engine."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .engine import EngineError, find_engine


def _default_ui() -> str:
    """GTK4 where it is importable (Linux), Tk otherwise (Windows, or no PyGObject)."""
    env = os.environ.get("FPAB_GUI_UI")
    if env in ("gtk", "tk"):
        return env
    if sys.platform == "win32":
        return "tk"
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        from gi.repository import Gtk  # noqa: F401
    except Exception:
        return "tk"
    return "gtk"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fpab-gui", description="FinalPass AudioBook — desktop QC for audiobook chapters.")
    parser.add_argument("--ui", choices=["gtk", "tk"], default=None,
                        help="frontend toolkit (default: GTK4 where available, else Tk)")
    parser.add_argument("--engine", help="path to the fpab executable (default: auto-detect)")
    parser.add_argument("--version", action="version", version=f"fpab-gui {__version__}")
    args = parser.parse_args(argv)
    try:
        engine = find_engine(args.engine)
    except EngineError as exc:
        print(f"fpab-gui: {exc}", file=sys.stderr)
        return 2
    if (args.ui or _default_ui()) == "gtk":
        from .gtk_app import run
    else:
        from .tk_app import run
    return run(engine)
