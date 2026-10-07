import os

import pytest

tk = pytest.importorskip("tkinter")

from fpab_gui.tk_app import TkApp  # noqa: E402


def test_window_builds():
    if os.name != "nt" and os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
        pytest.skip("no display")
    try:
        app = TkApp(["/bin/true"])
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    try:
        assert app.root.title() == "FinalPass AudioBook"
        assert app.tree["columns"] == ("file", "status", "summary", "report")
    finally:
        app.root.destroy()
