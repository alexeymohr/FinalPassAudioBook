import os

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from fpab_gui.gtk_app import MainWindow  # noqa: E402


def test_window_builds():
    if os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
        pytest.skip("no display")
    if not Gtk.init_check():
        pytest.skip("no working display")
    app = Gtk.Application(application_id="com.themactep.FinalPassAudioBook.test")
    app.register(None)
    window = MainWindow(app, ["/bin/true"])
    assert window.get_title() == "FinalPass AudioBook"
