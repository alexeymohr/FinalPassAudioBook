"""GTK4 frontend: a small window over the `fpab` engine.

Drop WAV files (or folders) in, press Go, get one CSV per WAV — the same per-file CSV
`fpab check --csv-per-file` writes: a summary, the problem events, then the informational
events; a switch adds the pause map's rows. Port of the macOS app's `ContentView` +
`FileDrop` + `FPABApp` onto GTK 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from . import __version__  # noqa: E402
from .engine import EngineError, find_engine  # noqa: E402
from .runmodel import RunModel  # noqa: E402

CSS = """
.drop-zone { border: 1px dashed alpha(currentColor, 0.35); border-radius: 10px; }
.drop-zone.drop-active { border: 2px solid @accent_color; background: alpha(@accent_color, 0.08); }
.hint { opacity: 0.6; }
.dim { opacity: 0.6; }
.ok { color: #2e9e44; }
.warn { color: #d68000; }
.err { color: #d64545; }
.mono { font-family: monospace; }
"""

ICONS = {
    "waiting": ("○", "dim"),
    "done": ("✓", "ok"),
    "failed": ("⚠", "err"),
    "unsaved": ("▣", "warn"),
    "cancelled": ("–", "dim"),
}


def _list_model_items(model: Gio.ListModel):
    return [model.get_item(i) for i in range(model.get_n_items())]


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, application: Gtk.Application, engine_argv: list[str]) -> None:
        super().__init__(application=application, title="FinalPass AudioBook")
        self.set_default_size(640, 620)
        self._updating = False
        self.model = RunModel(engine_argv, on_change=self._schedule_refresh,
                              confirm=self._confirm_discard)
        self._build()
        self._refresh()
        if self.model.has_leftovers():
            GLib.idle_add(self._offer_leftovers)

    # MARK: layout

    def _build(self) -> None:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        root.set_margin_top(20)
        root.set_margin_bottom(20)
        root.set_margin_start(20)
        root.set_margin_end(20)

        root.append(self._build_drop_zone())
        root.append(self._build_file_buttons())
        root.append(self._build_settings())
        root.append(self._build_progress())
        root.append(self._build_actions())
        self.set_child(root)
        self.connect("close-request", self._on_close_request)

    def _build_drop_zone(self) -> Gtk.Widget:
        self.drop_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.drop_box.add_css_class("drop-zone")
        self.drop_box.set_size_request(-1, 240)

        self.placeholder = Gtk.Label(
            label="Drop WAV files or folders here", halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
        self.placeholder.add_css_class("hint")
        self.placeholder.set_vexpand(True)
        self.drop_box.append(self.placeholder)

        self.listbox = Gtk.ListBox()
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self.listbox.add_css_class("background")
        self.scroller = Gtk.ScrolledWindow()
        self.scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.scroller.set_child(self.listbox)
        self.drop_box.append(self.scroller)

        target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        target.connect("drop", self._on_drop)
        target.connect("enter", self._on_drop_enter)
        target.connect("leave", self._on_drop_leave)
        self.drop_box.add_controller(target)
        return self.drop_box

    def _build_file_buttons(self) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.add_button = Gtk.Button(label="Add Files…")
        self.add_button.connect("clicked", self._on_add)
        self.clear_button = Gtk.Button(label="Clear")
        self.clear_button.connect("clicked", lambda _b: self.model.clear())
        row.append(self.add_button)
        row.append(self.clear_button)
        return row

    def _build_settings(self) -> Gtk.Widget:
        self.settings_frame = Gtk.Frame(label="Report")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(8)
        box.set_margin_end(8)

        self.radio_beside = Gtk.CheckButton(label="Next to its WAV (same name, .csv)")
        self.radio_folder = Gtk.CheckButton(label="In a folder")
        self.radio_folder.set_group(self.radio_beside)
        self.radio_beside.connect("toggled", self._on_output_toggled)
        box.append(self.radio_beside)
        box.append(self.radio_folder)

        self.folder_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.folder_row.set_margin_start(20)
        self.folder_label = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.MIDDLE, hexpand=True)
        choose = Gtk.Button(label="Choose…")
        choose.connect("clicked", self._on_choose_folder)
        self.folder_row.append(self.folder_label)
        self.folder_row.append(choose)
        box.append(self.folder_row)

        pause_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        pause_label = Gtk.Label(label="Include pause identification in report", xalign=0, hexpand=True)
        self.pauses_switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        self.pauses_switch.set_active(self.model.settings.with_pauses)
        self.pauses_switch.connect("notify::active",
                                   lambda s, _p: self.model.set_with_pauses(s.get_active()))
        pause_row.append(pause_label)
        pause_row.append(self.pauses_switch)
        box.append(pause_row)

        self.settings_frame.set_child(box)
        return self.settings_frame

    def _build_progress(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.progress = Gtk.ProgressBar()
        box.append(self.progress)
        self.status_label = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.status_label.add_css_class("dim")
        box.append(self.status_label)
        self.error_label = Gtk.Label(xalign=0, wrap=True, selectable=True)
        self.error_label.add_css_class("err")
        self.error_label.add_css_class("mono")
        box.append(self.error_label)
        self.notes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.append(self.notes_box)
        return box

    def _build_actions(self) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.save_unsaved_button = Gtk.Button(label="Save Unsaved CSVs…")
        self.save_unsaved_button.connect("clicked", self._on_save_unsaved)
        row.append(self.save_unsaved_button)
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        row.append(spacer)
        self.cancel_button = Gtk.Button(label="Cancel")
        self.cancel_button.connect("clicked", lambda _b: self.model.cancel())
        self.go_button = Gtk.Button(label="Go")
        self.go_button.add_css_class("suggested-action")
        self.go_button.connect("clicked", lambda _b: self.model.go())
        row.append(self.cancel_button)
        row.append(self.go_button)
        return row

    # MARK: refresh

    def _schedule_refresh(self) -> None:
        GLib.idle_add(self._refresh)

    def _refresh(self):
        model = self.model
        self.listbox.remove_all()
        for index, item in enumerate(model.items):
            self.listbox.append(self._row(index, item))
        self.placeholder.set_visible(not model.items)
        self.scroller.set_visible(bool(model.items))

        self._updating = True
        self.progress.set_fraction(model.progress)
        self.status_label.set_text(model.status)
        self.error_label.set_visible(bool(model.error_text))
        self.error_label.set_text(model.error_text or "")

        while (child := self.notes_box.get_first_child()) is not None:
            self.notes_box.remove(child)
        for note in model.run_notes:
            label = Gtk.Label(label=f"⚠ {note}", xalign=0)
            label.add_css_class("warn")
            self.notes_box.append(label)

        running = model.running
        self.radio_beside.set_active(model.settings.output == "beside")
        self.radio_folder.set_active(model.settings.output == "folder")
        self.folder_row.set_visible(model.settings.output == "folder")
        self.folder_label.set_text(model.settings.folder or "No folder chosen")
        self.folder_label.remove_css_class("err")
        if model.settings.output == "folder" and not model.settings.folder:
            self.folder_label.add_css_class("err")

        self.add_button.set_sensitive(not running)
        self.clear_button.set_sensitive(not running and bool(model.items))
        self.settings_frame.set_sensitive(not running)
        self.go_button.set_visible(not running)
        self.go_button.set_sensitive(model.can_go)
        self.cancel_button.set_visible(running)
        self.save_unsaved_button.set_visible(model.unsaved_count > 0 and not running)
        self._updating = False
        return False

    def _row(self, index: int, item) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        for edge in ("top", "bottom", "start", "end"):
            getattr(box, f"set_margin_{edge}")(4)

        if item.state == "running":
            spinner = Gtk.Spinner(valign=Gtk.Align.CENTER)
            spinner.start()
            box.append(spinner)
        else:
            glyph, css = ICONS.get(item.state, ("○", "dim"))
            icon = Gtk.Label(label=glyph)
            icon.add_css_class(css)
            box.append(icon)

        name = Gtk.Label(label=item.path.name, xalign=0, hexpand=True,
                         ellipsize=Pango.EllipsizeMode.MIDDLE)
        box.append(name)

        if item.notes:
            warn = Gtk.Label(label="⚠")
            warn.add_css_class("warn")
            warn.set_tooltip_text("\n".join(item.notes))
            box.append(warn)

        if item.state == "done":
            if item.summary:
                summary = Gtk.Label(label=item.summary)
                summary.add_css_class("dim")
                box.append(summary)
            if item.csv:
                show = Gtk.Button(label="Show")
                show.set_tooltip_text(str(item.csv))
                show.connect("clicked", lambda _b, p=item.csv: self._show(p))
                box.append(show)
        elif item.state == "unsaved":
            label = Gtk.Label(label=f"not saved: {item.why}", ellipsize=Pango.EllipsizeMode.END)
            label.add_css_class("warn")
            label.set_tooltip_text(item.why)
            box.append(label)
        elif item.state == "failed":
            label = Gtk.Label(label=item.why or "failed", ellipsize=Pango.EllipsizeMode.END)
            label.add_css_class("err")
            label.set_tooltip_text(item.why)
            box.append(label)
        elif item.state == "cancelled":
            label = Gtk.Label(label="cancelled")
            label.add_css_class("dim")
            box.append(label)
        elif item.state == "running":
            label = Gtk.Label(label="checking…")
            label.add_css_class("dim")
            box.append(label)

        remove = Gtk.Button(label="×")
        remove.set_tooltip_text("Remove")
        remove.add_css_class("flat")
        remove.set_sensitive(not self.model.running)
        remove.connect("clicked", lambda _b, i=index: self.model.remove(i))
        box.append(remove)

        row.set_child(box)
        return row

    # MARK: drop

    def _on_drop_enter(self, _target, _x, _y):
        if not self.model.running:
            self.drop_box.add_css_class("drop-active")
            return Gdk.DragAction.COPY
        return Gdk.DragAction()

    def _on_drop_leave(self, _target):
        self.drop_box.remove_css_class("drop-active")

    def _on_drop(self, _target, value, _x, _y):
        self.drop_box.remove_css_class("drop-active")
        if self.model.running:
            return False
        paths = [Path(f.get_path()) for f in value.get_files() if f.get_path()]
        if paths:
            self.model.add(paths)
        return True

    # MARK: dialogs

    def _on_add(self, _button) -> None:
        dialog = Gtk.FileDialog(title="Add WAV files")
        audio = Gtk.FileFilter()
        audio.set_name("Audio (WAV, BWF)")
        audio.add_suffix("wav")
        audio.add_suffix("bwf")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio)
        dialog.set_filters(filters)
        dialog.set_default_filter(audio)
        dialog.open_multiple(self, None, self._on_add_done)

    def _on_add_done(self, dialog, result) -> None:
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error:
            return
        paths = [Path(f.get_path()) for f in _list_model_items(files) if f.get_path()]
        if paths:
            self.model.add(paths)

    def _on_choose_folder(self, _button) -> None:
        dialog = Gtk.FileDialog(title="Save each CSV in this folder", accept_label="Use Folder")
        dialog.select_folder(self, None, self._on_choose_folder_done)

    def _on_choose_folder_done(self, dialog, result) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if folder.get_path():
            self.model.set_folder(Path(folder.get_path()))

    def _on_output_toggled(self, _button) -> None:
        if self._updating:
            return
        self.model.set_output("beside" if self.radio_beside.get_active() else "folder")

    def _on_save_unsaved(self, _button) -> None:
        dialog = Gtk.FileDialog(title="Save unsaved CSVs in this folder", accept_label="Save Here")
        dialog.select_folder(self, None, self._on_save_unsaved_done)

    def _on_save_unsaved_done(self, dialog, result) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if folder.get_path():
            self.model.save_unsaved(Path(folder.get_path()))

    def _confirm_discard(self, count: int, action: str) -> bool:
        if count == 0:
            return True
        return self._ask(f"{count} report{'' if count == 1 else 's'} {'is' if count == 1 else 'are'} not saved.",
                         f"{action} will discard {'it' if count == 1 else 'them'}. "
                         f"Use “Save Unsaved CSVs…” to keep {'it' if count == 1 else 'them'}.",
                         ["Cancel", f"Discard and {action}"], confirm=1) == 1

    def _ask(self, message: str, detail: str, buttons: list[str], confirm: int = -1) -> int:
        dialog = Gtk.AlertDialog(message=message, detail=detail, buttons=buttons)
        if confirm >= 0:
            dialog.set_default_button(confirm)
        dialog.set_cancel_button(0)
        loop = GLib.MainLoop()
        answer = {"index": -1}

        def done(dlg, result):
            try:
                answer["index"] = dlg.choose_finish(result)
            except GLib.Error:
                answer["index"] = -1
            loop.quit()

        dialog.choose(self, None, done)
        loop.run()
        return answer["index"]

    def _offer_leftovers(self) -> bool:
        entries = self.model.has_leftovers()
        if not entries:
            return False
        choice = self._ask("Reports from an earlier session were never saved.",
                           "Save them to a folder now, or discard them.", ["Save…", "Later", "Discard"])
        if choice == 0:
            dialog = Gtk.FileDialog(title="Save earlier reports in this folder", accept_label="Save Here")
            dialog.select_folder(self, None, self._on_save_leftovers_done)
        elif choice == 2:
            self.model.discard_leftovers()
        return False

    def _on_save_leftovers_done(self, dialog, result) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if folder.get_path():
            self.model.save_leftovers(Path(folder.get_path()))

    def _on_close_request(self, _window) -> bool:
        if not self.model.running:
            return False
        choice = self._ask("A check is still running.",
                           "Quit now and stop it? Files already finished keep their CSVs; reports not "
                           "yet saved are offered again when the app next opens.",
                           ["Keep Running", "Quit"], confirm=1)
        if choice == 1:
            self.model.stop_for_quit()
            return False
        return True

    def _show(self, path: Path) -> None:
        folder = path if path.is_dir() else path.parent
        Gio.AppInfo.launch_default_for_uri(folder.as_uri(), None)


class FPABApplication(Gtk.Application):
    def __init__(self, engine_argv: list[str]) -> None:
        super().__init__(application_id="com.themactep.FinalPassAudioBook",
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.engine_argv = engine_argv
        self.window: MainWindow | None = None

    def do_startup(self) -> None:
        Gtk.Application.do_startup(self)
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        display = Gdk.Display.get_default()
        if display is not None:
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def do_activate(self) -> None:
        if self.window is None:
            self.window = MainWindow(self, self.engine_argv)
        self.window.present()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="fpab-gui",
                                     description="FinalPass AudioBook — desktop QC for audiobook chapters.")
    parser.add_argument("--engine", help="path to the fpab executable (default: auto-detect)")
    parser.add_argument("--version", action="version", version=f"fpab-gui {__version__}")
    args = parser.parse_args(argv)
    try:
        engine = find_engine(args.engine)
    except EngineError as exc:
        print(f"fpab-gui: {exc}", file=sys.stderr)
        return 2
    return FPABApplication(engine).run([sys.argv[0]])


if __name__ == "__main__":
    raise SystemExit(main())
