"""Tkinter frontend: a small window over the `fpab` engine, for Windows (and anywhere
GTK is not available).

Reuses the same toolkit-agnostic core as the GTK frontend: `RunModel`, `placement`,
`config`, `engine`. Tkinter is single-threaded, so `RunModel`'s worker-thread
notifications are funnelled through a queue and drained on the main loop. Explorer
drag-and-drop is enabled when the optional `tkinterdnd2` package is installed;
"Add Files…" always works without it.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .runmodel import RunModel

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _HAS_DND = True
except Exception:                                    # the package is optional
    DND_FILES = None
    TkinterDnD = None
    _HAS_DND = False

STATE_LABEL = {
    "waiting": "",
    "running": "checking…",
    "done": "done",
    "cancelled": "cancelled",
}
STATE_TAG = {
    "waiting": "dim",
    "running": "run",
    "done": "ok",
    "unsaved": "warn",
    "failed": "err",
    "cancelled": "dim",
}


class TkApp:
    def __init__(self, engine_argv: list[str]) -> None:
        self.root = TkinterDnD.Tk() if _HAS_DND else tk.Tk()
        self.root.title("FinalPass AudioBook")
        self.root.minsize(680, 560)
        self._updating = False
        self._events: queue.Queue = queue.Queue()
        self.model = RunModel(engine_argv, on_change=self._notify, confirm=self._confirm)
        self._build()
        self._poll()
        self._refresh()
        self.root.after(200, self._offer_leftovers)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # MARK: layout

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        self._build_list(outer)
        self._build_file_buttons(outer)
        self._build_settings(outer)
        self._build_progress(outer)
        self._build_actions(outer)

    def _build_list(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)

        columns = ("file", "status", "summary", "report")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("file", text="File")
        self.tree.heading("status", text="Status")
        self.tree.heading("summary", text="Findings")
        self.tree.heading("report", text="Report")
        self.tree.column("file", width=240, anchor="w")
        self.tree.column("status", width=140, anchor="w")
        self.tree.column("summary", width=200, anchor="w")
        self.tree.column("report", width=240, anchor="w")
        self.tree.tag_configure("ok", foreground="#2e7d32")
        self.tree.tag_configure("warn", foreground="#b26a00")
        self.tree.tag_configure("err", foreground="#c62828")
        self.tree.tag_configure("run", foreground="#1565c0")
        self.tree.tag_configure("dim", foreground="#777777")
        self.tree.bind("<Double-1>", lambda _e: self._open_selected())
        self.tree.bind("<Button-3>", self._popup_menu)

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.menu = tk.Menu(self.root, tearoff=False)
        self.menu.add_command(label="Open report", command=self._open_selected)
        self.menu.add_command(label="Remove", command=self._remove_selected)

        if _HAS_DND:
            self.tree.drop_target_register(DND_FILES)
            self.tree.dnd_bind("<<Drop>>", self._on_drop)

    def _build_file_buttons(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))
        self.add_button = ttk.Button(row, text="Add Files…", command=self._on_add)
        self.clear_button = ttk.Button(row, text="Clear", command=self.model.clear)
        self.add_button.pack(side="left")
        self.clear_button.pack(side="left", padx=(6, 0))
        if not _HAS_DND:
            hint = ttk.Label(row, text="(install tkinterdnd2 to drag files from Explorer)", foreground="#777777")
            hint.pack(side="right")

    def _build_settings(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Report", padding=8)
        frame.pack(fill="x", pady=(8, 0))

        self.output_var = tk.StringVar(value="beside")
        radio = ttk.Frame(frame)
        radio.pack(fill="x")
        ttk.Radiobutton(radio, text="Next to its WAV (same name, .csv)", value="beside",
                        variable=self.output_var, command=self._on_output).pack(anchor="w")
        ttk.Radiobutton(radio, text="In a folder", value="folder",
                        variable=self.output_var, command=self._on_output).pack(anchor="w")

        self.folder_row = ttk.Frame(frame)
        self.folder_label = ttk.Label(self.folder_row, text="No folder chosen", anchor="w")
        self.folder_label.pack(side="left", fill="x", expand=True)
        ttk.Button(self.folder_row, text="Choose…", command=self._on_choose_folder).pack(side="right")

        self.pauses_var = tk.BooleanVar(value=self.model.settings.with_pauses)
        ttk.Checkbutton(frame, text="Include pause identification in report",
                        variable=self.pauses_var, command=self._on_pauses).pack(anchor="w", pady=(6, 0))

    def _build_progress(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(frame, maximum=1.0)
        self.progress.pack(fill="x")
        self.status_label = ttk.Label(frame, text="Drop WAV files to begin.", anchor="w")
        self.status_label.pack(fill="x", pady=(4, 0))
        self.error_label = tk.Label(frame, text="", anchor="w", justify="left", fg="#c62828",
                                    wraplength=620, font=("TkFixedFont",))
        self.notes_label = tk.Label(frame, text="", anchor="w", justify="left", fg="#b26a00",
                                    wraplength=620)

    def _build_actions(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))
        self.save_unsaved_button = ttk.Button(row, text="Save Unsaved CSVs…", command=self._on_save_unsaved)
        self.save_unsaved_button.pack(side="left")
        self.cancel_button = ttk.Button(row, text="Cancel", command=self.model.cancel)
        self.go_button = ttk.Button(row, text="Go", command=self.model.go)
        self.go_button.pack(side="right")
        self.cancel_button.pack(side="right", padx=(0, 6))

    # MARK: model notifications

    def _notify(self) -> None:
        self._events.put(True)

    def _poll(self) -> None:
        drained = False
        try:
            while True:
                self._events.get_nowait()
                drained = True
        except queue.Empty:
            pass
        if drained:
            self._refresh()
        self.root.after(80, self._poll)

    def _refresh(self) -> None:
        model = self.model
        selected = self.tree.selection()
        for row in self.tree.get_children():
            self.tree.delete(row)
        for index, item in enumerate(model.items):
            if item.state == "unsaved":
                status = f"not saved: {item.why}"
            elif item.state == "failed":
                status = item.why or "failed"
            else:
                status = STATE_LABEL.get(item.state, item.state)
            self.tree.insert("", "end", iid=str(index),
                             values=(item.path.name, status, item.summary, str(item.csv or "")),
                             tags=(STATE_TAG.get(item.state, "dim"),))
        for iid in selected:
            if self.tree.exists(iid):
                self.tree.selection_add(iid)

        self._updating = True
        self.progress["value"] = model.progress
        self.status_label.configure(text=model.status)
        self.output_var.set(model.settings.output)
        self._set_error(model.error_text)
        self._set_notes(model.run_notes)
        if model.settings.output == "folder":
            self.folder_row.pack(fill="x", padx=(20, 0))
            self.folder_label.configure(
                text=model.settings.folder or "No folder chosen",
                foreground="#c62828" if not model.settings.folder else "")
        else:
            self.folder_row.pack_forget()

        running = model.running
        self.add_button.state(["disabled" if running else "!disabled"])
        self.clear_button.state(["disabled" if running or not model.items else "!disabled"])
        if running:
            self.go_button.pack_forget()
            self.cancel_button.pack(side="right", padx=(0, 6))
        else:
            self.cancel_button.pack_forget()
            self.go_button.pack(side="right")
            self.go_button.state(["!disabled"] if model.can_go else ["disabled"])
        if model.unsaved_count and not running:
            self.save_unsaved_button.pack(side="left")
        else:
            self.save_unsaved_button.pack_forget()
        self._updating = False

    def _set_error(self, text: str | None) -> None:
        if text:
            self.error_label.configure(text=text)
            self.error_label.pack(fill="x", pady=(4, 0))
        else:
            self.error_label.pack_forget()

    def _set_notes(self, notes: list[str]) -> None:
        if notes:
            self.notes_label.configure(text="\n".join(f"⚠ {n}" for n in notes))
            self.notes_label.pack(fill="x", pady=(4, 0))
        else:
            self.notes_label.pack_forget()

    # MARK: actions

    def _on_add(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add WAV files",
            filetypes=[("Audio files", "*.wav *.bwf"), ("All files", "*.*")])
        if paths:
            self.model.add(paths)

    def _on_drop(self, event) -> None:
        if self.model.running:
            return
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.model.add(paths)

    def _on_output(self) -> None:
        if not self._updating:
            self.model.set_output(self.output_var.get())

    def _on_pauses(self) -> None:
        if not self._updating:
            self.model.set_with_pauses(self.pauses_var.get())

    def _on_choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Save each CSV in this folder")
        if folder:
            self.model.set_folder(Path(folder))

    def _on_save_unsaved(self) -> None:
        folder = filedialog.askdirectory(title="Save unsaved CSVs in this folder")
        if folder:
            self.model.save_unsaved(Path(folder))

    def _selected_index(self) -> int | None:
        selection = self.tree.selection()
        if not selection:
            return None
        try:
            return int(selection[0])
        except ValueError:
            return None

    def _open_selected(self) -> None:
        index = self._selected_index()
        if index is None or not (0 <= index < len(self.model.items)):
            return
        csv = self.model.items[index].csv
        if csv:
            self._open(csv)

    def _remove_selected(self) -> None:
        index = self._selected_index()
        if index is not None:
            self.model.remove(index)

    def _popup_menu(self, event) -> None:
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.menu.tk_popup(event.x_root, event.y_root)

    def _open(self, path: Path) -> None:
        target = path if path.is_dir() else path.parent
        try:
            if sys.platform == "win32":
                os.startfile(str(target))               # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            messagebox.showerror("Could not open", str(exc))

    # MARK: dialogs

    def _confirm(self, count: int, action: str) -> bool:
        if count == 0:
            return True
        them = "it" if count == 1 else "them"
        return messagebox.askyesno(
            "Unsaved reports",
            f"{count} report{'' if count == 1 else 's'} {'is' if count == 1 else 'are'} not saved.\n\n"
            f"{action} will discard {them}. Use “Save Unsaved CSVs…” to keep {them}.",
            default=messagebox.NO, icon=messagebox.WARNING)

    def _offer_leftovers(self) -> None:
        if not self.model.has_leftovers():
            return
        answer = messagebox.askyesnocancel(
            "Unsaved reports",
            "Reports from an earlier session were never saved.\n\n"
            "Yes: save them to a folder now.  No: discard them.  Cancel: decide later.")
        if answer is True:
            folder = filedialog.askdirectory(title="Save earlier reports in this folder")
            if folder:
                self.model.save_leftovers(Path(folder))
        elif answer is False:
            self.model.discard_leftovers()

    def _on_close(self) -> None:
        if self.model.running:
            if not messagebox.askyesno(
                    "A check is still running",
                    "Quit now and stop it? Files already finished keep their CSVs; reports not yet "
                    "saved are offered again when the app next opens."):
                return
            self.model.stop_for_quit()
        self.root.destroy()


def run(engine_argv: list[str]) -> int:
    TkApp(engine_argv).root.mainloop()
    return 0


if __name__ == "__main__":
    from .cli import main

    raise SystemExit(main())
