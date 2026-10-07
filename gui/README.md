# FinalPass AudioBook — desktop GUIs

Small windows over `fpab`: drop WAV files (or folders) in, press **Go**, get one CSV per
WAV — the same per-file CSV `fpab check --csv-per-file` writes: a summary, the problem
events, then the informational events; a switch adds the pause map's rows.

Two frontends share one toolkit-agnostic core:

| frontend | toolkit | platforms | notes |
|---|---|---|---|
| `fpab_gui/gtk_app.py` | GTK 4 / PyGObject | Linux (also other desktops with GTK) | native; the default where GTK is importable |
| `fpab_gui/tk_app.py` | Tkinter (stdlib) | **Windows**, and anywhere else | no install on python.org Python; Explorer drag-and-drop needs the optional `tkinterdnd2` |

`fpab-gui` picks GTK where available and Tk otherwise; force one with `--ui gtk|tk` or
`FPAB_GUI_UI`. An existing file is never replaced, not even this app's own earlier report:
a new report gets `<name> (2).csv`, `(3)`, …. A CSV that cannot be placed is kept by the
app and offered again, never discarded unasked.

## Requires

- The engine as usual: `uv sync` and `uv run fpab setup-model` in the project root.
- A frontend toolkit:
  - **GTK**: a Python with PyGObject + GTK 4 (`python3-gi`, `gir1.2-gtk-4.0` on
    Debian/Ubuntu). The window runs on that interpreter, not the project `.venv`.
  - **Tkinter**: bundled with python.org Python and in the `.venv` here. Drag-and-drop
    from the file manager needs `pip install tkinterdnd2` (optional).

The GUI finds the engine in this order: `--engine PATH`, `$FPAB_ENGINE`, the project's
`.venv/bin/fpab`, `uv run fpab`, then `fpab` on `$PATH`.

## Run

```
gui/fpab-gui                 # Linux/macOS: GTK if available, else Tk
gui/fpab-gui --ui tk         # force Tkinter
gui\fpab-gui.cmd             # Windows
pythonw -m fpab_gui          # Windows, no console window
gui/install-desktop.sh       # Linux: add it to the applications menu
```

Set `FPAB_GUI_PYTHON` (launcher) to pick the interpreter if `python3` is not the one with
PyGObject.

## Layout

```
fpab_gui/engine.py     run `fpab check --progress jsonl`, parse the events   (toolkit-agnostic)
fpab_gui/placement.py  put a CSV where asked, never replacing a file         (toolkit-agnostic)
fpab_gui/config.py     settings + kept reports, per-user (XDG / %APPDATA%)   (toolkit-agnostic)
fpab_gui/runmodel.py   the list, the two settings, one run at a time         (toolkit-agnostic)
fpab_gui/cli.py        shared command line, picks the frontend               (toolkit-agnostic)
fpab_gui/gtk_app.py    the GTK4 window
fpab_gui/tk_app.py     the Tkinter window (Windows)
fpab-gui / fpab-gui.cmd  launchers
```

Everything but `gtk_app.py` and `tk_app.py` is pure Python with no toolkit import, so
another frontend (Qt, web) could reuse it.

## Differences from the macOS app

- **No App Sandbox.** The Mac app blocks the network at the OS level and needs
  security-scoped bookmarks to write beside files. Linux and Windows have neither, so
  placement is plain POSIX/Win32 and numbered names (`… (2).csv`) are always allowed
  without a folder grant. The offline guarantee is the engine's own Python-level network
  guard, shown after each run as `network attempts: N`. For an OS-level block on Linux,
  run under `bwrap`/`firejail` or ship as a Flatpak without `--share=network`.
- No app icon, no code signing/notarization, no Finder/Explorer integration (double-click
  a row or use **Open report** to open the CSV's folder with the desktop file manager).

## Tests

```
uv run pytest gui/tests            # core modules (no display needed)
python3 -m pytest gui/tests        # also runs the window-build tests, if a display exists
```
