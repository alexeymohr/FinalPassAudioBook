# FinalPass AudioBook — Linux / desktop GUI

A small GTK4 window over `fpab`: drop WAV files (or folders) in, press **Go**, get one
CSV per WAV — the same CSV `fpab check --csv-per-file` writes: a summary, the problem
events, then the informational events; a switch adds the pause map's rows.

This is the Linux counterpart of the macOS app in `../macos`. An existing file is never
replaced, not even this app's own earlier report: a new report gets `<name> (2).csv`,
`(3)`, …. A CSV that cannot be placed is kept by the app and offered again, never
discarded unasked.

## Requires

- A Python with **PyGObject + GTK 4** (`python3-gi`, `gir1.2-gtk-4.0`; on Debian/Ubuntu
  `python3-gi`). The GUI runs on that interpreter, not on the project's `.venv`.
- The engine as usual: `uv sync` and `uv run fpab setup-model` in the project root.

The GUI finds the engine in this order: `--engine PATH`, `$FPAB_ENGINE`, the project's
`.venv/bin/fpab`, `uv run fpab`, then `fpab` on `$PATH`.

## Run

```
gui/fpab-gui                 # or: ./fpab-gui --engine /path/to/fpab
gui/install-desktop.sh       # optional: add it to the applications menu
```

Set `FPAB_GUI_PYTHON` to pick the interpreter if `python3` is not the one with PyGObject.

## Layout

```
fpab_gui/engine.py     run `fpab check --progress jsonl`, parse the events  (toolkit-agnostic)
fpab_gui/placement.py  put a CSV where asked, never replacing a file        (toolkit-agnostic)
fpab_gui/config.py     settings + kept reports, under the XDG dirs          (toolkit-agnostic)
fpab_gui/runmodel.py   the list, the two settings, one run at a time        (toolkit-agnostic)
fpab_gui/gtk_app.py    the GTK4 window (drop zone, list, settings, progress)
fpab-gui               launcher (system Python + PyGObject)
```

The first four modules are pure Python with no toolkit import, so another frontend
(Qt, tk) can reuse them; only `gtk_app.py` is GTK-specific.

## Differences from the macOS app

- **No App Sandbox.** The Mac app blocks the network at the OS level and needs
  security-scoped bookmarks to write beside files. Linux has neither, so placement is
  plain POSIX and numbered names (`… (2).csv`) are always allowed without a folder grant.
  The offline guarantee is the engine's own Python-level network guard, shown after each
  run as `network attempts: N`. For an OS-level block, run the app under `bwrap`/`firejail`
  or ship it as a Flatpak without `--share=network`.
- No app icon, no code signing/notarization, no Finder integration (the **Show** button
  opens the CSV's folder with the desktop's file manager).

## Tests

```
uv run pytest gui/tests            # core modules (no display needed)
python3 -m pytest gui/tests        # also runs the GTK window build, if a display exists
```
