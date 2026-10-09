# A one-click Windows installer

Builds `FinalPassAudioBook-<version>-setup.exe`: a per-user installer (no admin) that
drops a Start Menu entry (and an optional desktop shortcut) and installs the GUI, the
engine, and both model weights so the app runs fully offline. Double-click the setup,
then launch from the Start Menu.

## What goes in

| piece | how | where |
|---|---|---|
| `fpab-gui.exe` | PyInstaller, Tkinter frontend | `{app}\fpab-gui.exe` |
| `fpab.exe` | PyInstaller, the engine CLI | `{app}\fpab.exe` |
| model weights | `fpab setup-model` output | `{app}\models\` |
| licences | `licenses.py`, from the spec: the licence texts of every bundled component | `{app}\Licenses\` |

The GUI finds `fpab.exe` beside itself and points it at `{app}\models` via
`FPAB_MODEL_DIR`, so the installed app needs no network, no `.venv`, no `~/.cache`.

## Build (on Windows)

Needs [uv](https://docs.astral.sh/uv/), git (uv fetches the pinned FinalPass library with
it) and [Inno Setup](https://jrsoftware.org/isinfo.php) 6.3 or later (a per-user install is
found too).

```
powershell -ExecutionPolicy Bypass -File windows\build.ps1
```

It syncs the locked environment with the `windows-build` extra (PyInstaller, pinned and held
to the 7-day rule like every other dependency), downloads and verifies the models, builds the
two exes into `windows\dist\FinalPassAudioBook\`, gathers the licence texts of everything they
bundle into `windows\staging\Licenses\`, stages the two verified weight files in
`windows\staging\models\`, and runs Inno Setup (6.3 or later). Output: `windows\dist\FinalPassAudioBook-<version>-setup.exe`.

## Or build in CI

`.github/workflows/windows-installer.yml` runs the same build on `windows-latest` (real x64)
whenever a release is published and attaches the setup exe and its `.sha256` to that release.
Run it by hand from the Actions tab with a release tag to (re)attach to that release, or with
no tag to build the current branch as a downloadable artifact only.

## Notes

- **Experimental and unsupported.** The setup opens with `EXPERIMENTAL.txt`, which says so
  in plain words before anything is installed.
- **Unsigned.** The setup and the app are not code-signed, so SmartScreen shows a
  "unknown publisher" prompt. Signing needs a Windows code-signing certificate; add
  `SignTool` to `installer.iss` and sign the exes in `build.ps1` if you have one.
- The exes are **onedir** (a folder with the two exes and their DLLs), not single-file,
  so startup is fast; the installer packages that folder.
- Explorer drag-and-drop needs the optional `tkinterdnd2` package, which is not part of the
  locked build; "Add Files…" always works without it.
