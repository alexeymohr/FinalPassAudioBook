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

The GUI finds `fpab.exe` beside itself and points it at `{app}\models` via
`FPAB_MODEL_DIR`, so the installed app needs no network, no `.venv`, no `~/.cache`.

## Build (on Windows)

Needs [uv](https://docs.astral.sh/uv/) and
[Inno Setup 6](https://jrsoftware.org/isinfo.php).

```
powershell -ExecutionPolicy Bypass -File windows\build.ps1
```

It syncs the locked environment, downloads and verifies the models, installs PyInstaller,
builds the two exes into `windows\dist\FinalPassAudioBook\`, stages `windows\staging\models\`,
and runs Inno Setup. Output: `windows\dist\FinalPassAudioBook-<version>-setup.exe`.

## Or build in CI

`.github/workflows/windows-installer.yml` runs the same build on `windows-latest`
(triggered by a tag like `v0.2.4`, or manually) and uploads the setup exe as an artifact.

## Notes

- **Unsigned.** The setup and the app are not code-signed, so SmartScreen shows a
  "unknown publisher" prompt. Signing needs a Windows code-signing certificate; add
  `SignTool` to `installer.iss` and sign the exes in `build.ps1` if you have one.
- The exes are **onedir** (a folder with the two exes and their DLLs), not single-file,
  so startup is fast; the installer packages that folder.
- Explorer drag-and-drop needs the optional `tkinterdnd2` package. To include it, add it
  to the build environment (`uv pip install tkinterdnd2`) before PyInstaller runs; the
  spec collects it automatically when present.
