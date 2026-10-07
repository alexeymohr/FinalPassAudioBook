# Architecture

How the project is put together: the analysis pipeline, the checks, the models, the
network guard, and the desktop apps. For *why* the thresholds are what they are, see
[PLAN.md](PLAN.md); for using the tool, see the [README](../README.md).

## Shape

One Python package does the analysis; thin frontends drive it.

```
CLI (cli.py) ─────────────┐
GTK app  (gui/fpab_gui)   ├─► run() ──► checks ──► RunReport ──► output files
Tk app   (gui/fpab_gui)   │        (run.py)         (findings.py)  (output.py)
macOS app (macos/, Swift) ┘
```

Every frontend ends up calling the same engine: `fpab check …` runs `run()` over a set
of chapter files, inside the [network guard](#the-network-guard), and writes the same
output. Nothing about the analysis lives in a frontend.

## Repository layout

```
src/finalpass_audiobook/     the package (analysis + CLI)
  cli.py                     the `fpab` command line (click)
  run.py                     run every check over files, inside the network guard
  chapter.py                 one WAV loaded once, with the measurements checks share
  activity.py                where narration sounds and where it pauses
  checks/                    one module per check
  findings.py                the report data model (Finding, Pause, FileResult, RunReport)
  output.py                  write issues/pauses as text and CSV, everything as JSON
  netguard.py                fail-closed network/process guard for every run
  model.py, breath_model.py  install/verify/load the two local model weights
  truncation_np.py, breath_np.py  the two models as numpy (no PyTorch)
  vendor/                    upstream model code, vendored and pinned (see PROVENANCE)
  rules.py                   generic pause rule sets (a guess, informational only)
tests/                       pytest; synthetic audio only
gui/                         Linux/Windows desktop apps over the engine
windows/                     Windows installer build (PyInstaller + Inno Setup)
macos/                       macOS app (Swift/SwiftUI)
docs/                        this file, PLAN.md
```

## The pipeline

`cli.check` (in `cli.py`) expands the paths, plans where each CSV goes, and calls
`run.run` (`run.py`). `run()`:

1. Enters a `NetworkGuard`.
2. Loads the models (truncation, then breath), each time recording whether it ran.
3. For each file, calls `analyze_file`, catching any failure so one bad file is skipped
   with a note and never the batch.
4. Returns a `RunReport` (pydantic models in `findings.py`).

`analyze_file` loads a `Chapter` once (`chapter.py`), then runs the checks in a fixed
order (`STAGES`): `loading → breaths → pauses → hum → noise → dropouts → plosives →
clicks → ticks → chopped words`. Findings whose spans overlap are de-duplicated (for
example, a tick at a dropout's edge is the dropout's own edge, not a second event).
Everything comes back as `Finding`s with a time span, a plain-language `problem`, a
`severity` (3 worst, 1 lowest, 0 = informational) and the raw `measures`.

`Chapter` does the shared work: mono or identical dual-mono, invalid-sample detection,
a 1 ms energy envelope, a decimated signal for low-frequency checks, and the narration
level. Everything is numbers — nothing reads, transcribes or interprets the words.

## The checks

Each check is a module in `checks/` with its own frozen `Tunables` dataclass and a
`*_findings` function returning `Finding`s. The checks and their thresholds are listed
in the [README](../README.md#what-it-checks) table; the calibration is in
[PLAN.md](PLAN.md) §3.

| module | check |
|---|---|
| `breaths.py` | breath, mouth-click inhale, breath cut off into silence |
| `pauses.py` | pause map (informational) |
| `hum.py` | steady tone |
| `noise.py` | noisy section |
| `dropouts.py` | dead-silence dropout |
| `plosives.py` | plosive pop |
| `clicks.py` | click in a pause |
| `ticks.py` | digital tick |
| `truncation.py` | word ends abruptly at a clip end (uses the model) |

Adding a check: put it in `checks/`, give it tunables, return `Finding`s, wire it into
`STAGES` and `analyze_file` in `run.py`, and add it to the README table and PLAN.

## The models

Two local models run as numpy ports — no PyTorch at analysis time:

- **Chopped-word** (`model.py`, `truncation_np.py`): `mythicinfinity/speech-truncation-detection-12M`.
- **Breath** (`breath_model.py`, `breath_np.py`): Respiro-en fine-tuned, published as this repo's `breath-model-v2` release.

Weights are installed once by `fpab setup-model` (the only network user), verified
against a recorded size and SHA-256 on every install and every load, and read as
safetensors without pickle or remote code. The upstream Torch code is vendored
unmodified (`vendor/`) and used only as the reference the numpy ports are tested
against.

Where the weights live: `$FPAB_MODEL_DIR` if set, else `~/.cache/finalpass-audiobook`,
in a revision-named subfolder. A frozen/installed desktop app ships a `models/` folder
beside its executable and passes `FPAB_MODEL_DIR` so it runs fully offline.

## The network guard

`netguard.py` installs a `sys.addaudithook` while analysis runs. It refuses (and counts)
socket creation and connects, name lookups, starting processes, and reaching C network
or process functions through `ctypes`. Every run records the count, which must be `0`.

It is a guard *inside Python*, not a sandbox: native code calling the OS directly, or a
socket opened before the guard, is beyond it. The macOS app's OS sandbox is the hard
boundary; the CLI can be wrapped in `sandbox-exec`/`bwrap` too.

## Outputs

`output.py` writes the run report (`RUN_FILES`): `issues.txt`/`.csv`, `pauses.txt`/`.csv`
and `report.json`. `--csv-per-file` (and the desktop apps) write one CSV per WAV
instead: a summary, then problem events, then informational events. A per-file CSV is
never replaced — a new one gets `<name> (2).csv`, `(3)`, ….

`report.json` follows `findings.SCHEMA_VERSION`; it keeps every measure, while the CSVs
show the few a mixer uses (`CSV_MEASURES`).

## The desktop apps

The desktop apps share a toolkit-agnostic core in `gui/fpab_gui/`:

| module | role |
|---|---|
| `engine.py` | run `fpab check --progress jsonl`, parse the events; find the engine (explicit, `$FPAB_ENGINE`, a sibling `fpab.exe`, `.venv`, `uv`, `$PATH`); `app_dir()`/`bundled_model_dir()` for frozen installs |
| `placement.py` | put each finished CSV where asked, never replacing a file |
| `config.py` | settings and kept reports, per-user (XDG on Linux/macOS, `%APPDATA%`/`%LOCALAPPDATA%` on Windows) |
| `runmodel.py` | the file list, the settings, one run at a time, unsaved-report safety |
| `cli.py` | shared command line; picks the frontend |

Only the two frontends import a toolkit: `gtk_app.py` (GTK 4, Linux) and `tk_app.py`
(Tkinter, Windows). `cli.py` prefers GTK where it is importable and uses Tk otherwise.
Because the core holds no toolkit code, another frontend could reuse it.

The macOS app is separate (`macos/`, Swift/SwiftUI); it bundles the engine and models
into a sandboxed `.app`.

## Builds and CI

- `tests.yml` runs the test suite on Linux, macOS and Windows on every push/PR.
- `macos/build_app.sh` builds the macOS `.app`; `macos/release.sh` notarizes and wraps it.
- `windows/build.ps1` builds the Windows exes (PyInstaller spec `windows/fpab-windows.spec`)
  and the installer (`windows/installer.iss`, Inno Setup); `.github/workflows/windows-installer.yml`
  runs it on `windows-latest` and uploads the setup exe.

## Design constraints

These shape everything above (full list in [CLAUDE.md](../CLAUDE.md)):

- **Client audio never leaves the machine.** No upload, no transcription, no cloud.
  Only numbers (times, dBFS, scores) are reported. No client identifiers in the repo.
- **Numbers only.** No model outputs text, words, or anything that could give back the
  audio.
- **Thresholds are named and reported.** Every tunable is in `report.json`.
- **Supply chain.** Nothing published less than 7 days ago may resolve
  (`pyproject.toml` → `[tool.uv] exclude-newer`); ML deps are optional; model code is
  vendored after a full read.
