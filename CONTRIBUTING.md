# Contributing

Thanks for helping. This project is local, offline QC for AI-narrated audiobook
chapters. A few rules are stricter than usual — they are the reason the tool can be
trusted with client audio — so please read the short list below before the mechanics.

## Principles

- **Client audio never leaves the machine.** Never listen to, transcribe, or send the
  audio content anywhere; only numbers (times, dBFS, scores) are reported. This applies
  to issues, tests and pull requests: no real audio, no titles, no client names.
- **Numbers only.** No model output text, words, speaker identity or generated audio.
- **Thresholds are named and reported.** Every tunable appears in `report.json`; a new
  constant belongs in a `Tunables` dataclass, not inline.
- **Supply chain.** Nothing published less than 7 days ago may resolve — the cut-off is
  `[tool.uv] exclude-newer` in `pyproject.toml`; move it forward deliberately, never to
  within 7 days of today. ML dependencies stay an optional extra, never the base install.
- **Honest reporting.** Say plainly what failed or was not verified.

Full context, and why each exists: [CLAUDE.md](CLAUDE.md) and
[PLAN.md](docs/PLAN.md).

## Setup

Needs [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 if needed) and git.

```
uv sync                     # the package, tests and both numpy models (no PyTorch)
uv run fpab setup-model     # once: fetch and verify both models' weights (~62 MB)
uv run fpab check --help
```

## Everyday commands

```
uv run fpab check path/to/chapters/ --out report/     # analyse a folder
uv run fpab check --csv-per-file --with-pauses a.wav  # one CSV per file beside it
uv run fpab rules                                     # list pause rule sets
uv run pytest                                         # the whole suite (synthetic audio)
```

The suite also runs per-file: `uv run pytest tests/test_clicks.py`. Tests marked
`real_models` need the weights installed and are skipped where there are none.

The Python package lives in `src/finalpass_audiobook/` (src layout); see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map and the pipeline.

## Code conventions

- Match the surrounding code and keep the same voice in docstrings and messages.
- Library code returns data; only `cli.py` prints.
- Keep errors explicit (no silent failure paths) and avoid broad `except`.
- Add tests for new behaviour, using synthesized audio only (`tests/synth.py`,
  `tests/fixtures/`).
- Update the docs when behaviour or an operator workflow changes: the check table in
  [README.md](README.md), and [PLAN.md](docs/PLAN.md) for calibration.

## Adding a check

1. Add `src/finalpass_audiobook/checks/<name>.py` with a frozen `Tunables` dataclass
   (with `as_dict()`) and a function returning `Finding`s (`findings.py`).
2. Wire it into `STAGES` and `analyze_file` in `run.py`, and add its measures to
   `CSV_MEASURES` in `output.py` if the CSV needs a specific subset.
3. Add a row to the README table and a calibration note to PLAN.md §3.
4. Test it with synthetic audio; be explicit about what the check can and cannot see.

## Working on the desktop apps

The apps under `gui/` share a toolkit-agnostic core; only `gtk_app.py` and `tk_app.py`
import a toolkit. That is deliberate — keep logic in the core so every frontend gets it.

```
gui/fpab-gui               # GTK if available, else Tk
gui/fpab-gui --ui tk       # force Tkinter
uv run pytest gui/tests    # core + window-build tests (the GTK one skips without a display)
```

`gui/README.md` has the full picture; `windows/README.md` covers the installer build.

## Packaging

- macOS: `macos/build_app.sh` (Xcode + a uv-managed Python 3.12); see `macos/README.md`.
- Windows: `windows/build.ps1` (uv + Inno Setup 6) builds the exes and the installer;
  `.github/workflows/windows-installer.yml` runs the same in CI and attaches the installer to
  each published release.

Both bundle the model weights so the installed app runs offline.

## Pull requests

- One focused change per PR, with a clear description of what was verified.
- Run `uv run pytest` (and `uv run pytest gui/tests` for GUI changes) before opening it.
- Actions are pinned to commit hashes; if you add or bump one, pin it.
- By contributing you agree your work is licensed under the project's MIT licence
  (vendored model code keeps its own licence; see [LICENSE](LICENSE)).
