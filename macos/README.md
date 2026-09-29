# FinalPass AudioBook — macOS app

A small window over `fpab`: drop WAV files (or folders) in, press **Go**, get one CSV
per WAV. Same findings and columns as `fpab check`'s issues.csv; a checkbox adds the
pause map's rows.

- **Sandboxed, no network.** App Sandbox is on and the app has no network entitlement,
  so macOS itself refuses every connection. The analysis engine (Python with fpab, its
  locked dependencies and the verified model weights) is bundled inside the app and
  runs within the same sandbox. It can read only the files you give it. The
  chopped-word model runs as numpy (no PyTorch), held to the audited torch code.
- **Where CSVs go.** Next to each WAV as `<name>.csv` (macOS lets a sandboxed app
  create a same-name file beside one it was given), or into a folder you choose once.

## Build

```
uv sync --extra truncation      # the repo's environment
uv run fpab setup-model         # the model weights, verified
macos/build_app.sh              # -> macos/build/FinalPass AudioBook.app (about 185 MB)
```

The script installs exactly the versions in `uv.lock`, copies the real Python
interpreter (never through a link), precompiles it, refuses to continue if any link
points outside the bundle, and signs everything ad hoc for this Mac.

Scripted check of a build (runs, writes the CSVs, quits; result in the app's container
at `tmp/autorun_result.txt`):

```
open -W --env FPAB_AUTORUN=1 -a "macos/build/FinalPass AudioBook.app" a.wav b.wav
```
