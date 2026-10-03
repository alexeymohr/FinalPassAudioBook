# FinalPass AudioBook — macOS app

A small window over `fpab`: drop WAV files (or folders) in, press **Go**, get one CSV
per WAV, the same as `fpab check --csv-per-file` writes: a summary, the problem events,
then the informational events; a checkbox adds the pause map's rows.

- **Sandboxed, no network.** App Sandbox is on and the app has no network entitlement,
  so macOS itself refuses every connection. The analysis engine (Python with fpab, its
  locked dependencies and both models' verified weights) is bundled inside the app and
  runs within the same sandbox. It can read only the files you give it. The
  chopped-word model and the breath model run as numpy (no PyTorch), held to their
  audited torch code.
- **Where CSVs go.** Next to each WAV as `<name>.csv` (macOS lets a sandboxed app
  create a same-name file beside one it was given), or into a folder you choose once.
  An existing file is never replaced, not even this app's own earlier report: a new
  report gets `<name> (2).csv`, `(3)`, …. Beside the WAVs that needs access to the folder,
  which the app asks for once per folder and remembers. A CSV that cannot be placed is kept in the app until you save it
  ("Save Unsaved CSVs…"); the app asks before Go, Remove or Clear discard it, and any
  still unsaved when the app quits are offered again at the next launch.

## Build

Needs macOS 14 or later, Xcode 26 or later (Swift 6; its `actool` compiles the Icon
Composer app icon) and a uv-managed Python 3.12 (the script refuses any other).

```
uv sync --managed-python        # the repo's environment
uv run fpab setup-model         # both models' weights, verified
macos/build_app.sh              # -> macos/build/FinalPass AudioBook.app (about 190 MB)
```

The script installs exactly the versions in `uv.lock` (and the hash-pinned build
backends in `build-constraints.txt`), copies the real Python interpreter (never
through a link), precompiles it, refuses to continue if any link points outside the
bundle, any file names the folders it was built from (the repo, the Python and the
model it copied) or an untracked file sits in the package, gathers every bundled
component's licence texts into `Contents/Resources/Licenses`, and signs everything ad hoc
for this Mac.

## Release (for other Macs)

```
macos/build_app.sh --sign "Developer ID Application: NAME (TEAM)"
macos/release.sh                # -> macos/build/FinalPass-AudioBook-<version>.dmg
```

`--sign` signs every binary with the Developer ID, a secure timestamp and the hardened
runtime (the sandbox entitlements stay exactly as they are). `release.sh` notarizes the
app with Apple, staples the ticket into it, wraps it in a DMG (the app and a link to
Applications) that is itself signed, notarized and stapled, and checks both the way
Gatekeeper on a downloader's Mac will. It needs a notarization profile in the keychain,
made once (it asks for an app-specific password and stores it; the script never sees it):

```
xcrun notarytool store-credentials fpab-notary --apple-id APPLE_ID --team-id TEAM
```

It refuses a test-hook build or one not signed with a Developer ID.

Scripted check (only in a build made with `macos/build_app.sh --test-hooks`; the normal
build has no hook): runs, writes the CSVs, quits; result in the app's container at
`tmp/autorun_result.txt` (a Developer-ID-signed build's container is protected: reading it
from a terminal asks for permission, so check the CSVs beside the WAVs instead).

```
open -W --env FPAB_AUTORUN=1 -a "macos/build/FinalPass AudioBook.app" a.wav b.wav
```
