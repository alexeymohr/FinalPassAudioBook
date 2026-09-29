# FinalPassAudioBook

Local, offline QC for AI-narrated audiobook chapters. It lists what a QC
reviewer is likely to notice — with a time, a plain description and a severity
from 1 (worth a listen) to 3 (worst) — so a mixer can go straight to it. It is a
triage tool, not a gate: nothing it reports fails a file.

Everything runs on your machine. No audio is uploaded, transcribed or sent
anywhere; every analysis run refuses network access and records that it made
none.

## What it checks

| check | severity | status |
|---|---|---|
| Mouth-click inhale (breath opening with a click) | 3 after a clear gap and loud; 2 after a gap, quiet; 1 no gap (a hard consonant may run into the inhale) | calibrated on one title's operator labels, heard in context |
| Loud breath | 2 | calibrated on one title's operator labels |
| Possible chopped word (local model) | 3 | evaluated on two titles |
| Hum (a steady tone, also found in the pauses when speech covers it) | 3 | 12 of 12 listed hums confirmed by ear on one title; 7 lines that tracking alone found were not hum and are no longer listed |
| Noisy section (floor within 41 dB of the speech ≥ 3 s) | 3 / 2 / 1 by how close the floor comes to the speech | one title's QC room-tone notes |
| Sound cutting out abruptly (at least speech −31 dB) | 3 / 2 / 1 by the level that cuts out | first guess; needs real examples |
| Plosive pop (burst below 65 Hz louder than the speech) | 3 / 2 / 1 by how much louder | first guess; needs real examples |
| Pause map with a guess at each pause's kind | informational | two generic rule sets (`fpab rules`) |

Levels are judged against the speech around each spot (±10 s), not fixed dBFS,
so the checks follow the master. `--min-sev 2` lists only severity 2 and 3;
`report.json` always keeps everything.

Breath analysis comes from [FinalPass](https://github.com/alexeymohr/FinalPass).

## Use

A drag-and-drop macOS app (sandboxed, no network) is in [macos/](macos/README.md).

```
uv sync                         # every check, including the chopped-word model
uv run fpab setup-model         # once: fetch and verify the model weights
uv run fpab check path/to/chapters/ --out report/
```

Writes `issues.txt`, `issues.csv`, `pauses.txt`, `pauses.csv` and `report.json`.
Without the model installed, the chopped-word check is skipped with a note.

The model is [`mythicinfinity/speech-truncation-detection-12M`](https://huggingface.co/mythicinfinity/speech-truncation-detection-12M)
(Apache-2.0). It runs as our numpy port of its inference (no PyTorch); the
upstream torch code is vendored unmodified at a pinned revision as the reference
the port is tested against (`uv sync --extra truncation` installs it). The
weights are verified against a recorded SHA-256 on every load and read without
pickle or remote code.

Plan and calibration notes: [docs/PLAN.md](docs/PLAN.md).

License: MIT.
