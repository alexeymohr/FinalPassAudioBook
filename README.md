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
| Mouth-click inhale (a breath that opens with a click, after a pause since the word) | 3 when the click is loud (narration −20 dB or more), else 2 | calibrated on one title's operator labels; 32 of 32 held-out listings confirmed by ear |
| Breath (every breath, by its loudness against the narration) | quiet ones informational; 1 / 2 / 3 from −31.6 / −26.4 / −22.5 dB | a fixed scale set on one title's breaths (its 35th / 75th / 95th percentiles) |
| Plosive pop (a low thump below 100 Hz on its own, just before a word) | 1 / 2 / 3 from −42 / −34 / −26 dBFS | 10 of one title's 12 QC-noted pops; held out, 29 of 32 listings were plosives by ear |
| Click in a pause (a tick or click in the silence, 100 ms clear of words and breaths) | 3 | on one title's 12 chapters, the 2 ticks confirmed by ear and nothing else; a second title must confirm it |
| Digital tick (a spike above 16.5 kHz, a few samples long, standing alone) | 3 | a guard: none on a whole title; 40 of 40 planted spikes in pauses found |
| Dropout (dead digital silence for a frame or less, about 33 ms, inside audible sound) | 3 | a guard: none in 17 chapters of two titles; planted holes of 10 samples to 15 ms inside words all found |
| Word ends abruptly where a generated clip ends (local model) | 1 | four titles: found all 9 real chops on one; elsewhere only hard endings, nothing missing |
| Hum (a steady tone, also found in the pauses when speech covers it) | 1; 2 when strong (−55 dBFS or more); 3 when strong and it starts or stops abruptly | 12 of 12 listed hums confirmed by ear on one title |
| Noisy section (noise floor within 41 dB of the narration for about 5.5 s) | 3 / 2 / 1 as the floor comes within 25 / 32 / 41 dB of the speech | 6 of one title's 7 QC-noted noisy blocks |
| Pause map with a guess at each pause's kind | informational | two generic rule sets (`fpab rules`) |

Breath and noise levels are judged against the narration, so they follow the
master; the other checks use fixed dBFS limits. Quiet breaths are informational:
`report.json` and the per-file CSVs (`--csv-per-file`, and the app) list them
apart from the problems. `--min-sev 2` lists only severity 2 and 3;
`report.json` always keeps everything.

Breath analysis comes from [FinalPass](https://github.com/alexeymohr/FinalPass).

## Use

Needs [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 if needed) and git.
Tested on macOS.

```
uv sync                         # every check, including the chopped-word model
uv run fpab setup-model         # once: fetch and verify the model weights
uv run fpab check path/to/chapters/ --out report/
```

`fpab check` takes WAV (or BWF) files and folders of them (not searched
recursively); each file must be mono, or stereo with identical channels. It
writes `issues.txt`, `issues.csv`, `pauses.txt`, `pauses.csv` and `report.json`
to `--out` (default `./fpab-report`). `--csv-per-file` writes one CSV per WAV
beside it (or into `--csv-dir`): a summary, the problem events, then the
informational events; `--with-pauses` adds the pause map. `fpab check --help`
lists every option. Without the model installed, the chopped-word check is
skipped with a note. `fpab setup-model` is the only command that uses the
network.

A drag-and-drop macOS app (sandboxed, no network) that writes the same per-file
CSVs builds from [macos/](macos/README.md).

Tests use synthetic audio only: `uv run pytest`.

The model is [`mythicinfinity/speech-truncation-detection-12M`](https://huggingface.co/mythicinfinity/speech-truncation-detection-12M)
(Apache-2.0). It runs as our numpy port of its inference (no PyTorch); the
upstream torch code is vendored unmodified at a pinned revision as the reference
the port is tested against (`uv sync --extra truncation` installs PyTorch for
those tests; runs never need it). The weights are verified against a recorded
SHA-256 on every load and read without pickle or remote code.

Plan and calibration notes: [docs/PLAN.md](docs/PLAN.md).

License: MIT. The vendored model code keeps its Apache-2.0 licence
(`src/finalpass_audiobook/vendor/speech_truncation/LICENSE`).
