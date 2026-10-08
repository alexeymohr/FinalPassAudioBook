# FinalPassAudioBook

Local, offline QC for AI-narrated audiobook chapters. It lists what a QC
reviewer is likely to notice — with a time, a plain description and a severity
from 1 (worth a listen) to 3 (worst) — so a mixer can go straight to it. It is a
triage tool, not a gate: nothing it reports fails a file.

Everything runs on your machine. No audio is uploaded, transcribed or sent
anywhere. The macOS app runs in the OS sandbox with no network access at all.
The command line guards every analysis run from inside Python: it refuses the
network and process routes it knows of and records the count, which must be 0
(a Python-level guard, not a sandbox). For an OS-level block there too on macOS, run it under
`sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/fpab check …`.

## What it checks

| check | severity | status |
|---|---|---|
| Mouth-click inhale (a breath that opens with a click, after a pause since the word) | 3 when the click is loud (narration −20 dB or more), else 2; a click more than 40 dB under the narration is not counted | calibrated on one title's operator labels; 32 of 32 held-out listings confirmed by ear; with the breath model, 43 of 43 labelled ones kept and five of a second voice's seven false ones gone |
| Breath (every breath, by its loudness against the narration), confirmed by a local breath model so a consonant left alone at a word's end ("t", "k", "p", "ch", "s") is not listed | quiet ones informational; 1 / 2 / 3 from −31.6 / −26.4 / −22.5 dB | a fixed scale set on one title's breaths (its 35th / 75th / 95th percentiles); on tagged events never used to build the model: breaths kept 18 of 18 and 16 of 16 on two titles, consonants listed 5 of 28 (4 of them breathy), 0 of 14 on a breathless test voice |
| Breath cut off into silence (with the breath model: a breath running straight into a 50–80 ms hole of exact digital silence before the next word, not already listed) | informational | on one title's QC report, 8 of the breaths QC removed sat at such a hole; about 1 per 15 minutes there, 0.4 on a second title; heard by ear, none of 24 others was a defect on its own |
| Plosive pop (a low thump below 100 Hz on its own, just before a word) | 1 / 2 / 3 from −42 / −34 / −26 dBFS | 10 of one title's 12 QC-noted pops; held out, 29 of 32 listings were plosives by ear; at a generated clip's start (right after digital silence) 9 more found on 70 files, all 9 confirmed by ear |
| Click in the silence (between words, and in the room tone before the first word and after the last; 100 ms clear of words and breaths) | 3; 2 when it stands less than 30 dB over the file's room-tone floor | on one title's 12 chapters, the 2 ticks confirmed by ear and nothing else; on two titles, 15 of 15 heard confirmed, severities matched by ear |
| Digital tick (a spike above 16.5 kHz, a few samples long, standing alone) | 3 | a guard: none on a whole title; 40 of 40 planted spikes in pauses found |
| Dropout (dead digital silence for a frame or less, about 33 ms, inside audible sound) | 3 | a guard: none in 17 chapters of two titles; planted holes of 10 samples to 15 ms inside words all found |
| Word ends abruptly where a generated clip ends (local model) | 1 | four titles: found all 9 real chops on one; elsewhere only hard endings, nothing missing |
| Hum (a steady tone, also found in the pauses when speech covers it) | 1; 2 when strong (its loudest line, harmonics too, shown as −55 dBFS or more); 3 when strong and it starts or stops abruptly | 12 of 12 listed hums confirmed by ear on one title |
| Noisy section (noise floor within 41 dB of the narration for about 5.5 s) | 3 / 2 / 1 as the floor comes within 25 / 32 / 41 dB of the speech | 6 of one title's 7 QC-noted noisy blocks |
| File problem (corrupt or out-of-range samples, audio cut short of its header, no narration found, under 1 s) | 3 | checked on synthetic files |
| Pause map with a guess at each pause's kind | informational | two generic rule sets (`fpab rules`) |

Breath and noise levels are judged against the narration, so they follow the
master; the other checks use fixed dBFS limits. Severities are graded on the
values the report shows. Quiet breaths are informational: `report.json` and the
per-file CSVs (`--csv-per-file`, and the app) list them apart from the problems.
`--min-sev 2` lists only severity 2 and 3 in `issues.*`; the per-file CSVs and
`report.json` always keep everything.

Breath analysis comes from [FinalPass](https://github.com/alexeymohr/FinalPass); each breath it finds is then
confirmed by the breath model (below). Without the model, breaths are listed as FinalPass finds them, minus sounds as
loud as speech, and every report says "breath model: off".

## Use

Needs [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 if needed) and git.
Runs on macOS, Windows and Linux: the test suite runs on all three on every push.

```
uv sync                         # every check, including both local models
uv run fpab setup-model         # once: fetch and verify both models' weights
uv run fpab check path/to/chapters/ --out report/
```

`fpab check` takes WAV (or BWF) files and folders of them (not searched
recursively); each file must be mono, or have identical channels. It writes
`issues.txt`, `issues.csv`, `pauses.txt`, `pauses.csv` and `report.json` to
`--out` (default `./fpab-report`, or, with `--csv-per-file`, only when `--out`
is given); it never replaces files there that it did not write. `--csv-per-file`
writes one CSV per WAV beside it (or into `--csv-dir`): a summary, the problem
events, then the informational events; `--with-pauses` adds the pause map. A
per-file CSV never replaces an existing file, not even an earlier report: a new one
gets `<name> (2).csv`, `(3)`, …. (The run report in `--out` is the one exception: the
next run replaces its own files there.) `fpab check --help` lists every option.
Without a model installed, its check is skipped (the chopped-word check) or runs
without it (the breath model), with a note; `--no-breath-model` and `--no-truncation`
switch them off. It exits
1 when a file was skipped or a report could not be written, and 2 when nothing could
be checked (no audio found, an unusable `--out`); with `--progress jsonl`, which the
app uses, it exits 0 once the run completes and reports problems per file.
`fpab setup-model` is the only command that uses the network.

## Desktop apps

Drop WAV files (or folders) in, press Go, and get the same per-file CSVs.

- **macOS:** a sandboxed, no-network drag-and-drop app from [macos/](macos/README.md);
  releases carry the signed, notarized DMG.
- **Linux:** a GTK 4 app; **Windows:** a Tkinter app — both from [gui/](gui/README.md)
  (`gui/fpab-gui`, or `gui\fpab-gui.cmd` on Windows). Experimental: no OS sandbox (the
  engine's network guard still applies), and not yet part of a release.
- **Windows installer:** a one-click, per-user setup that bundles the GUI, the engine and
  both models, built from [windows/](windows/README.md) — unsigned, and built on demand,
  not with releases.

The Linux and Windows apps were contributed by Paul Philippov
([@themactep](https://github.com/themactep)).

Tests use synthetic audio only: `uv run pytest`.

Developer docs: [CONTRIBUTING.md](CONTRIBUTING.md) (setup, tests, conventions) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how the code is put together).

Two local models, both run as our numpy ports (no PyTorch), their upstream torch
code vendored unmodified at a pinned revision as the reference the ports are tested
against; the weights are verified against a recorded SHA-256 on every load and read
without pickle or remote code:

- The chopped-word model is [`mythicinfinity/speech-truncation-detection-12M`](https://huggingface.co/mythicinfinity/speech-truncation-detection-12M)
  (Apache-2.0); `uv sync --extra truncation` installs PyTorch for its reference tests
  (runs never need it).
- The breath model is [Respiro-en](https://github.com/ydqmkkx/Respiro-en) (Yang,
  Koriyama & Saito, Interspeech 2024; MIT), a frame-wise breath detector, with its
  weights fine-tuned for narration breaths and published as this repository's
  `breath-model-v2` release (weights only: no audio, labels or names). How they were
  made and judged: [docs/PLAN.md](docs/PLAN.md) §3.1 and
  `src/finalpass_audiobook/vendor/respiro/PROVENANCE.md`.

Plan and calibration notes: [docs/PLAN.md](docs/PLAN.md).

License: MIT. The vendored model code keeps its own licence (Apache-2.0,
`src/finalpass_audiobook/vendor/speech_truncation/LICENSE`; MIT,
`src/finalpass_audiobook/vendor/respiro/LICENSE`).
