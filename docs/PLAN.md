# FinalPassAudioBook — design and plan

**Status (2026-09-29):** milestone 1 built, followed by the severity scale, the hum
work, the macOS app, the numpy model port and a four-part adversarial audit with
its fixes. Evidence below is from one delivered audiobook (12 chapters used for
calibration), reported only as anonymous aggregates; the client-specific record
is kept privately, outside this repository.

## 1. What it does

```
fpab check CHAPTERS_OR_FOLDERS... [--rules standard|no-paragraph] [--out DIR] [--min-sev 1|2|3]
           [--csv-per-file] [--csv-dir DIR] [--with-pauses] [--progress text|jsonl] [--no-truncation]
fpab setup-model [--from-file model.safetensors]
fpab rules
```

- `issues.txt` / `issues.csv`: what a QC reviewer is likely to notice, per file, in
  time order (`file, time, problem, severity, end_time, check, measures`); skipped
  files are listed with the reason.
- `pauses.txt` / `pauses.csv`: the pause map (informational).
- `report.json`: everything — tunables, every measurement, all scored phrase ends,
  network attempts (must be 0).
- `--csv-per-file`: one CSV per WAV (`<name>.csv`, beside it or in `--csv-dir`),
  optionally with the pause rows; a CSV the tool did not write is never replaced
  (`<name> (2).csv` instead). `--progress jsonl` drives the macOS app.

One bad file (missing, empty, unreadable, corrupt, an unexpected error, an
unwritable CSV) is skipped with a note; the batch always finishes.

## 2. Severity

Nothing an automated check finds is a rejection: the features are subjective, and
the tool is triage for an experienced mixer. Every finding gets a severity
**1 (worth a listen) to 3 (worst)**. Normal narration produces no findings.
Levels are judged against the speech around each finding (±10 s), not fixed dBFS,
except hum. Severities are graded on the value the text shows (0.1 dB).

## 3. Checks

### 3.1 Breaths (FinalPass breath check)

- **Mouth-click inhale** (internally "T-inhale": a breath that opens with a mouth
  click). Heard alone, a word-final consonant running into a breath sounds almost
  the same; the difference is where the click stands. A consonant's release follows
  its vowel directly; a mouth click stands alone after a pause. So a click (flagged
  by FinalPass, or at least 8 dB sharp by the click-in-a-pause detector, from 100 ms
  before to 40 ms after the breath's start) counts only when at least 100 ms have
  passed since the word (level within 25 dB of the narration, 5 ms RMS). 3 = click
  ≥ narration −20 dB (0.73 ms RMS peak), 2 = quieter; a click right after its word
  is not listed. Evidence (operator's context-tagged breaths, one title, 35+
  chapters): mouth-click inhales 41/49, consonant-then-inhale 0/29, plain breaths
  3/127, barely audible ticks 0/4; FinalPass's own flag alone 30/39, 5/12, 8/127.
  Any pause from 100 to 150 ms gives the same result. On 12 chapters: 75 listed
  (60 at 3) where the old gap rule listed 126, most of them consonants. Held-out:
  32 of those 75, never heard before, all 32 confirmed by ear.
- **Loud breath** (grade 3): severity 2 — an artistic call, but clients dislike them.
  A breath that is both is one finding at the higher severity.

### 3.2 Possible chopped word (local model)

`mythicinfinity/speech-truncation-detection-12M`, run as our numpy port of its
inference (`truncation_np.py`); the audited torch code stays vendored, unmodified,
as the reference. Every phrase end is scored on the 5 s ending there; flagged at
score ≥ 0.979795 (10 ms decision window) with a final-30 ms peak ≥ −23.5 dBFS.
Severity 3 (if real, part of a word is missing). The port matches the reference
within 1e-4 on synthetic windows at 11.025-88.2 kHz, odd lengths, DC and rumble,
and on a real chapter's phrase ends (max 1e-5, identical flags); its tests catch
16 of 16 deliberately planted bugs.

### 3.3 Hum

Any steady tone 40 Hz-1 kHz, severity 3, at any level.
- Tracked in 2 s windows (a line ≥ 10 dB over its ±10 Hz neighbourhood, held within
  1 Hz for ≥ 3 s; pieces of one line up to 3 s apart joined). A tracked line must
  also be heard in a nearby pause, unless there are no pauses around it (a hum
  loud enough to fill them). Evidence: tracking alone found 8 lines on the
  calibration title; the operator heard only 1 as hum — the only one also present
  in a pause.
- Found in the pauses (speech hides lines in the voice's range): the same line in
  ≥ 2 pauses over ≥ 3 s, at least −70 dBFS (operator's floor). Evidence: 11 found,
  all 11 confirmed by ear.
- Described from its pauses: every steady line there, loudest first, harmonics
  named. Start and end from the tone's own level; "starts/cuts off abruptly" when
  it changes by ≥ 20 dB in 0.5 s from full level, or — when the cut lands on a
  word — by ≥ 10 dB in 0.2 s and the line is gone from the next pause.

### 3.4 Noisy section

Minimum statistics per ⅓-octave band (per-frame DC removed; a band's floor capped
at its running mean so a steady tone is not inflated; bands carrying a listed hum
left out while it sounds). Listed when the floor comes within 41 dB of the
narration for about 5.5 s; severity by the gap under the local speech: 3 < 25 dB,
2 < 32 dB, else 1. Files with no narration are graded on absolute level
(3 ≥ −45, 2 ≥ −51 dBFS). Evidence: 6 of the title's 7 QC-noted noisy blocks found;
the missed one had a normal floor (a room-character question, see §6).

### 3.5 Sound cutting out

A step of ≥ 20 dB within 5 ms from background (room tone, breath, word tail) to
well under the chapter floor, listed only from speech −31 dB (about −50 dBFS);
3 from speech −18 dB, 2 from −26 dB, else 1. First guess; needs labelled examples.

### 3.6 Plosive pop

A short (≤ 80 ms), fast-rising swell in a steep 20-65 Hz band, carrying ≥ −10 dB of
the moment's energy, not the first sound after digital black and not inside a
breath; listed from speech +3 dB (2 from +6, 3 from +9). The voice itself carries
almost nothing below 65 Hz; the loudest peak within 0.1 s is reported. Needs
labelled examples.

### 3.7 Click in a pause

A tick or click in the silence between words, severity 3. Inside speech a click is
not a defect (every t, k, p and ch is one): a speech-wide search listed almost only
consonants when heard. So only a click in a pause is listed, and only when it is
brief (in several half-octave bands from 1 to 16 kHz it rises at least 8 dB over
the same band's loudest level within 15 ms on both sides, the idea of a narration
de-clicker, reimplemented), at least −45 dBFS at its peak, and at least 100 ms from
the words and from any breath on both sides (a breath's own mouth click belongs to
the breath check). Evidence: 64 candidates from looser rules heard across 12
chapters, 2 confirmed ticks; this rule lists exactly those 2 and nothing else in the
12 chapters. Set on the same chapters, so a second title must confirm it.

### 3.8 Pause map (informational)

Every pause word to word, with its duration and a guess at its kind from the
chosen generic rule set; head/tail compared with the rule (±0.1 s). No severity.

## 4. macOS app

A small SwiftUI window (`macos/`): drop WAVs or folders (not searched recursively),
Go, a progress bar, one CSV per WAV beside it (`<name>.csv`) or in a chosen folder,
optional pause rows. **Fully sandboxed with no network entitlement**; the engine
(a copied Python with fpab, the `uv.lock` versions and the verified weights, about
185 MB in all) is bundled and runs inside the sandbox. The sandbox allows only
`<name>.csv` beside a WAV it was given, so where that would replace someone else's
file (or two WAVs share it) the app asks once for that folder and writes
`<name> (2).csv`. A CSV that cannot be placed is kept and can be saved later.
`macos/build_app.sh` assembles it in a staging folder, installs hash-checked
locked dependencies, signs every Mach-O, asserts exact entitlements and moves the
app into place only when all checks pass.

## 5. Safety and dependencies

- Client audio never leaves the machine: analysis runs under a network guard (a
  Python audit hook refusing DNS, connect, send, bind and child processes, counted
  in every report) and, in the app, the OS sandbox without network access.
- No client identifiers or client-derived specifics in this repository.
- 7-day package hold (`[tool.uv] exclude-newer`, asserted by the build); the build
  backend is pinned. Model weights: fetched only by `fpab setup-model` from a pinned
  revision, SHA-256 verified on every load, parsed strictly (no pickle, no code).

## 6. Open items

1. Dropout and plosive limits: calibrate on the operator's labelled examples.
2. The chopped-word model's precision on phrase ends into room tone (its
   evaluation covered clip ends into digital black).
3. "Starts/cuts off abruptly" on hums found only in pauses: not yet checked by ear; on
   synthetic audio a cut can be missed when a low-pitched word starts on it.
4. A further generic pause rule set (numbers pending from the operator).
5. Reverb / roominess (milestone 2): the QC-noted block the noise check misses.
6. Breath frames are computed twice per chapter (needs a small FinalPass API change).
7. Half-precision weights would save ~25 MB (changes the audited file; needs the
   operator's OK, a recorded SHA-256 and the same equivalence bar).
8. Click in a pause: confirm the limits on a second title.
9. The app's drag-and-drop, folder mode and the "(2)" folder prompt need a
   hands-on check by the operator.
