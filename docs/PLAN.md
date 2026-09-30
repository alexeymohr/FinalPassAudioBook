# FinalPassAudioBook — design and plan

**Status (2026-09-30):** milestone 1 built, followed by the severity scale, the hum
work, the macOS app, the numpy model port, a four-part adversarial audit and a
six-part pre-publish review, each with its fixes. Evidence below is from one delivered audiobook (12 chapters used for
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
  time order (`file, start_time, end_time, event, severity, check, measures`); skipped
  files are listed with the reason.
- `pauses.txt` / `pauses.csv`: the pause map (informational).
- `report.json`: everything — tunables, every measurement, all scored phrase ends,
  network attempts (must be 0).
- `--csv-per-file`: one CSV per WAV (`<name>.csv`, beside it or in `--csv-dir`). It
  opens with a summary (a label, then one short fact per cell: file, format, duration, problem
  and informational event counts, narration level, noise floor, rule set, whether the
  chopped-word check ran, when, a breaths line), then the problem events (every
  finding, severity 1-3) in time order, then, a few empty rows below, the
  informational events (quiet breaths, and the pause map when asked for). The CSVs
  show the two to four measures per check a mixer uses; `report.json` keeps them all. An
  existing file is never replaced (operator), not even an earlier report: a new report
  gets `<name> (2).csv`, `(3)`, …, created as a new file (a file that appears during the
  run is not overwritten either), skipping a "(k)" name that is another audio file's own.
  The summary names the WAV's file and folder. `--progress jsonl` drives the macOS app.
- The run report never replaces files in `--out` that the tool did not write; that is
  checked, with the folder's writability, before the analysis starts.

One bad file (missing, empty, unreadable, an unexpected error, an unwritable CSV) is
skipped with a note; the batch always finishes, and the command exits 1. A file that
can be read but is not usable narration is a severity-3 file event (§3.10).

## 2. Severity

Nothing an automated check finds is a rejection: the features are subjective, and
the tool is triage for an experienced mixer. Every finding gets a severity
**1 (worth a listen) to 3 (worst)**. Normal narration produces no findings.
Breath and noise levels are judged against the narration (a noisy section's
severity against the speech within ±10 s); the other checks use fixed dBFS limits.
Severities are graded on the value the report shows (0.1 dB; the whole dB where the
text rounds, as for hum and plosive).

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
- **Every other breath** is listed too, scored by loudness (FinalPass's
  noticeability: median level vs the narration, weighted by length): quiet below
  −31.6 dB (informational), 1 from −31.6, 2 from −26.4, 3 from −22.5. A fixed scale,
  not relative to each chapter: on the calibration title the lines sit at its 35th
  (the operator's suggested cut), 75th and 95th percentiles; a second title's
  breaths sit about 10 dB lower and list far fewer (per chapter about 73 quiet /
  68 / 38 / 9 against 61 / 2 / 1 / 3). A breath that is both is one finding at the
  higher severity. The per-file CSV lists quiet breaths as informational events
  and sums up the breaths in its summary.

### 3.2 Word ends abruptly at a clip end (local model)

`mythicinfinity/speech-truncation-detection-12M`, run as our numpy port of its
inference (`truncation_np.py`); the audited torch code stays vendored, unmodified,
as the reference. The model asks whether speech was still active when the audio
stopped, and it was evaluated on one kind of point only: where a generated clip ends
and digital black begins (the first sample of ≥ 50 ms of exact zeros after sound).
Only those points are scored, on the 5 s ending there; listed at score ≥ 0.979795
(10 ms decision window) with a final-30 ms peak ≥ −23.5 dBFS. Severity 1, a heads-up:
"word ends abruptly". Evidence: on the evaluation title fpab finds the same 118 clip
ends and lists the same 10 (9 real chops, 1 clean ending), on its second title 0 of 27;
on two further titles 8 listings, all hard abrupt endings, none missing part of a word
(operator). Scoring every pause instead, as fpab first did, gave 64 listings with no
real chop. The port matches the reference within 1e-4 on synthetic windows at
11.025-44.1 kHz (its resampler also at 88.2 kHz) and to 1e-7 on real windows; in the
audit its tests caught 16 of 16 planted bugs.

### 3.3 Hum

Any steady tone 40 Hz-1 kHz, at any level. Severity (operator, after hearing both
edges of all 12 hums on the calibration title): 1 for a low-level hum, most of them;
2 when strong (loudest line ≥ −55 dBFS; the two called strong measured −53.8 and
−54.3, the loudest of the rest −55.5); 3 when strong and it starts or cuts off
abruptly. On that title: 2 at severity 3, 10 at 1. Graded on the whole-dB level the
text shows; a hum that builds, on the level it reaches.
- Tracked in 2 s windows (a line ≥ 10 dB over its ±10 Hz neighbourhood, held within
  1 Hz for ≥ 3 s; pieces of one line up to 3 s apart joined). A tracked line must
  also be heard in a nearby pause, unless there are no pauses around it (a hum
  loud enough to fill them). Evidence: tracking alone found 8 lines on the
  calibration title; the operator heard only 1 as hum — the only one also present
  in a pause. A hum loud enough to fill the pauses inside it but not those around it
  (it starts and stops mid-chapter) is heard in the gaps it fills instead (≥ 0.4 s,
  ≥ 15 dB under the narration, within 6 dB of the hum, the line as loud there as under
  the words; not for a line within 15 dB of the narration, a held note): synthetic hums of −45 and
  −40 dBFS for 4-50 s were otherwise missed or listed only as noise; the calibration
  title still lists its 12 hums and nothing more.
- Found in the pauses (speech hides lines in the voice's range): the same line in
  ≥ 2 pauses over ≥ 3 s, at least −70 dBFS (operator's floor). Evidence: 11 found,
  all 11 confirmed by ear.
- Described from its pauses: every steady line there, loudest first, harmonics
  named (in the measures; the event text stays short: the tone or tones, level and
  edges, e.g. "hum 58.8 Hz, -70 dBFS, cuts off abruptly"). Start and end from the tone's own level; "starts/cuts off abruptly" when
  it changes by ≥ 20 dB in 0.5 s from full level, or — when the cut lands on a
  word — by ≥ 10 dB in 0.2 s and the line is gone from the next pause. Heard blind
  on all 24 edges: 6 of the 8 claimed abrupt were; 2 of the 16 others were abrupt
  too (7 could not be judged). None of the misses changes a severity.

### 3.4 Noisy section

Minimum statistics per ⅓-octave band (per-frame DC removed; a band's floor capped
at its running mean so a steady tone is not inflated; bands carrying a listed hum
left out while it sounds). Listed when the floor comes within 41 dB of the
narration for about 5.5 s; severity by the gap under the local speech: 3 < 25 dB,
2 < 32 dB, else 1. Files with no narration are graded on absolute level
(3 ≥ −45, 2 ≥ −51 dBFS). Evidence: 6 of the title's 7 QC-noted noisy blocks found;
the missed one had a normal floor (a room-character question, see §6).

### 3.5 Dropout

A classic dropout (operator): the sound falls to dead silence for a frame or less (29.97 fps,
~33 ms) and comes straight back. A run of at least 10 exact zeros (at 44.1 kHz, scaled; a slow
zero crossing leaves at most 9) lasting at most 33.4 ms, with at least −45 dBFS in the 5 ms before
and the 5 ms after. Always severity 3. Skipped for 8-bit audio (its steps round quiet sound to
exact zero), with a note. Generated narration has no room, so digital silence after a
word is just a pause and is not listed. None on the calibration title's 12 chapters or a second
title's 5 (a guard); planted in real narration, holes of 10 samples to 15 ms inside words are
listed 60/60, 30 ms 58/60, and 8-sample holes, 40 ms holes and holes in quiet audio 0/60.

### 3.6 Plosive pop

A low thump standing on its own just before a word (often breath, then pop, then word;
removing it leaves the word intact). A burst below 100 Hz of at least −42 dBFS, at most
40 ms wide, low-dominated (≥ 22 dB over 500–8000 Hz at its peak), over before the word
(the low band falls ≥ 30 dB) and followed 75–150 ms later by the word's high end (≥ 20 dB
above the high end at the pop). A normal p/b carries the word's high end with its burst.
Severity by level (the whole dB shown): 1 from −42 dBFS, 2 from −34, 3 from −26. A pop
overlapping a mouth-click inhale or up to 200 ms after it is part of that breath's entry. Evidence: 10 of 12
QC-noted pops; a blind round of 32 looser candidates: 10 of 11 wanted, 1 of 21 others
(limits set on those clips); held out, 29 of 32 listings were plosives by ear. About 11 per
15-minute chapter.

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
12 chapters. Set on the same chapters, so a second title must confirm it. The same
rule covers the room tone before the first word and after the last (only the word
side needs the 100 ms).

### 3.8 Digital tick

A guard for edit and processing glitches, severity 3. Rendered narration stops at
about 16 kHz, so a spike above 16.5 kHz did not come from the voice. Listed only
when it is extremely short (≤ 8 samples at 44.1 kHz, at half its peak), loud
(≥ −60 dBFS above 16.5 kHz) and alone (≥ 20 dB over everything else above
16.5 kHz within 0.5–10 ms on both sides, and the sharpest sample step there); a
tick inside a consonant blends in. Evidence: 32 candidates from a looser rule were
all consonants by ear; this rule lists none on the whole title (41 files). Planted
one-sample spikes: 100 % found in pauses at −40 dBFS, about half inside speech at
−30 dBFS. No real example has been heard yet. The step into and out of a dropout is
the dropout's own edge and is not listed again; a word that ends in a tick where its
clip stops is listed once, as the tick.

### 3.9 Pause map (informational)

Every pause word to word, with its duration and a guess at its kind from the
chosen generic rule set; head/tail compared with the rule (±0.1 s). No severity.
The guess reads the length as listed (0.01 s).

### 3.10 File problems

Severity 3, so an unusable file never reads as clean: invalid (NaN/Inf) samples
(zeroed for the analysis; the short events that causes — a dropout, ticks — are not
listed again, and a NaN in one channel of dual mono counts too), audio data that ends
at least one frame before its header says (RF64/W64 not yet covered), no narration
found, or a file under 1 s.

## 4. macOS app

A small SwiftUI window (`macos/`): drop WAVs or folders (not searched recursively),
Go, a progress bar, one CSV per WAV beside it (`<name>.csv`) or in a chosen folder,
optional pause rows. **Fully sandboxed with no network entitlement**; the engine
(a copied Python with fpab, the `uv.lock` versions and the verified weights, about
180 MB in all) is bundled and runs inside the sandbox. The sandbox allows only
`<name>.csv` beside a WAV it was given, so where that name already exists (or two WAVs
share it) the app asks once for that folder and writes `<name> (2).csv`. A CSV that
cannot be placed is kept and can be saved later.
`macos/build_app.sh` assembles it in a staging folder, installs hash-checked
locked dependencies and hash-pinned build backends, removes the building Mac's
folder names from what it bundles (and refuses to finish if any remain), signs
every Mach-O, asserts exact entitlements and moves the app into place only when all
checks pass. The app never replaces an existing file; reports it could not place are
kept until saved, it asks before discarding them, and any left
when it quits are offered again at the next launch.

## 5. Safety and dependencies

- Client audio never leaves the machine: analysis runs under a network guard (a
  Python audit hook refusing socket creation, DNS, connect, send, bind, child
  processes and ctypes routes to them, counted in every report) and, in the app,
  the OS sandbox without network access. The guard sees what goes through Python;
  the sandbox is the hard boundary (the command line can run under
  `sandbox-exec` with the network denied; see the README).
- No client identifiers or client-derived specifics in this repository.
- 7-day package hold (`[tool.uv] exclude-newer`, asserted by the build); fpab's
  build backend is pinned, and the app build pins the backends of what it builds
  from source by hash (`macos/build-constraints.txt`). Model weights: fetched only by `fpab setup-model` from a pinned
  revision, SHA-256 verified on every load, parsed strictly (no pickle, no code).

## 6. Open items

1. "Starts/cuts off abruptly": 6/8 right by ear, 2 missed (see §3.3); on synthetic
   audio a cut can be missed when a low-pitched word starts on it.
2. A further generic pause rule set (numbers pending from the operator).
3. Reverb / roominess (milestone 2): the QC-noted block the noise check misses.
4. Breath frames are computed twice per chapter (needs a small FinalPass API change).
5. Half-precision weights would save ~25 MB (changes the audited file; needs the
   operator's OK, a recorded SHA-256 and the same equivalence bar).
6. Click in a pause: confirm the limits on a second title. Digital tick: no real
   example yet (limits set with planted spikes).
7. The app's drag-and-drop, folder mode and the "(2)" folder prompt need a
   hands-on check by the operator.
