# FinalPassAudioBook — design and plan

**Status (2026-10-03):** milestone 1 built, followed by the severity scale, the hum
work, the macOS app, the numpy model port, a four-part adversarial audit, a
six-part pre-publish review, milestone 3 (every breath confirmed by a local breath
model, 0.2.0) and a seven-part review of 0.2.0, each with its fixes (0.2.1). Evidence below is from one delivered audiobook (12 chapters used for
calibration), reported only as anonymous aggregates; the client-specific record
is kept privately, outside this repository.

## 1. What it does

```
fpab check CHAPTERS_OR_FOLDERS... [--rules standard|no-paragraph] [--out DIR] [--min-sev 1|2|3]
           [--csv-per-file] [--csv-dir DIR] [--with-pauses] [--progress text|jsonl] [--no-truncation]
           [--no-breath-model]
fpab setup-model [--from-file model.safetensors] [--breath-from-file respiro-en-fpab-v2.safetensors]
fpab rules
```

- `issues.txt` / `issues.csv`: what a QC reviewer is likely to notice, per file, in
  time order (`file, start_time, end_time, event, severity, check, measures`); skipped
  files are listed with the reason.
- `pauses.txt` / `pauses.csv`: the pause map (informational).
- `report.json`: everything — tunables (as the run used them), every measurement, every
  scored clip end, whether the breath model ran on each file, network attempts (must be 0).
- `fpab setup-model` installs both models; one already installed and verified is left
  as it is, so `--from-file` alone needs no network once the breath model is in place.
- `--csv-per-file`: one CSV per WAV (`<name>.csv`, beside it or in `--csv-dir`). It
  opens with a summary (a label, then one short fact per cell: file, format, duration, problem
  and informational event counts, narration level, noise floor, rule set, whether the
  chopped-word check ran, whether the breath model ran on this file (or why not), when,
  a breaths line with any the model did not confirm), then the problem events (every
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
skipped with a note; the batch always finishes, and the command exits 1 (2 when nothing
could be checked; with `--progress jsonl` 0 once the run completes, problems per file).
A file that can be read but is not usable narration is a severity-3 file event (§3.10).
The run report in `--out` is written whole and moved into place, so an interrupted
write never leaves a cut-off file there.

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
  ≥ narration −20 dB (0.73 ms RMS peak), 2 = quieter, none below narration −40 dB (inaudible: two
  V4 transients at −50.7/−50.6 dB were heard as no click; the faintest confirmed was −36.6); a click right after its word
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
  and sums up the breaths in its summary. Below 16 kHz sampling the breath check is
  skipped with a note (its features need the band above 5 kHz).
- **A sound as loud as speech is not a breath** (FinalPass): when a breath's loudest
  quarter (75th-percentile frame) comes within 15 dB of the narration it is a speech
  sound, typically an "s", "sh" or "ch" left alone at the end of a word before a
  pause. Some voices' sibilants peak near 3 kHz instead of above 5 kHz, and a stop
  closure gives them their own gap from the word, so nothing else kept them out: a test
  book in a voice that makes no breaths listed 14 "breaths", 12 of them very loud, and
  every one the operator auditioned was a sibilant. Evidence: the operator's labelled
  breaths (523 on three chapters, mouth-click inhales and breaths straight after a
  consonant included) and the 16 QC-cut breaths in the 12 study chapters never come
  closer than 16.8 dB; the test voice's 12 loud ones all come within 12.8 dB. The rule
  also removes 3 of 1,717 unlabelled detections on 15 chapters of the calibration
  title and 12 of 331 on five chapters of the second title.
- **Every breath is confirmed by a local breath model** (milestone 3). On other voices
  the loudness rule is not enough: on the second title most "breaths" graded 2–3 were
  a consonant left alone at a word's end before a pause ("t", "k", "p", "ch", "s"),
  which no spectral or loudness line separates from breaths across voices. The model is
  Respiro-en (Yang, Koriyama & Saito, Interspeech 2024; MIT): a frame-wise breath
  detector (Conformer + BiLSTM, 2.9 M parameters, one probability per 10 ms of 16 kHz
  audio) trained on read audiobook speech with every frame inside an aligned word as
  "not breath". Its published weights missed most of the calibration title's loud
  breaths, so they were fine-tuned (the paper's recipe) on the operator's labelled
  breaths of that title plus the published model's confident verdicts on half of the
  second title; only the weights ship — no audio, labels or names, and nothing in them
  can give back the audio they were trained on.
  - Rule: a breath FinalPass finds is listed (as a problem or as a quiet breath) only if
    the model says breath (≥ 0.5) for at least 100 ms inside it. A mouth-click inhale is
    a breath, so its click is listed only with a confirmed breath after it. With the
    model, FinalPass's "as loud as speech" rule (above) is switched off: the model
    rejects all 33 tagged consonants that rule removes and keeps the two short loud
    breaths it wrongly drops. Without the model that rule stays.
  - Evidence, on tagged events never used to train or choose the model: breaths kept
    18/18 (calibration title) and 16/16 (second title); consonants listed 5/28 (second
    title; 4 of them "breathy", which the operator is content to see), 0/14 on the breathless
    test voice; mouth-click inhales kept 43/43 labelled + 8/8 heard; five of the second
    title's seven false mouth-click inhales (a word-final "p" held 80–125 ms, inaudible
    clicks) removed. At ≥ 150 ms one breath would be lost.
  - Run like the chopped-word model: our numpy port (`breath_np.py`), tested against the
    vendored, unmodified torch code; 16 kHz input in 30 s windows (the middle 20 s kept);
    weights installed only by `fpab setup-model` (size and SHA-256 checked); analysis
    under the network guard. Without the model the breaths are listed as before and every
    report says "breath model: off" and why; below 16 kHz the model does not run on that
    file, and a file it fails on keeps its breaths unconfirmed (with a note), never losing
    its other checks.
- **Breath cut off into silence** (with the model; informational). A breath (the model ≥ 0.5
  for ≥ 50 ms, ending ≤ 40 ms before) running straight into a 50–80 ms hole of exact digital
  silence with sound after it — a generated clip that ended mid-breath — is reported when
  nothing above lists it (it replaces a quiet breath at the same spot). On the calibration title
  the breaths QC removed by cutting them to silence mostly sat at such a hole (63/69 within
  0.3 s); holes alone are the generator's clip gaps (about 45 of 40–60 ms per 15 minutes), and
  nothing measured told QC's apart from the rest. By ear (operator): of 20 with holes from 27 ms,
  most were "too short to really matter" while QC's had distinct gaps → minimum 50 ms (31 there,
  1.0 per 15 minutes, 8 of them QC's; 8 on a second title; none on the test book); of the next 24,
  none was "truly rejectable" on its own → informational, not a problem. The model finds breaths
  shorter than FinalPass's 150 ms.

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
text shows; a hum that builds, on the level it reaches. The loudest line may be a
harmonic, shown as such ("-50 dBFS at 180 Hz"); a weak 60 Hz under strong 180/300 Hz
lines once read −70 dBFS, severity 1.
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
- A hum in the voice's range loud enough to fill every pause (less than 30 dB under
  the voice) and kept from tracking by speech at its pitch: its pauses are the gaps
  ≥ 15 dB under the voice that it fills — flat (within 3 dB), carried by one steady
  line (within 6 dB of the gap), and alike (the same line, ±0.5 Hz and ±3 dB, in
  another such gap within 10 s). A tracked line whose pauses carry it well beyond its
  track (speech broke it into pieces) is described from those pauses. Synthetic
  100-300 Hz hums at −47 and −40 dBFS, once missed or split into pieces that each
  "started abruptly", are each one hum at their level. On the calibration title 4
  gaps at the narrator's own pitch were flat and one line, none had a twin, and the
  12 chapters list exactly the same 12 hums, edges and severities as before. The
  limit: such a hum's start or cut under a word is placed between the pauses either
  side (and may go unnamed).
- Harmonics: a whole multiple within 0.6 Hz (tracked; 0.02 Hz per multiple beyond)
  or 1 Hz (in the pauses; 0.05 Hz per multiple), sounding mostly while the hum does.
  The 1 % once used let an unrelated 987 Hz hum pass as 41 Hz × 24, and any overlap
  let a later hum at a multiple vanish into an earlier one.
- Described from its pauses: every steady line there, loudest first, harmonics
  named (in the measures; the event text stays short: the tone or tones, level and
  edges, e.g. "hum 58.8 Hz, -70 dBFS, cuts off abruptly"). Start and end from the tone's own level; "starts/cuts off abruptly" when
  it changes by ≥ 20 dB in 0.5 s from full level, or — when the cut lands on a
  word — by ≥ 10 dB in 0.2 s and the line is gone from the next pause; a word in the
  tone's band above its full level is not part of the step (a hum running to the end
  of a file read "cuts off abruptly" from one). Heard blind
  on all 24 edges: 6 of the 8 claimed abrupt were; 2 of the 16 others were abrupt
  too (7 could not be judged). None of the misses changes a severity.

### 3.4 Noisy section

Minimum statistics per ⅓-octave band (per-frame DC removed; a band's floor capped
at its running mean so a steady tone is not inflated; bands carrying a listed hum
left out while it sounds, farther either side the louder it is: a −40 dBFS tone's
sidelobes lifted bands 110 Hz away by 35 dB). Listed when the floor comes within 41 dB of the
narration for about 5.5 s; severity by the gap under the local speech: 3 < 25 dB,
2 < 32 dB, else 1. Files with no narration are graded on absolute level
(3 ≥ −45, 2 ≥ −51 dBFS). Evidence: 6 of the title's 7 QC-noted noisy blocks found;
the missed one had a normal floor (a room-character question, see §6).

### 3.5 Dropout

A classic dropout (operator): the sound falls to dead silence for a frame or less (29.97 fps,
~33 ms) and comes straight back. A run of at least 10 exact zeros (at 44.1 kHz, scaled; a slow
zero crossing leaves at most 9) lasting at most 33.4 ms, with at least −45 dBFS in the 5 ms before
and the 5 ms after. Always severity 3. Skipped for 8-bit audio (its steps round quiet sound to
exact zero), with a note — so is the chopped-word check, which reads exact zeros too. Generated narration has no room, so digital silence after a
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
15-minute chapter. Skipped, with a note, at 16 kHz and below (the word's high band is not
there). At a generated clip's start — within 60 ms after ≥ 20 ms of exact digital
silence — the word's onset follows at once and overlaps the pop, so there it needs only 15 dB of
low-band dominance and the word may follow from 50 ms: over 4,193 clip starts of 70 sample files
this adds 9 listings, all 9 confirmed as pops by ear; elsewhere the rule is unchanged.

### 3.7 Click in a pause

A tick or click in the silence between words, severity 3 — 2 when its peak stands less than
30 dB over the file's room-tone floor (the noise check's floor, not the sound beside the click: the
prove-out's clicks heard as "low, 1 or 2" stood 23–29 dB over it, those left at 3 stood 36–50 dB, one
at 26). Inside speech a click is
not a defect (every t, k, p and ch is one): a speech-wide search listed almost only
consonants when heard. So only a click in a pause is listed, and only when it is
brief (in several half-octave bands from 1 to 16 kHz it rises at least 8 dB over
the same band's loudest level within 15 ms on both sides, the idea of a narration
de-clicker, reimplemented), at least −45 dBFS at its peak, and at least 100 ms from
the words and from any breath on both sides (a breath's own mouth click belongs to
the breath check). Evidence: 64 candidates from looser rules heard across 12
chapters, 2 confirmed ticks; this rule lists exactly those 2 and nothing else in the
12 chapters. Set on the same chapters, so a second title must confirm it. The same
rule covers the room tone before the first word and after the last, to the file's very
first and last milliseconds (only the word side needs the 100 ms).

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
clip stops is listed once, as the tick. A tick within 10 ms of the file's start or end
is judged on the one side there is. At 88.2 kHz and up the test looks only at
16.5-22.05 kHz and measures sample steps over one 44.1 kHz sample, so it behaves as at
44.1/48 kHz (a one-sample spike at 192 kHz holds a quarter of the energy and is listed
from about −34 dBFS). Skipped, with a note, at 34.7 kHz and below (no band above
16.5 kHz to look in).

### 3.9 Pause map (informational)

Every pause word to word, with its duration and a guess at its kind from the
chosen generic rule set; head/tail compared with the rule (±0.1 s). No severity.
The guess reads the length as listed (0.01 s). Sound is what stands 8 dB over the
quiet floor or within 45 dB of the narration. A file whose pauses are digital black
(≥ 5 % exact silence; the calibration chapters hold about 1 %, at the head and
tail) has no floor: its quietest sounding moments are speech, and taking them as
the floor trimmed every word (pauses 96 ms long) or, for evenly levelled speech,
found no narration at all. And no floor counts within 10 dB of the narration.

### 3.10 File problems

Severity 3, so an unusable file never reads as clean: invalid samples (NaN, Inf, or
beyond ±16, i.e. +24 dBFS, in a float file; zeroed for the analysis; the short events that causes — a dropout, ticks — are not
listed again, and a NaN in one channel of dual mono counts too), audio data that ends
at least one frame before its header says (RF64/W64 not yet covered), no narration
found, or a file under 1 s.

## 4. macOS app

A small SwiftUI window (`macos/`): drop WAVs or folders (not searched recursively),
Go, a progress bar, one CSV per WAV beside it (`<name>.csv`) or in a chosen folder,
optional pause rows. **Fully sandboxed with no network entitlement**; the engine
(a copied Python with fpab, the `uv.lock` versions and the verified weights, about
190 MB in all) is bundled and runs inside the sandbox. The sandbox allows only
`<name>.csv` beside a WAV it was given, so where that name already exists (or two WAVs
share it) the app asks once for that folder and writes `<name> (2).csv`. A CSV that
cannot be placed is kept and can be saved later.
`macos/build_app.sh` assembles it in a staging folder, installs hash-checked
locked dependencies and hash-pinned build backends, removes the building Mac's
folder names from what it bundles (and refuses to finish if the repo, Python or model
folders it copied from still appear, or if an untracked file sits in the package), signs
every Mach-O, asserts exact entitlements and moves the app into place only when all
checks pass. `--sign` signs with a Developer ID, secure timestamp and hardened runtime;
`macos/release.sh` notarizes and staples the app and wraps it in a signed, notarized,
stapled DMG; the app carries every bundled component's licence texts in
`Contents/Resources/Licenses`. The app never replaces an existing file; reports it could not place are
kept until saved, it asks before discarding them, and any left
when it quits are offered again at the next launch.

## 5. Safety and dependencies

- Client audio never leaves the machine: analysis runs under a network guard (a
  Python audit hook refusing socket creation, DNS, connect, sendto/sendmsg, bind,
  child processes and the common ctypes routes to them, counted in every run's
  report.json, terminal output and the app's status line) and, in the app,
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
