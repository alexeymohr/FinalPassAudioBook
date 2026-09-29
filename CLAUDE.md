# FinalPassAudioBook — Claude Code Context

## What this is

Local, offline QC for AI-narrated audiobook chapters. It lists what a QC
reviewer is likely to notice — timed, with a plain-language description and a
severity from 1 to 3 — and nothing else. It is a triage
tool for an experienced mixer, not a pass/fail gate, and it never "scores" normal
narration as bad.

Sibling of FinalPass (`~/programming/FinalPass`), which it uses as a library for
shared code: audio I/O, timecode/clock formatting, and the breath check
(`finalpass.breath_*`). Improve shared code in FinalPass, not by forking it here.

Plan and current milestone: `docs/PLAN.md`.

## ⛔ ABSOLUTE RULE — client audio never leaves this machine

The operator produces audiobooks for clients. That audio is confidential client
material, and so is everything derived from it (measurements of a specific
title, QC notes, file names): keep those in the private notes, never in this repo.

**NEVER listen to, transcribe, or otherwise ingest the audio content of a client
file, and NEVER let it leave this machine.** No multimodal audio read, no
speech-to-text, no cloud service. Do not send, attach or upload it into a
conversation, a file-send tool, an artifact, or any remote service — sending a
WAV "to the user" still uploads it and counts as a leak.

**Allowed:** run this tool and other fully local, deterministic analysis over
the files, and read/report numeric and textual results — times, sample indices,
dBFS, scores. When the operator needs to audition something, hand them times,
never audio. Excerpts written for the operator's own listening stay on disk and
are never read back.

A local model may process client audio only when it runs fully offline under a
network guard and emits numbers only (the truncation model below).

## Public repo: no client identifiers

This repository is public (MIT). Never commit book titles, ISBNs, project or
order IDs, file names of client deliverables, or anything derived from client
audio. Local evaluation data lives under `resources/` (gitignored).

## Models and dependencies

- Nothing installs a package version published less than 7 days ago (check the
  whole resolved graph, not just direct dependencies).
- ML dependencies (torch and friends) are an optional extra, never the base
  install. The base tool is numpy/scipy DSP.
- Model code is vendored after a full read (licence kept alongside). Never
  `trust_remote_code=True`; never unrestricted pickle — safetensors only, loaded
  with `load_state_dict(strict=True)`.
- Model weights are fetched only by an explicit setup command, from a pinned
  revision, and verified against a recorded SHA-256. Analysis runs never
  download anything; they run under a network guard that refuses sockets.

## How we work here

- One milestone at a time; the plan is locked before code is written. If the
  plan does not answer a question, stop and ask.
- Honest reporting: say plainly what failed or was not verified.
- Every threshold is a named, reported tunable, calibrated on the operator's
  labelled examples and stated with the numbers that justify it.
- Tests use synthesized audio only. Never commit real audio.

## Stack

Python 3.11+, `uv`, `pyproject.toml`, numpy/scipy/soundfile, `click`, `rich`,
`pydantic` v2, `pytest`. Library code returns data; only the CLI prints.
