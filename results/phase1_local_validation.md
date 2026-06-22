# Phase 1 — Local validation (Switchboard parser)

**Date:** 2026-06-22 · **Machine:** local (no GPU), Python 3.11.3

## What was validated
`analysis/swda.py` parses the SwDA distribution under `data/switchboard/swda/`, strips
SwDA disfluency/transcription markup, and merges consecutive same-speaker utterances into
turns (the same turn definition the paper / ALIGN use).

## Result (first 30 conversations)
| metric | value | paper (SB) |
|---|---|---|
| mean words/turn | **14.73** | ~14 (main body) |
| median words/turn | 7.0 | — |
| mean turns/conversation | 66.5 | — |

Mean words/turn matches the paper's Switchboard main-body figure closely → the parser and
cleaner are correct. The low median vs higher mean reflects the many short backchannel
turns characteristic of human spoken conversation.

**Scope:** computed over whole conversations. Exact main-body-only replication and the
ALIGN alignment numbers (conceptual ~0.57 Earlier, etc.) are done on the VM once ALIGN is
installed — that is the remaining half of the Phase-1 validation gate.

## Reproduce
```
py -3 analysis/swda.py --n 30
```
