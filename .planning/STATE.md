---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 10
  completed_plans: 5
  percent: 55
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Isolate the effect of generation architecture on conversational realism vs Switchboard
**Current focus:** Phase 2 — main generation (12 conditions), RESUMING after a GPU-driver crash

## Current Position

Phase: 2 of 3 (Main Experiment & Analysis)
Status: Generation was running on the VM in tmux but **crashed at C4-P0** (NVML driver/library
mismatch — C4 loads two models, torch's allocator nvmlInit asserted).
Done & safe on the VM (committed locally `b0ab7a0`, NOT yet pushed): **C1-P0, C2-P0, C3-P0 = 50
each (150 conversations)**. C4-P0 has 1. The P1 set and all of P2 were not started.
Last activity: 2026-06-24 — crash diagnosed; resume plan written to VM_TASKS.md.

## NEXT STEPS (resume)

1. **VM: push the 150 already-committed conversations** (b0ab7a0) so the local side gets them.
2. **VM: reboot** (fixes the NVML driver mismatch). Verify `nvidia-smi` + `torch.cuda.is_available()`.
3. **VM: re-run the generation loop** (resumable; do NOT delete data/generated). Then the 4 **P2** conditions. Push.
4. **Local: analyze.** `python analysis/analyze.py` (per-condition table) and `python analysis/stats.py`
   (mixed-effects + Independence Gradient + markers). Produce the ALIGN CSV on the VM (schema below).

See VM_TASKS.md for the exact VM commands.

## Local-session work done (2026-06-24, later)

- **Docs reconciled to the real 12-condition design** (C1-C4 × P0/P1/P2). ROADMAP/REQUIREMENTS/PROJECT
  previously said "6 conditions" (an unfinalized reduction). The actual reduction was convs/condition
  200→50; all 12 conditions are kept. P2 = robustness condition, **non-lexical metrics only** (its
  few-shot Switchboard excerpt makes marker measurement circular).
- **Added `analysis/stats.py`** — Phase 2 statistics (ANLY-01..05): words/turn mixed model
  `DV ~ Corpus*Section + (1|ConvID)`, marker rates vs SB (P2 auto-excluded), Independence Gradient
  trend test, optional figures. Runs on numpy+scipy locally today; `statsmodels`+`matplotlib` are
  optional upgrades (mixed model + PNGs); reads ALIGN output from `data/align/alignment_turns.csv`
  when present. Verified end-to-end on the spot-check data.
- **ALIGN export the VM should produce** (so stats.py picks it up): CSV `data/align/alignment_turns.csv`
  with columns `condition, conv_id, turn_index, n_turns, cosine_semanticL` (`condition`="SB" for the baseline).

## Findings so far (smoke + partial)

- **Turn length:** C1 (all-at-once) and **C2-P1 (our prompt) ≈ Switchboard ~14–15 words/turn**;
  C2-P0 ~64, C3/C4 ~75–80 (long). Clear architecture × prompt effect.
- **Coordination markers (oh/okay/uh-huh): ≈ 0 in all LLM conditions** vs Switchboard (uh-huh 0.85).
  Replicates the paper.
- ALIGN validated on Switchboard (~0.59–0.62 vs paper ~0.57). `analysis/analyze.py` ready.

## Blockers / Concerns

- NVML driver/library mismatch on the VM — reboot to fix; do NOT reboot mid-run.
- GitHub PATs were pasted in plaintext — revoke them.
- API access for the Phase-3 LLM-judge still unconfirmed.

## Session Continuity

Last session ended with generation crashed at C4-P0. Resume via VM_TASKS.md (reboot → resume loop → P2).
Resume files: VM_TASKS.md (VM steps), analysis/analyze.py (metrics), .planning/PROJECT.md & ROADMAP.md (design).
