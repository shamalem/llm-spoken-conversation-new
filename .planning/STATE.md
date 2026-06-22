---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 10
  completed_plans: 4
  percent: 40
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-22)

**Core value:** Isolate the effect of generation architecture on conversational realism vs Switchboard
**Current focus:** Phase 2 - Main Experiment & Analysis prep

## Current Position

Phase: 2 of 3 (Main Experiment & Analysis)
Plan: Phase 2 prep / generator smoke tests
Status: In progress - corrected prompts + C3/C4 generators ready for VM smoke testing
Last activity: 2026-06-22 - VM confirmed ALIGN validation and C2 multi_turn_emissions=0; local side added sb_prompt prompt fix, sampling cleanup, C3/C4 generators, and updated VM_TASKS.md

Progress: [####------] 40%

## Accumulated Context

### Decisions

Logged in PROJECT.md Key Decisions. Recent:
- Phase 1 gates cleared: parser, marker metrics, ALIGN, C1/C2 pilots.
- C2 pilot result: multi_turn_emissions=0 across all 10 conversations, so Vicuna remains viable for C2/C3.
- Customer-service genre issue came from using topic titles only; fixed by feeding the verbatim Switchboard instruction (`sb_prompt`).
- C3 = two first-person contexts using the same model; C4 = independent first-person agents using different models.
- The docs still need final confirmation of the condition matrix: "6 conditions" conflicts with "C1-C4 x P0/P1" (8 cells).

### Pending Todos

- Review VM smoke outputs for C1/C2/C3/C4 after the prompt fix.
- Confirm final Phase 2 condition matrix before 50-conversation scaling.
- Decide whether C4 can keep Vicuna + Mistral loaded together on the V100 or needs a memory workaround.

### Blockers/Concerns

- Final condition matrix ambiguity: "6 conditions" vs C1-C4 x P0/P1.
- API access for the Phase-3 LLM-judge is unconfirmed (course-staff meeting pending).

## Session Continuity

Last session: 2026-06-22
Stopped at: Phase 2 prep handoff written; VM should smoke-test corrected prompts and C3/C4 generators, then hold.
Resume file: VM_TASKS.md (VM tasks) / VM_REPORT.md (results)
