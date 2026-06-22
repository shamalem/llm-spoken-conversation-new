# LLM Spoken Conversation Simulation - Generation Architecture Study

## What This Is

A Technion course research project (NLP / computational learning) testing whether the
**generation architecture** used to produce LLM conversations - all-at-once vs turn-by-turn
vs two independent agents - changes how closely those conversations match real human
telephone conversations (Switchboard), independent of the prompt and the model. It extends
Mayor, Bietti & Bangerter (2025), whose design confounded model, prompt, and architecture.

## Core Value

Cleanly isolate the effect of **generation architecture** on conversational realism - the
confound the original paper could not separate - measured against the Switchboard corpus.

## Requirements

### Validated

- Switchboard parser reproduces the paper's baseline (~14 words/turn) - Phase 1 (local)
- Coordination-marker detectors reproduce the paper's Table 5 ranking - Phase 1 (local)
- ALIGN reproduces the Switchboard conceptual-alignment baseline (~0.57) - Phase 1 (VM)
- C2 turn-by-turn pilot shows Vicuna can emit one turn at a time (`multi_turn_emissions=0`) - Phase 1 (VM)

### Active

- [ ] Confirm final Phase 2 condition matrix ("6 conditions" vs C1-C4 x P0/P1)
- [ ] Generate conversations across the final architecture/prompt matrix, 50 per condition
- [ ] Compute turn length, ALIGN alignment, and marker rates per condition vs Switchboard
- [ ] Test the Independence Gradient hypothesis statistically (C1 >= C2 >= C3 >= C4 -> SB)
- [ ] Novel extension: LLM-as-judge humanness rating, or qualitative failure-mode coding
- [ ] Final poster / presentation

### Out of Scope

- Human evaluation study (Study 2 replication) - no ethics approval, budget, or time
- Fine-tuning a model - too risky for the semester timeline; changes the project
- Full 4x3 x 200-conversation design (2,400 convs) - too large; reduced condition matrix
- P2 few-shot as a headline condition - circular-evaluation risk (prompting for what we measure)

## Context

- Based on Mayor et al. (2025), "Can LLMs Simulate Spoken Human Conversations?" (Cognitive Science 49:e70106).
- Benchmark: Switchboard (SwDA distribution) - transcripts + topic/demographic metadata, local only.
- Compute: Azure NC6s v3 VM, 1x V100 16 GB; HuggingFace local inference (no API budget for generation).
- Models: Vicuna-13B (C1-C3), Mistral-7B-Instruct (C4 second agent).
- Workflow: two machines (local Windows + Linux VM) sharing a private GitHub repo; see CLAUDE.md / VM_TASKS.md.
- Team of 3 - Person A (generation), B (analysis), C (QC / extension / poster).

## Constraints

- **Tech stack**: HuggingFace local generation on a V100 16 GB - course constraint (no API keys for generation)
- **Models**: Vicuna-13B + Mistral-7B - fit 16 GB in 4-bit and match the paper's model family
- **Data**: Switchboard is LDC-licensed - kept local only, never committed or redistributed
- **Evaluation**: avoid circular evaluation - the P1 prompt omits the coordination markers we then measure
- **Timeline**: semester course, phase-based per the syllabus

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Reduce 12 conditions to a smaller matrix | Manageable scope, sufficient power at 50/condition | Needs final matrix confirmation |
| C3 = two first-person contexts, same model | Cleanly isolates the single-author effect vs C2 | Implemented; VM smoke pending |
| C4 = two first-person contexts, different models | Adds model-identity separation after the C3 comparison | Implemented; VM smoke pending |
| Headline = Independence Gradient hypothesis | Explains the paper's unexplained Vicuna result; one falsifiable prediction | Pending |
| Vicuna-13B as primary generator | Matches the paper, fits the VM | Good - C2 pilot had `multi_turn_emissions=0` |
| Validate-before-generate | Pipeline correctness before spending compute | Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? Move to Out of Scope with reason
2. Requirements validated? Move to Validated with phase reference
3. New requirements emerged? Add to Active
4. Decisions to log? Add to Key Decisions
5. "What This Is" still accurate? Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-22 after Phase 1 VM validation and Phase 2 prep.*
