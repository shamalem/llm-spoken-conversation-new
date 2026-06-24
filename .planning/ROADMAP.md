# Roadmap: LLM Spoken Conversation Simulation

## Overview

Three phases mapping onto the course structure: first validate the measurement pipeline and
prove generation works (Phase 1), then run the full 6-condition experiment and analyze it
(Phase 2), then add a novel extension and build the poster (Phase 3). Every comparison is
against the Switchboard corpus.

## Phases

- [x] **Phase 1: Pipeline Validation & Generation Pilots** - Validate metrics vs the paper; prove C1/C2 generation works on the VM
- [ ] **Phase 2: Main Experiment & Analysis** - Generate all 12 conditions (C1-C4 × P0/P1/P2); compute metrics; test the Independence Gradient
- [ ] **Phase 3: Extension & Presentation** - LLM-judge or qualitative coding; poster

## Phase Details

### Phase 1: Pipeline Validation & Generation Pilots
**Goal**: A measurement pipeline proven against the paper's Switchboard numbers, plus working C1/C2 generation pilots on the VM.
**Depends on**: Nothing (first phase)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, GEN-01, GEN-02
**Success Criteria** (what must be TRUE):
  1. Switchboard words/turn and marker rates reproduce the paper (✓ done locally)
  2. ALIGN reproduces SB conceptual alignment (~0.57 Earlier) on the VM
  3. C1 pilot produces coherent conversations matched to SB topic/demographics
  4. C2 pilot reveals whether Vicuna can do turn-by-turn (multi_turn_emissions)
**Plans**: 4

Plans:
- [x] 01-01: Switchboard parser + words/turn validation (local)
- [x] 01-02: Marker/sycophancy metrics + Table 5 validation (local)
- [x] 01-03: VM environment + ALIGN install + ALIGN validation
- [x] 01-04: Run C1 and C2 pilots; decide turn-by-turn model viability

### Phase 2: Main Experiment & Analysis
**Goal**: All 12 conditions (C1-C4 × P0/P1/P2) generated (50 each) and fully analyzed, with the Independence Gradient tested.
**Depends on**: Phase 1
**Requirements**: GEN-03, GEN-04, GEN-05, ANLY-01, ANLY-02, ANLY-03, ANLY-04, ANLY-05
**Condition matrix**: 4 architectures (C1 all-at-once, C2 turn-by-turn single-model,
  C3 two same-model agents, C4 two different-model agents) × 3 prompt levels (P0 paper-basic,
  P1 our intervention, P2 few-shot) = 12 conditions. The gradient is tested *within* a fixed
  prompt level. **P2 is a robustness condition, not headline**: it contains a verbatim
  Switchboard few-shot excerpt, so it is analyzed on **non-lexical metrics only**
  (words/turn, alignment) — never on the oh/okay/uh-huh marker rates, which would be circular.
**Success Criteria** (what must be TRUE):
  1. 12 conditions × 50 conversations generated and stored
  2. words/turn, alignment, and marker rates computed per condition vs SB
  3. Mixed-effects models run per condition × metric
  4. The Independence Gradient ordering is evaluated (supported or not)
**Plans**: 4

Plans:
- [ ] 02-01: C3 and C4 generators (gated on Phase 1 turn-by-turn result)
- [ ] 02-02: Generate all 6 conditions at scale
- [ ] 02-03: Run ALIGN + marker + turn-length metrics on all conditions
- [ ] 02-04: Mixed-effects models + gradient test + figures

### Phase 3: Extension & Presentation
**Goal**: One novel extension beyond the paper, plus the final poster.
**Depends on**: Phase 2
**Requirements**: EXT-01, EXT-02
**Success Criteria** (what must be TRUE):
  1. LLM-judge analysis (if API) or systematic qualitative failure-mode coding is complete
  2. Poster presents the research question, key figures, and the gradient finding clearly
**Plans**: 2

Plans:
- [ ] 03-01: Extension (LLM-judge or qualitative coding)
- [ ] 03-02: Poster + presentation

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pipeline Validation & Pilots | 4/4 | Complete | 2026-06-22 |
| 2. Main Experiment & Analysis | 0/4 | In progress | - |
| 3. Extension & Presentation | 0/2 | Not started | - |

