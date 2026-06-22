# Requirements: LLM Spoken Conversation Simulation

**Defined:** 2026-06-22
**Core Value:** Isolate the effect of generation architecture on conversational realism vs Switchboard

## v1 Requirements

### Pipeline & Validation
- [x] **PIPE-01**: Switchboard parser reproduces the paper's words/turn baseline (~14)
- [x] **PIPE-02**: oh/okay/uh-huh detectors reproduce the paper's Table 5 ranking
- [x] **PIPE-03**: ALIGN reproduces SB conceptual alignment (~0.57 Earlier) on the VM

### Generation
- [x] **GEN-01**: C1 all-at-once generator, conversations matched to SB topic + demographics
- [x] **GEN-02**: C2 turn-by-turn single-model generator with single-turn enforcement
- [ ] **GEN-03**: C3 two independent same-model agents (first-person contexts)
- [ ] **GEN-04**: C4 two independent different-model agents (Vicuna ↔ Mistral)
- [ ] **GEN-05**: 6 conditions × 50 conversations generated and stored

### Analysis
- [ ] **ANLY-01**: words/turn per condition vs SB (opening / main body / closing)
- [ ] **ANLY-02**: conceptual & syntactic alignment (Earlier/Later × Corpus) per condition
- [ ] **ANLY-03**: oh/okay/uh-huh rates per condition vs SB
- [ ] **ANLY-04**: mixed-effects models (DV ~ Corpus*Section + (1|ConvID)) per condition × metric
- [ ] **ANLY-05**: Independence Gradient tested (C1 ≥ C2 ≥ C3 ≥ C4 → SB)

### Extension & Delivery
- [ ] **EXT-01**: LLM-as-judge humanness rating (if API) OR qualitative failure-mode coding
- [ ] **EXT-02**: final poster / presentation with key figures and narrative

## v2 Requirements

Deferred — tracked, not in the current roadmap.

### Extensions
- **EXT2-01**: Turn-count ablation (30 vs 50 target turns) — does length pressure inflate alignment?
- **EXT2-02**: Closing-coordination analysis in C3/C4 (who ends the call? politeness loops?)
- **EXT2-03**: Continuous alignment trajectory across all turns vs architecture
- **EXT2-04**: P2 few-shot condition (non-lexical metrics only, to avoid circularity)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Human evaluation (Study 2 replication) | No ethics approval, budget, or time |
| Fine-tuning a model | Too risky for the semester timeline; changes the project |
| Full 4×3 × 200-conv design (2,400) | Too large; reduced to 6 × 50 |
| Additional models (Gemini, Llama-70B) | Multiplies conditions without payoff; V100 limits |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 1 | Complete |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| GEN-01 | Phase 1 | Complete |
| GEN-02 | Phase 1 | Complete |
| GEN-03 | Phase 2 | In Progress |
| GEN-04 | Phase 2 | In Progress |
| GEN-05 | Phase 2 | Pending |
| ANLY-01 | Phase 2 | Pending |
| ANLY-02 | Phase 2 | Pending |
| ANLY-03 | Phase 2 | Pending |
| ANLY-04 | Phase 2 | Pending |
| ANLY-05 | Phase 2 | Pending |
| EXT-01 | Phase 3 | Pending |
| EXT-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-22*
*Last updated: 2026-06-22 after GSD setup*
