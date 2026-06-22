# Project Plan — Isolating Generation Architecture in LLM-Simulated Spoken Conversation

**Course:** Project Design in Computational Learning (0950280), Technion, Spring 2026
**Base paper (ID 4):** Mayor, Bietti & Bangerter (2025), *Can Large Language Models Simulate Spoken Human Conversations?* Cognitive Science 49:e70106.
**Team:** [name 1], [name 2], [name 3]

---

## 1. Motivation — the confound in the original paper

Mayor et al. compared real Switchboard (SB) telephone conversations to LLM-generated
conversations and found LLMs differ on turn length, alignment, coordination markers
(`oh`, `okay`, `uh-huh`), and openings/closings. **But their design confounds three
factors**: model, prompt, and generation architecture.

- GPT-4 / Claude → generated **turn-by-turn**
- Vicuna / Wayfarer → generated **all-at-once**
- Some corpora got extra prompts (openings/closings, coordination markers, variation)

So when Vicuna (all-at-once) turned out *closest* to Switchboard on alignment, the authors
admit they cannot tell whether that is due to the model or the generation mode
(paper §2.5). That is the gap we close.

## 2. Research question

> Do **generation architecture** and **prompting strategy** independently affect how much
> LLM-generated telephone conversations differ from real Switchboard conversations, and
> which combination minimizes the gap?

## 3. Headline hypothesis — the Independence Gradient

> As the two speakers become more **independent** — from one model writing everything at
> once (C1), to one model alternating turns with full visibility (C2), to two first-person
> instances of the same model (C3), to two different models (C4) — the **exaggerated
> alignment** identified by the paper should progressively **decrease**, approaching human
> (Switchboard) levels: **C1 ≥ C2 ≥ C3 ≥ C4 → SB.**

This explains the paper's unexplained Vicuna result and turns the project from "can we fix
it?" (not guaranteed) into "what property of generation produces the gap?" (answerable).
A non-monotonic result is also a finding (it locates where independence stops mattering).

## 4. Experimental design

Four architectures × two prompt levels, reduced to **6 meaningful conditions**,
**50 conversations each** (300 total), benchmarked against **200 Switchboard conversations**.

| Condition | Architecture | Prompt | Model(s) |
|-----------|--------------|--------|----------|
| C1-P0 | All-at-once | Basic | Vicuna-13B |
| C1-P1 | All-at-once | Spoken | Vicuna-13B |
| C2-P0 | Turn-by-turn, single model | Basic | Vicuna-13B |
| C2-P1 | Turn-by-turn, single model | Spoken | Vicuna-13B |
| C3-P1 | Two independent agents, **same** model | Spoken | Vicuna-13B × 2 sessions |
| C4-P1 | Two independent agents, **different** models | Spoken | Vicuna-13B ↔ Mistral-7B |

**Clean comparisons:**
- C1-P0 vs C2-P0 → architecture effect, baseline prompt
- C2-P0 vs C2-P1 → prompt effect, fixed architecture
- C2-P1 vs C3-P1 → single-author vs independent-agents (same model)
- C3-P1 vs C4-P1 → same-model vs different-model agents

### Architecture definitions (critical)

- **C1 all-at-once:** one forward pass produces the whole dialogue.
- **C2 turn-by-turn single:** one model sees the full shared transcript as a *script* and
  is told "write the next turn for ParticipantX." It knows it authors both sides.
- **C3 two agents, same model:** two *separate* first-person sessions of Vicuna-13B. Each
  has its own persona system prompt and sees the conversation **only from its own point of
  view** (its turns = `assistant`, partner's turns = `user`). Neither sees a god's-eye
  script. This isolates the single-author effect.
- **C4 two agents, different models:** as C3, but the two sessions are different models
  (Vicuna ↔ Mistral).

**Consequence:** in C3/C4 no single controller decides length or ending — the two agents
must *coordinate* the closing, directly stress-testing the paper's headline finding.
A hard cap (~60 turns) prevents runaway; natural termination is what we measure.

## 5. Prompts

- **P0 (basic):** replicates the paper's basic prompt — ~50-turn target, "do not end too
  early," persona = gender/age/education, topic matched to an SB conversation.
- **P1 (spoken):** short turns (1–3 sentences), natural ending permitted, informal register,
  **no mention of specific coordination markers** (so marker rates are a clean outcome).

Exact builders live in `prompts/templates.py`.

## 6. Metrics

**Tier 1 (must run, directly comparable to paper):**
1. Words per turn — opening / main body / closing
2. Conceptual alignment — Earlier vs Later × Corpus (ALIGN package)
3. Syntactic alignment — same
4. `uh-huh` rate per 100 words
5. `okay` rate per 100 words

**Tier 2 (if time):** `oh` rate; topic-initiation turn; sycophancy-token frequency
(`absolutely`, `I couldn't agree more`, etc.) via regex.

**Skip:** full opening/closing annotation (labor-heavy); lexical alignment (least
informative in the paper).

**Stats:** mixed-effects regression per condition × outcome, paper's formula
`DV ~ Corpus * Section + (1 | ConvID)` (R + lme4 preferred for comparability).

## 7. Phase plan

### Phase 1 — Pipeline validation (week 1)
Set up VM; install stack; install ALIGN; download OSF data; run ALIGN on 30 SB
conversations and confirm numbers match the paper (SB main body ≈14 words/turn,
conceptual alignment Earlier ≈0.57); generate 10 pilot conversations C1-P0 and C2-P0 with
Vicuna and sanity-check against the paper's Vicuna / GPT4-1 cells.
**Gate:** do not proceed until SB baseline reproduces.

### Phase 2 — Main experiment (weeks 2–4)
Generate all 6 conditions (300 conversations); run ALIGN + marker metrics on everything;
run mixed-effects models; produce figures (one per metric, all conditions vs SB);
test the Independence Gradient ordering.

### Phase 3 — Extension (weeks 5–6)
**If API access:** LLM-as-judge — automated Study 2. Feed excerpts to a strong model,
ask "human or AI + why," test whether judge verdicts track ALIGN metrics and the gradient.
**Fallback (no API):** systematic qualitative coding of failure modes (sycophancy,
question patterns, closing-appointment phenomenon) across conditions.
**Free second analyses (same data):** closing-coordination behavior in C3/C4 (who ends the
call? politeness loops?); continuous alignment trajectory across all turns vs architecture.

## 8. Division of labor

- **Person A — Generation & infrastructure:** VM setup; generation scripts for C1, C2, C3, C4; conversation storage.
- **Person B — Analysis pipeline:** OSF data; ALIGN install + validation; SB baseline; run all metrics; mixed-effects models; figures.
- **Person C — QC, Phase 3 & presentation:** read/flag generated conversations; Tier-2 token metrics; lead LLM-judge design; lead poster + narrative; write comparison to the original paper.

All three share interpretation and final write-up.

## 9. Open items

- [ ] API access / budget — pending course-staff reply (only affects Phase 3 strength).
- [ ] Confirm HuggingFace-only generation is acceptable for grading — pending reply.
- [ ] VM GPU specs — confirm via `nvidia-smi` (expected: 1× V100 16GB).
- [ ] OSF data format — to inspect before writing the validation script.

## 10. Standing principles

1. **Validate before generate** — ALIGN must reproduce SB numbers first.
2. **Replicate one cell** — run a condition matching the paper (Vicuna all-at-once) as a pipeline check.
3. **No scope creep** — novelty comes from the gradient hypothesis and analyses, not more conditions.
4. **Switchboard is the fixed yardstick** — every comparison returns to SB.
5. **Avoid circular evaluation** — never prompt for a feature we then measure (this is why P1 omits marker instructions).

---
*Living document — update at each phase boundary.*
