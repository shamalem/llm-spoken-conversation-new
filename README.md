# LLM-Simulated Spoken Conversation — Generation Architecture Study

Technion course project (0950280), Spring 2026. Based on Mayor, Bietti & Bangerter (2025),
*Can Large Language Models Simulate Spoken Human Conversations?*

We isolate **generation architecture** (all-at-once vs turn-by-turn vs two independent
agents) from **prompting** and **model identity** — the three factors the original paper
confounds — and test whether more independent speakers produce conversations closer to the
real Switchboard corpus. Full plan: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**.

## Repository layout

```
PROJECT_PLAN.md        full locked plan (read this first)
requirements.txt       Python generation + metrics stack
prompts/templates.py   P0/P1 prompt + message builders for C1–C4
generation/            generation scripts per architecture        (added in Phase 1/2)
analysis/              ALIGN validation, metrics, stats, figures   (added in Phase 1/2)
data/switchboard/      Switchboard sample from OSF                  (downloaded)
data/generated/        generated conversations                     (produced)
results/               tables + figures                            (produced)
```

## Setup (on the Azure VM)

```bash
nvidia-smi                         # confirm GPU + CUDA version
conda create -n convsim python=3.10 -y
conda activate convsim
pip install torch --index-url https://download.pytorch.org/whl/cu118   # match CUDA
pip install -r requirements.txt
```

ALIGN is installed separately (see PROJECT_PLAN.md §7 / requirements.txt note).

## Run order (the golden rule: validate before generate)

1. **Phase 1 — validate.** Download OSF data; run ALIGN on Switchboard; confirm numbers
   match the paper (SB main body ≈14 words/turn, conceptual alignment Earlier ≈0.57).
   **Do not generate novel data until this passes.**
2. **Phase 1 — pilot.** Generate 10 conversations each for C1-P0 and C2-P0; sanity-check
   against the paper's Vicuna / GPT4-1 cells.
3. **Phase 2 — main run.** Generate all 6 conditions (50 each); run metrics; run stats.
4. **Phase 3 — extension.** LLM-as-judge (if API) or qualitative coding (fallback).

## Models

- Generation: **Vicuna-13B v1.5** (`lmsys/vicuna-13b-v1.5-16k`) for C1–C3;
  **Mistral-7B-Instruct** as the second agent in C4.
- Evaluation (Phase 3, optional): a strong API model as judge, if access is granted.
