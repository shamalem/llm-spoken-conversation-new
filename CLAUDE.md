# Project context for Claude (loaded automatically)

This repo is a Technion research project: simulating spoken telephone conversations with
LLMs and comparing them to the real **Switchboard** corpus, to isolate the effect of
**generation architecture** from prompting and model identity. **Read `PROJECT_PLAN.md`
before doing anything** — it holds the design, conditions, metrics, and phase plan.

## Two-machine workflow (IMPORTANT)

Work happens on two machines sharing this repo through git:

- **LOCAL (Windows):** planning, prompt design, writing instructions, analysis review.
- **VM (Azure, Linux, GPU):** environment setup, model download, generation, ALIGN metrics.

The two Claude sessions do **not** share memory — git is the only channel. Coordinate via:

1. On the VM, **read `VM_TASKS.md` first** — it holds the current concrete tasks.
2. Do the tasks. Write every result, spec, and blocker into **`VM_REPORT.md`**.
3. `git pull` before working; `git add -A && git commit` after each meaningful step; `git push`.
4. **Local side owns `VM_TASKS.md`; VM side owns `VM_REPORT.md`.** Do not edit the other
   side's file — this prevents merge conflicts.

## Golden rules (do not violate)

- **VALIDATE BEFORE GENERATE.** ALIGN must reproduce the paper's Switchboard numbers
  (main body ≈14 words/turn; conceptual alignment Earlier ≈0.57) **before** any novel
  generation. No exceptions.
- **WRITE EACH CONVERSATION TO DISK IMMEDIATELY**, as its own file
  `data/generated/<condition>/<id>.json`. A dropped SSH or kernel must never lose more
  than the one conversation in flight. Generation scripts resume by skipping existing ids.
- **Run long generation in `tmux`** (or `nohup`) so it survives disconnects.
- **NEVER commit model weights or Switchboard source data.** `.gitignore` enforces this —
  keep it. Switchboard is LDC-licensed; it stays local only.
- **Avoid circular evaluation:** never prompt for a feature we then measure (this is why
  the P1 prompt deliberately omits coordination markers).

## Code conventions

- Shared logic lives in `.py` modules (`prompts/`, `generation/`, `analysis/`). No
  copy-paste across notebooks.
- **Generation = resumable `.py` scripts**, one per architecture, run in `tmux`.
- **Validation / metrics / stats / figures = notebooks**, one per task, importing the
  shared modules. Clear notebook outputs before committing to keep diffs small.
- Models: **Vicuna-13B v1.5** (C1–C3) and **Mistral-7B-Instruct** (C4 second agent),
  4-bit quantized. Build chat inputs with `tokenizer.apply_chat_template(...)`.
