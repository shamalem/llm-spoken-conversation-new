# VM Tasks — current

Owner of this file: **local side** (do not edit on the VM).
VM-side agent: read `CLAUDE.md` and `PROJECT_PLAN.md` first, then work these tasks and
record everything in `VM_REPORT.md`.

> Phase 1 goal: a validated measurement pipeline. **Do NOT generate novel conversations
> until ALIGN reproduces the Switchboard baseline** (a later task, added once setup is done).

## TASK 1 — Environment
- Run `nvidia-smi`; record GPU model, VRAM, driver, and CUDA version in `VM_REPORT.md`.
- Create conda env `convsim` (Python 3.10).
- Install `torch` matching the CUDA version, then `pip install -r requirements.txt`.
- Report: torch version, `torch.cuda.is_available()`, and any install errors.

## TASK 2 — Switchboard data (osf.io/zxwtr)
- Download the OSF project files. In `VM_REPORT.md`, list the **folder tree** and the
  **file formats** (CSV / TXT / JSON?).
- Identify (a) the 200-conversation Switchboard sample, (b) the authors' generated corpora,
  (c) their ALIGN + generation **Colab notebooks**.
- Put the Switchboard sample under `data/switchboard/`. **Do not commit it** (gitignored,
  LDC-licensed).

## TASK 3 — ALIGN install
- Install the ALIGN package (Duran et al., 2019) the way the authors' OSF Colab does.
- Record the exact install commands that worked, plus the version, in `VM_REPORT.md`.
- Note any dependency conflicts with the generation stack (a separate env is fine).

## TASK 4 — Download Vicuna (no generation yet)
- Pre-download `lmsys/vicuna-13b-v1.5-16k` in 4-bit and confirm it loads and produces a
  one-line test completion. Report VRAM used and tokens/sec.
- **Stop there.** Do not run any experimental generation.

## TASK 5 — Report and wait
- Commit + push. Confirm in `VM_REPORT.md` that TASKS 1–4 are done.
- The local side will then add the **Phase-1 validation notebook** (ALIGN on 30 SB
  conversations) and the **C1-P0 pilot generation script**.
