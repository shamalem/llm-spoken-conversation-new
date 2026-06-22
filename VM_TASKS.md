# VM Tasks - Phase 2 prep

Owner of this file: **local side** (do not edit on the VM).
VM-side agent: read `CLAUDE.md`, `.planning/PROJECT.md`, `.planning/ROADMAP.md`, and this
file first. Do the tasks below and append every result/blocker to `VM_REPORT.md`.

Phase 1 gates are cleared:
- ALIGN validation matches the Switchboard baseline.
- C2 pilot produced `multi_turn_emissions = 0` across 10/10 conversations.
- Vicuna remains viable for C2/C3.

Important local changes in this handoff:
- C1/C2 now feed the verbatim Switchboard `prompt` (`sb_prompt`) instead of only the topic
  title, to avoid the customer-service genre drift.
- Sampling is passed explicitly per generation call to avoid the transformers temperature
  warning.
- New generators exist: `generation/generate_c3.py` and `generation/generate_c4.py`.

Do **not** start the full Phase 2 scale run yet. First run smoke tests and report. The
project docs still say "6 conditions" while also saying `C1-C4 x P0/P1` (8 cells), so the
local side will confirm the exact final condition matrix after these smoke results.

## TASK 1 - Pull and verify code

```bash
git pull --ff-only
python -m py_compile generation/*.py prompts/*.py analysis/*.py
```

Report:
- current commit hash
- whether syntax checks pass
- whether the VM still imports `transformers`, `bitsandbytes`, and `torch`

## TASK 2 - Retire old P0 pilot outputs

The committed `data/generated/C1-P0/` and `data/generated/C2-P0/` files were generated
before the `sb_prompt` fix and show customer-service framing. Before new smoke tests:

```bash
rm -rf data/generated/C1-P0 data/generated/C2-P0
```

This lets the corrected C1/C2 scripts regenerate those first ids instead of skipping them.

## TASK 3 - Smoke-test corrected Phase 2 generators

Run in `tmux` or another persistent session:

```bash
python generation/generate_c1.py --prompt P0 --n 2 --max-new-tokens 1024
python generation/generate_c2.py --prompt P0 --n 2 --max-turns 12
python generation/generate_c3.py --prompt P0 --n 2 --max-turns 12
python generation/generate_c4.py --prompt P0 --n 2 --max-turns 12
```

Report for each architecture:
- did the output use the Switchboard instruction as a caller discussion task?
- did the customer-service/help-desk framing disappear?
- `n_turns`
- `multi_turn_emissions`
- any model-load or VRAM issues, especially for C4 loading Vicuna + Mistral together

## TASK 4 - Hold before scaling

Do **not** run 50 conversations/condition yet.

Append the smoke-test summary to `VM_REPORT.md`, commit, and push:

```bash
git status
git add -A
git commit -m "feat(vm): smoke-test Phase 2 generators"
git push
```

Then hold. The local side will review the raw smoke conversations and send the final
Phase 2 scale instructions.
