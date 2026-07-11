# VM Tasks — Final pre-regen check: C4 + C3 at ngram6 (2026-07-06)

Owner: local side. Read `CLAUDE.md` first. **Small, fire-and-forget job — launch detached,
confirm it started, then stop.** It commits and pushes by itself.

## What & why
The C2 drift A/B settled the last decoding knob: **no-repeat-ngram = 6** (kills the Spanish
drift, keeps verbatim loops at 0, labels clean). That is now the default for the whole
turn-by-turn family (C2/C3/C4), alongside min-new-tokens 16, stop-at-sentence, repetition 1.15,
and natural termination.

Before the full regeneration, this one small check confirms the two untested pieces:
- **C4** at the final settings (never validated), and
- **C3** stays loop-free at the looser ngram6 (it was validated at 4).

Writes to `data/generated_test/finalcheck/`. Does **not** touch `data/generated/`.

## TASK 1 — Pull and verify GPU
```bash
cd ~/llm-spoken-conversation
git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py && echo "SYNTAX OK"
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an **NVML driver/library mismatch**, `sudo reboot`, reconnect,
`conda activate convsim`, then continue. Never reboot mid-run. (C4 loads Vicuna + Mistral,
~15 GB peak on the 16 GB V100 — this is the run most likely to hit the NVML issue.)

## TASK 2 — Launch detached, then leave
```bash
tmux new-session -d -s finalcheck 'cd ~/llm-spoken-conversation && bash generation/run_final_check.sh'
```

## TASK 3 — Confirm it started, then STOP
```bash
sleep 20
tmux ls                          # expect a "finalcheck" session
tail -n 15 run_final_check.log   # expect cuda True and "C3 re-check at ngram6"
```
If the session exists and generation is running, **you are done — disconnect.** The script
generates ~7 conversations (C4 is slower — two models), prints a per-conversation quality line,
and **commits + pushes** by itself. Do not wait for it.

## Do NOT
- Do **not** touch `data/generated/` or the earlier `data/generated_test/` runs.
- Do **not** start the full P0 regeneration yet — that's the next step after local reviews this.
- Do **not** run in the foreground or babysit.

## After it pushes
Local pulls `data/generated_test/finalcheck/`, confirms C4 is coherent and C3 has 0 loops at
ngram6, then we run the full P0 regeneration (`run_p0_v2.sh`) into `data/generated_v2/`.
