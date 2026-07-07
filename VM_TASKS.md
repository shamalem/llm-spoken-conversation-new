# VM Tasks — C2 language-drift A/B (2026-07-06)

Owner: local side. Read `CLAUDE.md` first. **Small, fire-and-forget job — launch detached,
confirm it started, then stop.** It commits and pushes by itself.

## What & why
The C2 fix test fixed fragmentation and (with a stronger cleaner) the label leaks, but ~10 turns
still drifted into **Spanish**, almost all in conv 2095 — likely the model escaping the
`no-repeat-ngram` ban once English saturates. This A/B regenerates the drift-prone conversations
under **looser** n-gram settings (6 and 0) and reports Spanish-drift turns AND verbatim-loop
turns, so we can pick a setting that removes the drift **without** bringing the old loops back.
(Baseline at ngram=4 for these ids: 10 Spanish turns, 0 loops.)

Writes to `data/generated_test/c2drift/ngram6/` and `.../ngram0/`. Does **not** touch
`data/generated/` or the earlier `data/generated_test/c2fix/`.

## TASK 1 — Pull and verify GPU
```bash
cd ~/llm-spoken-conversation
git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py && echo "SYNTAX OK"
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an **NVML driver/library mismatch**, `sudo reboot`, reconnect,
`conda activate convsim`, then continue. Never reboot mid-run.

## TASK 2 — Launch detached, then leave
```bash
tmux new-session -d -s c2drift 'cd ~/llm-spoken-conversation && bash generation/run_c2_drift_test.sh'
```

## TASK 3 — Confirm it started, then STOP
```bash
sleep 20
tmux ls                            # expect a "c2drift" session
tail -n 15 run_c2_drift_test.log   # expect cuda True and "generating with --no-repeat-ngram 6"
```
If the session exists and generation is running, **you are done — disconnect.** The script
regenerates 4 conversations twice (~8 generations, a few minutes), prints a Spanish-drift +
verbatim-loop count per setting, and **commits + pushes** by itself. Do not wait for it.

## Do NOT
- Do **not** touch `data/generated/` or `data/generated_test/c2fix/`.
- Do **not** start the full P0 regeneration yet.
- Do **not** run in the foreground or babysit.

## After it pushes
Local pulls `data/generated_test/c2drift/`, compares the two settings (drift down? loops still
0?), picks the n-gram value for the turn-by-turn family, then we validate C4, glance at C1, and
run the full P0 regeneration.
