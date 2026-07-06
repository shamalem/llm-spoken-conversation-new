# VM Tasks — Regenerate the full P0 set into generated_v2 (2026-07-06)

Owner: local side. Read `CLAUDE.md` first. **This job is fire-and-forget — launch it detached,
confirm it started, then stop. Do NOT stay attached, do NOT poll, do NOT run it in the
foreground.** The script commits and pushes by itself.

## What & why
Regenerate all four P0 conditions (C1, C2, C3, C4 — 50 conversations each, 30-turn cap) with
**unified decoding**. `repetition_penalty` is now a default inside `chat()`, so every
architecture uses the same decoding — this removes the earlier confound where only C2/C3 had
the repetition fix. Output goes to a **separate directory `data/generated_v2/`** so the current
pilot in `data/generated/` is left untouched (a teammate is inspecting it in parallel).

The runner `generation/run_p0_v2.sh` does everything: it writes each conversation immediately,
resumes on rerun (existing ids are skipped), and **auto-commits + pushes after each condition**.

## TASK 1 — Pull and verify GPU
```bash
cd ~/llm-spoken-conversation          # (or wherever this repo lives on the VM)
git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py prompts/*.py && echo "SYNTAX OK"
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an **NVML driver/library mismatch**, run `sudo reboot`, wait, reconnect,
`conda activate convsim`, and re-check before launching. (This is the crash that killed the last
C4 run — C4 loads two models. Reboot fixes it. Never reboot mid-run.)

## TASK 2 — Launch detached, then leave
```bash
tmux new-session -d -s genp0 'cd ~/llm-spoken-conversation && bash generation/run_p0_v2.sh'
```

## TASK 3 — Confirm it started, then STOP
```bash
sleep 20
tmux ls                         # expect a "genp0" session
tail -n 15 run_p0_v2.log        # expect "run_p0_v2 START", cuda_available True, C1-P0 START
```
If the session exists and the log shows generation starting, **you are done — disconnect.**
The script runs for a few hours, pushes after each condition, and writes a final
`run_p0_v2.status` when all four are done. Do not wait for it.

## Do NOT
- Do **not** touch, delete, or regenerate `data/generated/` (the current pilot — keep it).
- Do **not** run the generators in the foreground or babysit the tmux session.
- Do **not** start P1/P2 yet — that is a separate task after local reviews P0-v2.

## How the user checks progress later (no Codex needed)
```bash
tmux attach -t genp0            # watch live (Ctrl-b then d to detach)
cat run_p0_v2.status           # final per-condition counts once finished
git -C ~/llm-spoken-conversation log --oneline -5   # the auto-pushed data commits
```
