# VM Tasks — Regenerate C1 with the no-repeat-ngram fix (2026-07-11)

Owner: local side. Read `CLAUDE.md` first. **Fire-and-forget job — launch detached, confirm it
started, then STOP.** It commits and pushes by itself.

## What & why
The full 600-conversation regen ran C1 with the OLD generator, which let C1 fall through to
chat()'s per-turn default `no_repeat_ngram_size=3`. Over a whole ~2000-token conversation that
hard 3-gram ban is unsatisfiable, so C1 generation died almost immediately — **72% of C1-P0,
90% of C1-P1, 86% of C1-P2 came out as 1-2 line fragments.** C2/C3/C4 are fine and are NOT
touched by this job.

The fix is now on `main`: `generate_c1.py` sets its own `--repetition-penalty 1.15` and
`--no-repeat-ngram 6` (same value the turn-by-turn family uses) and passes them into chat().
The broken output was moved to `data/generated_v2/C1-P{0,1,2}.broken/`, so C1-P0/P1/P2 start
fresh (50 each). This job regenerates ONLY C1.

## TASK 1 — Pull and verify GPU
```bash
cd ~/llm-spoken-conversation
git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py && echo "SYNTAX OK"
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
Confirm the fix is present before launching:
```bash
grep -n "no-repeat-ngram" generation/generate_c1.py   # expect the arg with default=6
```
If `nvidia-smi` shows an **NVML driver/library mismatch**, `sudo reboot`, reconnect,
`conda activate convsim`, then continue. Never reboot mid-run. (C1 loads only Vicuna, ~9 GB —
this is a light run, not the C4 two-model case.)

## TASK 2 — Launch detached, then leave
```bash
tmux new-session -d -s c1regen 'cd ~/llm-spoken-conversation && bash generation/run_c1_regen.sh'
```

## TASK 3 — Confirm it started, then STOP
```bash
sleep 20
tmux ls                        # expect a "c1regen" session
tail -n 15 run_c1_regen.log    # expect cuda True, then "C1 P0" and "saved <id> (NNN words)"
```
The word count on each `saved` line is the health signal: full C1 conversations are **~200-400
words**. If you see `saved <id> (5 words)` style fragments, the fix did NOT take — STOP and tell
Diyar. If the saved lines show a few hundred words each, **you are done — disconnect.** The
script does C1-P0, then P1, then P2 (50 each) and commits + pushes after each. Do not babysit.

## TASK 4 — Verify when done (no GPU needed; safe to run anytime)
```bash
/anaconda/envs/convsim/bin/python analysis/quality_report.py data/generated_v2
```
**GO check:** C1-P0/P1/P2 must now **appear** in the table (they were silently skipped before
the regex fix) with **medWPT ~15-20** and frag/loop/drift/leak all near 0. If C1 rows are
missing or medWPT is tiny (<8), the regen did not work — report it.

## Do NOT
- Do **not** touch `data/generated/` (pre-fix baseline) or the `.broken` C1 dirs (kept for
  before/after comparison).
- Do **not** regenerate C2/C3/C4 — they are fine. (C2's not-closing-at-cap issue is a SEPARATE,
  later task; do not start it here.)
- Do **not** run in the foreground or babysit.

## After it pushes
Local pulls the fresh C1 data, re-runs the quality report + `analysis/evaluate_generated.py`,
and confirms C1 words/turn is in range before the analysis phase. Then the only open generation
item is C2 natural-termination (separate task).
