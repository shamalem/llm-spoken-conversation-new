# VM Tasks — Regenerate C1 with repetition controls OFF (2026-07-11, CORRECTED)

Owner: local side. Read `CLAUDE.md` first. **Fire-and-forget job — launch detached, confirm it
started, then STOP.** It commits and pushes by itself.

## What & why — READ THIS, the earlier fix was wrong
The first fix attempt loosened C1's n-gram ban from 3 to 6 and kept a `repetition_penalty` of
1.15. **It did NOT work** — the re-run still produced fragments (median 9 words). Root cause was
misdiagnosed: the problem is not the n-gram *size*, it is that C1 must repeat the speaker labels
(`ParticipantA:` / `ParticipantB:`) on every line, and ANY `repetition_penalty > 1.0` (plus the
n-gram ban) punishes that required repetition, so the model hits EOS after 1-2 turns.

Proof: the early C1 pilots that produced full 250-400-word conversations ran BEFORE these
controls were added (they were added 2026-07-01 to fix the C3 two-agent loops). C1 all-at-once
does not have C3's loop failure mode, so it does not need them.

**Corrected fix (now on `main`):** `generate_c1.py` defaults are now `--repetition-penalty 1.0`
(off) and `--no-repeat-ngram 0` (off) — the exact known-good pilot decoding. This job
regenerates ONLY C1. C2/C3/C4 are fine and are NOT touched.

## TASK 0 — Clear the failed partial output first (IMPORTANT)
The previous run wrote ~33 fragment files into `data/generated_v2/C1-P0/` before it was killed.
The runner resumes by SKIPPING existing ids, so those fragments must be removed or the rerun
will keep them. The good originals are safe in the `.broken` dirs; only clear the active dirs:
```bash
cd ~/llm-spoken-conversation && git pull --ff-only origin main
rm -rf data/generated_v2/C1-P0 data/generated_v2/C1-P1 data/generated_v2/C1-P2
```
Do NOT touch `data/generated_v2/C1-P0.broken` (etc.) — those are the archived comparison copies.

## TASK 1 — Pull and verify GPU
```bash
cd ~/llm-spoken-conversation
git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py && echo "SYNTAX OK"
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
Confirm the CORRECTED fix is present before launching (both must read default 1.0 / 0):
```bash
grep -nE "repetition-penalty|no-repeat-ngram" generation/generate_c1.py
# expect: --repetition-penalty ... default=1.0   AND   --no-repeat-ngram ... default=0
```
If either still shows 1.15 or 6, you did not pull the corrected fix — `git pull` again.
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
