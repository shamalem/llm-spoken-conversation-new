# HANDOFF — task list for Codex (2026-07-06)

Diyar is out until his usage limit resets. This is the ordered task list to continue. Read
`CLAUDE.md` and `.planning/STATE.md` for context. **Golden rules still apply:** run generation
in `tmux`, write each conversation to disk immediately, never commit model weights or the
Switchboard source data, and never edit the other machine's coordination file.

## Where things stand (all pushed to `main`)
- **Generation fixes are DONE for the turn-by-turn family (C2/C3/C4).** Defaults now: min-new-tokens
  16, stop-at-sentence on, repetition-penalty 1.15, no-repeat-ngram 6, natural termination
  (stop at goodbye), aggressive label cleaner. Validated on the worst conversations:
  - C3: word-salad fixed (median words/turn 2→15, 0 fragments), ends naturally.
  - C2: fragmentation fixed; label leaks 49→3 via the stronger cleaner; Spanish language-drift
    fixed by ngram6 (10→0 turns) with 0 verbatim loops.
- **C4 fix (natural termination) is committed but only the `run_final_check.sh` test confirms it.**
  The OLD `data/generated/C4-P0` shows the pre-fix bug: conversations pad the 32-turn cap with
  ~20 turns of repetitive "thank you / you're welcome / goodbye for now" (Mistral won't stop).
- **C1 is all-at-once and coherent.** Its "help-desk / advisor" feel in P0 is NOT a bug to fix —
  P0 is the deliberately weak baseline (no peer guard). P1's peer guard is what makes C1/C4 sound
  like two equals. Do NOT try to "fix" C1-P0.
- The OLD data in `data/generated/` is PRE-FIX. The clean data goes to `data/generated_v2/`.

## HOW TO RUN (every VM job)
Fire-and-forget: launch in a detached tmux session, confirm it started from the log, then STOP.
Each runner writes conversations immediately, resumes on rerun (skips existing ids), and
auto-commits + pushes by itself. If `nvidia-smi` shows an NVML driver/library mismatch,
`sudo reboot`, reconnect, `conda activate convsim`, retry. Never reboot mid-run. Use
`/anaconda/envs/convsim/bin/python`. Never touch `data/generated/` (the pre-fix baseline is kept
for comparison).

---

## PHASE A — VM generation (Codex can do this autonomously)

### A1. Final pre-regen check  ← DO THIS FIRST
```bash
cd ~/llm-spoken-conversation && git pull --ff-only origin main
tmux new-session -d -s finalcheck 'cd ~/llm-spoken-conversation && bash generation/run_final_check.sh'
```
It regenerates a few C3 + C4 conversations at the final settings and pushes them to
`data/generated_test/finalcheck/`. **GO / NO-GO gate — read the log's quality lines:**
- **GO** if every conversation shows `loops=0` and `spanish=0` (C4 should also end early, `n<32`).
- **NO-GO** if any conversation shows `loops>0` or `spanish>0`. If NO-GO, STOP and wait for Diyar
  — do not start the full regen.

### A2. Full regeneration (only if A1 is GO)  ← the big job
```bash
tmux new-session -d -s regen 'cd ~/llm-spoken-conversation && bash generation/run_full_regen.sh'
```
Regenerates all 12 conditions (C1-C4 × P0/P1/P2, 50 each) into `data/generated_v2/`, auto-pushing
after each condition. This runs for hours — leave it. When done, `run_full_regen.log` prints a
count per condition (expect 50 each, 600 total). If it dies, just relaunch the same command (it
resumes).

### A3. ALIGN export  ← NEEDS Diyar + Claude to build the script; do NOT improvise
The stats need `data/align/alignment_turns.csv` with columns
`condition, conv_id, turn_index, n_turns, cosine_semanticL` (condition="SB" for the Switchboard
baseline), produced with the ALIGN package (already installed, validated on SB — see VM_REPORT).
There is not yet a script that emits this for the generated corpora. **Leave this for Diyar's
return** unless he left a script named `analysis/export_align.py`.

---

## PHASE B — analysis (LOCAL, no GPU; needs Diyar's judgment — list only)
Do NOT run these autonomously; they need review. When Diyar is back:
1. Decide whether to promote `data/generated_v2/` → `data/generated/` (the analysis scripts read
   `data/generated/*/`). Keep the old data archived first.
2. `python analysis/evaluate_generated.py` — per-condition metric table.
3. `python analysis/stats.py` — mixed models + Independence Gradient (needs the ALIGN CSV from A3
   for the alignment part; runs without it for words/turn + markers).
4. `python analysis/social_metrics.py --n-sb 50 --embedding-backend sentence-transformers` — the
   extension metrics (CED is the headline; TSI/ADI). Never commit `turn_labels.csv` (LDC text).

---

## NOT for Codex (waiting on Diyar / Claude)
- Quality-reviewing the regenerated conversations (P1 is where C1/C4 should stop sounding like
  assistants — confirm that when the data lands).
- Building the ALIGN export script (A3).
- Interpreting stats / building figures / the poster.
- **Security to-do:** revoke the GitHub PATs that were pasted in plaintext earlier.

## Quick status check anytime (no Codex needed)
```bash
tmux ls
tail -n 20 run_full_regen.log
cat run_full_regen.log | grep -E "^(C1|C2|C3|C4)-P[012]:"   # per-condition counts when done
git -C ~/llm-spoken-conversation log --oneline -8
```
