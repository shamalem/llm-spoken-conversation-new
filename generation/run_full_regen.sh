#!/usr/bin/env bash
# FULL regeneration of all 12 conditions (C1-C4 x P0/P1/P2, 50 each) into data/generated_v2/
# with ALL fixes baked into the generator defaults (min-new-tokens 16, stop-at-sentence,
# repetition 1.15, no-repeat-ngram 6, natural termination, aggressive label cleaner).
# Resumable (skips existing ids); auto-commits + pushes after EACH condition so a crash never
# loses more than the condition in flight. Run detached and leave:
#   tmux new-session -d -s regen 'cd ~/llm-spoken-conversation && bash generation/run_full_regen.sh'
# This is a LONG run (12 x 50 conversations). It survives disconnects in tmux.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_v2"
LOG="run_full_regen.log"
N=50; MT=30

echo "=== full regen START $(date -Is) ===" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

commit_push() {
  git add "$OUT" 2>&1 | tee -a "$LOG"
  if git commit -m "$1" 2>&1 | tee -a "$LOG"; then
    git pull --rebase origin main 2>&1 | tee -a "$LOG"
    git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
  else
    echo "(nothing new to commit for: $1)" | tee -a "$LOG"
  fi
}

for P in P0 P1 P2; do
  echo "=========== PROMPT $P  $(date -Is) ===========" | tee -a "$LOG"
  "$PY" generation/generate_c1.py --prompt "$P" --n "$N"                    --out-root "$OUT" 2>&1 | tee -a "$LOG"
  commit_push "data: regen C1-$P into generated_v2 (all fixes)"
  "$PY" generation/generate_c2.py --prompt "$P" --n "$N" --max-turns "$MT" --out-root "$OUT" 2>&1 | tee -a "$LOG"
  commit_push "data: regen C2-$P into generated_v2 (all fixes)"
  "$PY" generation/generate_c3.py --prompt "$P" --n "$N" --max-turns "$MT" --out-root "$OUT" 2>&1 | tee -a "$LOG"
  commit_push "data: regen C3-$P into generated_v2 (all fixes)"
  "$PY" generation/generate_c4.py --prompt "$P" --n "$N" --max-turns "$MT" --out-root "$OUT" 2>&1 | tee -a "$LOG"
  commit_push "data: regen C4-$P into generated_v2 (all fixes)"
done

echo "=== full regen DONE $(date -Is) — counts ===" | tee -a "$LOG"
for d in "$OUT"/*/; do echo "$(basename "$d"): $(ls "$d" 2>/dev/null | wc -l)" | tee -a "$LOG"; done
