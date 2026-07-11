#!/usr/bin/env bash
# C1-ONLY regeneration into data/generated_v2/ after the no-repeat-ngram fix.
#
# WHY: C1 builds the WHOLE ~30-turn conversation in one generate() call. It used to fall
# through to chat()'s per-turn default no_repeat_ngram_size=3 — a hard "never repeat any
# 3-gram" ban that is unsatisfiable over ~2000 tokens of natural dialogue, so generation hit
# EOS almost immediately and produced 1-2 line fragments. generate_c1.py now sets its own
# --repetition-penalty 1.15 and --no-repeat-ngram 6 (the same value the C2/C3/C4 family uses)
# and passes them into chat(). The old broken output was moved to data/generated_v2/C1-P*.broken,
# so C1-P0/P1/P2 start fresh (50 each).
#
# Resumable (skips existing ids); auto-commits + pushes after each prompt so a crash never
# loses more than the prompt in flight. Run detached in tmux and LEAVE IT:
#   tmux new-session -d -s c1regen 'cd ~/llm-spoken-conversation && bash generation/run_c1_regen.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_v2"
LOG="run_c1_regen.log"
N=50

echo "=== C1 regen START $(date -Is) ===" | tee -a "$LOG"
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
  echo "=========== C1 $P  $(date -Is) ===========" | tee -a "$LOG"
  "$PY" generation/generate_c1.py --prompt "$P" --n "$N" --out-root "$OUT" 2>&1 | tee -a "$LOG"
  commit_push "data: regen C1-$P into generated_v2 (ngram6 fix)"
done

echo "=== C1 regen DONE $(date -Is) — counts (want 50 each) ===" | tee -a "$LOG"
for d in "$OUT"/C1-P?/; do echo "$(basename "$d"): $(ls "$d" 2>/dev/null | wc -l)" | tee -a "$LOG"; done
echo "Now verify: $PY analysis/quality_report.py $OUT   (C1-* must APPEAR with medWPT ~15-20)" | tee -a "$LOG"
