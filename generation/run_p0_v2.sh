#!/usr/bin/env bash
# Regenerate the FULL P0 set (C1-C4, 50 convs each, 30-turn cap) with unified decoding
# (repetition_penalty is now a default in chat(), so every architecture uses it), writing to
# a SEPARATE directory so the current pilot in data/generated/ is never touched.
#
# Designed to be launched detached (tmux) and left alone:
#   - each conversation is written to disk immediately; rerun resumes (existing ids skipped)
#   - it auto-commits + pushes after EACH condition, so a late crash still keeps earlier work
#   - it logs everything to run_p0_v2.log and writes run_p0_v2.status on finish
#
# Launch (from the repo root):
#   tmux new-session -d -s genp0 'cd ~/llm-spoken-conversation && bash generation/run_p0_v2.sh'
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
REPO="$(pwd)"
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_v2"
LOG="$REPO/run_p0_v2.log"
STATUS="$REPO/run_p0_v2.status"
N=50
MT=30

echo "=== run_p0_v2 START $(date -Is) ===" | tee -a "$LOG"
echo "python : $PY" | tee -a "$LOG"
echo "outdir : $OUT" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda_available', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

commit_push() {   # commit_push <message>
  git add "$OUT" 2>&1 | tee -a "$LOG"
  if git commit -m "$1" 2>&1 | tee -a "$LOG"; then
    git pull --rebase origin main 2>&1 | tee -a "$LOG"
    git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
  else
    echo "(nothing new to commit)" | tee -a "$LOG"
  fi
}

run() {           # run <label> <cmd...>
  echo "--- $1 START $(date -Is) ---" | tee -a "$LOG"
  "${@:2}" 2>&1 | tee -a "$LOG"
  echo "--- $1 END   $(date -Is)  count=$(ls "$OUT/$1" 2>/dev/null | wc -l) ---" | tee -a "$LOG"
  commit_push "data: regenerate $1 into generated_v2 (50, 30-turn cap, unified decoding)"
}

run C1-P0 "$PY" generation/generate_c1.py --prompt P0 --n "$N"                 --out-root "$OUT"
run C2-P0 "$PY" generation/generate_c2.py --prompt P0 --n "$N" --max-turns "$MT" --out-root "$OUT"
run C3-P0 "$PY" generation/generate_c3.py --prompt P0 --n "$N" --max-turns "$MT" --out-root "$OUT"
run C4-P0 "$PY" generation/generate_c4.py --prompt P0 --n "$N" --max-turns "$MT" --out-root "$OUT"

{
  echo "=== run_p0_v2 DONE $(date -Is) ==="
  for d in C1-P0 C2-P0 C3-P0 C4-P0; do
    echo "$d: $(ls "$OUT/$d" 2>/dev/null | wc -l) / $N"
  done
} | tee -a "$LOG" | tee "$STATUS"
