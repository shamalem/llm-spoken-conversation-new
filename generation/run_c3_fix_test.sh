#!/usr/bin/env bash
# C3 fragmentation-fix VALIDATION (small, ~8 conversations). Regenerates the worst
# word-salad C3 conversations with the new turn-quality settings (min-length floor +
# stop-at-sentence + softer repetition) into a SEPARATE test dir, then commits + pushes
# so the local side can read them and judge whether turns now sound like two people.
#
# Does NOT touch data/generated/. Safe to launch detached and leave:
#   tmux new-session -d -s c3fix 'cd ~/llm-spoken-conversation && bash generation/run_c3_fix_test.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_test/c3fix"
# 7 worst word-salad convs + 4325 as a control (was already clean — must stay clean)
IDS="4104,4333,4321,4372,4877,4166,4109,4325"
LOG="run_c3_fix_test.log"

echo "=== c3 fix test START $(date -Is) ===" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

# Force a fresh regeneration (generators skip existing ids).
rm -f "$OUT/C3-P0"/*.json

# The new defaults (min-new-tokens 16, stop-at-sentence on, rep-penalty 1.15, ngram 4)
# are baked into generate_c3.py, so no extra flags needed here.
"$PY" generation/generate_c3.py --prompt P0 --ids "$IDS" --max-turns 30 --out-root "$OUT" 2>&1 | tee -a "$LOG"

echo "=== median words/turn: NEW (fix) vs OLD (current data/generated) ===" | tee -a "$LOG"
"$PY" - "$OUT/C3-P0" "data/generated/C3-P0" <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os, statistics, sys
new, old = sys.argv[1], sys.argv[2]
def med(p):
    b = json.load(open(p, encoding="utf-8")).get("turns", [])[2:]
    return round(statistics.median([len(t.split()) for _, t in b]), 1) if b else 0
for f in sorted(glob.glob(os.path.join(new, "*.json"))):
    cid = os.path.basename(f)
    o = os.path.join(old, cid)
    print(f"  {cid:12} NEW_med_wpt={med(f):<5} OLD_med_wpt={med(o) if os.path.exists(o) else 'NA'}")
PY

git add "$OUT" 2>&1 | tee -a "$LOG"
git commit -m "test: C3 fragmentation-fix validation on worst conversations" 2>&1 | tee -a "$LOG"
git pull --rebase origin main 2>&1 | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
echo "=== c3 fix test DONE $(date -Is) ===" | tee -a "$LOG"
