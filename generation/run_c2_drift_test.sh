#!/usr/bin/env bash
# C2 language-drift A/B. The fix test (no-repeat-ngram=4) left ~10 turns drifting into Spanish,
# concentrated in conv 2095 — likely the model escaping the n-gram ban once English saturates.
# This regenerates the drift-prone convs under LOOSER n-gram settings (6 and 0) into separate
# dirs and reports Spanish-drift turns + verbatim-duplicate turns (the thing n-gram blocking
# guards against), so we can pick a setting that kills the drift WITHOUT bringing loops back.
# Baseline for these ids at ngram=4 (from the earlier c2fix run): 10 Spanish turns, 0 loops.
# Does NOT touch data/generated/. Launch detached and leave:
#   tmux new-session -d -s c2drift 'cd ~/llm-spoken-conversation && bash generation/run_c2_drift_test.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
IDS="2095,4321,4382,4316"   # the drift-prone convs (2095 worst) + 4316 as a clean control
LOG="run_c2_drift_test.log"

echo "=== c2 drift A/B START $(date -Is) ===" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

for NG in 6 0; do
  OUT="data/generated_test/c2drift/ngram${NG}"
  rm -f "$OUT/C2-P0"/*.json
  echo "--- generating with --no-repeat-ngram ${NG} ---" | tee -a "$LOG"
  "$PY" generation/generate_c2.py --prompt P0 --ids "$IDS" --max-turns 30 \
        --no-repeat-ngram "$NG" --out-root "$OUT" 2>&1 | tee -a "$LOG"
done

echo "=== drift + loop check (baseline ngram=4: spanish=10 loops=0) ===" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os, re
SPAN = re.compile(r"\b(?:el|la|que|de|en|es|con|clima|acuerdo|cambio|todo|nuestra|para|una|los|las|del)\b", re.I)
def spanish(t): return len(SPAN.findall(t)) >= 4
def norm(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
for ng in ["6", "0"]:
    d = f"data/generated_test/c2drift/ngram{ng}/C2-P0"
    sp = dup = tot = 0
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = json.load(open(f, encoding="utf-8")).get("turns", [])[2:]
        seen = set()
        for _, t in b:
            tot += 1
            if spanish(t): sp += 1
            k = norm(t)
            if len(k) > 12 and k in seen: dup += 1
            seen.add(k)
    print(f"  ngram={ng}: turns={tot}  spanish_turns={sp}  verbatim_loops={dup}")
PY

git add data/generated_test/c2drift 2>&1 | tee -a "$LOG"
git commit -m "test: C2 no-repeat-ngram A/B for language drift" 2>&1 | tee -a "$LOG"
git pull --rebase origin main 2>&1 | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
echo "=== c2 drift A/B DONE $(date -Is) ===" | tee -a "$LOG"
