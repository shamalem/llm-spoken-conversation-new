#!/usr/bin/env bash
# Pre-full-regen check at the FINAL turn-by-turn family settings (min-new-tokens 16,
# stop-at-sentence, repetition 1.15, no-repeat-ngram 6, natural termination):
#   - validate C4 (never tested at these settings), and
#   - re-confirm C3 stays loop-free at the looser ngram6 (it was validated at 4; 6 is looser
#     and C3 = two identical Vicunas is the most loop-prone).
# Regenerates a few of each into a test dir and reports loops / spanish / fragments / leaks /
# termination. Does NOT touch data/generated/. Launch detached and leave:
#   tmux new-session -d -s finalcheck 'cd ~/llm-spoken-conversation && bash generation/run_final_check.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_test/finalcheck"
LOG="run_final_check.log"

echo "=== final check START $(date -Is) ===" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"
rm -f "$OUT/C3-P0"/*.json "$OUT/C4-P0"/*.json

echo "--- C3 re-check at ngram6 (worst word-salad convs) ---" | tee -a "$LOG"
"$PY" generation/generate_c3.py --prompt P0 --ids "4104,4333,4321" --max-turns 30 --out-root "$OUT" 2>&1 | tee -a "$LOG"

echo "--- C4 validation at final settings ---" | tee -a "$LOG"
"$PY" generation/generate_c4.py --prompt P0 --ids "4104,4321,4325,4333" --max-turns 30 --out-root "$OUT" 2>&1 | tee -a "$LOG"

echo "=== quality check (want: loops=0, spanish=0, leaks=0; convs should end early n<32) ===" | tee -a "$LOG"
"$PY" - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os, re, sys, statistics
root = sys.argv[1]
SPAN = re.compile(r"\b(?:el|la|que|de|en|es|con|para|una|los|las|del|pero|como|muy)\b", re.I)
LEAK = re.compile(r"\bparticip\w*[\s_]*[ab]?\s*[:,]|\bpart[ab]\b", re.I)
def norm(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
def frag(t):
    w = t.split(); return 0 < len(w) <= 3 and not t.rstrip().endswith((".", "?", "!"))
for cond in ["C3-P0", "C4-P0"]:
    for f in sorted(glob.glob(os.path.join(root, cond, "*.json"))):
        b = json.load(open(f, encoding="utf-8")).get("turns", [])[2:]
        seen = set(); loop = sp = lk = fr = 0
        for _, t in b:
            k = norm(t)
            if len(k) > 12 and k in seen: loop += 1
            seen.add(k)
            if len(SPAN.findall(t)) >= 4: sp += 1
            if LEAK.search(t): lk += 1
            if frag(t): fr += 1
        n = len(b) + 2
        med = statistics.median([len(t.split()) for _, t in b]) if b else 0
        tag = "(ended early)" if n < 32 else "(hit cap)"
        print(f"  {cond} {os.path.basename(f).replace('.json',''):6} n={n:2} med={med:<4} loops={loop} spanish={sp} leaks={lk} frags={fr} {tag}")
PY

git add "$OUT" 2>&1 | tee -a "$LOG"
git commit -m "test: final pre-regen check (C4 + C3 at ngram6)" 2>&1 | tee -a "$LOG"
git pull --rebase origin main 2>&1 | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
echo "=== final check DONE $(date -Is) ===" | tee -a "$LOG"
