"""Automated quality checker for generated conversations.

Reproduces the manual review used during the C2/C3/C4 fixes so Codex (or a teammate) can
verify any generated corpus without judgement calls. For each condition it reports the signals
we care about and prints a PASS/FLAG verdict per condition, plus the worst conversations.

Detected problems:
  - fragments        : 1-3 word turns that don't end a sentence (the C3 word-salad signature)
  - verbatim loops   : a turn that exactly repeats an earlier turn (the old C3 disease)
  - language drift   : a turn that slipped into Spanish / non-Latin script
  - label leaks      : speaker labels leaking into a turn (the C2 disease: PartB:, Participants:)
  - no natural close : conversation did NOT end on a goodbye (padded to the turn cap)
  - assistant tells  : help-desk / advice phrasing ("how can I help", "I recommend", "feel free
                       to reach out"). NOTE: high in P0 by design (no peer guard) — expected.

Usage:
    python analysis/quality_report.py                      # checks data/generated_v2
    python analysis/quality_report.py data/generated       # any directory
    python analysis/quality_report.py data/generated_v2 --worst 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.model_utils import looks_like_closing  # noqa: E402

SPAN = re.compile(r"\b(?:el|la|que|de|en|es|con|para|una|los|las|del|pero|como|muy|nuestra)\b", re.I)
LEAK = re.compile(r"\bparticip\w*[\s_]*[ab]?\s*[:,]|\bpart[ab]\b", re.I)
TELLS = re.compile(
    r"how can i help|how may i help|is this the|i'd be happy to help|i can help you|"
    r"i recommend|i would recommend|you should|you could try|here are (?:a few|some|three)|"
    r"feel free to|don't hesitate|as an ai|hope this helps|glad i could help|happy to assist",
    re.I,
)


def turns_of(rec: dict) -> list[tuple[str, str]]:
    if rec.get("turns"):
        return [(s, t) for s, t in rec["turns"]]
    out = []
    for line in rec.get("raw_output", "").splitlines():
        m = re.match(r"\s*(Participant[AB])\s*:\s*(.*)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def is_fragment(t: str) -> bool:
    w = t.split()
    return 0 < len(w) <= 3 and not t.rstrip().endswith((".", "?", "!"))


def has_nonlatin(t: str) -> bool:
    for c in t:
        if c.isalpha() and "LATIN" not in unicodedata.name(c, "LATIN"):
            return True
    return False


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def analyze_conv(path: str, is_c1: bool) -> dict:
    rec = json.load(open(path, encoding="utf-8"))
    turns = turns_of(rec)
    body = turns if is_c1 else turns[2:]      # C1 has no Hello seeds
    if not body:
        return {"id": os.path.basename(path).replace(".json", ""), "empty": True}
    wpt = [len(t.split()) for _, t in body]
    seen, loops = set(), 0
    for _, t in body:
        k = norm(t)
        if len(k) > 12 and k in seen:
            loops += 1
        seen.add(k)
    last = body[-1][1]
    return {
        "id": os.path.basename(path).replace(".json", ""),
        "empty": False,
        "n_turns": len(turns),
        "median_wpt": statistics.median(wpt),
        "fragments": sum(is_fragment(t) for _, t in body),
        "loops": loops,
        "drift": sum(1 for _, t in body if len(SPAN.findall(t)) >= 4 or has_nonlatin(t)),
        "leaks": sum(1 for _, t in body if LEAK.search(t)),
        "tells": sum(1 for _, t in body if TELLS.search(t)),
        "closed_naturally": looks_like_closing(last),  # ended on a goodbye?
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/generated_v2",
                    help="directory of <condition>/<id>.json (default data/generated_v2)")
    ap.add_argument("--worst", type=int, default=3, help="worst N conversations to list per condition")
    args = ap.parse_args()

    conds = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    if not conds:
        print(f"No condition folders under {args.root}")
        return

    print(f"quality report for {args.root}")
    print("=" * 100)
    hdr = f"{'cond':8} {'n':>3} {'turns':>6} {'medWPT':>7} {'frag':>5} {'loop':>5} {'drift':>6} {'leak':>5} {'noEnd':>6} {'tells':>6}  verdict"
    print(hdr)
    print("-" * 100)
    for cdir in conds:
        cond = os.path.basename(cdir)
        is_c1 = cond.startswith("C1")
        rows = [analyze_conv(f, is_c1) for f in sorted(glob.glob(os.path.join(cdir, "*.json")))]
        rows = [r for r in rows if not r.get("empty")]
        if not rows:
            continue
        n = len(rows)
        frag = sum(r["fragments"] for r in rows)
        loop = sum(r["loops"] for r in rows)
        drift = sum(r["drift"] for r in rows)
        leak = sum(r["leaks"] for r in rows)
        tells = sum(r["tells"] for r in rows)
        no_end = sum(1 for r in rows if not r["closed_naturally"] and not is_c1)
        med = statistics.mean(r["median_wpt"] for r in rows)
        turns = statistics.mean(r["n_turns"] for r in rows)
        # verdict: real bugs are frag / loop / drift / leak / no_end. tells are expected in P0.
        bad = frag + loop + drift + leak + no_end
        verdict = "PASS" if bad == 0 else ("FLAG" if bad <= n else "BAD")
        print(f"{cond:8} {n:3d} {turns:6.1f} {med:7.1f} {frag:5d} {loop:5d} {drift:6d} {leak:5d} {no_end:6d} {tells:6d}  {verdict}")

        worst = sorted(rows, key=lambda r: -(r["fragments"] + r["loops"] + r["drift"] + r["leaks"]))[: args.worst]
        for r in worst:
            score = r["fragments"] + r["loops"] + r["drift"] + r["leaks"]
            if score:
                print(f"         worst {r['id']}: frag={r['fragments']} loop={r['loops']} "
                      f"drift={r['drift']} leak={r['leaks']} noEnd={not r['closed_naturally']}")
    print("-" * 100)
    print("frag/loop/drift/leak/noEnd = real bugs (want 0). tells = help-desk/advice phrasing,")
    print("EXPECTED high in P0 (no peer guard), should DROP in P1/P2. C1 is all-at-once (no noEnd).")


if __name__ == "__main__":
    main()
