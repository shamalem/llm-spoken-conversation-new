"""
Conversation metrics: words/turn, coordination-marker rates, sycophancy tokens.

Used for both the Switchboard baseline and generated conversations. Marker rates are
validated locally against the paper's Switchboard Table 5
(oh 0.57, okay 0.16, uh-huh 1.03 per 100 words).

Note on scope: the paper counts markers in the MAIN BODY only (from topic initiation).
We don't have topic-initiation annotation locally, so the script below counts over whole
conversations — expect oh/uh-huh to land close, and `okay` to run a bit higher (okay
clusters at the opening/closing transitions the paper excludes). Exact main-body
replication happens alongside ALIGN (analysis/export_align.py).

Run as a script to validate against Switchboard (stdlib only, no GPU):
    python analysis/metrics.py --n 50
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.swda import iter_conversation_files, parse_conversation

MARKER_PATTERNS = {
    "oh": re.compile(r"\boh\b", re.I),
    "okay": re.compile(r"\b(?:okay|ok)\b", re.I),
    "uh-huh": re.compile(r"\buh-?\s?huh\b", re.I),
}

# Turn-initial agreement / sycophancy tokens (seed list; expand from Supp. File 2).
SYCOPHANCY = [
    "absolutely",
    "definitely",
    "exactly",
    "for sure",
    "great point",
    "i couldn't agree more",
    "i completely agree",
    "totally",
]
SYCOPHANCY_PATTERNS = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in SYCOPHANCY]


def words_per_turn(turns: list[tuple[str, str]]) -> list[int]:
    return [len(txt.split()) for _, txt in turns]


def marker_counts(text: str) -> dict[str, int]:
    return {name: len(pat.findall(text)) for name, pat in MARKER_PATTERNS.items()}


def marker_rates(turns: list[tuple[str, str]]) -> dict[str, float]:
    """Occurrences per 100 words, per marker, over the whole conversation."""
    full = " ".join(txt for _, txt in turns)
    total = max(len(full.split()), 1)
    return {name: 100.0 * cnt / total for name, cnt in marker_counts(full).items()}


def sycophancy_rate(turns: list[tuple[str, str]]) -> float:
    """Fraction of turns that OPEN with a sycophancy/agreement token."""
    if not turns:
        return 0.0
    hits = sum(
        1 for _, txt in turns if any(p.match(txt.lower().lstrip()) for p in SYCOPHANCY_PATTERNS)
    )
    return hits / len(turns)


def _validate(n: int) -> None:
    files = []
    for fp in iter_conversation_files():
        files.append(fp)
        if len(files) >= n:
            break
    if not files:
        print("No conversation files found. Did you extract swda.zip?")
        return

    rates = {"oh": [], "okay": [], "uh-huh": []}
    syco, wpt = [], []
    for fp in files:
        turns = parse_conversation(fp)
        r = marker_rates(turns)
        for k in rates:
            rates[k].append(r[k])
        syco.append(sycophancy_rate(turns))
        wpt.extend(words_per_turn(turns))

    print(f"conversations        : {len(files)}")
    print(f"mean words/turn      : {statistics.mean(wpt):.2f}   (paper SB ~14)")
    print("marker rates per 100 words (mean) vs paper SB Table 5:")
    print(f"  oh     : {statistics.mean(rates['oh']):.2f}   (paper 0.57)")
    print(
        f"  okay   : {statistics.mean(rates['okay']):.2f}   (paper 0.16; higher here = incl. openings/closings)"
    )
    print(f"  uh-huh : {statistics.mean(rates['uh-huh']):.2f}   (paper 1.03)")
    print(f"sycophancy turn-openers: {statistics.mean(syco):.3f}   (SB expected ~0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of conversations to check")
    _validate(ap.parse_args().n)
