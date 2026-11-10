"""
Aggregate metrics for the generated conversations vs the Switchboard baseline.

For every data/generated/<condition>/*.json it computes mean words/turn and oh/okay/uh-huh
rates, then prints a per-condition table next to the Switchboard reference. Handles both
turn-based conditions (C2/C3/C4, which store `turns`) and C1 (which stores `raw_output`).

    python analysis/analyze.py

This runs on whatever conditions are present, so it can be used incrementally as the full
generation completes.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.metrics import marker_rates, words_per_turn
from analysis.swda import (
    iter_conversation_files,
    parse_conversation,
)
from analysis.swda import (
    words_per_turn as sb_words_per_turn,
)

GEN_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "generated"


def conversation_turns(rec: dict) -> list[tuple[str, str]]:
    """(speaker, text) turns for a generated record — parses C1 raw_output if needed.

    Drops the scripted opening greetings (v3 records carry `seed_turns`): at most that many
    LEADING turns, and only while they are literally the scripted "Hello!". C2/C3/C4 store
    the seeds in `turns`; C1 only has them in its prompt, so its raw_output may or may not
    echo them — a model-written opener is never dropped. Records without `seed_turns`
    (v2 and earlier) are unchanged.
    """
    if rec.get("turns"):
        turns = [(t[0], t[1]) for t in rec["turns"]]
    else:
        turns = []
        for line in rec.get("raw_output", "").split("\n"):
            line = line.strip()
            if ":" in line:
                spk, txt = line.split(":", 1)
                if txt.strip():
                    turns.append((spk.strip(), txt.strip()))
    n_seed = 0
    while (
        n_seed < int(rec.get("seed_turns", 0))
        and n_seed < len(turns)
        and turns[n_seed][1].strip().strip('"').lower() == "hello!"
    ):
        n_seed += 1
    turns = turns[n_seed:]
    # P3 shows examples with "[backchannel]"-style labels; if the model copies one into its
    # own turn, strip it (only our exact label vocabulary, so ordinary text is untouched).
    turns = [(s, _P3_LABEL_RE.sub("", t).strip()) for s, t in turns]
    # Agents sometimes echo the turn countdown back as a stage direction — "(Turn 14)",
    # "(Conversation ends here)", "[Your turn to respond. This conversation has 7 turns so
    # far…]". Not speech: strip any bracketed note mentioning turn(s)/conversation. Same rule
    # for every condition, applied in analysis only (generation stays frozen).
    turns = [(s, _META_NOTE_RE.sub("", t).strip()) for s, t in turns]
    # Language drift (10 C3/C4 conversations): Vicuna appends "번역결과" ("translation
    # result") + a Korean translation, and the session then fills with Korean. Keep the
    # English speech before the first non-Latin character; see has_language_drift().
    turns = [(s, _cut_language_drift(t)) for s, t in turns]
    return [(s, t) for s, t in turns if t]


def _cut_language_drift(text: str) -> str:
    text = text.replace("�", "")  # undecodable-byte debris ("�����")
    m = _NON_LATIN_RE.search(text)
    return text[: m.start()].strip() if m else text


def has_language_drift(rec: dict) -> bool:
    """Does the raw record contain non-Latin script (the drift flag for sensitivity runs)?"""
    texts = [t[1] for t in rec.get("turns") or []] or [rec.get("raw_output", "")]
    return any(_NON_LATIN_RE.search(t.replace("�", "")) for t in texts)


def _p3_label_re():
    """Build the regex that matches P3's bracketed gold-label tags, e.g. "[backchannel]"
    or "[statement, opinion]" — used by conversation_turns() to strip them if a model
    echoes one of its own shown examples back into a turn instead of just the dialogue."""
    import re

    from analysis.swda import ACT_NAMES

    names = sorted(set(ACT_NAMES.values()) | {"other"}, key=len, reverse=True)
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"\s*\[(?:{alt})(?:,\s*(?:{alt}))*\]", re.I)


_P3_LABEL_RE = _p3_label_re()

_META_NOTE_RE = re.compile(
    r"\s*(?:\([^()\[\]]{0,400}?\b(?:turns?|conversation)\b[^()\[\]]{0,400}?\)"
    r"|\[[^()\[\]]{0,400}?\b(?:turns?|conversation)\b[^()\[\]]{0,400}?\])",
    re.I,
)

# Anything outside Latin script + general punctuation (€, ™) — Hangul, CJK, Hebrew, …
_NON_LATIN_RE = re.compile(r"[^\x00-ɏ -⁯€™\s]")


def switchboard_baseline(n: int = 50) -> dict:
    """Words/turn and oh/okay/uh-huh marker rates for the first `n` Switchboard
    conversations — the human reference the generated conditions get compared against."""
    wpt, rates = [], {"oh": [], "okay": [], "uh-huh": []}
    for fp in list(iter_conversation_files())[:n]:
        turns = parse_conversation(fp)
        wpt.extend(sb_words_per_turn(turns))
        r = marker_rates(turns)
        for k in rates:
            rates[k].append(r[k])
    return {"n": n, "wpt": wpt, "rates": rates}


def main() -> None:
    """CLI entry point: load every generated conversation under GEN_ROOT, compute
    per-condition mean words/turn and oh/okay/uh-huh rates, and print a table of
    all conditions alongside the Switchboard baseline for a quick side-by-side read."""
    conds: dict[str, dict] = {}
    for f in sorted(glob.glob(str(GEN_ROOT / "*" / "*.json"))):
        rec = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        turns = conversation_turns(rec)
        if not turns:
            continue
        cond = rec["condition"]
        c = conds.setdefault(cond, {"n": 0, "wpt": [], "oh": [], "okay": [], "uh-huh": []})
        c["n"] += 1
        c["wpt"].extend(words_per_turn(turns))
        r = marker_rates(turns)
        for k in ("oh", "okay", "uh-huh"):
            c[k].append(r[k])

    sb = switchboard_baseline()
    hdr = f"{'condition':10} {'n':>3} {'w/turn':>7} {'oh':>6} {'okay':>6} {'uh-huh':>7}"
    print(hdr)
    print("-" * len(hdr))
    sb_r = {k: statistics.mean(v) for k, v in sb["rates"].items()}
    print(
        f"{'SWITCHBD':10} {sb['n']:>3} {statistics.mean(sb['wpt']):7.1f} "
        f"{sb_r['oh']:6.2f} {sb_r['okay']:6.2f} {sb_r['uh-huh']:7.2f}"
    )
    print("-" * len(hdr))
    for cond in sorted(conds):
        c = conds[cond]
        print(
            f"{cond:10} {c['n']:>3} {statistics.mean(c['wpt']):7.1f} "
            f"{statistics.mean(c['oh']):6.2f} {statistics.mean(c['okay']):6.2f} "
            f"{statistics.mean(c['uh-huh']):7.2f}"
        )
    print("\n(Switchboard ref from paper: ~14 w/turn; oh 0.57, okay 0.16, uh-huh 1.03)")


if __name__ == "__main__":
    main()
