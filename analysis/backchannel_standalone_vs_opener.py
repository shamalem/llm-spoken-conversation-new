"""Reviewer checklist: humans and LLMs are cut into units differently, and backchannel
words at the start of a long LLM turn ("Yeah, totally. I think...") get the same
Backchannel tag as a real standalone "uh-huh" even though the speaker keeps the floor.

Steps implemented (reviewer's own wording):
  1. Document exactly how LLM text is split into units.
  2. Apply that same splitting method to Switchboard text; report whether the human
     backchannel rate barely changes vs. the project's existing gold-utterance-based rate.
  3. For every backchannel, label it Standalone (whole turn is just the reaction) or
     Turn opener (the reaction starts a longer turn).
  4. Report the share of each type for humans and every one of the 12 conditions.

Method
------
Step 1: LLM turns are split into units by ``analysis.dialogue_acts.sentence_units`` --
a dependency-free regex sentence-boundary splitter: ``(?<=[.!?])\\s+(?=["'(]*[A-Z0-9])``
(split after ., !, or ?, then whitespace, then an optional quote/paren and a capital
letter or digit). This is the SAME function the project's own tagging pipeline already
uses for generated turns (analysis/dialogue_acts.py:load_generated).

Confirmed confound this reviewer item caught: the project's gold human reference does
NOT go through this splitter at all -- analysis/dialogue_acts.py:load_switchboard() uses
SwDA's own expert-annotated utterance rows directly. Step 2 below applies the LLM
splitter to human turn text too, so the comparison in step 3-4 uses ONE method on both
sides, as instructed.

A turn is judged "backchannel-initiated" using the project's own transparent (non-ML)
rule (analysis.dialogue_acts.BACKCHANNEL_FORMS: uh huh / yeah / right / okay / mm hm /
i see, exact match after punctuation-stripped casefolding) applied to the FIRST unit of
the turn:
  - exactly 1 unit, and it's a backchannel form            -> Standalone
  - 2+ units, and the FIRST one is a backchannel form       -> Turn opener
  - first unit is not a backchannel form                    -> not counted (not
    backchannel-initiated at all)

Usage
-----
    python analysis/backchannel_standalone_vs_opener.py --sb-root <path to local swda/>
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns  # noqa: E402
from analysis.dialogue_acts import BACKCHANNEL_FORMS, sentence_units  # noqa: E402
from analysis.swda import clean_text, iter_conversation_files, parse_conversation  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", clean_text(text).casefold()).strip()


def is_backchannel_form(unit: str) -> bool:
    normalized = normalize(unit)
    return normalized in BACKCHANNEL_FORMS and len(normalized.split()) <= 3


def classify_turn(turn_text: str) -> str | None:
    """None (not backchannel-initiated), 'standalone', or 'opener'."""
    units = sentence_units(turn_text)
    if not units or not is_backchannel_form(units[0]):
        return None
    return "standalone" if len(units) == 1 else "opener"


def tally(turns_list: list[list[tuple[str, str]]]) -> dict[str, int]:
    counts = {"standalone": 0, "opener": 0, "n_turns": 0}
    for turns in turns_list:
        for _speaker, text in turns:
            counts["n_turns"] += 1
            label = classify_turn(text)
            if label:
                counts[label] += 1
    return counts


def load_llm_condition(cond_dir: pathlib.Path) -> list[list[tuple[str, str]]]:
    out = []
    for fp in sorted(cond_dir.glob("*.json")):
        import json
        rec = json.loads(fp.read_text(encoding="utf-8"))
        out.append(conversation_turns(rec))
    return out


def load_sb(root: pathlib.Path, n: int | None) -> list[list[tuple[str, str]]]:
    files = list(iter_conversation_files(root))
    if n:
        files = files[:n]
    return [parse_conversation(fp) for fp in files]


def rule_backchannel_rate_gold_units(root: pathlib.Path, n: int | None) -> tuple[int, int]:
    """Method (a): existing approach -- rate over SwDA's own per-utterance gold rows,
    NOT turn-merged, NOT sentence_units()-split. Returns (backchannel_count, n_units)."""
    import csv

    files = list(iter_conversation_files(root))
    if n:
        files = files[:n]
    backchannels = units = 0
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                txt = clean_text(row["text"])
                if not txt:
                    continue
                units += 1
                normalized = re.sub(r"[^a-z0-9']+", " ", txt.casefold()).strip()
                if normalized in BACKCHANNEL_FORMS and len(normalized.split()) <= 3:
                    backchannels += 1
    return backchannels, units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb-root", required=True, help="local path to the extracted swda/ directory (LDC-licensed, not committed)")
    ap.add_argument("--sb-n", type=int, default=None, help="limit to first N SB conversations (default: all)")
    ap.add_argument("--gen-root", default=str(GEN_ROOT), help="root containing <condition>/*.json generated conversations")
    args = ap.parse_args()
    sb_root = pathlib.Path(args.sb_root)
    gen_root = pathlib.Path(args.gen_root)

    print("=== Step 1: LLM unit-splitting method ===")
    print("sentence_units(): regex split after [.!?] + whitespace + optional quote/paren + [A-Z0-9]")
    print("(analysis.dialogue_acts.SENTENCE_BOUNDARY, same function the project's own tagging pipeline uses)\n")

    print("=== Step 2: apply the SAME splitter to Switchboard turn text; compare to the gold-unit rate ===")
    bc_gold, n_gold = rule_backchannel_rate_gold_units(sb_root, args.sb_n)
    print(f"Method (a) existing: gold SwDA per-utterance rows (not turn-merged, not sentence_units()-split)")
    print(f"  backchannel rate = {bc_gold}/{n_gold} = {bc_gold / n_gold:.4f}")

    sb_turns = load_sb(sb_root, args.sb_n)
    sb_units_total = sum(len(sentence_units(t)) for turns in sb_turns for _, t in turns)
    sb_bc_units = sum(
        1 for turns in sb_turns for _, t in turns for u in sentence_units(t) if is_backchannel_form(u)
    )
    print(f"Method (b) unified: turn-merged (parse_conversation) + sentence_units()-split -- same method as LLM")
    print(f"  backchannel rate = {sb_bc_units}/{sb_units_total} = {sb_bc_units / sb_units_total:.4f}")
    delta = abs(bc_gold / n_gold - sb_bc_units / sb_units_total)
    print(f"  delta = {delta:.4f}  ({'barely changes' if delta < 0.02 else 'CHANGES MEANINGFULLY -- use method (b) for the comparison below'})\n")

    print("=== Steps 3-4: Standalone vs. Turn-opener share, per condition (method b, same splitter both sides) ===")
    print(f"{'condition':10s} {'n_turns':>8s} {'standalone':>11s} {'opener':>8s} {'standalone%':>13s} {'opener%':>9s}")

    sb_counts = tally(sb_turns)
    bc_total = sb_counts["standalone"] + sb_counts["opener"]
    print(f"{'SB':10s} {sb_counts['n_turns']:8d} {sb_counts['standalone']:11d} {sb_counts['opener']:8d} "
          f"{(sb_counts['standalone'] / bc_total if bc_total else 0):13.1%} {(sb_counts['opener'] / bc_total if bc_total else 0):9.1%}")

    conditions = sorted(p.name for p in gen_root.glob("*") if p.is_dir())
    for cond in conditions:
        turns_list = load_llm_condition(gen_root / cond)
        c = tally(turns_list)
        bc_total = c["standalone"] + c["opener"]
        print(f"{cond:10s} {c['n_turns']:8d} {c['standalone']:11d} {c['opener']:8d} "
              f"{(c['standalone'] / bc_total if bc_total else 0):13.1%} {(c['opener'] / bc_total if bc_total else 0):9.1%}")


if __name__ == "__main__":
    main()
