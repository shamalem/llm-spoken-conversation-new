"""Reviewer checklist: is the "LLMs have more opinions" finding just tagger error on
Statement vs. Opinion?

Two things live in this one file, run as two independent steps -- they do NOT both need
to be re-run every time you regenerate conversations:

  1. --confusion : reports the Statement/Opinion confusion table from the hand-labelled
     annotation set (annotation/filled/*.xlsx) vs. the tagger. This only changes if you
     relabel/add to the hand-check (e.g. after breaking the two current ties) -- it does
     NOT depend on which conversations were generated.

  2. --merge-recompute : merges Statement+Opinion in the coarse dialogue-act distribution
     and recomputes the Jensen-Shannon distance from humans for all 12 conditions, then
     reports whether C1 is still closest and whether "agents farthest" still holds. This
     DOES depend on the current conversations -- re-run it after every regeneration.

Workflow after regenerating conversations
------------------------------------------
    # 1. (VM, isolated DialogTag env -- see requirements-dialogue-acts.txt) refresh the
    #    per-condition dialogue-act distributions for the new data/generated/:
    python analysis/dialogue_acts.py

    # 2. (anywhere, no heavy deps) redo the merge + recompute + decide:
    python analysis/check_statement_opinion_merge.py --merge-recompute

Re-run --confusion separately only when annotation/filled/*.xlsx changes (e.g. once the
two tied sentences get a tie-breaking label) -- it needs annotation/_key.csv (produced by
analysis/make_annotation_sheets.py when the sheets were built) which is not committed
here; without it this step reports the confusion counts already on file in
results/tagger_handcheck.txt instead of recomputing them live.

    python analysis/check_statement_opinion_merge.py --confusion
    python analysis/check_statement_opinion_merge.py --all
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST_CSV = ROOT / "results" / "dialogue_acts" / "da_distribution_coarse_by_condition.csv"
JSD_CSV = ROOT / "results" / "dialogue_acts" / "da_jsd_vs_sb.csv"
HANDCHECK_TXT = ROOT / "results" / "tagger_handcheck.txt"
ANNOTATION_KEY = ROOT / "annotation" / "_key.csv"
ANNOTATION_SHEETS = ROOT / "annotation" / "filled"

ACT_COLS = [
    "Statement", "Opinion", "Backchannel", "YesNoQuestion", "WhQuestion/OpenQuestion",
    "Agreement", "Answer", "Directive", "Repair/Hedge", "Abandoned/Other",
]


# --------------------------------------------------------------------------- confusion --

def report_confusion() -> None:
    """Statement<->Opinion confusion, human hand-labels vs. tagger.

    Recomputes live from annotation/filled/*.xlsx + annotation/_key.csv when the key file
    is present (it maps each sheet row to a stable item_id, its tagger prediction, and
    whether it's one of the 20 cross-annotator "shared" items -- see
    analysis/make_annotation_sheets.py). Falls back to the last computed summary
    (results/tagger_handcheck.txt) otherwise, since that key file is not committed here.
    """
    if not ANNOTATION_KEY.exists():
        print(f"{ANNOTATION_KEY} not found (not committed -- see analysis/make_annotation_sheets.py).")
        print(f"Showing the last computed confusion summary from {HANDCHECK_TXT} instead:\n")
        if HANDCHECK_TXT.exists():
            print(HANDCHECK_TXT.read_text(encoding="utf-8"))
        else:
            print("(no cached summary found either -- run analysis/score_annotation.py once "
                  "you have annotation/_key.csv)")
        return

    # Live path: delegate to the project's own scorer, which already builds exactly this
    # table (and the Statement/Opinion rows within it) from the sheets + key.
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "score_annotation.py"),
         "--sheets", str(ANNOTATION_SHEETS), "--key", str(ANNOTATION_KEY)],
        check=True,
    )


# ---------------------------------------------------------------- merge + recompute -----

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence in [0, 1] -- same formula as analysis/dialogue_acts.py."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_total, q_total = p.sum(), q.sum()
    if p_total == 0 and q_total == 0:
        return 0.0
    if p_total == 0 or q_total == 0:
        return 1.0
    p, q = p / p_total, q / q_total
    midpoint = 0.5 * (p + q)

    def _kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def _load_distributions() -> dict[str, dict[str, float]]:
    rows = {}
    with open(DIST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["condition"]] = {c: float(row[c]) for c in ACT_COLS}
    return rows


def _merged_vector(dist: dict[str, float]) -> np.ndarray:
    merged = dist["Statement"] + dist["Opinion"]
    rest = [dist[c] for c in ACT_COLS if c not in ("Statement", "Opinion")]
    return np.array([merged] + rest)


def _load_original_coarse_jsd() -> dict[str, float]:
    out = {}
    with open(JSD_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label_set"] == "coarse" and row["view"] == "tagger_human_vs_tagger_llm_primary":
                out[row["condition"]] = float(row["jsd_dist"])
    return out


def report_merge_recompute() -> None:
    if not DIST_CSV.exists() or not JSD_CSV.exists():
        print(f"Missing {DIST_CSV} or {JSD_CSV}.")
        print("Run `python analysis/dialogue_acts.py` first (needs the isolated DialogTag "
              "env -- see analysis/requirements-dialogue-acts.txt) to tag the current "
              "data/generated/ conversations and produce these files.")
        return

    dists = _load_distributions()
    sb_vec = _merged_vector(dists["SB-tagger"])
    conditions = sorted(c for c in dists if c not in ("SB-gold", "SB-tagger"))
    orig = _load_original_coarse_jsd()

    print(f"{'condition':10s} {'orig_jsd':>10s} {'merged_jsd':>12s} {'delta':>8s}")
    merged = {}
    for cond in conditions:
        jsd = js_divergence(sb_vec, _merged_vector(dists[cond]))
        merged[cond] = jsd
        o = orig.get(cond, float("nan"))
        print(f"{cond:10s} {o:10.4f} {jsd:12.4f} {jsd - o:8.4f}")

    def arch_avgs(values: dict[str, float]) -> dict[str, float]:
        archs: dict[str, list[float]] = {}
        for cond, v in values.items():
            archs.setdefault(cond.split("-")[0], []).append(v)
        return {a: sum(v) / len(v) for a, v in archs.items()}

    merged_avgs = arch_avgs(merged)
    orig_avgs = arch_avgs(orig)

    print("\nArchitecture averages:")
    print("  merged :  " + "  ".join(f"{a}={v:.4f}" for a, v in sorted(merged_avgs.items(), key=lambda kv: kv[1])))
    print("  original: " + "  ".join(f"{a}={v:.4f}" for a, v in sorted(orig_avgs.items(), key=lambda kv: kv[1])))

    closest_m, farthest_m = min(merged_avgs, key=merged_avgs.get), max(merged_avgs, key=merged_avgs.get)
    closest_o, farthest_o = min(orig_avgs, key=orig_avgs.get), max(orig_avgs, key=orig_avgs.get)
    print(f"\nClosest to humans:  merged={closest_m}  original={closest_o}")
    print(f"Farthest from humans: merged={farthest_m}  original={farthest_o}")

    print()
    if closest_m == closest_o and farthest_m == farthest_o:
        print('DECIDE: "the result holds even when statement and opinion are merged."')
    else:
        print("DECIDE: ranking changed -- we need to think. "
              f"(closest unchanged={closest_m == closest_o}, farthest unchanged={farthest_m == farthest_o})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confusion", action="store_true", help="report the Statement/Opinion confusion table")
    ap.add_argument("--merge-recompute", action="store_true",
                     help="merge Statement+Opinion and recompute JSD-from-humans for all 12 conditions")
    ap.add_argument("--all", action="store_true", help="run both steps")
    args = ap.parse_args()

    if not (args.confusion or args.merge_recompute or args.all):
        ap.print_help()
        return

    if args.confusion or args.all:
        print("=== Statement/Opinion confusion ===")
        report_confusion()
        print()

    if args.merge_recompute or args.all:
        print("=== Merge Statement+Opinion, recompute JSD vs. humans ===")
        report_merge_recompute()


if __name__ == "__main__":
    main()
