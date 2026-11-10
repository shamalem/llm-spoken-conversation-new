"""Score the tagger hand-check (sheets from analysis/make_annotation_sheets.py).

Reads annotation/sheets/annotator_*.xlsx (filled) + annotation/_key.csv and reports:
  * tagger accuracy vs humans on all 100 units (shared units: the humans' majority label),
    overall and per tagger label, with a 95% Wilson interval;
  * human-human agreement on the 20 shared units (mean pairwise % + Fleiss' kappa) — the
    ceiling the tagger should be compared against;
  * the confusion table (human label -> tagger label).

    python analysis/score_annotation.py [--sheets annotation/sheets] [--out results]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fleiss_kappa(items: list[list[str]]) -> float:
    cats = sorted({c for it in items for c in it})
    n = len(items[0])
    p_i = [(sum(v * v for v in Counter(it).values()) - n) / (n * (n - 1)) for it in items]
    p_j = [sum(it.count(c) for it in items) / (len(items) * n) for c in cats]
    pbar, pe = sum(p_i) / len(items), sum(p * p for p in p_j)
    return (pbar - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    from openpyxl import load_workbook

    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default=str(ROOT / "annotation" / "sheets"))
    ap.add_argument("--key", default=str(ROOT / "annotation" / "_key.csv"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    with open(args.key, encoding="utf-8") as f:
        key = list(csv.DictReader(f))
    labels: dict[str, dict[int, str]] = defaultdict(dict)  # item -> annotator -> label
    tagger, shared = {}, {}
    missing = 0
    for a in sorted({int(k["annotator"]) for k in key}):
        path = pathlib.Path(args.sheets) / f"annotator_{a}.xlsx"
        if not path.exists():
            print(f"(annotator {a}: no sheet yet — skipped)")
            continue
        ws = load_workbook(path)["Label these"]
        answers = {
            int(r[0]): (r[4] or "").strip() for r in ws.iter_rows(min_row=2, values_only=True)
        }
        for k in (k for k in key if int(k["annotator"]) == a):
            lab = answers.get(int(k["row"]), "")
            tagger[k["item_id"]], shared[k["item_id"]] = k["tagger_coarse"], k["shared"] == "1"
            if lab:
                labels[k["item_id"]][a] = lab
            else:
                missing += 1

    # Human reference per item: own items -> that annotator; shared -> majority (ties: skip).
    human = {}
    for item, by_a in labels.items():
        top = Counter(by_a.values()).most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            human[item] = top[0][0]
    hits = sum(human[i] == tagger[i] for i in human)
    lo, hi = wilson(hits, len(human))
    lines = [
        f"# Tagger hand-check ({len(human)} units scored; {missing} blank answers)",
        "",
        f"Tagger accuracy vs humans: {hits}/{len(human)} = {hits / len(human):.0%} "
        f"(95% CI {lo:.0%}-{hi:.0%})",
        "",
    ]

    sh = [i for i in labels if shared[i] and len(labels[i]) >= 2]
    if sh:
        pairs = [
            labels[i][x] == labels[i][y]
            for i in sh
            for x, y in itertools.combinations(sorted(labels[i]), 2)
        ]
        complete = [
            list(labels[i].values())
            for i in sh
            if len(labels[i]) == max(len(labels[j]) for j in sh)
        ]
        lines += [
            f"Human-human agreement on {len(sh)} shared units: {sum(pairs) / len(pairs):.0%} pairwise"
            + (f", Fleiss' kappa {fleiss_kappa(complete):.2f}" if len(complete) >= 2 else ""),
            f"Tagger vs human majority on the shared units: "
            f"{sum(human[i] == tagger[i] for i in sh if i in human)}/{sum(i in human for i in sh)}",
            "",
        ]

    lines += ["Per human label (how often the tagger gave the same label):"]
    by_h = defaultdict(list)
    for i, h in human.items():
        by_h[h].append(tagger[i] == h)
    for h, v in sorted(by_h.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {h:26} {sum(v)}/{len(v)}")
    lines += ["", "Confusion (human -> tagger), disagreements only:"]
    for (h, t), n in Counter(
        (human[i], tagger[i]) for i in human if human[i] != tagger[i]
    ).most_common():
        lines.append(f"  {h:26} -> {t:26} {n}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tagger_handcheck.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
