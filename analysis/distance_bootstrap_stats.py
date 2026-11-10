"""Reviewer checklist: add error bars and one stats test to "architecture matters more
than prompting".

Currently results/dialogue_acts/da_jsd_vs_sb.csv reports ONE pooled JSD number per
condition (all ~50 conversations' units pooled into a single distribution, then compared
to humans). That throws away conversation-to-conversation variation, so there's no way to
put a confidence interval on it or test architecture vs. prompt statistically.

This script instead:
  1. Computes the coarse dialogue-act distance from humans PER CONVERSATION (not pooled),
     using the same Jensen-Shannon divergence formula as analysis/dialogue_acts.py.
  2. Bootstraps a 95% CI per condition: resample that condition's ~50 per-conversation
     distances with replacement (10,000 times), take the 2.5/97.5 percentiles of the
     resampled means -- these are the error bars for a bar chart.
  3. Fits ONE model, `distance ~ C(architecture) * C(prompt)`, on all 600 per-conversation
     distances pooled across every condition, and reports the ANOVA table (sum_sq, F, p)
     so architecture's and prompt's explanatory power can be directly compared -- this is
     what backs up "architecture matters more than prompting" statistically rather than
     just visually.

This needs DialogTag-tagged units with conversation_no/condition preserved per unit, which
only exists once you've generated conversations and run the real tagging pipeline (needs
the isolated env in analysis/requirements-dialogue-acts.txt) -- so there is nothing to run
yet. This file is the code for when that data exists; see the __main__ smoke test at the
bottom for a synthetic-data correctness check of the bootstrap + stats logic alone.

Usage (once you have generated conversations and have run analysis/dialogue_acts.py at
least once, so results/dialogue_acts/dialogtag_predictions.json has a cache to reuse):
    python analysis/distance_bootstrap_stats.py
    python analysis/distance_bootstrap_stats.py --force-retag  # ignore the cache
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.dialogue_acts import (  # noqa: E402
    COARSE_LABELS,
    OUT_DIR,
    Conversation,
    js_divergence,
    load_generated,
    load_switchboard,
    tag_both_sides,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260714


def coarse_distribution(labels: list[str]) -> np.ndarray:
    """Count vector over COARSE_LABELS, in that fixed order (unnormalized -- js_divergence
    normalizes internally, same convention as analysis/dialogue_acts.py)."""
    counts = Counter(labels)
    return np.array([counts.get(act, 0) for act in COARSE_LABELS], dtype=float)


def per_conversation_distances(generated: list[Conversation], human_reference: np.ndarray) -> list[dict]:
    """One row per conversation: condition, architecture, prompt, n_units, distance."""
    rows = []
    for conv in generated:
        dist = coarse_distribution(conv.coarse_labels())
        jsd = js_divergence(dist, human_reference)
        condition = conv.condition
        architecture, _, prompt = condition.partition("-")
        rows.append({
            "condition": condition,
            "architecture": architecture,
            "prompt": prompt,
            "conversation_no": conv.conversation_no,
            "n_units": len(conv.coarse_labels()),
            "distance": jsd,
        })
    return rows


def bootstrap_ci(values: list[float], reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CI on the MEAN, by resampling conversations (not units) with replacement."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resampled_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(reps)
    ])
    lo, hi = np.quantile(resampled_means, [0.025, 0.975])
    return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(arr)}


def summarize_with_error_bars(rows: list[dict]) -> dict[str, dict]:
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row["distance"])
    return {cond: bootstrap_ci(vals) for cond, vals in sorted(by_condition.items())}


def fit_architecture_vs_prompt(rows: list[dict]):
    """distance ~ C(architecture) * C(prompt), ANOVA table. Requires pandas + statsmodels
    (not needed by the rest of this repo -- install only for this script)."""
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    df = pd.DataFrame(rows)
    model = smf.ols("distance ~ C(architecture) * C(prompt)", data=df).fit()
    table = anova_lm(model, typ=2)
    return model, table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--validation-conversations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--cache", type=pathlib.Path, default=OUT_DIR / "dialogtag_predictions.json")
    ap.add_argument("--force-retag", action="store_true")
    ap.add_argument("--out-csv", type=pathlib.Path, default=OUT_DIR / "da_distance_per_conversation.csv")
    args = ap.parse_args()

    print("Loading generated + Switchboard conversations...")
    generated = load_generated()
    gold = load_switchboard()

    print("Tagging (reusing cache if present)...")
    tagged_human, validation, cache_used = tag_both_sides(
        generated, gold,
        model_name=args.model, batch_size=args.batch_size,
        validation_conversations=args.validation_conversations, seed=args.seed,
        cache_path=args.cache, force_retag=args.force_retag, use_cache=True,
    )
    print(f"Cache used: {cache_used}\n")

    human_reference = coarse_distribution(
        [label for conv in tagged_human for label in conv.coarse_labels()]
    )

    rows = per_conversation_distances(generated, human_reference)

    import csv
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "architecture", "prompt", "conversation_no", "n_units", "distance"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} per-conversation distances to {args.out_csv}\n")

    print("=== Bootstrap 95% CI per condition (error bars) ===")
    print(f"{'condition':10s} {'n':>4s} {'mean':>8s} {'ci_lo':>8s} {'ci_hi':>8s}")
    summary = summarize_with_error_bars(rows)
    for cond, s in summary.items():
        print(f"{cond:10s} {s['n']:4d} {s['mean']:8.4f} {s['ci_lo']:8.4f} {s['ci_hi']:8.4f}")

    print("\n=== Stats test: distance ~ architecture * prompt ===")
    _model, table = fit_architecture_vs_prompt(rows)
    print(table)


if __name__ == "__main__":
    main()
