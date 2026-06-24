"""Phase 2 statistics: per-turn dataset, mixed-effects models, Independence Gradient test.

Builds one tidy per-turn table from the Switchboard baseline + every
``data/generated/<condition>/*.json``, then:

  1. words/turn:   DV ~ Corpus*Section + (1|ConvID) per architecture vs SB  (ANLY-01, -04)
  2. markers:      oh/okay/uh-huh rate per condition vs SB                   (ANLY-03)
  3. alignment:    same mixed model on ALIGN's cosine_semanticL, if present  (ANLY-02)
  4. gradient:     Independence Gradient C1->C2->C3->C4 vs SB                 (ANLY-05)

Design notes
------------
* **Runs on whatever conditions are present** — use it incrementally as generation lands.
* **Pure numpy + scipy core** so it runs in the local Windows env today. ``statsmodels``
  (the formal mixed-effects model) and ``matplotlib`` (figures) are *optional* upgrades; if
  missing, the script falls back to a per-conversation Welch test and skips figures, telling
  you so. Install the full stack (``pip install pandas statsmodels matplotlib``) or run on
  the VM ``convsim`` env for the mixed-effects numbers that go in the write-up.
* **P2 is a robustness condition.** Its prompt embeds a verbatim Switchboard few-shot
  excerpt, so measuring oh/okay/uh-huh on P2 would be circular. P2 is therefore **excluded
  from the marker analysis** and kept only for non-lexical metrics (words/turn, alignment).
* **Section** = within-conversation half: Earlier = first half of turns, Later = second
  half. This mirrors the paper's Earlier/Later split while staying robust to length.

Usage
-----
    python analysis/stats.py                 # full text report
    python analysis/stats.py --figures       # also write PNGs to analysis/figures/
    python analysis/stats.py --n-sb 50       # number of Switchboard convs for the baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns                 # noqa: E402  (C1 raw parse reuse)
from analysis.metrics import marker_counts                      # noqa: E402
from analysis.swda import (                                     # noqa: E402
    iter_conversation_files, parse_conversation, conversation_no_of,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated"
ALIGN_CSV = ROOT / "data" / "align" / "alignment_turns.csv"     # VM writes this; see load_align_rows
FIG_DIR = ROOT / "analysis" / "figures"

ARCH_ORDER = ["C1", "C2", "C3", "C4"]            # independence gradient order
MARKERS = ("oh", "okay", "uh-huh")

# Paper Switchboard references (Mayor et al. 2025), for context in the report.
PAPER_SB = {"words_per_turn": 14.0, "oh": 0.57, "okay": 0.16, "uh-huh": 1.03, "align_earlier": 0.57}


# --------------------------------------------------------------------------- #
# 1. Build the tidy per-turn table                                            #
# --------------------------------------------------------------------------- #

def _section(turn_index: int, n_turns: int) -> str:
    """Earlier = first half of the conversation, Later = second half."""
    return "Earlier" if turn_index < n_turns / 2 else "Later"


def _rows_from_turns(turns, *, conv_id, condition, architecture, prompt_level, corpus):
    n = len(turns)
    rows = []
    for i, (_, text) in enumerate(turns):
        mc = marker_counts(text)
        rows.append({
            "conv_id": conv_id,
            "condition": condition,
            "architecture": architecture,
            "prompt_level": prompt_level,
            "corpus": corpus,                      # "SB" or "LLM"
            "turn_index": i,
            "n_turns": n,
            "section": _section(i, n),
            "words": len(text.split()),
            "oh": mc["oh"],
            "okay": mc["okay"],
            "uh-huh": mc["uh-huh"],
        })
    return rows


def generated_rows() -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(GEN_ROOT / "*" / "*.json"))):
        rec = json.load(open(f, encoding="utf-8"))
        turns = conversation_turns(rec)
        if not turns:
            continue
        rows += _rows_from_turns(
            turns,
            conv_id=f"{rec['condition']}/{rec.get('conversation_no', pathlib.Path(f).stem)}",
            condition=rec["condition"],
            architecture=rec.get("architecture", rec["condition"].split("-")[0]),
            prompt_level=rec.get("prompt_level", rec["condition"].split("-")[-1]),
            corpus="LLM",
        )
    return rows


def switchboard_rows(n: int = 50) -> list[dict]:
    rows: list[dict] = []
    for fp in list(iter_conversation_files())[:n]:
        turns = parse_conversation(fp)
        if not turns:
            continue
        rows += _rows_from_turns(
            turns,
            conv_id=f"SB/{conversation_no_of(fp)}",
            condition="SB", architecture="SB", prompt_level="SB", corpus="SB",
        )
    return rows


def conv_table(rows: list[dict]) -> list[dict]:
    """Collapse per-turn rows to one record per conversation (means / rates)."""
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_conv[r["conv_id"]].append(r)
    out = []
    for conv_id, rs in by_conv.items():
        total_words = sum(r["words"] for r in rs) or 1
        head = rs[0]
        rec = {
            "conv_id": conv_id,
            "condition": head["condition"],
            "architecture": head["architecture"],
            "prompt_level": head["prompt_level"],
            "corpus": head["corpus"],
            "n_turns": head["n_turns"],
            "mean_words": statistics.mean(r["words"] for r in rs),
        }
        for m in MARKERS:
            rec[f"{m}_rate"] = 100.0 * sum(r[m] for r in rs) / total_words
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# 2. Models / tests                                                           #
# --------------------------------------------------------------------------- #

def _cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else float("nan")


def words_mixedlm(rows: list[dict], architecture: str, prompt_level: str) -> str | None:
    """Fit ``words ~ C(corpus)*C(section) + (1|conv_id)`` (SB vs one LLM condition).

    Returns a formatted summary, or ``None`` if statsmodels/pandas are unavailable
    (caller falls back to :func:`welch_words`).
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    sub = [r for r in rows
           if r["corpus"] == "SB"
           or (r["architecture"] == architecture and r["prompt_level"] == prompt_level)]
    df = pd.DataFrame(sub)
    if df.empty or df["corpus"].nunique() < 2:
        return f"  (insufficient data for {architecture}-{prompt_level} mixed model)"
    df["corpus"] = pd.Categorical(df["corpus"], categories=["SB", "LLM"])
    df["section"] = pd.Categorical(df["section"], categories=["Earlier", "Later"])
    try:
        md = smf.mixedlm("words ~ C(corpus) * C(section)", df, groups=df["conv_id"])
        res = md.fit(reml=True, method="lbfgs")
    except Exception as e:                                    # singular fits, tiny n, etc.
        return f"  (mixed model failed for {architecture}-{prompt_level}: {e})"
    lines = [f"  {architecture}-{prompt_level} vs SB  -  words ~ Corpus*Section + (1|ConvID)"]
    for name in res.params.index:
        if name.startswith("Group") or name == "Intercept":
            continue
        lines.append(f"    {name:32} beta={res.params[name]:8.2f}  p={res.pvalues[name]:.3g}")
    return "\n".join(lines)


def welch_words(conv_rows: list[dict], architecture: str, prompt_level: str) -> str:
    """scipy fallback: per-conversation mean words, Welch t-test vs SB + Cohen's d."""
    sb = [c["mean_words"] for c in conv_rows if c["corpus"] == "SB"]
    cond = [c["mean_words"] for c in conv_rows
            if c["architecture"] == architecture and c["prompt_level"] == prompt_level]
    if len(sb) < 2 or len(cond) < 2:
        return f"  {architecture}-{prompt_level}: n too small (SB={len(sb)}, cond={len(cond)})"
    t, p = stats.ttest_ind(cond, sb, equal_var=False)
    d = _cohens_d(cond, sb)
    return (f"  {architecture}-{prompt_level}: mean w/turn {statistics.mean(cond):5.1f} "
            f"vs SB {statistics.mean(sb):5.1f}  Welch t={t:6.2f} p={p:.3g}  d={d:5.2f}")


def marker_summary(conv_rows: list[dict], condition: str) -> str:
    sb_rows = [c for c in conv_rows if c["corpus"] == "SB"]
    cond_rows = [c for c in conv_rows if c["condition"] == condition]
    if len(sb_rows) < 2 or len(cond_rows) < 2:
        return f"  {condition}: n too small"
    parts = [f"  {condition}:"]
    for m in MARKERS:
        sb = [c[f"{m}_rate"] for c in sb_rows]
        cd = [c[f"{m}_rate"] for c in cond_rows]
        try:
            _, p = stats.mannwhitneyu(cd, sb, alternative="two-sided")
        except ValueError:                                   # all-identical (e.g. all zero)
            p = float("nan")
        parts.append(f"{m} {statistics.mean(cd):.2f} vs {statistics.mean(sb):.2f} (p={p:.2g})")
    return "  ".join(parts)


def independence_gradient(conv_rows: list[dict], prompt_level: str, value_key: str) -> dict:
    """Ordered trend test of ``value_key`` across C1->C2->C3->C4 at one prompt level.

    Returns means per architecture, distance to the SB mean, a Spearman trend
    (architecture-index vs value) and a Kruskal-Wallis omnibus.
    """
    sb_vals = [c[value_key] for c in conv_rows if c["corpus"] == "SB"]
    sb_mean = statistics.mean(sb_vals) if sb_vals else float("nan")

    groups, means, dist, x, y = {}, {}, {}, [], []
    for idx, arch in enumerate(ARCH_ORDER):
        vals = [c[value_key] for c in conv_rows
                if c["architecture"] == arch and c["prompt_level"] == prompt_level]
        if not vals:
            continue
        groups[arch] = vals
        means[arch] = statistics.mean(vals)
        dist[arch] = abs(means[arch] - sb_mean)
        x += [idx] * len(vals)
        y += vals

    res = {"prompt_level": prompt_level, "value_key": value_key, "sb_mean": sb_mean,
           "means": means, "dist_to_sb": dist, "n_arch": len(groups),
           "trend_rho": None, "trend_p": None, "kw_p": None, "closest_to_sb": None}
    if len(groups) >= 2 and len(set(x)) >= 2:
        rho, p = stats.spearmanr(x, y)
        res["trend_rho"], res["trend_p"] = float(rho), float(p)
        if all(len(v) >= 1 for v in groups.values()) and len(groups) >= 2:
            try:
                res["kw_p"] = float(stats.kruskal(*groups.values()).pvalue)
            except ValueError:
                res["kw_p"] = float("nan")
        res["closest_to_sb"] = min(dist, key=dist.get)
    return res


# --------------------------------------------------------------------------- #
# 3. Optional ALIGN integration (VM produces the input)                       #
# --------------------------------------------------------------------------- #

def load_align_rows() -> list[dict]:
    """Load ALIGN per-turn output if the VM has produced it.

    Expected CSV at ``data/align/alignment_turns.csv`` with columns:
        condition, conv_id, turn_index, n_turns, cosine_semanticL
    (``condition`` "SB" for the baseline). Returns conv-level rows shaped like
    :func:`conv_table` with an ``align`` value, or ``[]`` if the file is absent.
    """
    if not ALIGN_CSV.exists():
        return []
    import csv
    by_conv: dict[str, list[dict]] = defaultdict(list)
    with open(ALIGN_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_conv[(row["condition"], row["conv_id"])].append(row)
    out = []
    for (condition, conv_id), rs in by_conv.items():
        vals = [float(r["cosine_semanticL"]) for r in rs if r.get("cosine_semanticL")]
        if not vals:
            continue
        arch = "SB" if condition == "SB" else condition.split("-")[0]
        plv = "SB" if condition == "SB" else condition.split("-")[-1]
        out.append({"conv_id": f"{condition}/{conv_id}", "condition": condition,
                    "architecture": arch, "prompt_level": plv,
                    "corpus": "SB" if condition == "SB" else "LLM",
                    "align": statistics.mean(vals)})
    return out


# --------------------------------------------------------------------------- #
# 4. Figures (optional)                                                       #
# --------------------------------------------------------------------------- #

def make_figures(conv_rows: list[dict]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sb_mean = statistics.mean([c["mean_words"] for c in conv_rows if c["corpus"] == "SB"] or [0])
    conds = sorted({c["condition"] for c in conv_rows if c["corpus"] == "LLM"})
    means = [statistics.mean([c["mean_words"] for c in conv_rows if c["condition"] == k]) for k in conds]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(conds, means, color="#4c72b0")
    ax.axhline(sb_mean, color="crimson", ls="--", label=f"Switchboard ({sb_mean:.1f})")
    ax.set_ylabel("mean words / turn"); ax.set_title("Words per turn by condition vs Switchboard")
    ax.legend(); plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "words_per_turn.png", dpi=130); plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sb", type=int, default=50, help="Switchboard convs for the baseline")
    ap.add_argument("--figures", action="store_true", help="also write PNGs to analysis/figures/")
    args = ap.parse_args()

    rows = generated_rows() + switchboard_rows(args.n_sb)
    if not any(r["corpus"] == "LLM" for r in rows):
        print("No generated conversations found under data/generated/. Nothing to analyze.")
        return
    convs = conv_table(rows)
    prompt_levels = sorted({c["prompt_level"] for c in convs if c["corpus"] == "LLM"})

    print("=" * 72)
    print("PHASE 2 STATISTICS")
    n_llm = sum(c["corpus"] == "LLM" for c in convs)
    n_sb = sum(c["corpus"] == "SB" for c in convs)
    print(f"generated convs: {n_llm}   Switchboard convs: {n_sb}   prompt levels: {prompt_levels}")
    print("=" * 72)

    # --- words/turn: mixed model (or Welch fallback) ---
    print("\n[1] WORDS / TURN  -  DV ~ Corpus*Section + (1|ConvID)")
    have_smf = words_mixedlm(rows, "C1", prompt_levels[0]) is not None if prompt_levels else False
    if not have_smf:
        print("  (statsmodels/pandas not installed — using per-conversation Welch t-test fallback)")
    for plv in prompt_levels:
        for arch in ARCH_ORDER:
            if not any(c["architecture"] == arch and c["prompt_level"] == plv for c in convs):
                continue
            if have_smf:
                print(words_mixedlm(rows, arch, plv))
            else:
                print(welch_words(convs, arch, plv))

    # --- markers (P2 excluded as circular) ---
    print("\n[2] MARKER RATES per 100 words vs SB  (Mann-Whitney; SB ref oh/okay/uh-huh "
          f"= {PAPER_SB['oh']}/{PAPER_SB['okay']}/{PAPER_SB['uh-huh']})")
    for c in sorted({c["condition"] for c in convs if c["corpus"] == "LLM"}):
        if c.endswith("P2"):
            print(f"  {c}: SKIPPED (few-shot excerpt -> circular for lexical markers)")
            continue
        print(marker_summary(convs, c))

    # --- alignment (if VM produced it) ---
    align = load_align_rows()
    print("\n[3] ALIGNMENT (cosine_semanticL)")
    if not align:
        print(f"  ALIGN output not present yet - expected at {ALIGN_CSV.relative_to(ROOT)} "
              "(produced on the VM). Skipping.")
    else:
        for plv in sorted({a["prompt_level"] for a in align if a["corpus"] == "LLM"}):
            g = independence_gradient(align, plv, "align")
            print(f"  {plv}: SB={g['sb_mean']:.3f}  " +
                  "  ".join(f"{a}={m:.3f}" for a, m in g["means"].items()))

    # --- independence gradient on words/turn ---
    print("\n[4] INDEPENDENCE GRADIENT  C1->C2->C3->C4  (words/turn, per prompt level)")
    print("    rho<0: w/turn falls along the gradient; dist = abs(mean - SB mean)")
    for plv in prompt_levels:
        g = independence_gradient(convs, plv, "mean_words")
        if g["n_arch"] < 2:
            print(f"  {plv}: <2 architectures present, skipping trend")
            continue
        order = "  ".join(f"{a}={m:.1f}(dist {g['dist_to_sb'][a]:.1f})" for a, m in g["means"].items())
        rho = g["trend_rho"]; tp = g["trend_p"]; kw = g["kw_p"]
        print(f"  {plv}: SB={g['sb_mean']:.1f} | {order}")
        print(f"        Spearman rho={rho:+.2f} p={tp:.3g}  Kruskal p={kw:.3g}  "
              f"closest to SB: {g['closest_to_sb']}")

    if args.figures:
        ok = make_figures(convs)
        print(f"\n[figures] {'written to ' + str(FIG_DIR.relative_to(ROOT)) if ok else 'matplotlib not installed - skipped'}")

    print("\nNote: spot-check data (n=2/cond) gives unstable estimates; numbers stabilize at "
          "the full 50/condition. P2 shown for non-lexical metrics only.")


if __name__ == "__main__":
    main()
