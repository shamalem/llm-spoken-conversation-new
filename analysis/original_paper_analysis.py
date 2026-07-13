"""Exact replication of Mayor, Bietti & Bangerter (2025) Study 1's statistical analyses.

This file is a DISTINCT, deliberately narrow module: it does not compute anything new, and
it does not use this project's own metric conventions (see evaluate_generated.py / stats.py
for those). Instead it ports the paper's *own* released analysis code as closely as
possible, so our Vicuna/Mistral conditions can be run through literally the same
statistical models the paper ran GPT-4/Claude/Vicuna/Wayfarer through.

Source of truth for each piece:
  - Coordination markers: a line-for-line port of the released `MarkersAnalyses.R`
    (simple OLS, SB as the reference level, all corpora in one model).
  - Alignment: a line-for-line port of the released `AlignAnalyseOSF.R` (mixed-effects
    model per LLM corpus vs. SB, with the *LLM* corpus as the reference level -- note this
    is the OPPOSITE reference convention from the markers script; that is not a bug, it is
    what the paper's own two scripts each did).
  - Turn length: no R file for this was released to us. The paper's Section 2.4.1 describes
    "regression models comparing SB with each other corpus" per phase, with an omnibus
    F-test (matching the exact one-model-many-corpora design of MarkersAnalyses.R) -- so
    this function mirrors that design, not a literal ported script.

Known scope differences from the paper (documented rather than silently glossed over):
  1. The paper splits turn length by phase (opening / main body / closing) using hand
     annotation. We don't have opening/closing phase annotation, so turn length here is
     computed over the whole conversation. Treat it as one phase, not three.
  2. The paper's alignment "Section" is turn-pairs 1-5 (Earlier) vs. 6-10 (Later) *within
     the main body*, itself starting from a hand-annotated "topic initiation" utterance.
     We don't have topic-initiation annotation, so Earlier/Later here uses turn-pairs 0-4
     vs. 5-9 of the conversation as exported by export_align.py -- a proxy for "main body,"
     not an exact match.
  3. Syntactic/lexical alignment require export_align.py's syntax_stan_tok2/tok3/lem2/lem3
     and lexical_tok2/tok3/lem2/lem3 columns (added alongside this file). Older
     alignment_turns.csv exports only have cosine_semanticL -- re-run export_align.py on
     the VM to get syntax_stan/lexical populated before this script's alignment analysis
     will produce anything beyond conceptual alignment.

Usage
-----
    python analysis/original_paper_analysis.py --markers --turn-length
    python analysis/original_paper_analysis.py --alignment --align-csv data/align/alignment_turns.csv
    python analysis/original_paper_analysis.py --all --prompt-level P0
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated"
ALIGN_CSV_DEFAULT = ROOT / "data" / "align" / "alignment_turns.csv"
SWDA_ROOT = ROOT / "data" / "switchboard" / "swda"

# --------------------------------------------------------------------------------------- #
# Inlined from swda.py / analyze.py / metrics.py (those files were removed from the repo;  #
# this module is meant to be self-contained). Kept verbatim so the replication logic below #
# still matches what was validated earlier -- see git history for the original modules.    #
# --------------------------------------------------------------------------------------- #

def _clean_text(raw: str) -> str:
    """Remove SwDA transcription markup, keep the spoken words."""
    t = raw
    t = re.sub(r"<+[^>]*>+", " ", t)      # <beep>, <<long pause>>
    t = re.sub(r"\{[A-Z]\s", " ", t)       # opening {D {F {C {E {A ...
    t = t.replace("}", " ")
    for ch in "[]+#":
        t = t.replace(ch, " ")
    t = re.sub(r"-?/", " ", t)             # slash-unit and -/ interruption
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.?!])", r"\1", t)   # tidy space before punctuation
    return t


def iter_conversation_files(root: pathlib.Path = SWDA_ROOT):
    yield from sorted(root.rglob("sw_*.utt.csv"))


def conversation_no_of(csv_path: pathlib.Path) -> int:
    """sw_0001_4325.utt.csv -> 4325 (the SwDA conversation_no, the metadata join key)."""
    return int(csv_path.stem.split("_")[2].split(".")[0])


def parse_conversation(csv_path: pathlib.Path) -> list[tuple[str, str]]:
    """Return [(caller, text), ...] with consecutive same-caller utterances merged."""
    turns: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spk = row["caller"].strip()
            txt = _clean_text(row["text"])
            if not txt:
                continue
            if turns and turns[-1][0] == spk:
                turns[-1] = (spk, turns[-1][1] + " " + txt)
            else:
                turns.append((spk, txt))
    return turns


def conversation_turns(rec: dict) -> list[tuple[str, str]]:
    """(speaker, text) turns for a generated record -- parses C1 raw_output if needed."""
    if rec.get("turns"):
        return [(t[0], t[1]) for t in rec["turns"]]
    turns = []
    for line in rec.get("raw_output", "").split("\n"):
        line = line.strip()
        if ":" in line:
            spk, txt = line.split(":", 1)
            if txt.strip():
                turns.append((spk.strip(), txt.strip()))
    return turns


MARKER_PATTERNS = {
    "oh": re.compile(r"\boh\b", re.I),
    "okay": re.compile(r"\b(?:okay|ok)\b", re.I),
    "uh-huh": re.compile(r"\buh-?\s?huh\b", re.I),
}


def marker_counts(text: str) -> dict[str, int]:
    return {name: len(pat.findall(text)) for name, pat in MARKER_PATTERNS.items()}


def _wpt(turns: list[tuple[str, str]]) -> list[int]:
    return [len(txt.split()) for _, txt in turns]

ARCHITECTURES = ["C1", "C2", "C3", "C4"]  # C3/C4 = A3/A4 in the presentation, same data


# --------------------------------------------------------------------------------------- #
# Data loading -- one row per conversation (turn length, markers) or per turn-pair (align) #
# --------------------------------------------------------------------------------------- #

def load_per_conversation_table(n_sb: int = 50, prompt_level: str = "P0") -> pd.DataFrame:
    """One row per conversation: condition, mean words/turn, oh/okay/uh-huh rate per 100 words.

    condition is "SB" or one of ARCHITECTURES -- matches MarkersAnalyses.R's `Model` column
    and is exactly the DV table `words_per_turn ~ Model` / `oh ~ Model` etc. need.
    """
    rows = []

    for fp in list(iter_conversation_files())[:n_sb]:
        turns = parse_conversation(fp)
        if not turns:
            continue
        rows.append(_conversation_row("SB", conversation_no_of(fp), turns))

    for arch in ARCHITECTURES:
        cond_dir = GEN_ROOT / f"{arch}-{prompt_level}"
        if not cond_dir.exists():
            continue
        for fp in sorted(cond_dir.glob("*.json")):
            import json
            rec = json.loads(fp.read_text(encoding="utf-8"))
            turns = conversation_turns(rec)
            if not turns:
                continue
            rows.append(_conversation_row(arch, fp.stem, turns))

    return pd.DataFrame(rows)


def _conversation_row(condition: str, conv_id, turns) -> dict:
    wpt = _wpt(turns)
    full_text = " ".join(t for _, t in turns)
    total_words = max(len(full_text.split()), 1)
    mc = marker_counts(full_text)
    return {
        "condition": condition,
        "conv_id": conv_id,
        "mean_words_per_turn": float(np.mean(wpt)) if wpt else 0.0,
        "oh": 100.0 * mc["oh"] / total_words,
        "okay": 100.0 * mc["okay"] / total_words,
        "uh_huh": 100.0 * mc["uh-huh"] / total_words,
    }


def load_alignment_table(csv_path: pathlib.Path = ALIGN_CSV_DEFAULT) -> pd.DataFrame:
    """Turn-pair-level alignment table from export_align.py's CSV.

    Restricts to turn-pairs 0-9 (the paper's "10 successive turn-pairs" cap) and labels
    Section exactly as the paper does: Earlier = pairs 0-4, Later = pairs 5-9. This is a
    proxy for the paper's own Earlier/Later (see module docstring, point 2).
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ti = int(row["turn_index"])
            if ti >= 10:
                continue
            rows.append({
                "condition": row["condition"],
                "conv_id": row["conv_id"],
                "turn_index": ti,
                "section": "Earlier" if ti < 5 else "Later",
                "cosine_semanticL": _to_float(row.get("cosine_semanticL")),
                "syntax_stan": _to_float(row.get("syntax_stan")),
                "lexical": _to_float(row.get("lexical")),
            })
    return pd.DataFrame(rows)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------------------- #
# 1. Turn length -- port of the paper's Section 2.4.1 design (no R file was released)      #
# --------------------------------------------------------------------------------------- #

def run_turn_length_analysis(df: pd.DataFrame) -> None:
    """words_per_turn ~ Model, SB as reference, all corpora in ONE model (omnibus F-test).

    This is the same one-model-many-corpora design as MarkersAnalyses.R's `lm(okay ~ Model)`
    -- the paper's own Section 2.4.1 reports a single F-test per phase (e.g., main body:
    F(6, 1378) = 2471.38), which only makes sense if all corpora were dummy-coded into one
    regression, exactly like the markers script. We do the same, over the whole conversation
    (see module docstring, point 1, for why this isn't split by phase).
    """
    print("\n=== Turn length: mean_words_per_turn ~ Model (SB = reference) ===")
    model = smf.ols(
        "mean_words_per_turn ~ C(condition, Treatment(reference='SB'))", data=df
    ).fit()
    print(model.summary())
    print(f"\nOmnibus F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.2f}, "
          f"p = {model.f_pvalue:.3g}")


# --------------------------------------------------------------------------------------- #
# 2. Coordination markers -- line-for-line port of MarkersAnalyses.R                        #
# --------------------------------------------------------------------------------------- #

def run_marker_analysis(df: pd.DataFrame) -> None:
    """oh/okay/uh_huh ~ Model, SB releveled as the reference (exactly MarkersAnalyses.R)."""
    for marker in ("okay", "uh_huh", "oh"):
        print(f"\n=== Coordination marker: {marker} ~ Model (SB = reference) ===")
        model = smf.ols(
            f"{marker} ~ C(condition, Treatment(reference='SB'))", data=df
        ).fit()
        print(model.summary())
        print(f"\nF({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.2f}, "
              f"p = {model.f_pvalue:.3g}   (paper's okay: F=45.97, uh-huh: F=317.93, oh: F=244.09)")


# --------------------------------------------------------------------------------------- #
# 3. Alignment -- line-for-line port of AlignAnalyseOSF.R                                   #
# --------------------------------------------------------------------------------------- #

def run_alignment_analysis(df: pd.DataFrame, architecture: str) -> None:
    """cosine_semanticL / syntax_stan / lexical ~ Condition * Section + (1|Conv).

    One LLM corpus at a time vs. SB -- exactly AlignAnalyseOSF.R's AGGa..AGGf subsetting.
    NOTE the reference level here is the LLM corpus, not SB (relevel(...,"GPT4A") etc. in
    the R script) -- the opposite convention from run_marker_analysis above. That is a
    faithful replication of what the paper's own two scripts each did, not an inconsistency
    introduced here.
    """
    sub = df[df["condition"].isin(["SB", architecture])].copy()
    if sub.empty:
        print(f"\n=== Alignment vs. SB for {architecture}: no data (run export_align.py first) ===")
        return

    for dv in ("cosine_semanticL", "syntax_stan", "lexical"):
        sub_dv = sub.dropna(subset=[dv])
        if sub_dv.empty:
            print(f"\n=== {dv} ({architecture} vs SB): no data -- "
                  f"re-run export_align.py to populate this column ===")
            continue
        print(f"\n=== Alignment: {dv} ~ Condition * Section + (1|Conv), "
              f"reference = {architecture} ===")
        formula = f"{dv} ~ C(condition, Treatment(reference='{architecture}')) * section"
        model = smf.mixedlm(formula, data=sub_dv, groups=sub_dv["conv_id"])
        result = model.fit(reml=True)
        print(result.summary())

        # VIF on the fixed-effects design matrix, as vif(mAGa.Cosine) etc. did in the R script.
        try:
            import patsy
            X = patsy.dmatrix(
                f"C(condition, Treatment(reference='{architecture}')) * section",
                sub_dv, return_type="dataframe",
            )
            vifs = {
                col: variance_inflation_factor(X.values, i)
                for i, col in enumerate(X.columns) if col != "Intercept"
            }
            print("VIF:", {k: round(v, 2) for k, v in vifs.items()})
        except Exception as e:
            print(f"(VIF skipped: {e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turn-length", action="store_true")
    ap.add_argument("--markers", action="store_true")
    ap.add_argument("--alignment", action="store_true")
    ap.add_argument("--all", action="store_true", help="run all three analyses")
    ap.add_argument("--prompt-level", default="P0", choices=["P0", "P1", "P2"])
    ap.add_argument("--n-sb", type=int, default=50)
    ap.add_argument("--align-csv", default=str(ALIGN_CSV_DEFAULT))
    args = ap.parse_args()

    if not (args.turn_length or args.markers or args.alignment or args.all):
        ap.error("pick at least one of --turn-length / --markers / --alignment / --all")

    if args.turn_length or args.markers or args.all:
        conv_df = load_per_conversation_table(n_sb=args.n_sb, prompt_level=args.prompt_level)
        if conv_df.empty:
            print(f"No conversations found for prompt level {args.prompt_level}.")
        else:
            if args.turn_length or args.all:
                run_turn_length_analysis(conv_df)
            if args.markers or args.all:
                run_marker_analysis(conv_df)

    if args.alignment or args.all:
        align_path = pathlib.Path(args.align_csv)
        if not align_path.exists():
            print(f"\n{align_path} not found -- run export_align.py first.")
        else:
            align_df = load_alignment_table(align_path)
            for arch in ARCHITECTURES:
                run_alignment_analysis(align_df, arch)


if __name__ == "__main__":
    main()
