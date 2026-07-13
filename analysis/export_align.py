"""Run ALIGN (Duran et al., 2019) on the generated corpora and/or Switchboard.

Produces the per-turn CSV analysis/stats.py expects at data/align/alignment_turns.csv:
    condition, conv_id, turn_index, n_turns, cosine_semanticL   (condition="SB" for the baseline)

This mirrors the SB pipeline already validated on the VM (see VM_REPORT.md "ALIGN validation"):
    1. write each conversation as a tab-separated participant/content .txt file
       (same format as analysis/prepare_align_input.py)
    2. align.prepare_transcripts()  -- cleaning, tokenizing, lemmatizing, POS-tagging
    3. align.calculate_alignment()  -- turn-by-turn lexical/syntactic/conceptual alignment,
       using the same pretrained word2vec-google-news-300 vectors as the validated SB run

Needs the `ALIGN` package + NLTK data (punkt_tab, wordnet, averaged_perceptron_tagger[_eng])
+ the pretrained word2vec file. All three exist already in the VM's `convsim` env per
VM_REPORT.md -- run this there. `--pretrained-vectors` defaults to the path VM_REPORT.md
recorded (~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz).

Usage:
    python analysis/export_align.py --data-dir data/generated_v2 --include-sb --n-sb 50
    python analysis/export_align.py --data-dir data/generated_v2 --conditions C2-P0 C2-P1
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns  # noqa: E402
from analysis.swda import iter_conversation_files, parse_conversation, conversation_no_of  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALIGN_OUT = ROOT / "data" / "align"


def generated_records(data_dir: pathlib.Path, conditions: list[str] | None):
    """(condition, conv_id, turns) for every data_dir/<condition>/*.json.

    Skips any condition directory whose name doesn't look like a real condition (e.g. the
    `.broken` backups left behind by the C1 regeneration) unless explicitly requested.
    """
    for cond_dir in sorted(p for p in data_dir.glob("*") if p.is_dir()):
        cond = cond_dir.name
        if conditions and cond not in conditions:
            continue
        if not conditions and (".broken" in cond or cond.endswith(".broken")):
            continue
        for fp in sorted(cond_dir.glob("*.json")):
            rec = json.load(open(fp, encoding="utf-8"))
            turns = conversation_turns(rec)
            if len(turns) >= 4:  # ALIGN needs turn pairs; too-short convs aren't useful anyway
                yield cond, fp.stem, turns


def switchboard_records(n: int):
    for fp in list(iter_conversation_files())[:n]:
        turns = parse_conversation(fp)
        if len(turns) >= 4:
            yield "SB", str(conversation_no_of(fp)), turns


def _sanitize_for_align_tsv(text: str) -> str:
    """Make a turn safe for align.prepare_transcripts()'s pd.read_csv(sep='\\t').

    Some generated turns carry leftover artifacts (label leaks, stray meta-instructions)
    with an odd number of literal '"' characters or embedded newlines -- e.g. C2 turns like
    '...getting worse." (Label for ParticipantB)'. An unmatched quote makes pandas' C parser
    treat everything after it as inside one open quoted field and read straight past EOF
    (ParserError: "EOF inside string"); an embedded newline creates a bogus extra row. Neither
    character carries meaning ALIGN's word2vec scoring needs, so strip both rather than trying
    to preserve them.
    """
    return text.replace("\n", " ").replace("\r", " ").replace('"', "").replace("\t", " ").strip()


def write_input_files(records, raw_dir: pathlib.Path) -> dict[str, tuple[str, str, int]]:
    """Write one ALIGN-format .txt per conversation; return basename -> (condition, conv_id, n_turns)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, tuple[str, str, int]] = {}
    for condition, conv_id, turns in records:
        fname = f"{condition}__{conv_id}.txt"
        with open(raw_dir / fname, "w", encoding="utf-8") as f:
            f.write("participant\tcontent\n")
            for speaker, text in turns:
                text_clean = _sanitize_for_align_tsv(text)
                if text_clean:
                    f.write(f"{speaker}\t{text_clean}\n")
        meta[fname] = (condition, conv_id, len(turns))
    return meta


def run_align(raw_dir: pathlib.Path, work_dir: pathlib.Path, pretrained_vectors: str | None):
    import align

    prepared_dir = work_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Same settings as the validated SB run (VM_REPORT.md): spell-check off.
    align.prepare_transcripts(
        input_files=str(raw_dir),
        output_file_directory=str(prepared_dir),
        run_spell_check=False,
        input_as_directory=True,
    )
    concatenated = prepared_dir / ".." / "align_concatenated_dataframe.txt"

    turn_df, _convo_df = align.calculate_alignment(
        input_files=str(prepared_dir),
        output_file_directory=str(results_dir) + "/",
        semantic_model_input_file=str(concatenated),
        pretrained_input_file=pretrained_vectors,
        use_pretrained_vectors=pretrained_vectors is not None,
    )
    return turn_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/generated_v2")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="restrict to these condition dirs (default: all non-.broken dirs)")
    ap.add_argument("--include-sb", action="store_true", help="also export the Switchboard baseline")
    ap.add_argument("--n-sb", type=int, default=50)
    ap.add_argument("--pretrained-vectors", default=None,
                    help="path to word2vec-google-news-300(.gz); matches the validated SB run. "
                         "Omit to build a from-corpus model instead (NOT comparable to the SB "
                         "baseline -- only for a mechanical smoke test of this script).")
    ap.add_argument("--work-dir", default=None, help="scratch dir for intermediate ALIGN files")
    ap.add_argument("--out", default=str(ALIGN_OUT / "alignment_turns.csv"))
    args = ap.parse_args()

    records = list(generated_records(pathlib.Path(args.data_dir), args.conditions))
    if args.include_sb:
        records += list(switchboard_records(args.n_sb))
    if not records:
        print(f"No conversations found under {args.data_dir} (n_turns>=4).")
        return
    print(f"Preparing {len(records)} conversations for ALIGN...")

    work_dir = pathlib.Path(args.work_dir) if args.work_dir else pathlib.Path(ROOT / "data" / "align" / "_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    raw_dir = work_dir / "raw"
    meta = write_input_files(records, raw_dir)

    turn_df = run_align(raw_dir, work_dir, args.pretrained_vectors)

    # Sub-measure columns ALIGN's calculate_alignment() emits (Duran et al., 2019). The paper
    # (and its released AlignAnalyseOSF.R) aggregates the tok2/tok3/lem2/lem3 sub-measures into
    # one "syntax_stan" and one "lexical" column because the sub-measures correlated r > .6 --
    # replicated here as a plain mean so analysis/original_paper_analysis.py can run the exact
    # same mixed models the paper did, not just on conceptual (cosine_semanticL) alignment.
    SYNTAX_COLS = ["syntax_stan_tok2", "syntax_stan_lem2", "syntax_stan_tok3", "syntax_stan_lem3"]
    LEXICAL_COLS = ["lexical_tok2", "lexical_lem2", "lexical_tok3", "lexical_lem3"]

    out_rows = []
    fieldnames = (["condition", "conv_id", "turn_index", "n_turns", "cosine_semanticL"]
                  + SYNTAX_COLS + LEXICAL_COLS + ["syntax_stan", "lexical"])
    for _, row in turn_df.iterrows():
        fname = row["condition_info"]
        if fname not in meta:
            continue
        condition, conv_id, n_turns = meta[fname]
        syntax_vals = [row.get(c) for c in SYNTAX_COLS if c in row and row.get(c) is not None]
        lexical_vals = [row.get(c) for c in LEXICAL_COLS if c in row and row.get(c) is not None]
        out_row = {
            "condition": condition,
            "conv_id": conv_id,
            "turn_index": int(row["time"]),
            "n_turns": n_turns,
            "cosine_semanticL": row.get("cosine_semanticL", ""),
            "syntax_stan": (sum(syntax_vals) / len(syntax_vals)) if syntax_vals else "",
            "lexical": (sum(lexical_vals) / len(lexical_vals)) if lexical_vals else "",
        }
        for c in SYNTAX_COLS + LEXICAL_COLS:
            out_row[c] = row.get(c, "")
        out_rows.append(out_row)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} turn-pair rows to {out_path}")


if __name__ == "__main__":
    main()
