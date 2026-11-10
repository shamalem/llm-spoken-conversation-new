"""Reviewer checklist: is "conceptual alignment" just reflecting turn length?

Steps implemented (reviewer's own wording):
  1. Baseline: per conversation, similarity between turns that are NOT next to each
     other (distance >= 5 turns apart).
  2. Real alignment = adjacent-turn score - baseline.
  3. Length check: repeat 1-2 twice more, both (a) truncating every turn to its first 15
     words, and (b) only comparing turn pairs of similar length (ratio >= 0.5, no
     truncation) -- the two alternatives the reviewer offered.
  4. Decide: if every LLM condition's corrected score still beats the human corrected
     score under BOTH length-control methods, keep the figure with the corrected numbers.

Embedding method: average of in-vocabulary word2vec-google-news-300 vectors per turn
(lowercased, punctuation stripped, whitespace split), L2-normalized, cosine similarity.
Deliberately simpler than the full ALIGN pipeline (no lemmatization/POS-filtering) so
adjacent and non-adjacent scores are computed by the exact same method -- this diagnostic
cares about the *relative* comparison, not matching ALIGN's published decimal values.

Reuses the project's own canonical turn-loaders (analysis.analyze.conversation_turns for
generated conversations, analysis.swda.parse_conversation for Switchboard) so the turns
being compared here are identical to the ones every other analysis script in this repo
uses -- not a separately-parsed copy.

Usage (after regenerating conversations)
-----------------------------------------
    python analysis/alignment_confound_check.py \\
        --sb-root /path/to/local/swda \\
        --w2v-path /path/to/word2vec-google-news-300.gz

--gen-root defaults to this repo's data/generated/. --sb-root has no default: Switchboard
is LDC-licensed and never committed here (CLAUDE.md), so point it at wherever you keep
your own local extraction. --w2v-path has no default either (it's a ~1.7GB download, not
committed); get it from https://github.com/RaRe-Technologies/gensim-data (release tag
"word2vec-google-news-300").
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import statistics as stats
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns  # noqa: E402
from analysis.swda import iter_conversation_files, parse_conversation  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated"

random.seed(42)
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str, max_words: int | None = None) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    return words[:max_words] if max_words is not None else words


def turn_vector(kv, text: str, max_words: int | None = None):
    import numpy as np

    words = tokenize(text, max_words)
    vecs = [kv[w] for w in words if w in kv.key_to_index]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b))


def load_llm_conversations(cond_dir: pathlib.Path) -> list[list[tuple[str, str]]]:
    out = []
    for fp in sorted(cond_dir.glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        out.append(conversation_turns(rec))
    return out


def load_sb_conversations(sb_root: pathlib.Path, n: int | None) -> list[list[tuple[str, str]]]:
    files = list(iter_conversation_files(sb_root))
    if n:
        files = files[:n]
    return [parse_conversation(fp) for fp in files]


def conversation_scores(kv, turns, max_words, min_nonadjacent_dist=5, n_baseline_samples=10):
    n = len(turns)
    if n < 2:
        return None, None
    vecs = [turn_vector(kv, txt, max_words) for _, txt in turns]

    adj_sims = [cosine(vecs[i], vecs[i + 1]) for i in range(n - 1) if vecs[i] is not None and vecs[i + 1] is not None]
    if not adj_sims:
        return None, None

    candidates = [(i, j) for i in range(n) for j in range(i + min_nonadjacent_dist, n)]
    if not candidates:
        return stats.mean(adj_sims), None
    sample = random.sample(candidates, min(n_baseline_samples, len(candidates)))
    base_sims = [cosine(vecs[i], vecs[j]) for i, j in sample if vecs[i] is not None and vecs[j] is not None]
    if not base_sims:
        return stats.mean(adj_sims), None
    return stats.mean(adj_sims), stats.mean(base_sims)


def conversation_scores_matched(kv, turns, min_nonadjacent_dist=5, n_baseline_samples=10,
                                 min_len_ratio=0.5, min_words=3):
    n = len(turns)
    if n < 2:
        return None, None
    lens = [len(tokenize(txt)) for _, txt in turns]
    vecs = [turn_vector(kv, txt) for _, txt in turns]

    def length_ok(i, j):
        li, lj = lens[i], lens[j]
        return li >= min_words and lj >= min_words and min(li, lj) / max(li, lj) >= min_len_ratio

    adj_sims = [cosine(vecs[i], vecs[i + 1]) for i in range(n - 1)
                if length_ok(i, i + 1) and vecs[i] is not None and vecs[i + 1] is not None]
    if not adj_sims:
        return None, None

    candidates = [(i, j) for i in range(n) for j in range(i + min_nonadjacent_dist, n) if length_ok(i, j)]
    if not candidates:
        return stats.mean(adj_sims), None
    sample = random.sample(candidates, min(n_baseline_samples, len(candidates)))
    base_sims = [cosine(vecs[i], vecs[j]) for i, j in sample if vecs[i] is not None and vecs[j] is not None]
    if not base_sims:
        return stats.mean(adj_sims), None
    return stats.mean(adj_sims), stats.mean(base_sims)


def run_scores(kv, conversations, max_words):
    adj_vals, base_vals, corrected = [], [], []
    for turns in conversations:
        adj, base = conversation_scores(kv, turns, max_words)
        if adj is None or base is None:
            continue
        adj_vals.append(adj)
        base_vals.append(base)
        corrected.append(adj - base)
    return {
        "n": len(corrected),
        "adjacent_mean": stats.mean(adj_vals) if adj_vals else float("nan"),
        "baseline_mean": stats.mean(base_vals) if base_vals else float("nan"),
        "corrected_mean": stats.mean(corrected) if corrected else float("nan"),
    }


def run_scores_matched(kv, conversations):
    adj_vals, base_vals, corrected = [], [], []
    for turns in conversations:
        adj, base = conversation_scores_matched(kv, turns)
        if adj is None or base is None:
            continue
        adj_vals.append(adj)
        base_vals.append(base)
        corrected.append(adj - base)
    return {
        "n": len(corrected),
        "adjacent_mean": stats.mean(adj_vals) if adj_vals else float("nan"),
        "baseline_mean": stats.mean(base_vals) if base_vals else float("nan"),
        "corrected_mean": stats.mean(corrected) if corrected else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sb-root", required=True, help="local path to the extracted swda/ directory (not committed)")
    ap.add_argument("--sb-n", type=int, default=50, help="number of SB conversations to use (default 50, matches the project's usual baseline sample)")
    ap.add_argument("--gen-root", default=str(GEN_ROOT), help="root containing <condition>/*.json generated conversations")
    ap.add_argument("--w2v-path", required=True, help="path to word2vec-google-news-300.gz (not committed, ~1.7GB)")
    args = ap.parse_args()

    from gensim.models import KeyedVectors

    print("Loading word2vec-google-news-300 (~40s)...")
    kv = KeyedVectors.load_word2vec_format(args.w2v_path, binary=True)
    print(f"Loaded {len(kv)} vectors.\n")

    gen_root = pathlib.Path(args.gen_root)
    sb_root = pathlib.Path(args.sb_root)
    conditions = sorted(p.name for p in gen_root.glob("*") if p.is_dir())
    print(f"Conditions found: {conditions}\n")

    sb_conversations = load_sb_conversations(sb_root, args.sb_n)
    print(f"SB conversations loaded: {len(sb_conversations)}\n")

    all_llm_convs = {cond: load_llm_conversations(gen_root / cond) for cond in conditions}

    results = {}
    for label, max_words in (("FULL TURNS (no length control)", None), ("LENGTH-CONTROLLED (first 15 words)", 15)):
        print(f"=== {label} ===")
        print(f"{'condition':10s} {'n':>4s} {'adjacent':>10s} {'baseline':>10s} {'corrected':>10s}")
        results[label] = {"SB": run_scores(kv, sb_conversations, max_words)}
        r = results[label]["SB"]
        print(f"{'SB':10s} {r['n']:4d} {r['adjacent_mean']:10.4f} {r['baseline_mean']:10.4f} {r['corrected_mean']:10.4f}")
        for cond in conditions:
            r = run_scores(kv, all_llm_convs[cond], max_words)
            results[label][cond] = r
            print(f"{cond:10s} {r['n']:4d} {r['adjacent_mean']:10.4f} {r['baseline_mean']:10.4f} {r['corrected_mean']:10.4f}")
        print()

    label = "LENGTH-MATCHED PAIRS (ratio >= 0.5, no truncation)"
    print(f"=== {label} ===")
    print(f"{'condition':10s} {'n':>4s} {'adjacent':>10s} {'baseline':>10s} {'corrected':>10s}")
    results[label] = {"SB": run_scores_matched(kv, sb_conversations)}
    r = results[label]["SB"]
    print(f"{'SB':10s} {r['n']:4d} {r['adjacent_mean']:10.4f} {r['baseline_mean']:10.4f} {r['corrected_mean']:10.4f}")
    for cond in conditions:
        r = run_scores_matched(kv, all_llm_convs[cond])
        results[label][cond] = r
        print(f"{cond:10s} {r['n']:4d} {r['adjacent_mean']:10.4f} {r['baseline_mean']:10.4f} {r['corrected_mean']:10.4f}")
    print()

    print("=== DECIDE: corrected condition vs. SB, under BOTH length-control methods ===")
    trunc_label = "LENGTH-CONTROLLED (first 15 words)"
    match_label = "LENGTH-MATCHED PAIRS (ratio >= 0.5, no truncation)"
    sb_trunc = results[trunc_label]["SB"]["corrected_mean"]
    sb_match = results[match_label]["SB"]["corrected_mean"]
    print(f"SB corrected: truncated={sb_trunc:.4f}  length-matched={sb_match:.4f}\n")
    all_survive = True
    for cond in conditions:
        c_trunc = results[trunc_label][cond]["corrected_mean"]
        c_match = results[match_label][cond]["corrected_mean"]
        n_match = results[match_label][cond]["n"]
        survives_trunc = c_trunc > sb_trunc
        survives_match = c_match > sb_match
        all_survive = all_survive and survives_trunc and survives_match
        verdict = "BOTH survive" if (survives_trunc and survives_match) else (
            "ONLY truncated survives" if survives_trunc else (
                "ONLY length-matched survives" if survives_match else "GAP GONE in both"))
        print(f"{cond:10s} truncated={c_trunc:.4f}  length-matched={c_match:.4f} (n={n_match})  -> {verdict}")
    print()
    print("FINAL VERDICT:", "KEEP THE FIGURE -- gap survives for every condition under both corrections."
          if all_survive else "Mixed/failed -- inspect per-condition rows above before deciding.")


if __name__ == "__main__":
    main()
