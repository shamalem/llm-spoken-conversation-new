"""
Evaluate all generated conversations against the full metric set.

Tier 1 (original paper, independently justified):
  M1  Turn length          – mean words/turn, full distribution
  M2  Coordination markers – oh / okay / uh-huh per 100 words
  M3  Alignment trajectory – lexical Jaccard overlap earlier vs later halves

Tier 2 (new criteria, evaluation_criteria.pdf):
  M4  Turn-taking economy        – % turns < 5 words
  M5  Backchannel standalone ratio
  M6  "Oh" epistemic contingency
  M7  Self-repair rate
  M8  Closing sequence order     – stages in correct order
  M9  Sycophantic over-validation rate
  M10 Information density variance (CV of content-word ratio)

Output:
  results/metrics_per_conversation.json  – one record per conversation
  results/metrics_summary.json           – per-condition means ± SD
  results/metrics_summary.csv            – same, as CSV

Usage:
  python analysis/evaluate_generated.py
  python analysis/evaluate_generated.py --data_dir data/generated --out_dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

MARKER_PAT = {
    "oh":     re.compile(r"\boh\b", re.I),
    "okay":   re.compile(r"\b(?:okay|ok)\b", re.I),
    "uh_huh": re.compile(r"\buh-?\s?huh\b", re.I),
}

BACKCHANNEL_FULL = re.compile(
    r"^\s*(uh-?huh|okay|ok|mm-?hm|yeah|right|sure)[\s.,!?]*$", re.I
)
BACKCHANNEL_PRESENCE = re.compile(
    r"\b(uh-?huh|okay|ok|mm-?hm)\b", re.I
)

REPAIR_REPEATS_RE = re.compile(r"\b(\w+)\s+\1\b", re.I)
REPAIR_MARKERS_RE = re.compile(r"-\s|I mean\b|or rather\b", re.I)

CLOSING_STAGES = {
    "pre_closing":   re.compile(r"\bokay\b|\ball right\b|\bwell\b", re.I),
    "justification": re.compile(r"\bi (should|have to|need to|gotta) go\b", re.I),
    "well_wishing":  re.compile(r"\bhave a (nice|good|great)\b|\btake care\b|\bgood luck\b", re.I),
    "goodbye":       re.compile(r"\bbye\b|\bgoodbye\b|\bsee you\b|\bsee ya\b", re.I),
}
CLOSING_ORDER = ["pre_closing", "justification", "well_wishing", "goodbye"]

SYCOPHANCY_PHRASES = [
    "that's a great point", "i really agree", "absolutely",
    "you're right that", "great question", "i love that",
    "that's a really good point", "i completely agree",
    "that's so true", "i totally agree", "great point",
    "definitely", "exactly right",
]

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "it", "i", "you", "he", "she", "we",
    "they", "was", "are", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "that", "this", "these", "those", "my", "your", "his",
    "her", "our", "their", "its", "not", "no", "so", "as", "if",
    "up", "out", "about", "just", "like", "what", "when", "how",
    "there", "then", "than", "from", "by", "into", "over", "also",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

TURN_RE = re.compile(r"^(Participant[AB])\s*:\s*(.*)", re.MULTILINE)


def parse_turns(raw_output: str) -> list[tuple[str, str]]:
    """Return [(speaker, text), ...] from raw_output string (C1 format)."""
    turns = []
    for m in TURN_RE.finditer(raw_output):
        text = m.group(2).strip()
        if text:
            turns.append((m.group(1), text))
    return turns


def load_conversations(data_dir: pathlib.Path) -> list[dict]:
    convs = []
    for json_file in sorted(data_dir.rglob("*.json")):
        with json_file.open(encoding="utf-8") as f:
            data = json.load(f)
        # C2/C3/C4 store turns as [["Speaker", "text"], ...] directly
        if "turns" in data and isinstance(data["turns"], list) and data["turns"]:
            raw = data["turns"]
            if isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2:
                turns = [(str(spk), str(txt).strip()) for spk, txt in raw if str(txt).strip()]
            else:
                turns = parse_turns(data.get("raw_output", ""))
        else:
            turns = parse_turns(data.get("raw_output", ""))
        if turns:
            convs.append({**data, "turns": turns, "file": str(json_file)})
    return convs


# ---------------------------------------------------------------------------
# Metric functions (all take `turns: list[tuple[str, str]]`)
# ---------------------------------------------------------------------------

def m1_turn_length(turns):
    wpt = [len(txt.split()) for _, txt in turns]
    return {
        "mean_words_per_turn": round(statistics.mean(wpt), 3) if wpt else 0,
        "median_words_per_turn": round(statistics.median(wpt), 3) if wpt else 0,
        "stdev_words_per_turn": round(statistics.stdev(wpt), 3) if len(wpt) > 1 else 0,
        "n_turns": len(wpt),
        "words_per_turn_list": wpt,
    }


def m2_marker_rates(turns):
    full_text = " ".join(txt for _, txt in turns)
    total_words = max(len(full_text.split()), 1)
    return {
        f"rate_{name}": round(100.0 * len(pat.findall(full_text)) / total_words, 4)
        for name, pat in MARKER_PAT.items()
    }


def _lexical_overlap(turns_a: list[tuple], turns_b: list[tuple]) -> float:
    """Mean pairwise Jaccard on content words between adjacent cross-speaker turns."""
    pairs = []
    all_turns = turns_a + turns_b
    for i in range(len(all_turns) - 1):
        sa, ta = all_turns[i]
        sb, tb = all_turns[i + 1]
        if sa != sb:
            words_a = {w.lower() for w in ta.split() if w.lower() not in STOP_WORDS}
            words_b = {w.lower() for w in tb.split() if w.lower() not in STOP_WORDS}
            union = words_a | words_b
            if union:
                pairs.append(len(words_a & words_b) / len(union))
    return statistics.mean(pairs) if pairs else 0.0


def m3_alignment_trajectory(turns):
    if len(turns) < 4:
        return {"alignment_earlier": None, "alignment_later": None, "alignment_delta": None}
    mid = len(turns) // 2
    earlier = turns[:mid]
    later = turns[mid:]
    a_early = _lexical_overlap(earlier, [])
    a_late = _lexical_overlap(later, [])
    # recompute cleanly over each half
    def half_overlap(half):
        pairs = []
        for i in range(len(half) - 1):
            sa, ta = half[i]
            sb, tb = half[i + 1]
            if sa != sb:
                wa = {w.lower() for w in ta.split() if w.lower() not in STOP_WORDS}
                wb = {w.lower() for w in tb.split() if w.lower() not in STOP_WORDS}
                u = wa | wb
                if u:
                    pairs.append(len(wa & wb) / len(u))
        return statistics.mean(pairs) if pairs else 0.0
    a_early = half_overlap(earlier)
    a_late = half_overlap(later)
    return {
        "alignment_earlier": round(a_early, 4),
        "alignment_later":   round(a_late, 4),
        "alignment_delta":   round(a_late - a_early, 4),
    }


def m4_turn_taking_economy(turns):
    wpt = [len(txt.split()) for _, txt in turns]
    if not wpt:
        return {"pct_short_turns": None}
    pct = sum(1 for w in wpt if w < 5) / len(wpt)
    return {"pct_short_turns": round(pct, 4)}


def m5_backchannel_standalone(turns):
    texts = [txt for _, txt in turns]
    has_bc = [t for t in texts if BACKCHANNEL_PRESENCE.search(t)]
    if not has_bc:
        return {"backchannel_standalone_ratio": None, "n_backchannel_turns": 0}
    standalone = [t for t in has_bc if BACKCHANNEL_FULL.match(t)]
    return {
        "backchannel_standalone_ratio": round(len(standalone) / len(has_bc), 4),
        "n_backchannel_turns": len(has_bc),
        "n_standalone": len(standalone),
    }


def m6_oh_epistemic(turns):
    texts = [txt for _, txt in turns]
    total_oh, contingent = 0, 0
    for i, turn_text in enumerate(texts):
        if MARKER_PAT["oh"].search(turn_text):
            total_oh += 1
            if i > 0:
                prev = texts[i - 1]
                # proxy: prior turn introduced a named entity or number
                if re.search(r"\b[A-Z][a-z]{2,}\b|\d+", prev):
                    contingent += 1
    if total_oh == 0:
        return {"oh_epistemic_ratio": None, "n_oh": 0}
    return {
        "oh_epistemic_ratio": round(contingent / total_oh, 4),
        "n_oh": total_oh,
        "n_oh_contingent": contingent,
    }


def m7_self_repair(turns):
    all_text = " ".join(txt for _, txt in turns)
    words = all_text.split()
    if not words:
        return {"self_repair_per_100": None}
    repeats = len(REPAIR_REPEATS_RE.findall(all_text))
    markers = len(REPAIR_MARKERS_RE.findall(all_text))
    rate = (repeats + markers) / len(words) * 100
    return {
        "self_repair_per_100": round(rate, 4),
        "n_word_repeats": repeats,
        "n_repair_markers": markers,
    }


def m8_closing_sequence(turns):
    if len(turns) < 4:
        return {"closing_order_correct": None}
    closing_turns = [txt for _, txt in turns[-6:]]
    closing_text = " ".join(closing_turns).lower()
    stage_positions = {}
    for stage, pat in CLOSING_STAGES.items():
        m = pat.search(closing_text)
        if m:
            stage_positions[stage] = m.start()
    found = [s for s in CLOSING_ORDER if s in stage_positions]
    if len(found) < 2:
        return {"closing_order_correct": None, "closing_stages_found": found}
    sorted_found = sorted(found, key=lambda s: stage_positions[s])
    correct = sorted_found == found
    return {
        "closing_order_correct": int(correct),
        "closing_stages_found": found,
    }


def m9_sycophancy(turns):
    texts = [txt for _, txt in turns]
    if not texts:
        return {"sycophancy_rate_per_100w": None, "sycophancy_turn_open_frac": None}
    full_text = " ".join(texts).lower()
    total_words = max(len(full_text.split()), 1)
    phrase_count = sum(full_text.count(ph) for ph in SYCOPHANCY_PHRASES)
    rate_per_100 = phrase_count / total_words * 100

    # turn-opening sycophancy
    n_syco_open = 0
    for txt in texts:
        first_50 = txt[:80].lower()
        if any(ph in first_50 for ph in SYCOPHANCY_PHRASES):
            n_syco_open += 1
    return {
        "sycophancy_rate_per_100w": round(rate_per_100, 4),
        "sycophancy_turn_open_frac": round(n_syco_open / len(texts), 4),
    }


def m10_info_density_variance(turns):
    texts = [txt for _, txt in turns]
    if not texts:
        return {"info_density_cv": None}
    ratios = []
    for txt in texts:
        words = txt.lower().split()
        if words:
            content = [w for w in words if w not in STOP_WORDS]
            ratios.append(len(content) / len(words))
    if not ratios:
        return {"info_density_cv": None}
    mean_r = statistics.mean(ratios)
    if mean_r == 0:
        return {"info_density_cv": None}
    sd_r = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    return {"info_density_cv": round(sd_r / mean_r, 4)}


# ---------------------------------------------------------------------------
# Per-conversation evaluation
# ---------------------------------------------------------------------------

def evaluate_conversation(conv: dict) -> dict:
    turns = conv["turns"]
    metrics = {
        "condition":       conv.get("condition"),
        "architecture":    conv.get("architecture"),
        "prompt_level":    conv.get("prompt_level"),
        "conversation_no": conv.get("conversation_no"),
        "topic":           conv.get("topic"),
        "file":            conv.get("file"),
    }
    metrics.update(m1_turn_length(turns))
    metrics.update(m2_marker_rates(turns))
    metrics.update(m3_alignment_trajectory(turns))
    metrics.update(m4_turn_taking_economy(turns))
    metrics.update(m5_backchannel_standalone(turns))
    metrics.update(m6_oh_epistemic(turns))
    metrics.update(m7_self_repair(turns))
    metrics.update(m8_closing_sequence(turns))
    metrics.update(m9_sycophancy(turns))
    metrics.update(m10_info_density_variance(turns))
    # drop the raw list to keep JSON small (kept in m1 for internal use only)
    metrics.pop("words_per_turn_list", None)
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

NUMERIC_KEYS = [
    "mean_words_per_turn", "median_words_per_turn", "stdev_words_per_turn",
    "n_turns",
    "rate_oh", "rate_okay", "rate_uh_huh",
    "alignment_earlier", "alignment_later", "alignment_delta",
    "pct_short_turns",
    "backchannel_standalone_ratio", "n_backchannel_turns",
    "oh_epistemic_ratio", "n_oh",
    "self_repair_per_100",
    "closing_order_correct",
    "sycophancy_rate_per_100w", "sycophancy_turn_open_frac",
    "info_density_cv",
]


def summarise(records: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = {}
    for r in records:
        cond = r.get("condition", "unknown")
        by_condition.setdefault(cond, []).append(r)

    summary = {}
    for cond, recs in sorted(by_condition.items()):
        summary[cond] = {"n": len(recs)}
        for key in NUMERIC_KEYS:
            vals = [r[key] for r in recs if r.get(key) is not None]
            if vals:
                summary[cond][f"{key}_mean"] = round(statistics.mean(vals), 4)
                summary[cond][f"{key}_sd"] = round(
                    statistics.stdev(vals) if len(vals) > 1 else 0.0, 4
                )
    return summary


def summary_to_csv(summary: dict, out_path: pathlib.Path) -> None:
    conditions = sorted(summary.keys())
    if not conditions:
        return
    # collect all stat columns
    all_cols: list[str] = []
    for cond in conditions:
        for k in summary[cond]:
            if k not in all_cols and k != "n":
                all_cols.append(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n"] + all_cols)
        for cond in conditions:
            row = [cond, summary[cond].get("n", "")]
            for col in all_cols:
                row.append(summary[cond].get(col, ""))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate generated conversations")
    ap.add_argument("--data_dir", default="data/generated")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parent.parent
    data_dir = base / args.data_dir
    out_dir  = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading conversations from {data_dir} …")
    convs = load_conversations(data_dir)
    print(f"  {len(convs)} conversations found")
    if not convs:
        print("Nothing to evaluate.")
        sys.exit(1)

    print("Computing metrics …")
    records = [evaluate_conversation(c) for c in convs]

    per_conv_path = out_dir / "metrics_per_conversation.json"
    with per_conv_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved per-conversation results → {per_conv_path}")

    summary = summarise(records)
    summary_json_path = out_dir / "metrics_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved condition summary       → {summary_json_path}")

    summary_csv_path = out_dir / "metrics_summary.csv"
    summary_to_csv(summary, summary_csv_path)
    print(f"  Saved condition summary CSV   → {summary_csv_path}")

    print("\n=== Quick summary ===")
    for cond, stats in sorted(summary.items()):
        print(f"\n{cond}  (n={stats['n']})")
        print(f"  mean words/turn          : {stats.get('mean_words_per_turn_mean', '—')}")
        print(f"  % turns <5 words         : {stats.get('pct_short_turns_mean', '—')}")
        print(f"  oh/100w                  : {stats.get('rate_oh_mean', '—')}")
        print(f"  okay/100w                : {stats.get('rate_okay_mean', '—')}")
        print(f"  uh-huh/100w              : {stats.get('rate_uh_huh_mean', '—')}")
        print(f"  alignment earlier        : {stats.get('alignment_earlier_mean', '—')}")
        print(f"  alignment later          : {stats.get('alignment_later_mean', '—')}")
        print(f"  alignment delta          : {stats.get('alignment_delta_mean', '—')}")
        print(f"  backchannel standalone   : {stats.get('backchannel_standalone_ratio_mean', '—')}")
        print(f"  oh epistemic ratio       : {stats.get('oh_epistemic_ratio_mean', '—')}")
        print(f"  self-repair/100w         : {stats.get('self_repair_per_100_mean', '—')}")
        print(f"  closing order correct    : {stats.get('closing_order_correct_mean', '—')}")
        print(f"  sycophancy/100w          : {stats.get('sycophancy_rate_per_100w_mean', '—')}")
        print(f"  sycophancy turn-open frac: {stats.get('sycophancy_turn_open_frac_mean', '—')}")
        print(f"  info density CV          : {stats.get('info_density_cv_mean', '—')}")


if __name__ == "__main__":
    main()
