"""Social-conversation extension metrics: TSI/ADI, CAS, and CED.

This script implements the three proposed extension metrics from RESEARCH.md:

1. TSI/ADI: rule-based proxy labels for social-vs-assistant turn function.
2. CAS: Context Anchoring Score using real previous turns vs shuffled wrong contexts.
3. CED: Conversation Embedding Dispersion for full-conversation diversity/stereotypy.

The embedding backend is deliberately practical:
  - If sentence-transformers is installed, use it.
  - Otherwise, fall back to a local TF-IDF vectorizer implemented with numpy.

Usage:
    python analysis/social_metrics.py --n-sb 50
    python analysis/social_metrics.py --n-sb 50 --embedding-backend sentence-transformers --embedding-model all-MiniLM-L6-v2

Outputs:
    results/social_metrics/turn_labels.csv
    results/social_metrics/conversation_metrics.csv
    results/social_metrics/group_metrics.csv
    results/social_metrics/ced_by_condition.csv
    results/social_metrics/ced_by_topic_condition.csv
    results/social_metrics/ced_topic_matched_by_condition.csv
    results/social_metrics/ced_projection.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pathlib
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns  # noqa: E402
from analysis.swda import (  # noqa: E402
    conversation_no_of,
    iter_conversation_files,
    load_metadata,
    parse_conversation,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated"
OUT_DIR = ROOT / "results" / "social_metrics"

LABELS = [
    "A_task_help_advice",
    "B_social_continuation",
    "C_phatic_bonding",
    "D_grounding_backchannel",
    "E_personal_stance_experience",
    "G_repair_clarification",
    "H_assistant_explanation",
]

ASSISTANT_LABELS = {"A_task_help_advice", "H_assistant_explanation"}
SOCIAL_LABELS = {
    "B_social_continuation",
    "C_phatic_bonding",
    "E_personal_stance_experience",
    "G_repair_clarification",
}

WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class Conversation:
    source: str
    condition: str
    architecture: str
    prompt_level: str
    conv_id: str
    topic: str
    turns: list[tuple[str, str]]
    meta: dict = field(default_factory=dict)


class TfidfEmbedder:
    """Small dependency-free TF-IDF vectorizer used when sentence-transformers is absent."""

    def __init__(self, max_features: int = 12000, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: Iterable[str]) -> None:
        texts = list(texts)
        df: Counter[str] = Counter()
        tf_total: Counter[str] = Counter()
        for text in texts:
            toks = self._tokens(text)
            df.update(set(toks))
            tf_total.update(toks)

        n_docs = max(len(texts), 1)
        terms = [
            term for term, count in df.items()
            if count >= self.min_df
        ]
        terms.sort(key=lambda t: (df[t], tf_total[t], t), reverse=True)
        terms = terms[: self.max_features]
        self.vocab = {term: i for i, term in enumerate(terms)}
        self.idf = np.array(
            [math.log((1 + n_docs) / (1 + df[term])) + 1 for term in terms],
            dtype=float,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode().")
        mat = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for row, text in enumerate(texts):
            counts = Counter(tok for tok in self._tokens(text) if tok in self.vocab)
            if not counts:
                continue
            total = sum(counts.values())
            for tok, count in counts.items():
                mat[row, self.vocab[tok]] = count / total
        mat *= self.idf
        return _l2_normalize(mat)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [m.group(0).lower() for m in WORD_RE.finditer(text)]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: Iterable[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=float)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_embedder(backend: str, model_name: str) -> tuple[object, str]:
    if backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEmbedder(model_name), f"sentence-transformers:{model_name}"
        except ImportError:
            if backend == "sentence-transformers":
                raise
        except Exception as exc:
            if backend == "sentence-transformers":
                raise RuntimeError(f"Failed to load sentence-transformers model {model_name}: {exc}") from exc
    return TfidfEmbedder(), "tfidf-fallback"


def load_generated() -> list[Conversation]:
    conversations: list[Conversation] = []
    for path in sorted(glob.glob(str(GEN_ROOT / "*" / "*.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        turns = conversation_turns(rec)
        if not turns:
            continue
        condition = rec.get("condition", pathlib.Path(path).parent.name)
        conversations.append(
            Conversation(
                source="LLM",
                condition=condition,
                architecture=rec.get("architecture", condition.split("-")[0]),
                prompt_level=rec.get("prompt_level", condition.split("-")[-1]),
                conv_id=f"{condition}/{rec.get('conversation_no', pathlib.Path(path).stem)}",
                topic=rec.get("topic", ""),
                turns=turns,
                meta=rec,
            )
        )
    return conversations


def load_switchboard(n: int) -> list[Conversation]:
    conversations: list[Conversation] = []
    try:
        metadata = load_metadata()
    except FileNotFoundError:
        metadata = {}

    # Match generation/*.py's target_conversations(): skip files with no metadata row
    # *before* counting toward n, don't just take the literal first n files. Every
    # generation script builds its 50-conversation set this way; if even one early file
    # lacks metadata, taking a plain [:n] slice here silently samples a different, offset
    # set of conversation_no's than what the LLM data was actually generated from -- which
    # would starve the topic-matched CED comparison of any real overlap.
    picked = 0
    for fp in iter_conversation_files():
        if picked >= n:
            break
        conv_no = conversation_no_of(fp)
        if conv_no not in metadata:
            continue
        meta = metadata[conv_no]
        turns = parse_conversation(fp)
        if not turns:
            continue
        conversations.append(
            Conversation(
                source="SB",
                condition="SB",
                architecture="SB",
                prompt_level="SB",
                conv_id=f"SB/{conv_no}",
                topic=(meta.get("topic_description") or meta.get("prompt") or "").strip(),
                turns=turns,
                meta=meta,
            )
        )
        picked += 1
    return conversations


def clean_lower(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text)}


def has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pat, text, flags=re.I) for pat in patterns)


TASK_PATTERNS = [
    r"\byou should\b",
    r"\byou could\b",
    r"\byou need to\b",
    r"\btry to\b",
    r"\btry\b.+\b(?:doing|using|making|calling|asking)\b",
    r"\bi recommend\b",
    r"\bmy recommendation\b",
    r"\bthe best (?:way|option|approach|solution)\b",
    r"\bhere are\b",
    r"\bsteps?\b",
    r"\bsolution\b",
    r"\bconsider\b",
    r"\bmake sure\b",
]

ASSISTANT_EXPLANATION_PATTERNS = [
    r"\bthere are (?:several|many|a few|three|two)\b",
    r"\bit is important to\b",
    r"\bit's important to\b",
    r"\boverall\b",
    r"\bin conclusion\b",
    r"\bto summarize\b",
    r"\bthis can be (?:understood|explained)\b",
    r"\bmany people\b",
    r"\bin general\b",
    r"\bgenerally speaking\b",
    r"\b(?:first|second|third|finally),\b",
    r"^\s*\d+[\).\s]",
    r"\bas an ai\b",
]

PHATIC_PATTERNS = [
    r"\boh wow\b",
    r"\bwow\b",
    r"\bthat's funny\b",
    r"\bthat is funny\b",
    r"\bno kidding\b",
    r"\bsame here\b",
    r"\bi know\b",
    r"\bexactly\b",
    r"\bright\b",
    r"\byeah\b",
    r"\bhaha\b",
    r"\blaugh",
]

BACKCHANNEL_PATTERNS = [
    r"^(?:uh-?huh|mm-?hm|mhm|yeah|right|okay|ok|i see|sure|exactly|oh|wow)[.!?,\s]*$",
    r"^(?:yeah|right|okay|ok|uh-?huh|mm-?hm),?\s+(?:yeah|right|okay|ok|uh-?huh)[.!?,\s]*$",
]

PERSONAL_PATTERNS = [
    r"\bi (?:think|feel|guess|remember|like|love|hate|prefer|used to|had|have|was|am)\b",
    r"\bi've\b",
    r"\bi'd\b",
    r"\bmy (?:mom|dad|mother|father|brother|sister|family|friend|friends|wife|husband|kids|job|school|house|car)\b",
    r"\bfor me\b",
    r"\bwhen i\b",
    r"\bwhere i\b",
    r"\bwe (?:had|have|used to|were|are|went|did)\b",
]

REPAIR_PATTERNS = [
    r"\bwhat do you mean\b",
    r"\bwhat d'you mean\b",
    r"\bwait\b",
    r"\bsorry\b",
    r"\bi mean\b",
    r"\byou mean\b",
    r"\bwhich one\b",
    r"\bsay that again\b",
    r"\bpardon\b",
    r"\bclarif",
    r"\bno,? i meant\b",
]


def label_turn(text: str, prev_text: str | None = None) -> dict:
    """Rule-based proxy labels for the TSI/ADI coding scheme."""
    t = clean_lower(text)
    n_words = len(t.split())
    labels = {label: 0 for label in LABELS}

    if has_pattern(t, TASK_PATTERNS):
        labels["A_task_help_advice"] = 1
    if has_pattern(t, ASSISTANT_EXPLANATION_PATTERNS) or (
        n_words >= 45 and has_pattern(t, [r"\b(?:because|therefore|however|for example)\b"])
    ):
        labels["H_assistant_explanation"] = 1
    if has_pattern(t, REPAIR_PATTERNS):
        labels["G_repair_clarification"] = 1
    if has_pattern(t, BACKCHANNEL_PATTERNS) or (n_words <= 4 and has_pattern(t, [r"\b(?:yeah|right|okay|ok|uh-?huh|i see)\b"])):
        labels["D_grounding_backchannel"] = 1
    if has_pattern(t, PERSONAL_PATTERNS):
        labels["E_personal_stance_experience"] = 1
    if has_pattern(t, PHATIC_PATTERNS):
        labels["C_phatic_bonding"] = 1

    is_question = "?" in text
    reciprocal_question = has_pattern(
        t,
        [
            r"\bwhat about you\b",
            r"\bhow about you\b",
            r"\bdid you\b",
            r"\bdo you\b",
            r"\bhave you\b",
            r"\bwhat kind\b",
            r"\bwhere do you\b",
            r"\bhow do you\b",
        ],
    )
    if (is_question or reciprocal_question or labels["C_phatic_bonding"] or labels["E_personal_stance_experience"]) and not labels["H_assistant_explanation"]:
        labels["B_social_continuation"] = 1

    labels["topic_code"] = topic_code(prev_text, text)
    return labels


def topic_code(prev_text: str | None, text: str) -> str:
    if not prev_text:
        return "F1_maintain"
    prev_tokens = token_set(prev_text)
    cur_tokens = token_set(text)
    if not cur_tokens:
        return "F4_loop"
    if clean_lower(prev_text) == clean_lower(text):
        return "F4_loop"
    overlap = len(prev_tokens & cur_tokens) / max(len(cur_tokens), 1)
    if overlap >= 0.22:
        return "F1_maintain"
    if overlap >= 0.07:
        return "F2_smooth_shift"
    return "F3_abrupt_drift"


def build_turn_rows(conversations: list[Conversation]) -> list[dict]:
    rows: list[dict] = []
    for conv in conversations:
        for idx, (speaker, text) in enumerate(conv.turns):
            prev_text = conv.turns[idx - 1][1] if idx > 0 else None
            labels = label_turn(text, prev_text)
            row = {
                "source": conv.source,
                "condition": conv.condition,
                "architecture": conv.architecture,
                "prompt_level": conv.prompt_level,
                "conversation_id": conv.conv_id,
                "topic": conv.topic,
                "turn_index": idx,
                "speaker": speaker,
                "text": text,
            }
            row.update(labels)
            row["assistant_like"] = int(any(row[label] for label in ASSISTANT_LABELS))
            row["social_like"] = int(any(row[label] for label in SOCIAL_LABELS))
            rows.append(row)
    return rows


def apply_cas(
    rows: list[dict],
    embedder: object,
    *,
    k_wrong: int,
    seed: int,
    same_topic_negatives: bool,
) -> None:
    eligible = [r for r in rows if int(r["turn_index"]) > 0]
    if not eligible:
        return

    prev_by_key: dict[tuple[str, int], str] = {}
    topic_by_conv: dict[str, str] = {}
    for r in rows:
        conv_id = r["conversation_id"]
        idx = int(r["turn_index"])
        topic_by_conv[conv_id] = r["topic"]
        prev_by_key[(conv_id, idx + 1)] = r["text"]

    contexts = []
    for r in eligible:
        r["real_previous_text"] = prev_by_key.get((r["conversation_id"], int(r["turn_index"])), "")
        contexts.append(
            {
                "conversation_id": r["conversation_id"],
                "topic": r["topic"],
                "text": r["real_previous_text"],
            }
        )

    all_texts = [r["real_previous_text"] for r in eligible] + [r["text"] for r in eligible]
    embedder.fit(all_texts)
    real_vecs = embedder.encode([r["real_previous_text"] for r in eligible])
    response_vecs = embedder.encode([r["text"] for r in eligible])

    rng = random.Random(seed)
    global_contexts = contexts
    contexts_by_topic: dict[str, list[dict]] = defaultdict(list)
    for ctx in contexts:
        contexts_by_topic[ctx["topic"]].append(ctx)

    for i, row in enumerate(eligible):
        real_score = cosine(real_vecs[i], response_vecs[i])
        pool = global_contexts
        if same_topic_negatives and row["topic"]:
            same_topic = [
                ctx for ctx in contexts_by_topic.get(row["topic"], [])
                if ctx["conversation_id"] != row["conversation_id"]
            ]
            if len(same_topic) >= max(2, min(k_wrong, 5)):
                pool = same_topic
        pool = [ctx for ctx in pool if ctx["conversation_id"] != row["conversation_id"] and ctx["text"]]
        if not pool:
            row["cas_real_similarity"] = real_score
            row["cas_wrong_similarity"] = ""
            row["cas"] = ""
            row["cas_negatives"] = 0
            continue
        sample = rng.sample(pool, k=min(k_wrong, len(pool)))
        wrong_vecs = embedder.encode([ctx["text"] for ctx in sample])
        wrong_scores = [cosine(vec, response_vecs[i]) for vec in wrong_vecs]
        wrong_mean = statistics.mean(wrong_scores) if wrong_scores else 0.0
        row["cas_real_similarity"] = real_score
        row["cas_wrong_similarity"] = wrong_mean
        row["cas"] = real_score - wrong_mean
        row["cas_negatives"] = len(sample)


def mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.mean(vals) if vals else float("nan")


def conversation_metrics(rows: list[dict]) -> list[dict]:
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_conv[row["conversation_id"]].append(row)

    out: list[dict] = []
    for conv_id, conv_rows in sorted(by_conv.items()):
        n = len(conv_rows)
        head = conv_rows[0]
        rec = {
            "source": head["source"],
            "condition": head["condition"],
            "architecture": head["architecture"],
            "prompt_level": head["prompt_level"],
            "conversation_id": conv_id,
            "topic": head["topic"],
            "n_turns": n,
        }
        for label in LABELS:
            rec[f"rate_{label}"] = sum(int(r[label]) for r in conv_rows) / max(n, 1)
        rec["ADI"] = min(1.0, rec["rate_A_task_help_advice"] + rec["rate_H_assistant_explanation"])
        rec["social_rate"] = (
            rec["rate_B_social_continuation"]
            + rec["rate_C_phatic_bonding"]
            + rec["rate_E_personal_stance_experience"]
            + rec["rate_G_repair_clarification"]
            + 0.5 * rec["rate_D_grounding_backchannel"]
        )
        rec["topic_health"] = topic_health(conv_rows)
        rec["reciprocity"] = reciprocity(conv_rows)
        rec["TSI_proxy"] = tsi_proxy(rec)
        cas_values = [float(r["cas"]) for r in conv_rows if r.get("cas") not in {"", None}]
        rec["CAS"] = mean(cas_values)
        rec["cas_turns"] = len(cas_values)
        out.append(rec)
    return out


def topic_health(rows: list[dict]) -> float:
    counts = Counter(r["topic_code"] for r in rows)
    n = max(len(rows), 1)
    return (
        (counts["F1_maintain"] + counts["F2_smooth_shift"])
        - counts["F3_abrupt_drift"]
        - counts["F4_loop"]
    ) / n


def reciprocity(rows: list[dict]) -> float:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    if len(by_speaker) < 2:
        return float("nan")
    rates = []
    for speaker_rows in by_speaker.values():
        rates.append(sum(int(r["social_like"]) for r in speaker_rows) / max(len(speaker_rows), 1))
    if len(rates) < 2:
        return float("nan")
    return max(0.0, 1.0 - (max(rates) - min(rates)))


def tsi_proxy(rec: dict) -> float:
    social_norm = min(1.0, rec["social_rate"])
    topic_norm = max(0.0, min(1.0, (rec["topic_health"] + 1.0) / 2.0))
    reciprocity_norm = 0.5 if math.isnan(rec["reciprocity"]) else rec["reciprocity"]
    return 100.0 * (
        0.35 * social_norm
        + 0.20 * topic_norm
        + 0.20 * reciprocity_norm
        + 0.25 * (1.0 - rec["ADI"])
    )


def group_metrics(conv_rows: list[dict]) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in conv_rows:
        by_group[row["condition"]].append(row)

    out = []
    numeric_keys = [
        "ADI",
        "social_rate",
        "topic_health",
        "reciprocity",
        "TSI_proxy",
        "CAS",
    ]
    for condition, rows in sorted(by_group.items()):
        head = rows[0]
        rec = {
            "condition": condition,
            "source": head["source"],
            "architecture": head["architecture"],
            "prompt_level": head["prompt_level"],
            "n_conversations": len(rows),
        }
        for key in numeric_keys:
            rec[f"mean_{key}"] = mean([float(r[key]) for r in rows if r.get(key) not in {"", None}])
        out.append(rec)
    return out


def conversation_text(conv: Conversation, max_chars: int) -> str:
    text = "\n".join(f"{speaker}: {turn}" for speaker, turn in conv.turns)
    return text[:max_chars] if max_chars > 0 else text


def embed_conversations(
    conversations: list[Conversation],
    embedder: object,
    *,
    text_mode: str,
    max_chars: int,
) -> np.ndarray:
    """Embed each conversation to a single vector.

    ``concat`` joins the whole conversation into one blob and embeds that (the original
    approach) — sentence encoders only attend to their first ~256 tokens, so this silently
    reduces a long conversation to its opening turns. ``turns`` (the default) embeds every
    turn independently and mean-pools per conversation, so the full conversation counts.
    """
    if text_mode == "concat":
        texts = [conversation_text(conv, max_chars) for conv in conversations]
        embedder.fit(texts)
        return _l2_normalize(embedder.encode(texts))

    turn_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for conv in conversations:
        start = len(turn_texts)
        turn_texts.extend(text for _, text in conv.turns)
        spans.append((start, len(turn_texts)))

    embedder.fit(turn_texts)
    turn_vecs = embedder.encode(turn_texts)
    dim = turn_vecs.shape[1] if turn_vecs.ndim == 2 else 0
    conv_vecs = np.zeros((len(conversations), dim), dtype=float)
    for i, (start, end) in enumerate(spans):
        if end > start:
            conv_vecs[i] = turn_vecs[start:end].mean(axis=0)
    return _l2_normalize(conv_vecs)


def nearest_neighbor_distances(vecs: np.ndarray) -> list[float]:
    """Cosine distance from each row to its closest other row (vecs must be L2-normalized)."""
    n = len(vecs)
    if n < 2:
        return []
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    return (1.0 - sims.max(axis=1)).tolist()


def compute_ced(
    conversations: list[Conversation],
    embedder: object,
    *,
    text_mode: str,
    max_chars: int,
) -> tuple[list[dict], list[dict], np.ndarray]:
    vecs = embed_conversations(conversations, embedder, text_mode=text_mode, max_chars=max_chars)

    by_condition: dict[str, list[int]] = defaultdict(list)
    by_topic_condition: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, conv in enumerate(conversations):
        by_condition[conv.condition].append(i)
        if conv.topic:
            by_topic_condition[(conv.topic, conv.condition)].append(i)

    condition_rows = [
        ced_for_indices(conversations, vecs, indices, {"condition": condition})
        for condition, indices in sorted(by_condition.items())
    ]
    topic_rows = [
        ced_for_indices(conversations, vecs, indices, {"topic": topic, "condition": condition})
        for (topic, condition), indices in sorted(by_topic_condition.items())
        if len(indices) >= 2
    ]
    return condition_rows, topic_rows, vecs


def ced_for_indices(
    conversations: list[Conversation],
    vecs: np.ndarray,
    indices: list[int],
    base: dict,
) -> dict:
    selected = vecs[indices]
    centroid = np.mean(selected, axis=0)
    distances = [1.0 - cosine(vec, centroid) for vec in selected]
    nn_distances = nearest_neighbor_distances(selected)
    head = conversations[indices[0]]
    rec = {
        "source": head.source,
        "architecture": head.architecture if "condition" in base else "",
        "prompt_level": head.prompt_level if "condition" in base else "",
        "n_conversations": len(indices),
        "CED": mean(distances),
        "min_distance": min(distances) if distances else float("nan"),
        "max_distance": max(distances) if distances else float("nan"),
        # nearest-neighbor distance: low mean/min => conversations are near-duplicates,
        # which is exactly the LLM-clumping failure mode CED is meant to catch.
        "mean_nn_distance": mean(nn_distances),
        "min_nn_distance": min(nn_distances) if nn_distances else float("nan"),
    }
    rec.update(base)
    return rec


def matched_topic_summary(topic_rows: list[dict]) -> list[dict]:
    """Average CED/NN-distance over topics Switchboard also has, so dispersion is compared
    within the same topics rather than confounded by each condition's own topic mix."""
    sb_topics = {row["topic"] for row in topic_rows if row["condition"] == "SB"}
    if not sb_topics:
        return []

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in topic_rows:
        if row["topic"] in sb_topics:
            by_condition[row["condition"]].append(row)

    out = []
    for condition, rows in sorted(by_condition.items()):
        out.append(
            {
                "condition": condition,
                "n_topics": len(rows),
                "mean_CED_topic_matched": mean([r["CED"] for r in rows]),
                "mean_nn_distance_topic_matched": mean([r["mean_nn_distance"] for r in rows]),
            }
        )
    return out


def plot_ced_projection(
    conversations: list[Conversation],
    vecs: np.ndarray,
    out_path: pathlib.Path,
    *,
    method: str,
) -> pathlib.Path | None:
    """2-D scatter of every conversation, colored by condition (PCA or UMAP)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping CED projection plot.")
        return None

    reducer = None
    if method == "umap":
        try:
            from umap import UMAP

            reducer = UMAP(n_components=2, random_state=7)
        except ImportError:
            print("umap-learn not installed; falling back to PCA for the CED projection plot.")
            method = "pca"
    if reducer is None:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("scikit-learn not installed; skipping CED projection plot.")
            return None
        reducer = PCA(n_components=2, random_state=7)

    coords = reducer.fit_transform(vecs)
    conditions = [conv.condition for conv in conversations]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, condition in enumerate(sorted(set(conditions))):
        idx = [j for j, c in enumerate(conditions) if c == condition]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            label=condition, s=18, alpha=0.75, color=cmap(i % 10),
        )
    ax.set_title(f"Conversation embedding dispersion ({method.upper()})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(groups: list[dict], ceds: list[dict], embedder_name: str) -> None:
    print("=" * 86)
    print("SOCIAL CONVERSATION EXTENSION METRICS")
    print(f"embedding backend: {embedder_name}")
    print("=" * 86)
    print(f"{'condition':12} {'n':>4} {'ADI':>7} {'TSI':>7} {'CAS':>8} {'CED':>8} {'NN-dist':>8}")
    print("-" * 86)
    ced_by_condition = {row["condition"]: row for row in ceds}
    for row in groups:
        condition = row["condition"]
        ced_row = ced_by_condition.get(condition, {})
        print(
            f"{condition:12} {int(row['n_conversations']):4d} "
            f"{float(row['mean_ADI']):7.3f} "
            f"{float(row['mean_TSI_proxy']):7.1f} "
            f"{float(row['mean_CAS']):8.3f} "
            f"{float(ced_row.get('CED', float('nan'))):8.3f} "
            f"{float(ced_row.get('mean_nn_distance', float('nan'))):8.3f}"
        )
    print("-" * 86)
    print("ADI: higher = more assistant-like advice/explanation turns")
    print("TSI: higher = more social-conversation behavior (rule-based proxy)")
    print("CAS: higher = real previous turn beats shuffled wrong contexts more strongly")
    print("CED: higher = full conversations are more dispersed/diverse within condition")
    print("NN-dist: higher = conversations are less likely to be near-duplicates of each other")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sb", type=int, default=50, help="number of Switchboard conversations")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory of <condition>/*.json to read instead of data/generated "
        "(e.g. data/generated_v2, before that's promoted to data/generated)",
    )
    parser.add_argument("--k-wrong", type=int, default=20, help="wrong contexts sampled per CAS turn")
    parser.add_argument("--seed", type=int, default=7, help="random seed for CAS negatives")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence-transformers", "tfidf"],
        default="tfidf",
        help="embedding backend; tfidf is dependency-free; sentence-transformers gives stronger semantic embeddings",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name when that backend is available",
    )
    parser.add_argument(
        "--global-negatives",
        action="store_true",
        help="sample CAS wrong contexts globally instead of preferring same-topic negatives",
    )
    parser.add_argument(
        "--conversation-max-chars",
        type=int,
        default=12000,
        help="truncate full conversation text before CED embedding when --ced-text-mode=concat; 0 means no truncation",
    )
    parser.add_argument(
        "--ced-embedding-backend",
        choices=["auto", "sentence-transformers", "tfidf"],
        default="auto",
        help="embedding backend for CED specifically; auto prefers sentence-transformers "
        "(captures style) and falls back to tfidf (captures topic-word overlap) if it isn't installed",
    )
    parser.add_argument(
        "--ced-text-mode",
        choices=["turns", "concat"],
        default="turns",
        help="turns: embed each turn and mean-pool per conversation (full conversation counts); "
        "concat: embed the whole joined transcript (sentence encoders effectively only see the opening)",
    )
    parser.add_argument(
        "--projection",
        choices=["pca", "umap", "none"],
        default="pca",
        help="2-D projection of conversation embeddings colored by condition; requires scikit-learn (pca) or umap-learn (umap)",
    )
    args = parser.parse_args()

    if args.data_dir:
        global GEN_ROOT
        GEN_ROOT = pathlib.Path(args.data_dir)

    conversations = load_generated() + load_switchboard(args.n_sb)
    if not conversations:
        print("No conversations found. Expected data/generated or data/switchboard/swda.")
        return

    turn_rows = build_turn_rows(conversations)
    cas_embedder, cas_embedder_name = build_embedder(args.embedding_backend, args.embedding_model)
    apply_cas(
        turn_rows,
        cas_embedder,
        k_wrong=args.k_wrong,
        seed=args.seed,
        same_topic_negatives=not args.global_negatives,
    )

    conv_rows = conversation_metrics(turn_rows)
    group_rows = group_metrics(conv_rows)

    ced_embedder, ced_embedder_name = build_embedder(args.ced_embedding_backend, args.embedding_model)
    ced_condition_rows, ced_topic_rows, ced_vecs = compute_ced(
        conversations,
        ced_embedder,
        text_mode=args.ced_text_mode,
        max_chars=args.conversation_max_chars,
    )
    ced_matched_rows = matched_topic_summary(ced_topic_rows)

    write_csv(OUT_DIR / "turn_labels.csv", turn_rows)
    write_csv(OUT_DIR / "conversation_metrics.csv", conv_rows)
    write_csv(OUT_DIR / "group_metrics.csv", group_rows)
    write_csv(OUT_DIR / "ced_by_condition.csv", ced_condition_rows)
    write_csv(OUT_DIR / "ced_by_topic_condition.csv", ced_topic_rows)
    write_csv(OUT_DIR / "ced_topic_matched_by_condition.csv", ced_matched_rows)

    plot_path = None
    if args.projection != "none":
        plot_path = plot_ced_projection(
            conversations, ced_vecs, OUT_DIR / "ced_projection.png", method=args.projection
        )

    backend_name = cas_embedder_name
    if ced_embedder_name != cas_embedder_name:
        backend_name = f"CAS={cas_embedder_name}; CED={ced_embedder_name} ({args.ced_text_mode})"
    print_summary(group_rows, ced_condition_rows, backend_name)
    if ced_matched_rows:
        print("\ntopic-matched CED (averaged over topics Switchboard also has):")
        for row in ced_matched_rows:
            print(
                f"  {row['condition']:12} n_topics={row['n_topics']:3d} "
                f"CED={row['mean_CED_topic_matched']:.3f} "
                f"NN-dist={row['mean_nn_distance_topic_matched']:.3f}"
            )
    print(f"\nWrote CSV outputs to {OUT_DIR.relative_to(ROOT)}")
    if plot_path:
        print(f"Wrote CED projection plot to {plot_path.relative_to(ROOT)}")
    print("\nNote: TSI/ADI is a rule-based proxy until the team creates human labels. "
          "CAS and CED are automatic embedding metrics.")


if __name__ == "__main__":
    main()
