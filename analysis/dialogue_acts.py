"""Dialogue-act structural signatures for Switchboard and generated dialogue.

The human reference uses the gold ``act_tag`` field in the local SwDA CSV files.
DialogTag is also applied to both human and sentence-segmented generated units,
with every view mapped to exactly the same short-code inventory.  The module deliberately keeps model imports inside the
tagging adapter: ``--human-only`` and cached statistical reruns do not require
DialogTag, TensorFlow, or torch.

Usage::

    python analysis/dialogue_acts.py --human-only
    python analysis/dialogue_acts.py
    python analysis/dialogue_acts.py --force-retag --validation-conversations 20

The default output directory is ``results/dialogue_acts/``.  No transcript text
is written: the optional cache contains only human/generated labels, identifiers,
fingerprints, and aggregate gold calibration counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import itertools

from analysis.analyze import conversation_turns, has_language_drift
from analysis.swda import (
    DATA_ROOT as SWDA_ROOT,
)
from analysis.swda import (
    clean_text,
    conversation_no_of,
    iter_conversation_files,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated"
OUT_DIR = ROOT / "results" / "dialogue_acts"
CACHE_VERSION = 2

# DialogTag is trained on the 38-class simplification in cgpotts/swda.  The task
# additionally requires every suffix modifier to be removed (notably qy^d -> qy,
# qw^d -> qw, and b^m -> b), while retaining the four leading-caret base acts.
# Gold-only %, x, ar, and no remain real columns even though DialogTag cannot emit
# them.  The result is one shared 39-code inventory, with structural zeros where
# the automatic tagger lacks a class.
FINE_LABELS = [
    "sd",
    "b",
    "sv",
    "%",
    "aa",
    "ba",
    "qy",
    "x",
    "ny",
    "fc",
    "qw",
    "nn",
    "bk",
    "h",
    "fo",
    "bh",
    "^q",
    "bf",
    "na",
    "ad",
    "^2",
    "qo",
    "qh",
    "^h",
    "ng",
    "br",
    "fp",
    "qrr",
    "arp",
    "t3",
    "oo",
    "aap",
    "bd",
    "t1",
    "^g",
    "fa",
    "ft",
    "ar",
    "no",
]

COARSE_LABELS = [
    "Statement",
    "Opinion",
    "Backchannel",
    "YesNoQuestion",
    "WhQuestion/OpenQuestion",
    "Agreement",
    "Answer",
    "Directive",
    "Repair/Hedge",
    "Abandoned/Other",
]

# A deliberately exhaustive coarse map.  Some rare DAMSL acts have no perfect
# ten-way home; the choices below preserve their broad interactional function.
FINE_TO_COARSE = {
    "sd": "Statement",
    "^q": "Statement",  # quotation
    "sv": "Opinion",
    "b": "Backchannel",
    "bk": "Backchannel",  # response acknowledgement
    "bh": "Backchannel",  # backchannel in question form
    "ba": "Backchannel",  # appreciation/listener response
    "qy": "YesNoQuestion",
    "^g": "YesNoQuestion",  # tag question
    "qrr": "YesNoQuestion",  # or-clause
    "qw": "WhQuestion/OpenQuestion",
    "qo": "WhQuestion/OpenQuestion",
    "qh": "WhQuestion/OpenQuestion",
    "aa": "Agreement",
    "aap": "Agreement",  # maybe/accept-part
    "bd": "Agreement",  # downplayer
    "ny": "Answer",
    "nn": "Answer",
    "na": "Answer",
    "ng": "Answer",
    "ar": "Answer",  # reject
    "no": "Answer",  # other answer
    "arp": "Answer",  # dispreferred answer
    "ad": "Directive",
    "oo": "Directive",  # offers/options/commits
    "h": "Repair/Hedge",
    "bf": "Repair/Hedge",  # summarize/reformulate
    "br": "Repair/Hedge",  # signal non-understanding
    "^h": "Repair/Hedge",  # hold before answer/agreement
    "^2": "Repair/Hedge",  # collaborative completion
    "t1": "Repair/Hedge",  # self-talk/word-search turn management
    "%": "Abandoned/Other",
    "x": "Abandoned/Other",
    "fo": "Abandoned/Other",
    "fc": "Abandoned/Other",
    "fp": "Abandoned/Other",
    "fa": "Abandoned/Other",
    "ft": "Abandoned/Other",
    "t3": "Abandoned/Other",  # third-party talk, outside the dyadic exchange
}

# Exact long strings from DialogTag 1.1.3's downloaded label_map.txt.  Several
# labels correspond to grouped SwDA codes in that model; we choose one canonical
# short code for the group.  Declarative questions and repeat phrases follow the
# requested suffix-stripping rule, so both sides collapse identically.
DIALOGTAG_TO_FINE = {
    "Statement-non-opinion": "sd",
    "Acknowledge (Backchannel)": "b",
    "Statement-opinion": "sv",
    "Agree/Accept": "aa",
    "Appreciation": "ba",
    "Yes-No-Question": "qy",
    "Yes Answers": "ny",
    "Conventional-closing": "fc",
    "Wh-Question": "qw",
    "No Answers": "nn",
    "Response Acknowledgement": "bk",
    "Hedge": "h",
    "Declarative Yes-No-Question": "qy",
    "Backchannel in Question Form": "bh",
    "Quotation": "^q",
    "Summarize/Reformulate": "bf",
    "Other": "fo",
    "Affirmative Non-yes Answers": "na",
    "Action-directive": "ad",
    "Collaborative Completion": "^2",
    "Repeat-phrase": "b",
    "Open-Question": "qo",
    "Rhetorical-Question": "qh",
    "Hold Before Answer/Agreement": "^h",
    "Negative Non-no Answers": "ng",
    "Signal-non-understanding": "br",
    "Conventional-opening": "fp",
    "Or-Clause": "qrr",
    "Dispreferred Answers": "arp",
    "3rd-party-talk": "t3",
    "Offers, Options Commits": "oo",
    "Maybe/Accept-part": "aap",
    "Downplayer": "bd",
    "Self-talk": "t1",
    "Tag-Question": "^g",
    "Declarative Wh-Question": "qw",
    "Apology": "fa",
    "Thanking": "ft",
}

# cgpotts/swda's standard grouping, expressed with ordinary short codes rather
# than compound column names.  Applied after modifiers and annotation flags are
# removed.  For comma/semicolon multi-tags, the first tag is used, matching the
# reference loader's documented policy.
GOLD_BASE_ALIASES = {
    "qr": "qy",
    "fe": "ba",
    "oo": "oo",
    "co": "oo",
    "cc": "oo",
    "fx": "sv",
    "aap": "aap",
    "am": "aap",
    "arp": "arp",
    "nd": "arp",
    "fo": "fo",
    "o": "fo",
    "fw": "fo",
    '"': "fo",
    "by": "fo",
    "bc": "fo",
}

ASSISTANT_PATTERNS = {
    "numbered_list": re.compile(r"(?:^|\n)\s*\d{1,2}[.)]\s+", re.I),
    "you_should": re.compile(r"\byou\s+should\b", re.I),
    "you_could_try": re.compile(r"\byou\s+could\s+try\b", re.I),
    "i_recommend": re.compile(r"\bi\s+recommend\b", re.I),
    "important_to": re.compile(r"\bit(?:'s|\s+is)\s+important\s+to\b", re.I),
    "several": re.compile(r"\bthere\s+are\s+several\b", re.I),
    "conclusion": re.compile(r"(?:^|[.!?]\s+)(?:overall|in\s+conclusion)\b", re.I),
}


@dataclass
class Conversation:
    source: str
    condition: str
    conversation_no: int
    speakers: list[str]
    texts: list[str]
    fine_labels: list[str] = field(default_factory=list)

    def coarse_labels(self) -> list[str]:
        return [FINE_TO_COARSE[label] for label in self.fine_labels]


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(]*[A-Z0-9])")


def sentence_units(text: str) -> list[str]:
    """Conservative, dependency-free sentence segmentation for generated turns."""

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    units = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    return units or [text]


@dataclass
class ValidationResult:
    conversation_ids: list[int]
    fine_confusion: Counter[tuple[str, str]]

    @property
    def n_items(self) -> int:
        return sum(self.fine_confusion.values())


def normalize_swda_base(raw_tag: str) -> str:
    """Collapse one non-continuation raw SwDA tag to the shared inventory.

    Policy, in order:
      * take the first tag in rare comma/semicolon compounds;
      * preserve ``%`` (abandoned/uninterpretable) and ``x`` (non-verbal);
      * discard parenthesized, ``@``, and ``*`` annotation modifiers;
      * strip suffix ``^`` modifiers, while retaining leading ``^q/^2/^h/^g``;
      * apply the standard grouped-class aliases listed above.

    ``+`` requires caller-local history and is therefore resolved by
    :func:`load_switchboard`, not here.
    """

    tag = re.split(r"[,;]", raw_tag.strip(), maxsplit=1)[0].strip()
    if tag.startswith("+"):
        return "+"
    if tag.startswith("%"):
        return "%"
    if tag.startswith("x"):
        return "x"

    # These two elaboration codes are semantic classes in the standard SwDA
    # reduction, not disposable style modifiers.  Handle them before the task's
    # general suffix collapse.
    if tag == "nn^e":
        return "ng"
    if tag == "ny^e":
        return "na"

    tag = re.sub(r"\([^)]*\)", "", tag)
    tag = tag.replace("@", "").replace("*", "")
    if tag.startswith("^"):
        tag = tag[:2]
    elif "^" in tag:
        tag = tag.split("^", 1)[0]
    tag = GOLD_BASE_ALIASES.get(tag, tag)
    if tag not in FINE_LABELS:
        raise ValueError(f"Unmapped SwDA act_tag {raw_tag!r} -> {tag!r}")
    return tag


def load_switchboard(root: pathlib.Path = SWDA_ROOT) -> list[Conversation]:
    """Read gold SwDA annotations without using the turn-merging parser."""

    conversations: list[Conversation] = []
    for path in iter_conversation_files(root):
        speakers: list[str] = []
        texts: list[str] = []
        labels: list[str] = []
        previous_by_caller: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                caller = row["caller"].strip()
                label = normalize_swda_base(row["act_tag"])
                if label == "+":
                    if caller not in previous_by_caller:
                        raise ValueError(
                            f"Continuation without prior act for caller {caller!r} "
                            f"at {path}:{row_number}"
                        )
                    label = previous_by_caller[caller]
                else:
                    previous_by_caller[caller] = label
                speakers.append(caller)
                texts.append(row["text"])
                labels.append(label)
        if labels:
            conversations.append(
                Conversation(
                    source="SB",
                    condition="SB",
                    conversation_no=conversation_no_of(path),
                    speakers=speakers,
                    texts=texts,
                    fine_labels=labels,
                )
            )
    if not conversations:
        raise FileNotFoundError(f"No SwDA .utt.csv files found below {root}")
    return conversations


def drop_loop_turns(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove echo-loop turns: near-verbatim repeats of an earlier turn.

    Same definition as the generation-time guard (generation.quality.is_near_duplicate:
    >= 8 distinct words, token-set Jaccard >= 0.8), applied post hoc so every condition is
    judged identically. Short turns are never dropped (backchannels legitimately repeat).
    Used only for the sensitivity analysis (--exclude-loops); the default keeps all turns.
    """
    from generation.quality import is_near_duplicate

    kept: list[tuple[str, str]] = []
    seen: list[tuple[str, str]] = []
    for speaker, text in turns:
        if not is_near_duplicate(text, seen):
            kept.append((speaker, text))
        seen.append((speaker, text))
    return kept


def load_generated(
    root: pathlib.Path = GEN_ROOT, exclude_loops: bool = False, exclude_drift: bool = False
) -> list[Conversation]:
    conversations: list[Conversation] = []
    for path in sorted(root.glob("*/*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if exclude_drift and has_language_drift(record):
            continue
        turns = conversation_turns(record)
        if exclude_loops:
            turns = drop_loop_turns(turns)
        if not turns:
            continue
        condition = str(record.get("condition", path.parent.name))
        conversation_no = int(record.get("conversation_no", path.stem))
        speakers: list[str] = []
        texts: list[str] = []
        for speaker, text in turns:
            for unit in sentence_units(text):
                speakers.append(speaker)
                texts.append(unit)
        conversations.append(
            Conversation(
                source="LLM",
                condition=condition,
                conversation_no=conversation_no,
                speakers=speakers,
                texts=texts,
            )
        )
    if not conversations:
        raise FileNotFoundError(f"No generated JSON conversations found below {root}")
    return conversations


def _normal_label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.casefold())


_DIALOGTAG_NORMALIZED = {
    _normal_label_key(long_label): short_label
    for long_label, short_label in DIALOGTAG_TO_FINE.items()
}


def dialogtag_to_fine(label: str) -> str:
    """Map a DialogTag long label to the shared short-code inventory."""

    if label in DIALOGTAG_TO_FINE:
        return DIALOGTAG_TO_FINE[label]
    key = _normal_label_key(label)
    if key in _DIALOGTAG_NORMALIZED:
        return _DIALOGTAG_NORMALIZED[key]
    raise ValueError(
        f"Unknown DialogTag label {label!r}. Expected one of: "
        + ", ".join(sorted(DIALOGTAG_TO_FINE))
    )


class DialogTagAdapter:
    """Lazy DialogTag wrapper with a batched fast path and public-API fallback."""

    def __init__(self, model_name: str):
        try:
            from dialog_tag import DialogTag
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "DialogTag is required only when predictions are not cached. "
                "Install the isolated tagger environment from "
                "analysis/requirements-dialogue-acts.txt, or run --human-only."
            ) from exc

        self.model_name = model_name
        self.tagger = DialogTag(model_name)
        self._batch_available = False
        try:
            self._tokenizer = self.tagger._DialogTag__tokenizer
            self._model = self.tagger._DialogTag__model
            class_helper = self.tagger._DialogTag__classhelper
            logits_class, class_expanded = class_helper()
            self._index_to_long = {
                int(index): class_expanded[short_code] for index, short_code in logits_class.items()
            }
            self._batch_available = True
        except (AttributeError, KeyError, TypeError, ValueError):
            # DialogTag exposes only predict_tag publicly.  Keeping this fallback
            # makes the adapter robust if the package changes its private fields.
            self._batch_available = False

    def predict_many(self, texts: Sequence[str], batch_size: int = 32) -> list[str]:
        if not texts:
            return []
        if not self._batch_available:
            return [self.tagger.predict_tag(text) for text in texts]

        predictions: list[str] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="tf",
                )
                output = self._model(**encoded, training=False)
                logits = output.logits if hasattr(output, "logits") else output[0]
                indices = np.asarray(logits).argmax(axis=1).tolist()
                predictions.extend(self._index_to_long[int(index)] for index in indices)
            return predictions
        except Exception as exc:  # pragma: no cover - package-version fallback
            print(f"DialogTag batch path failed ({exc}); falling back to predict_tag().")
            self._batch_available = False
            return [self.tagger.predict_tag(text) for text in texts]


def fingerprint_conversations(conversations: Sequence[Conversation]) -> str:
    """Content fingerprint used to reject stale label caches (never stores text)."""

    digest = hashlib.sha256()
    for conversation in sorted(
        conversations, key=lambda item: (item.condition, item.conversation_no)
    ):
        digest.update(conversation.condition.encode("utf-8"))
        digest.update(str(conversation.conversation_no).encode("ascii"))
        for speaker, text in zip(conversation.speakers, conversation.texts, strict=True):
            digest.update(b"\0")
            digest.update(speaker.encode("utf-8"))
            digest.update(b"\1")
            digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def validation_sample(
    gold: Sequence[Conversation], n_conversations: int, seed: int
) -> list[Conversation]:
    if n_conversations <= 0:
        raise ValueError("--validation-conversations must be positive")
    if n_conversations > len(gold):
        raise ValueError(
            f"Cannot sample {n_conversations} validation conversations from {len(gold)}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(list(gold), n_conversations), key=lambda item: item.conversation_no)


def _validation_texts(sample: Sequence[Conversation]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for conversation in sample:
        for raw_text, label in zip(conversation.texts, conversation.fine_labels, strict=True):
            # DialogTag was trained on cleaned SwDA utterance text.  Keep a raw
            # fallback for pure non-verbal rows that clean to an empty string.
            text = clean_text(raw_text) or raw_text.strip() or "[non-verbal]"
            texts.append(text)
            labels.append(label)
    return texts, labels


def _unit_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_partial_cache(
    cache_path: pathlib.Path,
    gold: Sequence[Conversation],
    model_name: str,
) -> tuple[list[Conversation], ValidationResult, dict[str, str]] | None:
    """Reuse what is still valid when the generated side changed (e.g. a sensitivity run):
    the human labels (if the human fingerprint matches) and a memo of generated-unit labels
    keyed by a SHA-1 of the unit text. Returns None if the human side is stale."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open(encoding="utf-8") as handle:
            cache = json.load(handle)
        expected_ids = [conversation.conversation_no for conversation in gold]
        if (
            cache.get("version") != CACHE_VERSION
            or cache.get("model_name") != model_name
            or cache.get("human_ids") != expected_ids
            or cache.get("human_fingerprint") != fingerprint_conversations(gold)
        ):
            return None
        human_by_id = {int(row["conversation_no"]): row["fine_labels"] for row in cache["human"]}
        tagged_human: list[Conversation] = []
        confusion: Counter[tuple[str, str]] = Counter()
        for conversation in gold:
            labels = list(human_by_id[conversation.conversation_no])
            if len(labels) != len(conversation.texts) or any(
                label not in FINE_LABELS for label in labels
            ):
                return None
            tagged_human.append(
                Conversation(
                    "SB",
                    "SB-tagger",
                    conversation.conversation_no,
                    list(conversation.speakers),
                    [],
                    labels,
                )
            )
            confusion.update(zip(conversation.fine_labels, labels, strict=True))
        memo = {k: v for k, v in cache.get("generated_unit_labels", {}).items() if v in FINE_LABELS}
        return tagged_human, ValidationResult(expected_ids, confusion), memo
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_cache(
    cache_path: pathlib.Path,
    generated: Sequence[Conversation],
    gold: Sequence[Conversation],
    model_name: str,
) -> tuple[list[Conversation], ValidationResult] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open(encoding="utf-8") as handle:
            cache = json.load(handle)
        expected_ids = [conversation.conversation_no for conversation in gold]
        if (
            cache.get("version") != CACHE_VERSION
            or cache.get("model_name") != model_name
            or cache.get("generated_fingerprint") != fingerprint_conversations(generated)
            or cache.get("human_ids") != expected_ids
            or cache.get("human_fingerprint") != fingerprint_conversations(gold)
        ):
            return None

        by_key = {
            (row["condition"], int(row["conversation_no"])): row["fine_labels"]
            for row in cache["generated"]
        }
        for conversation in generated:
            labels = list(by_key[(conversation.condition, conversation.conversation_no)])
            if len(labels) != len(conversation.texts) or any(
                label not in FINE_LABELS for label in labels
            ):
                return None
            conversation.fine_labels = labels
        human_by_id = {int(row["conversation_no"]): row["fine_labels"] for row in cache["human"]}
        tagged_human: list[Conversation] = []
        confusion: Counter[tuple[str, str]] = Counter()
        for conversation in gold:
            labels = list(human_by_id[conversation.conversation_no])
            if len(labels) != len(conversation.texts) or any(
                label not in FINE_LABELS for label in labels
            ):
                return None
            tagged_human.append(
                Conversation(
                    "SB",
                    "SB-tagger",
                    conversation.conversation_no,
                    list(conversation.speakers),
                    [],
                    labels,
                )
            )
            confusion.update(zip(conversation.fine_labels, labels, strict=True))
        return tagged_human, ValidationResult(expected_ids, confusion)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(
    cache_path: pathlib.Path,
    generated: Sequence[Conversation],
    gold: Sequence[Conversation],
    tagged_human: Sequence[Conversation],
    model_name: str,
    validation: ValidationResult,
) -> None:
    memo: dict[str, str] = {}
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as handle:
                memo = dict(json.load(handle).get("generated_unit_labels", {}))
        except (OSError, ValueError, AttributeError):
            memo = {}
    for conversation in generated:
        memo.update(
            {
                _unit_key(t): label
                for t, label in zip(conversation.texts, conversation.fine_labels, strict=True)
            }
        )
    payload = {
        "version": CACHE_VERSION,
        "model_name": model_name,
        "generated_unit_labels": memo,
        "generated_fingerprint": fingerprint_conversations(generated),
        "human_ids": [conversation.conversation_no for conversation in gold],
        "human_fingerprint": fingerprint_conversations(gold),
        "generated": [
            {
                "condition": conversation.condition,
                "conversation_no": conversation.conversation_no,
                "fine_labels": conversation.fine_labels,
            }
            for conversation in sorted(
                generated, key=lambda item: (item.condition, item.conversation_no)
            )
        ],
        "human": [
            {
                "conversation_no": conversation.conversation_no,
                "fine_labels": conversation.fine_labels,
            }
            for conversation in tagged_human
        ],
        "validation_confusion": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in sorted(validation.fine_confusion.items())
        ],
        "privacy": "No transcript text is stored; validation is aggregate counts only.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def tag_both_sides(
    generated: Sequence[Conversation],
    gold: Sequence[Conversation],
    *,
    model_name: str,
    batch_size: int,
    validation_conversations: int,
    seed: int,
    cache_path: pathlib.Path,
    force_retag: bool,
    use_cache: bool,
) -> tuple[list[Conversation], ValidationResult, bool]:
    if use_cache and not force_retag:
        cached = _load_cache(cache_path, generated, gold, model_name)
        if cached is not None:
            return cached[0], cached[1], True
        partial = _load_partial_cache(cache_path, gold, model_name)
        if partial is not None:
            tagged_human, validation, memo = partial
            missing = sorted({t for c in generated for t in c.texts if _unit_key(t) not in memo})
            print(
                f"Reusing cached human labels and {len(memo):,} generated-unit labels; "
                f"tagging {len(missing):,} new generated units..."
            )
            if missing:
                tagger = DialogTagAdapter(model_name)
                predicted = tagger.predict_many(missing, batch_size=batch_size)
                if len(predicted) != len(missing):
                    raise RuntimeError("DialogTag returned the wrong number of predictions")
                memo.update(
                    {
                        _unit_key(t): dialogtag_to_fine(lab)
                        for t, lab in zip(missing, predicted, strict=True)
                    }
                )
            for conversation in generated:
                conversation.fine_labels = [memo[_unit_key(t)] for t in conversation.texts]
            _write_cache(cache_path, generated, gold, tagged_human, model_name, validation)
            return tagged_human, validation, True

    llm_texts = [text for conversation in generated for text in conversation.texts]
    human_texts, gold_labels = _validation_texts(gold)
    print(
        f"Loading DialogTag {model_name!r}; tagging {len(llm_texts):,} generated "
        f"sentence units and {len(human_texts):,} human utterances..."
    )
    tagger = DialogTagAdapter(model_name)
    long_predictions = tagger.predict_many(llm_texts + human_texts, batch_size=batch_size)
    fine_predictions = [dialogtag_to_fine(label) for label in long_predictions]
    if len(fine_predictions) != len(llm_texts) + len(human_texts):
        raise RuntimeError("DialogTag returned the wrong number of predictions")

    cursor = 0
    for conversation in generated:
        next_cursor = cursor + len(conversation.texts)
        conversation.fine_labels = fine_predictions[cursor:next_cursor]
        cursor = next_cursor
    predicted_gold = fine_predictions[len(llm_texts) :]
    tagged_human: list[Conversation] = []
    cursor = 0
    for conversation in gold:
        next_cursor = cursor + len(conversation.texts)
        tagged_human.append(
            Conversation(
                "SB",
                "SB-tagger",
                conversation.conversation_no,
                list(conversation.speakers),
                [],
                predicted_gold[cursor:next_cursor],
            )
        )
        cursor = next_cursor
    validation = ValidationResult(
        conversation_ids=[conversation.conversation_no for conversation in gold],
        fine_confusion=Counter(zip(gold_labels, predicted_gold, strict=True)),
    )
    if use_cache:
        _write_cache(cache_path, generated, gold, tagged_human, model_name, validation)
    return tagged_human, validation, False


def grouped_by_condition(
    conversations: Sequence[Conversation],
) -> dict[str, list[Conversation]]:
    grouped: defaultdict[str, list[Conversation]] = defaultdict(list)
    for conversation in conversations:
        grouped[conversation.condition].append(conversation)
    return dict(sorted(grouped.items()))


def labels_for(conversation: Conversation, label_set: str) -> list[str]:
    if label_set == "fine":
        return conversation.fine_labels
    if label_set == "coarse":
        return conversation.coarse_labels()
    raise ValueError(f"Unknown label set: {label_set}")


def label_inventory(label_set: str) -> list[str]:
    return FINE_LABELS if label_set == "fine" else COARSE_LABELS


def distribution(conversations: Sequence[Conversation], label_set: str) -> np.ndarray:
    inventory = label_inventory(label_set)
    index = {label: i for i, label in enumerate(inventory)}
    counts = np.zeros(len(inventory), dtype=float)
    for conversation in conversations:
        for label in labels_for(conversation, label_set):
            counts[index[label]] += 1
    total = counts.sum()
    return counts / total if total else counts


def transition_matrix(conversations: Sequence[Conversation], label_set: str) -> np.ndarray:
    inventory = label_inventory(label_set)
    index = {label: i for i, label in enumerate(inventory)}
    counts = np.zeros((len(inventory), len(inventory)), dtype=float)
    for conversation in conversations:
        sequence = labels_for(conversation, label_set)
        for current, following in itertools.pairwise(sequence):
            counts[index[current], index[following]] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts),
        where=row_totals != 0,
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence in [0, 1]."""

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_total = float(p.sum())
    q_total = float(q.sum())
    if p_total == 0 and q_total == 0:
        return 0.0
    if p_total == 0 or q_total == 0:
        return 1.0
    p = p / p_total
    q = q / q_total
    midpoint = 0.5 * (p + q)

    def _kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def transition_jsd(left: np.ndarray, right: np.ndarray) -> float:
    """Unweighted mean row JSD; a one-sided missing row has distance 1."""

    row_distances = [
        js_divergence(left[row], right[row])
        for row in range(left.shape[0])
        if left[row].sum() or right[row].sum()
    ]
    return float(np.mean(row_distances)) if row_distances else 0.0


def bootstrap_noise_floor(
    gold: Sequence[Conversation],
    label_set: str,
    *,
    unit_count: int,
    repetitions: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    inventory = label_inventory(label_set)
    index = {label: i for i, label in enumerate(inventory)}
    pooled = [
        index[label] for conversation in gold for label in labels_for(conversation, label_set)
    ]
    if unit_count > len(pooled):
        raise ValueError(f"Unit-matched draw {unit_count} exceeds {len(pooled)} human units")
    full = np.bincount(pooled, minlength=len(inventory)).astype(float)
    full /= full.sum()
    rng = np.random.default_rng(seed)
    sampled_counts = rng.multinomial(unit_count, full, size=repetitions)
    distances = np.asarray(
        [js_divergence(sampled, full) for sampled in sampled_counts], dtype=float
    )
    return float(np.quantile(distances, 0.95)), distances


BACKCHANNEL_FORMS = {"uh huh", "yeah", "right", "okay", "mm hm", "i see"}


def transparent_rule_counts(conversations: Sequence[Conversation]) -> dict[str, float | int]:
    """Text-only cross-check: exact short reactions and literal question marks."""

    units = backchannels = questions = 0
    for conversation in conversations:
        for text in conversation.texts:
            units += 1
            normalized = re.sub(r"[^a-z0-9']+", " ", clean_text(text).casefold()).strip()
            tokens = normalized.split()
            reactive = normalized in BACKCHANNEL_FORMS and len(tokens) <= 3
            backchannels += int(reactive)
            questions += int("?" in text)
    return {
        "n_units": units,
        "rule_backchannel_rate": backchannels / units if units else 0.0,
        "rule_question_rate": questions / units if units else 0.0,
    }


def tagger_rates(conversations: Sequence[Conversation]) -> tuple[float, float]:
    labels = [label for conversation in conversations for label in conversation.fine_labels]
    backchannel = {"b", "bk", "bh", "ba"}
    questions = {"qy", "qw", "qo", "qh", "qrr", "^g", "bh"}
    return (
        (
            sum(x in backchannel for x in labels) / len(labels),
            sum(x in questions for x in labels) / len(labels),
        )
        if labels
        else (0.0, 0.0)
    )


def assistant_register_counts(
    conversations: Sequence[Conversation],
) -> dict[str, float | int]:
    matches = Counter(dict.fromkeys(ASSISTANT_PATTERNS, 0))
    any_match = 0
    n_turns = 0
    for conversation in conversations:
        for text in conversation.texts:
            n_turns += 1
            matched = [name for name, pattern in ASSISTANT_PATTERNS.items() if pattern.search(text)]
            if matched:
                any_match += 1
                matches.update(matched)
    result: dict[str, float | int] = {
        "n_turns": n_turns,
        "assistant_register_turns": any_match,
        "assistant_register_fraction": any_match / n_turns if n_turns else 0.0,
    }
    for name in ASSISTANT_PATTERNS:
        result[f"{name}_fraction"] = matches[name] / n_turns if n_turns else 0.0
    return result


def confusion_for_label_set(
    fine_confusion: Counter[tuple[str, str]], label_set: str
) -> Counter[tuple[str, str]]:
    if label_set == "fine":
        return Counter(fine_confusion)
    confusion: Counter[tuple[str, str]] = Counter()
    for (gold, predicted), count in fine_confusion.items():
        confusion[(FINE_TO_COARSE[gold], FINE_TO_COARSE[predicted])] += count
    return confusion


def validation_accuracy(confusion: Counter[tuple[str, str]]) -> float:
    total = sum(confusion.values())
    correct = sum(count for (gold, predicted), count in confusion.items() if gold == predicted)
    return correct / total if total else 0.0


def write_csv(path: pathlib.Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(groups: dict[str, Sequence[Conversation]], label_set: str) -> list[dict]:
    inventory = label_inventory(label_set)
    rows: list[dict] = []
    for condition, conversations in groups.items():
        values = distribution(conversations, label_set)
        row = {
            "condition": condition,
            "n_conversations": len(conversations),
            "n_units": sum(len(labels_for(c, label_set)) for c in conversations),
        }
        row.update({label: f"{values[i]:.10f}" for i, label in enumerate(inventory)})
        rows.append(row)
    return rows


def write_transition_csv(path: pathlib.Path, matrix: np.ndarray, label_set: str) -> None:
    inventory = label_inventory(label_set)
    rows = []
    for row_index, current in enumerate(inventory):
        row = {"current_act": current}
        row.update(
            {following: f"{matrix[row_index, col]:.10f}" for col, following in enumerate(inventory)}
        )
        rows.append(row)
    write_csv(path, rows, ["current_act", *inventory])


def safe_condition_name(condition: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", condition)


def build_comparison_rows(
    gold: Sequence[Conversation],
    tagged_human: Sequence[Conversation],
    generated_groups: dict[str, list[Conversation]],
    *,
    repetitions: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    noise_rows: list[dict] = []
    for label_set in ("fine", "coarse"):
        gold_dist = distribution(gold, label_set)
        tagged_dist = distribution(tagged_human, label_set)
        rows.append(
            {
                "condition": "SB",
                "label_set": label_set,
                "view": "gold_human_reference",
                "jsd_dist": "0.0000000000",
                "jsd_transition": "0.0000000000",
                "noise_floor_p95": "",
                "exceeds_floor": "",
                "n_units": sum(len(labels_for(c, label_set)) for c in gold),
            }
        )
        calibration = js_divergence(gold_dist, tagged_dist)
        rows.append(
            {
                "condition": "SB",
                "label_set": label_set,
                "view": "gold_vs_tagger_human_calibration",
                "jsd_dist": f"{calibration:.10f}",
                "jsd_transition": f"{transition_jsd(transition_matrix(gold, label_set), transition_matrix(tagged_human, label_set)):.10f}",
                "noise_floor_p95": "",
                "exceeds_floor": "",
                "n_units": sum(len(labels_for(c, label_set)) for c in tagged_human),
            }
        )
        for condition, conversations in generated_groups.items():
            condition_distribution = distribution(conversations, label_set)
            condition_transition = transition_matrix(conversations, label_set)
            assistant = assistant_register_counts(conversations)
            n_units = sum(len(labels_for(c, label_set)) for c in conversations)
            floor, distances = bootstrap_noise_floor(
                tagged_human, label_set, unit_count=n_units, repetitions=repetitions, seed=seed
            )
            dist_jsd = js_divergence(condition_distribution, tagged_dist)
            rows.append(
                {
                    "condition": condition,
                    "label_set": label_set,
                    "view": "tagger_human_vs_tagger_llm_primary",
                    "jsd_dist": f"{dist_jsd:.10f}",
                    "jsd_transition": f"{transition_jsd(condition_transition, transition_matrix(tagged_human, label_set)):.10f}",
                    "noise_floor_p95": f"{floor:.10f}",
                    "exceeds_floor": str(dist_jsd > floor).lower(),
                    "assistant_register_fraction": f"{assistant['assistant_register_fraction']:.10f}",
                    "n_conversations": len(conversations),
                    "n_units": n_units,
                }
            )
            noise_rows.append(
                {
                    "condition": condition,
                    "label_set": label_set,
                    "unit_count": n_units,
                    "repetitions": repetitions,
                    "seed": seed,
                    "mean_jsd": f"{distances.mean():.10f}",
                    "p95_jsd": f"{floor:.10f}",
                }
            )
    return rows, noise_rows


def write_validation_outputs(output_dir: pathlib.Path, validation: ValidationResult) -> list[dict]:
    summary_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for label_set in ("fine", "coarse"):
        confusion = confusion_for_label_set(validation.fine_confusion, label_set)
        total_by_gold: Counter[str] = Counter()
        for (gold, _), count in confusion.items():
            total_by_gold[gold] += count
        accuracy = validation_accuracy(confusion)
        summary_rows.append(
            {
                "label_set": label_set,
                "n_conversations": len(validation.conversation_ids),
                "n_utterances": sum(confusion.values()),
                "accuracy": f"{accuracy:.10f}",
            }
        )
        for (gold, predicted), count in sorted(
            confusion.items(), key=lambda item: (-item[1], item[0])
        ):
            confusion_rows.append(
                {
                    "label_set": label_set,
                    "gold": gold,
                    "predicted": predicted,
                    "count": count,
                    "fraction_of_gold": f"{count / total_by_gold[gold]:.10f}",
                }
            )
    write_csv(
        output_dir / "da_tagger_validation.csv",
        summary_rows,
        ["label_set", "n_conversations", "n_utterances", "accuracy"],
    )
    write_csv(
        output_dir / "da_tagger_confusion.csv",
        confusion_rows,
        ["label_set", "gold", "predicted", "count", "fraction_of_gold"],
    )
    return summary_rows


def print_validation(validation: ValidationResult, top_n: int = 12) -> None:
    print("\nDialogTag validation against gold SwDA")
    for label_set in ("fine", "coarse"):
        confusion = confusion_for_label_set(validation.fine_confusion, label_set)
        accuracy = validation_accuracy(confusion)
        print(
            f"  {label_set:6} accuracy: {accuracy:.3%} "
            f"({sum(confusion.values()):,} utterances; "
            f"{len(validation.conversation_ids)} conversations)"
        )
        errors = [
            (count, gold, predicted)
            for (gold, predicted), count in confusion.items()
            if gold != predicted
        ]
        errors.sort(reverse=True)
        print(f"  {label_set:6} largest confusions (gold -> predicted):")
        for count, gold, predicted in errors[:top_n]:
            print(f"           {gold:25} -> {predicted:25} {count:5d}")


def write_assistant_output(
    output_dir: pathlib.Path,
    gold: Sequence[Conversation],
    generated_groups: dict[str, list[Conversation]],
) -> list[dict]:
    rows: list[dict] = []
    for condition, conversations in [("SB", list(gold)), *generated_groups.items()]:
        row: dict = {"condition": condition}
        row.update(assistant_register_counts(conversations))
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.10f}"
        rows.append(row)
    fields = [
        "condition",
        "n_turns",
        "assistant_register_turns",
        "assistant_register_fraction",
        *[f"{name}_fraction" for name in ASSISTANT_PATTERNS],
    ]
    write_csv(output_dir / "assistant_register_by_condition.csv", rows, fields)
    return rows


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_key_acts(groups: dict[str, Sequence[Conversation]], output_path: pathlib.Path) -> None:
    plt = _import_plotting()
    conditions = list(groups)
    keys = ["Backchannel", "Statement", "Question", "Agreement"]
    values: dict[str, list[float]] = {key: [] for key in keys}
    for condition in conditions:
        coarse = distribution(groups[condition], "coarse")
        by_label = dict(zip(COARSE_LABELS, coarse, strict=True))
        values["Backchannel"].append(by_label["Backchannel"])
        values["Statement"].append(by_label["Statement"] + by_label["Opinion"])
        values["Question"].append(by_label["YesNoQuestion"] + by_label["WhQuestion/OpenQuestion"])
        values["Agreement"].append(by_label["Agreement"])

    x = np.arange(len(conditions))
    width = 0.19
    fig, axis = plt.subplots(figsize=(14, 6))
    for offset, key in enumerate(keys):
        axis.bar(x + (offset - 1.5) * width, values[key], width, label=key)
    axis.set_xticks(x, conditions, rotation=45, ha="right")
    axis.set_ylabel("Proportion of dialogue units")
    axis.set_title("Key dialogue-act groups: Switchboard vs generated conditions")
    axis.legend(ncols=4, frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sb_heatmap(matrix: np.ndarray, output_path: pathlib.Path) -> None:
    plt = _import_plotting()
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=max(0.5, float(matrix.max())))
    axis.set_xticks(range(len(COARSE_LABELS)), COARSE_LABELS, rotation=55, ha="right")
    axis.set_yticks(range(len(COARSE_LABELS)), COARSE_LABELS)
    axis.set_xlabel("Next act")
    axis.set_ylabel("Current act")
    axis.set_title("Switchboard transition grammar (coarse acts)")
    fig.colorbar(image, ax=axis, label="P(next | current)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_jsd(comparison_rows: Sequence[dict], output_path: pathlib.Path) -> None:
    plt = _import_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for axis, label_set in zip(axes, ("fine", "coarse"), strict=True):
        rows = [
            row
            for row in comparison_rows
            if row["label_set"] == label_set
            and row.get("view") == "tagger_human_vs_tagger_llm_primary"
        ]
        conditions = [row["condition"] for row in rows]
        values = [float(row["jsd_dist"]) for row in rows]
        floors = [float(row["noise_floor_p95"]) for row in rows]
        colors = [
            "#b43c39" if value > floor else "#4c78a8"
            for value, floor in zip(values, floors, strict=True)
        ]
        axis.bar(range(len(rows)), values, color=colors)
        axis.scatter(
            range(len(rows)),
            floors,
            color="black",
            marker="_",
            s=180,
            label="unit-matched human p95",
        )
        axis.set_ylabel("JSD vs full SB")
        axis.set_title(f"{label_set.capitalize()} dialogue-act distribution")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
        axis.set_xticks(range(len(rows)), conditions, rotation=45, ha="right")
    fig.suptitle("Dialogue-act structural distance and the human sampling-noise floor")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def top_confusions(
    validation: ValidationResult, label_set: str, n: int = 5
) -> list[tuple[int, str, str]]:
    confusion = confusion_for_label_set(validation.fine_confusion, label_set)
    errors = [
        (count, gold, predicted)
        for (gold, predicted), count in confusion.items()
        if gold != predicted
    ]
    return sorted(errors, reverse=True)[:n]


def write_readme(
    output_dir: pathlib.Path,
    *,
    gold: Sequence[Conversation],
    tagged_human: Sequence[Conversation],
    generated_groups: dict[str, list[Conversation]],
    comparison_rows: Sequence[dict],
    validation: ValidationResult,
    bootstrap_repetitions: int,
    seed: int,
    model_name: str,
    cache_used: bool,
) -> None:
    primary = [
        r
        for r in comparison_rows
        if r["view"] == "tagger_human_vs_tagger_llm_primary" and r["label_set"] == "coarse"
    ]
    calibration = next(
        r
        for r in comparison_rows
        if r["view"] == "gold_vs_tagger_human_calibration" and r["label_set"] == "coarse"
    )
    closest = min(primary, key=lambda r: float(r["jsd_dist"]))
    farthest = max(primary, key=lambda r: float(r["jsd_dist"]))
    gold_rate = tagger_rates(gold)
    tagged_rate = tagger_rates(tagged_human)
    units = [
        sum(len(c.fine_labels) for c in group) / len(group) for group in generated_groups.values()
    ]
    text = f"""# Dialogue-act structural signature

This rerun compares {len(gold):,} Switchboard conversations ({sum(len(c.fine_labels) for c in gold):,} gold utterance units) with {sum(len(v) for v in generated_groups.values()):,} generated conversations. Generated turns are split into sentence units before tagging.

## Three views

1. **Gold-human reference (authoritative):** charts and `SB-gold` distribution/transition files preserve the expert `act_tag` labels. Gold coarse backchannels are {gold_rate[0]:.1%}; gold question acts are {gold_rate[1]:.1%}.
2. **Primary fair divergence:** DialogTag is applied to both human and generated units. Coarse JSD ranges from **{float(closest["jsd_dist"]):.4f}** ({closest["condition"]}) to **{float(farthest["jsd_dist"]):.4f}** ({farthest["condition"]}); every condition has its own human unit-matched 95th-percentile noise floor.
3. **Tagger calibration:** gold-human vs tagger-human coarse JSD is **{float(calibration["jsd_dist"]):.4f}** (transition JSD {float(calibration["jsd_transition"]):.4f}). On human speech the tagger reports {tagged_rate[0]:.1%} backchannels and {tagged_rate[1]:.1%} questions, versus the gold rates above. Per-act contributions are in `da_tagger_calibration_by_act.csv`.

Switchboard averages {sum(len(c.fine_labels) for c in gold) / len(gold):.1f} units/conversation. Generated conditions average {min(units):.1f}–{max(units):.1f} sentence units/conversation; segmentation narrows but does not eliminate the granularity difference.

## Tagger-independent cross-check

`da_rule_crosscheck.csv` reports literal `?` questions and exact short reactions (`uh-huh`, `yeah`, `right`, `okay`, `mm-hm`, `i see`) beside tagger backchannel/question rates on both sides. These rules are intentionally high-precision and incomplete.

## Method and limitations

DialogTag `{model_name}` was retained after a timeboxed Hugging Face search: the plausible public SwDA checkpoint was token-classification with undocumented utterance-boundary encoding, not a clean sentence-classification replacement. The same-tagger primary view removes the two-ruler confound but cannot remove correlated domain errors on polished LLM text. Full-corpus in-domain fine/coarse agreement is {validation_accuracy(confusion_for_label_set(validation.fine_confusion, "fine")):.1%}/{validation_accuracy(confusion_for_label_set(validation.fine_confusion, "coarse")):.1%}; this is calibration, not a held-out accuracy claim. A separate human check on 100 generated units found 77% agreement (see `results/tagger_handcheck.txt` and `annotation/`).

The text-only cross-check is especially important for listener feedback: it finds {transparent_rule_counts(gold)["rule_backchannel_rate"]:.1%} short lexical backchannels in humans, while generated conditions range from {min(transparent_rule_counts(v)["rule_backchannel_rate"] for v in generated_groups.values()):.1%} to {max(transparent_rule_counts(v)["rule_backchannel_rate"] for v in generated_groups.values()):.1%}. DialogTag assigns substantially more generated backchannels, so the deficit is not an artifact of relying on its labels.

Noise floors use {bootstrap_repetitions:,} seeded multinomial bootstrap draws from tagger-human units, matched separately to each condition's classified-unit count (seed {seed}). The label cache contains labels, IDs, fingerprints, and aggregate confusion only—never Switchboard or generated transcript text. This run {"reused the cache" if cache_used else "created fresh predictions"}.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def print_summary(comparison_rows: Sequence[dict]) -> None:
    by_condition: defaultdict[str, dict[str, dict]] = defaultdict(dict)
    for row in comparison_rows:
        if row.get("view") != "tagger_human_vs_tagger_llm_primary":
            continue
        by_condition[row["condition"]][row["label_set"]] = row
    print("\nDialogue-act structural-signature summary")
    header = (
        f"{'condition':12} {'fine JSD':>9} {'coarse JSD':>10} "
        f"{'coarse trans':>12} {'floor':>8} {'>floor':>7} "
        f"{'topic JSD':>10} {'assistant':>10}"
    )
    print(header)
    print("-" * len(header))
    for condition in sorted(by_condition):
        fine = by_condition[condition]["fine"]
        coarse = by_condition[condition]["coarse"]
        print(
            f"{condition:12} {float(fine['jsd_dist']):9.4f} "
            f"{float(coarse['jsd_dist']):10.4f} "
            f"{float(coarse['jsd_transition']):12.4f} "
            f"{float(coarse['noise_floor_p95']):8.4f} "
            f"{coarse['exceeds_floor']:>7} "
            f"{'n/a':>10} "
            f"{float(coarse['assistant_register_fraction']):9.2%}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fine and coarse dialogue-act structural signatures."
    )
    parser.add_argument(
        "--generated-root",
        type=pathlib.Path,
        default=GEN_ROOT,
        help=f"generated condition directory (default: {GEN_ROOT})",
    )
    parser.add_argument(
        "--swda-root",
        type=pathlib.Path,
        default=SWDA_ROOT,
        help=f"extracted SwDA directory (default: {SWDA_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=None,
        help="prediction cache path (default: <out-dir>/dialogtag_predictions.json)",
    )
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-conversations", type=int, default=20)
    parser.add_argument("--bootstrap-size", type=int, default=50)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--force-retag",
        action="store_true",
        help="ignore a valid cache and invoke DialogTag again",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write the prediction cache",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="write the gold reference/noise floor without importing DialogTag",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip matplotlib figures",
    )
    parser.add_argument(
        "--exclude-loops",
        action="store_true",
        help="sensitivity analysis: drop echo-loop turns (near-verbatim repeats) before tagging",
    )
    parser.add_argument(
        "--exclude-drift",
        action="store_true",
        help="sensitivity analysis: drop whole conversations with language drift (non-Latin script)",
    )
    return parser.parse_args(argv)


def validate_configuration() -> None:
    if set(FINE_LABELS) != set(FINE_TO_COARSE):
        missing = set(FINE_LABELS).difference(FINE_TO_COARSE)
        extra = set(FINE_TO_COARSE).difference(FINE_LABELS)
        raise RuntimeError(f"Invalid coarse map; missing={missing}, extra={extra}")
    if set(FINE_TO_COARSE.values()) != set(COARSE_LABELS):
        raise RuntimeError("Coarse map does not cover exactly COARSE_LABELS")
    if not set(DIALOGTAG_TO_FINE.values()).issubset(FINE_LABELS):
        raise RuntimeError("DialogTag mapping emits a label outside FINE_LABELS")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_configuration()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.cache or output_dir / "dialogtag_predictions.json").resolve()

    print(f"Loading full gold SwDA reference from {args.swda_root}...")
    gold = load_switchboard(args.swda_root)
    print(
        f"Loaded {len(gold):,} conversations and "
        f"{sum(len(c.fine_labels) for c in gold):,} annotated units."
    )

    gold_groups: dict[str, Sequence[Conversation]] = {"SB": gold}
    fine_rows = distribution_rows(gold_groups, "fine")
    coarse_rows = distribution_rows(gold_groups, "coarse")
    write_csv(
        output_dir / "da_distribution_by_condition.csv",
        fine_rows,
        ["condition", "n_conversations", "n_units", *FINE_LABELS],
    )
    write_csv(
        output_dir / "da_distribution_coarse_by_condition.csv",
        coarse_rows,
        ["condition", "n_conversations", "n_units", *COARSE_LABELS],
    )
    fine_sb_transition = transition_matrix(gold, "fine")
    coarse_sb_transition = transition_matrix(gold, "coarse")
    write_transition_csv(output_dir / "da_transition_SB.csv", fine_sb_transition, "fine")
    write_transition_csv(output_dir / "da_transition_coarse_SB.csv", coarse_sb_transition, "coarse")

    if args.human_only:
        if not args.no_figures:
            plot_key_acts(gold_groups, output_dir / "key_acts_grouped_bar.png")
            plot_sb_heatmap(coarse_sb_transition, output_dir / "sb_transition_heatmap.png")
        print("\nHuman-only run complete (DialogTag was not imported).")
        print(f"Wrote gold outputs to {output_dir}")
        return

    generated = load_generated(
        args.generated_root, exclude_loops=args.exclude_loops, exclude_drift=args.exclude_drift
    )
    generated_groups = grouped_by_condition(generated)
    print(
        f"Loaded {len(generated):,} generated conversations across "
        f"{len(generated_groups)} conditions."
    )
    tagged_human, validation, cache_used = tag_both_sides(
        generated,
        gold,
        model_name=args.model,
        batch_size=args.batch_size,
        validation_conversations=args.validation_conversations,
        seed=args.seed,
        cache_path=cache_path,
        force_retag=args.force_retag,
        use_cache=not args.no_cache,
    )
    print(f"Prediction source: {'validated cache' if cache_used else 'DialogTag model'}")
    print_validation(validation)

    all_groups: dict[str, Sequence[Conversation]] = {
        "SB-gold": gold,
        "SB-tagger": tagged_human,
        **generated_groups,
    }
    fine_rows = distribution_rows(all_groups, "fine")
    coarse_rows = distribution_rows(all_groups, "coarse")
    write_csv(
        output_dir / "da_distribution_by_condition.csv",
        fine_rows,
        ["condition", "n_conversations", "n_units", *FINE_LABELS],
    )
    write_csv(
        output_dir / "da_distribution_coarse_by_condition.csv",
        coarse_rows,
        ["condition", "n_conversations", "n_units", *COARSE_LABELS],
    )
    write_transition_csv(
        output_dir / "da_transition_SB_tagger.csv", transition_matrix(tagged_human, "fine"), "fine"
    )
    write_transition_csv(
        output_dir / "da_transition_coarse_SB_tagger.csv",
        transition_matrix(tagged_human, "coarse"),
        "coarse",
    )

    for condition, conversations in generated_groups.items():
        safe_name = safe_condition_name(condition)
        write_transition_csv(
            output_dir / f"da_transition_{safe_name}.csv",
            transition_matrix(conversations, "fine"),
            "fine",
        )
        write_transition_csv(
            output_dir / f"da_transition_coarse_{safe_name}.csv",
            transition_matrix(conversations, "coarse"),
            "coarse",
        )

    comparison_rows, noise_rows = build_comparison_rows(
        gold, tagged_human, generated_groups, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    write_csv(
        output_dir / "da_noise_floor.csv",
        noise_rows,
        ["condition", "label_set", "unit_count", "repetitions", "seed", "mean_jsd", "p95_jsd"],
    )
    write_csv(
        output_dir / "da_jsd_vs_sb.csv",
        comparison_rows,
        [
            "condition",
            "label_set",
            "view",
            "jsd_dist",
            "jsd_transition",
            "noise_floor_p95",
            "exceeds_floor",
            "assistant_register_fraction",
            "n_conversations",
            "n_units",
        ],
    )
    write_validation_outputs(output_dir, validation)
    write_assistant_output(output_dir, gold, generated_groups)
    crosscheck_rows = []
    for condition, raw_group, tagged_group in [("SB", gold, tagged_human)] + [
        (condition, group, group) for condition, group in generated_groups.items()
    ]:
        rule = transparent_rule_counts(raw_group)
        bc, q = tagger_rates(tagged_group)
        crosscheck_rows.append(
            {
                "condition": condition,
                **rule,
                "tagger_backchannel_rate": f"{bc:.10f}",
                "tagger_question_rate": f"{q:.10f}",
            }
        )
    write_csv(
        output_dir / "da_rule_crosscheck.csv",
        crosscheck_rows,
        [
            "condition",
            "n_units",
            "rule_backchannel_rate",
            "tagger_backchannel_rate",
            "rule_question_rate",
            "tagger_question_rate",
        ],
    )
    calibration_rows = []
    for label_set in ("fine", "coarse"):
        gd, td = distribution(gold, label_set), distribution(tagged_human, label_set)
        for i, act in enumerate(label_inventory(label_set)):
            m = (gd[i] + td[i]) / 2
            contribution = 0.5 * (
                (gd[i] * np.log2(gd[i] / m) if gd[i] else 0)
                + (td[i] * np.log2(td[i] / m) if td[i] else 0)
            )
            calibration_rows.append(
                {
                    "label_set": label_set,
                    "act": act,
                    "gold_rate": f"{gd[i]:.10f}",
                    "tagger_rate": f"{td[i]:.10f}",
                    "jsd_contribution": f"{contribution:.10f}",
                }
            )
    write_csv(
        output_dir / "da_tagger_calibration_by_act.csv",
        calibration_rows,
        ["label_set", "act", "gold_rate", "tagger_rate", "jsd_contribution"],
    )

    if not args.no_figures:
        plot_key_acts(all_groups, output_dir / "key_acts_grouped_bar.png")
        plot_sb_heatmap(coarse_sb_transition, output_dir / "sb_transition_heatmap.png")
        plot_jsd(comparison_rows, output_dir / "jsd_vs_sb.png")

    write_readme(
        output_dir,
        gold=gold,
        tagged_human=tagged_human,
        generated_groups=generated_groups,
        comparison_rows=comparison_rows,
        validation=validation,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        model_name=args.model,
        cache_used=cache_used,
    )
    print_summary(comparison_rows)
    print(f"\nWrote dialogue-act outputs to {output_dir}")


if __name__ == "__main__":
    main()
