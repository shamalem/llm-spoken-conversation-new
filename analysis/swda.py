"""
Switchboard (SwDA) loader + cleaner for the project's Switchboard baseline.

Source: Christopher Potts' SwDA distribution (swda.zip) — the same source the paper used
(compprag.christopherpotts.net/swda.html). Data lives under data/switchboard/swda/ and is
NOT committed (LDC-licensed; gitignored).

This module:
  - load_metadata()      : conversation-level topic, verbatim SB prompt, demographics.
  - clean_text()         : strip SwDA disfluency/transcription markup, keep spoken words.
  - parse_conversation() : rows -> turns (consecutive same-speaker utterances merged),
                           matching how the paper / ALIGN treat turns.
  - words_per_turn()     : descriptive check to validate the pipeline against the paper
                           (SB main-body mean is ~14 words/turn).

Run as a script for a quick local validation (stdlib only — no GPU, no pandas):
    python analysis/swda.py --n 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "switchboard" / "swda"

# SwDA caller education codes, rendered as text in the personas (9 = not recorded).
EDU = {
    "0": "less than a high-school education",
    "1": "a high-school education",
    "2": "some college",
    "3": "a college degree",
    "9": "an unspecified education",
}


def clean_text(raw: str) -> str:
    """Remove SwDA markup, keep the spoken words.

    Handles: <beep>/<<pause>> non-verbals; {D ..}/{F ..}/{C ..} annotations (keep inner
    words); [ .. + .. ] repair brackets; trailing slash-units '/' and interruptions '-/'.
    """
    t = raw
    t = re.sub(r"<+[^>]*>+", " ", t)  # <beep>, <<long pause>>
    t = re.sub(r"\{[A-Z]\s", " ", t)  # opening {D {F {C {E {A ...
    t = t.replace("}", " ")
    t = re.sub(r"\(\(\s*(.*?)\s*\)\)", r"\1", t)  # (( uncertain )) -> keep the words
    for ch in "[]+#":
        t = t.replace(ch, " ")
    t = re.sub(r"-?/", " ", t)  # slash-unit and -/ interruption
    t = t.replace("--", " ")  # interruption dashes
    t = re.sub(r"(?:^|\s)-+(?=\s|$)", " ", t)  # stray standalone hyphens
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.?!])", r"\1", t)  # tidy space before punctuation
    t = t.strip(" ,")  # no dangling leading/trailing commas
    # Drop turns that are only non-verbals: e.g. '<Breathing>.' / '<Lipsmack>' clean to
    # bare punctuation. Left in, they become phantom 1-word "turns" that inflate the human
    # short-turn/backchannel rate (2.85% of SB turns) and poison the P2 few-shot example.
    if not re.search(r"[A-Za-z0-9]", t):
        return ""
    return t


def load_metadata(root: Path = DATA_ROOT) -> dict[int, dict]:
    """conversation_no -> metadata row (topic_description, prompt, demographics)."""
    out: dict[int, dict] = {}
    with open(root / "swda-metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["conversation_no"])] = row
    return out


def parse_conversation(csv_path: Path) -> list[tuple[str, str]]:
    """Return [(caller, text), ...] with consecutive same-caller utterances merged."""
    turns: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spk = row["caller"].strip()
            txt = clean_text(row["text"])
            if not txt:
                continue
            if turns and turns[-1][0] == spk:
                turns[-1] = (spk, turns[-1][1] + " " + txt)
            else:
                turns.append((spk, txt))
    return turns


def parse_conversation_tagged(csv_path: Path) -> list[tuple[str, str, list[str]]]:
    """Like parse_conversation, but also returns each merged turn's gold SwDA act tags
    (one per utterance, in order). Skips exactly the same rows, so turn i here is turn i
    of parse_conversation — the P3 labeled excerpts line up with P2's unlabeled ones."""
    turns: list[tuple[str, str, list[str]]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spk = row["caller"].strip()
            txt = clean_text(row["text"])
            if not txt:
                continue
            tag = row.get("act_tag", "").strip()
            if turns and turns[-1][0] == spk:
                turns[-1] = (spk, turns[-1][1] + " " + txt, turns[-1][2] + [tag])
            else:
                turns.append((spk, txt, [tag]))
    return turns


# Plain-language names for the SwDA act tags shown in P3's labeled examples. Base tag =
# the part before any ^ ( @ modifier. Anything unlisted renders as "other".
ACT_NAMES = {
    "sd": "statement",
    "sv": "opinion",
    "b": "backchannel",
    "bh": "backchannel question",
    "bk": "acknowledgment",
    "aa": "agreement",
    "aap": "partial agreement",
    "ba": "appreciation",
    "%": "unfinished",
    "+": "continues their earlier turn",
    "qy": "yes/no question",
    "qw": "wh-question",
    "qo": "open question",
    "qh": "rhetorical question",
    "qrr": "or-question",
    "ny": "yes answer",
    "nn": "no answer",
    "na": "affirmative answer",
    "ng": "negative answer",
    "no": "other answer",
    "h": "hedge",
    "fc": "closing",
    "fp": "opening",
    "bf": "summary",
    "br": "didn't understand",
    "ar": "disagreement",
    "arp": "partial disagreement",
    "bd": "downplayer",
    "ft": "thanks",
    "fa": "apology",
    "ad": "suggestion",
    "oo": "offer",
    "x": "non-verbal",
    "t1": "self-talk",
    "t3": "aside",
    "^2": "completes the other's sentence",
    "^q": "quotation",
}


def act_name(tag: str) -> str:
    """'qy^d' -> 'yes/no question', 'sd(^q)' -> 'statement', '^2' -> ..."""
    tag = tag.strip()
    if tag in ACT_NAMES:
        return ACT_NAMES[tag]
    base = re.split(r"[\^(@]", tag, maxsplit=1)[0].strip().split("_")[0]
    return ACT_NAMES.get(base, "other")


def words_per_turn(turns: list[tuple[str, str]]) -> list[int]:
    return [len(txt.split()) for _, txt in turns]


def make_personas(meta: dict):
    """SwDA metadata row -> (PersonaA, PersonaB, topic, sb_prompt).

    age is derived from talk_day (YYMMDD year) minus birth_year; education via EDU codes.
    The verbatim SB `prompt` is returned for fidelity / future use.
    """
    from prompts.templates import Persona  # lazy import keeps this module standalone

    year = 1900 + int(str(meta["talk_day"])[:2])

    def _age(birth_year: str) -> int:
        try:
            return year - int(birth_year)
        except (ValueError, TypeError):
            return 40

    def _sex(s: str) -> str:
        return "woman" if s.strip().upper().startswith("F") else "man"

    def _edu(code: str) -> str:
        return EDU.get(code.strip(), "an unspecified education")

    a = Persona(
        "ParticipantA",
        _sex(meta["from_caller_sex"]),
        _age(meta["from_caller_birth_year"]),
        _edu(meta["from_caller_education"]),
    )
    b = Persona(
        "ParticipantB",
        _sex(meta["to_caller_sex"]),
        _age(meta["to_caller_birth_year"]),
        _edu(meta["to_caller_education"]),
    )
    # word-wise capitalize, not .title() — .title() mangles apostrophes ("WOMEN'S" ->
    # "Women'S"); .capitalize() per word gives "Women's Roles".
    topic_desc = meta["topic_description"].strip()
    topic = " ".join(w.capitalize() for w in topic_desc.split())
    prompt = _PROMPT_FIXES.get(topic_desc, meta["prompt"].strip())
    return a, b, topic, prompt


# Seven SwDA topic prompts are TRUNCATED in the source metadata, leaving a garbage tail
# (a dangling "ORY" / "FOR EXAMPLE", or a word cut mid-token: "TEN Y[EARS AGO]" -> "TENY",
# "FOR YOU?" -> "YOUY"). These are source-data artifacts, not prompt-design choices, so we
# restore each topic's clean intended instruction — verbatim style (ALL CAPS) preserved so
# P0 stays a faithful replication anchor and P1/P2 naturalize the casing. Keyed by topic.
_PROMPT_FIXES = {
    "PETS": "FIND OUT WHAT KIND OF PETS THE OTHER CALLER HAS.",
    "TRIAL BY JURY": "DISCUSS POSSIBLE CHANGES IN THE WAY TRIALS BY JURY ARE CONDUCTED.",
    "JOB BENEFITS": (
        "WHAT DO YOU CONSIDER THE MOST IMPORTANT BENEFITS BESIDES SALARY IN A "
        "JOB WITH A LARGE ORGANIZATION?  HOW SATISFIED ARE YOU WITH THE CURRENT "
        "BENEFITS OF YOUR JOB?"
    ),
    "DRUG TESTING": (
        "HOW DO YOU FEEL ABOUT THE PRACTICE OF SOME COMPANIES OR GOVERNMENT "
        "AGENCIES TESTING EMPLOYEES OR PROSPECTIVE EMPLOYEES FOR DRUGS?  IS "
        "RANDOM SPOT TESTING JUSTIFIED?  WHAT LIMITS SHOULD THERE BE?"
    ),
    "POLITICS": (
        "DISCUSS ANY RECENT POLITICAL ELECTIONS OR MOVEMENT THAT YOU AND THE OTHER "
        "CALLER CONSIDER INTERESTING OR IMPORTANT."
    ),
    "SOCIAL CHANGE": (
        "DISCUSS RECENT SOCIAL CHANGES.  HOW IS LIFE IN AMERICA DIFFERENT "
        "TODAY COMPARED TO LIVING TEN YEARS AGO?"
    ),
    "WOODWORKING": "PLEASE DISCUSS WOODWORKING.  IS IT A HOBBY FOR YOU?",
}


def iter_conversation_files(root: Path = DATA_ROOT):
    yield from sorted(root.rglob("sw_*.utt.csv"))


def conversation_no_of(csv_path: Path) -> int:
    """sw_0001_4325.utt.csv -> 4325 (the SwDA conversation_no, the metadata join key)."""
    return int(csv_path.stem.split("_")[2].split(".")[0])


# --- P2 few-shot pool (v3) ------------------------------------------------------------
#
# The pool is a committed RECIPE (generation/fewshot_pool.json: conversation ids + turn
# offsets only, NO transcript text — Switchboard is LDC-licensed and never committed). At
# generation time we reconstruct each excerpt's text from the local corpus. Each generated
# conversation draws k excerpts, seeded by its conversation_no, so the draw is deterministic
# and identical across architectures (paired), yet no single excerpt dominates all of P2.

_POOL_PATH = Path(__file__).resolve().parent.parent / "generation" / "fewshot_pool.json"
_FILE_INDEX: dict[int, Path] | None = None
_POOL_CACHE: list[dict] | None = None
_LABEL = {"A": "ParticipantA", "B": "ParticipantB"}


def _file_index() -> dict[int, Path]:
    global _FILE_INDEX
    if _FILE_INDEX is None:
        _FILE_INDEX = {conversation_no_of(fp): fp for fp in iter_conversation_files()}
    return _FILE_INDEX


def load_fewshot_pool() -> list[dict]:
    """Reconstruct the P2 excerpt pool from the committed recipe + local Switchboard.

    Returns [{conversation_no, topic, text}] where text is the rendered excerpt
    (ParticipantA/B labels). Cached. Empty list if the recipe or corpus is absent.
    """
    global _POOL_CACHE
    if _POOL_CACHE is not None:
        return _POOL_CACHE
    pool: list[dict] = []
    try:
        recipe = json.loads(_POOL_PATH.read_text(encoding="utf-8"))
        idx = _file_index()
        for e in recipe["excerpts"]:
            fp = idx.get(e["conversation_no"])
            if fp is None:
                continue
            turns = parse_conversation_tagged(fp)[e["start"] : e["start"] + e["window"]]
            if not turns:
                continue
            text = "\n".join(f"{_LABEL.get(spk, spk)}: {txt}" for spk, txt, _ in turns)
            # P3 only: the same excerpt with each turn's gold act(s) in plain language.
            labeled = "\n".join(
                f"{_LABEL.get(spk, spk)}: {txt}   "
                f"[{', '.join(dict.fromkeys(act_name(t) for t in tags))}]"
                for spk, txt, tags in turns
            )
            pool.append(
                {
                    "conversation_no": e["conversation_no"],
                    "topic": e["topic"],
                    "text": text,
                    "labeled_text": labeled,
                }
            )
    except Exception:
        pool = []
    _POOL_CACHE = pool
    return pool


def fewshot_examples(conversation_no: int, k: int = 2) -> list[dict]:
    """k pool excerpts for one conversation, drawn seeded by conversation_no (deterministic,
    same for that id across all architectures). Pool topics are already disjoint from every
    target topic, so no drawn excerpt can share the generated conversation's topic."""
    pool = load_fewshot_pool()
    if not pool:
        return []
    rng = random.Random(f"fewshot:v3:{conversation_no}")
    return rng.sample(pool, min(k, len(pool)))


def fewshot_example(
    turns: int = 10,
    exclude_ids: set[int] | None = None,
    exclude_topics: set[str] | None = None,
    skip: int = 80,
) -> str:
    """A cleaned Switchboard excerpt for the P2 few-shot style example.

    With `exclude_ids`/`exclude_topics` (from the sampling manifest): the excerpt is the
    first conversation in sorted order that is not a generation target or dev id AND whose
    topic appears nowhere in the target set — so the style example can never be the
    conversation being imitated, nor even share its topic. Deterministic.
    Without them: legacy behavior (skip the first `skip` files), kept for old callers.
    """
    meta = load_metadata() if exclude_topics else {}
    files = list(iter_conversation_files())
    if exclude_ids is None and exclude_topics is None:
        files = files[skip:]
    for fp in files:
        cno = conversation_no_of(fp)
        if exclude_ids and cno in exclude_ids:
            continue
        if exclude_topics and meta.get(cno, {}).get("topic_description") in exclude_topics:
            continue
        convo = parse_conversation(fp)
        if len(convo) >= turns + 2:
            label = {"A": "ParticipantA", "B": "ParticipantB"}
            return "\n".join(f"{label.get(spk, spk)}: {txt}" for spk, txt in convo[:turns])
    return ""


def _validate(n: int) -> None:
    files = list(iter_conversation_files())[:n]
    if not files:
        print(f"No conversation files under {DATA_ROOT}. Did you extract swda.zip?")
        return
    all_wpt: list[int] = []
    n_turns: list[int] = []
    for fp in files:
        turns = parse_conversation(fp)
        all_wpt.extend(words_per_turn(turns))
        n_turns.append(len(turns))
    print(f"conversations parsed : {len(files)}")
    print(f"mean turns/conv      : {statistics.mean(n_turns):.1f}")
    print(f"mean words/turn      : {statistics.mean(all_wpt):.2f}  (paper SB main body ~14)")
    print(f"median words/turn    : {statistics.median(all_wpt):.1f}")
    print("\nsample cleaned turns (first conversation):")
    for spk, txt in parse_conversation(files[0])[:4]:
        print(f"  {spk}: {txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of conversations to check")
    _validate(ap.parse_args().n)
