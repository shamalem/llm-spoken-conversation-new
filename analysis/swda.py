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
import re
import statistics
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "switchboard" / "swda"

# SwDA caller education codes -> text. TODO: verify exact labels against SwDA docs.
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
    t = re.sub(r"<+[^>]*>+", " ", t)      # <beep>, <<long pause>>
    t = re.sub(r"\{[A-Z]\s", " ", t)       # opening {D {F {C {E {A ...
    t = t.replace("}", " ")
    for ch in "[]+#":
        t = t.replace(ch, " ")
    t = re.sub(r"-?/", " ", t)             # slash-unit and -/ interruption
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.?!])", r"\1", t)   # tidy space before punctuation
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

    a = Persona("ParticipantA", _sex(meta["from_caller_sex"]),
                _age(meta["from_caller_birth_year"]), _edu(meta["from_caller_education"]))
    b = Persona("ParticipantB", _sex(meta["to_caller_sex"]),
                _age(meta["to_caller_birth_year"]), _edu(meta["to_caller_education"]))
    topic = meta["topic_description"].strip().title()
    return a, b, topic, meta["prompt"].strip()


def iter_conversation_files(root: Path = DATA_ROOT):
    yield from sorted(root.rglob("sw_*.utt.csv"))


def conversation_no_of(csv_path: Path) -> int:
    """sw_0001_4325.utt.csv -> 4325 (the SwDA conversation_no, the metadata join key)."""
    return int(csv_path.stem.split("_")[2].split(".")[0])


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
