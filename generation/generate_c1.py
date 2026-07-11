"""
C1 (all-at-once) pilot generation with Vicuna-13B.

Each conversation is written to disk IMMEDIATELY as
    data/generated/C1-<prompt>/<conversation_no>.json
and already-present ids are skipped, so the job is resumable after an SSH/kernel drop.
Run inside tmux on the VM:

    python generation/generate_c1.py --prompt P0 --n 10     # Phase-1 pilot
    python generation/generate_c1.py --prompt P1 --n 50     # Phase-2 scale

Conversations are matched to the first N SwDA conversations (topic + demographics) via the
verified `conversation_no` join.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from prompts.templates import build_c1                                   # noqa: E402
from analysis.swda import (                                              # noqa: E402
    load_metadata, make_personas, iter_conversation_files, conversation_no_of,
)
from generation.model_utils import load_model, chat, VICUNA             # noqa: E402

OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "generated"


def target_conversations(n: int, meta: dict) -> list[int]:
    """First N SwDA conversation_no's that exist in the metadata (the match set)."""
    out: list[int] = []
    for fp in iter_conversation_files():
        cno = conversation_no_of(fp)
        if cno in meta:
            out.append(cno)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="P0", choices=["P0", "P1", "P2"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    # C1 generates the WHOLE ~30-turn conversation in one generate() call, unlike C2/C3/C4
    # which call chat() once per short turn. It must repeat the speaker labels ("ParticipantA:"
    # / "ParticipantB:") on EVERY line. chat()'s per-turn defaults (repetition_penalty=1.2,
    # no_repeat_ngram_size=3) — added on 2026-07-01 to stop the C3 two-agent loops — punish
    # exactly that required label repetition, so after a turn or two the model can no longer
    # cheaply start a new labelled line and hits EOS instead → the 1-2 line fragments.
    # The early C1 pilots (which produced full 250-400-word conversations) ran with BOTH of
    # these OFF; that is the known-good config, restored here. NOTE: loosening only the n-gram
    # ban to 6 was tried (commit 32ca994) and did NOT fix it — because repetition_penalty was
    # still on. C1 all-at-once does not fall into the verbatim-loop failure mode that motivated
    # these controls (that was C3's two-agents-feeding-each-other spiral), so it does not need
    # them. If a regenerated C1 conversation ever shows a real verbatim loop, add a MILD penalty
    # (e.g. 1.05) rather than reinstating the 1.15/6 that truncates.
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = off (the known-good C1 setting). >1.0 penalizes the repeated "
                         "speaker labels C1 needs and truncates the conversation.")
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="0 = off (the known-good C1 setting). A hard n-gram ban over a whole "
                         "2000-token conversation fights natural phrase reuse and truncates it.")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="output root; point at a separate dir to avoid overwriting existing data")
    args = ap.parse_args()

    cond = f"C1-{args.prompt}"
    out_dir = pathlib.Path(args.out_root) / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    ids = target_conversations(args.n, meta)
    todo = [c for c in ids if not (out_dir / f"{c}.json").exists()]
    print(f"[{cond}] target={len(ids)} todo={len(todo)} done={len(ids) - len(todo)}")
    if not todo:
        return

    model, tok = load_model(VICUNA)
    for cno in todo:
        a, b, topic, sb_prompt = make_personas(meta[cno])
        messages = build_c1(args.prompt, a, b, topic, sb_prompt)
        text = chat(
            model, tok, messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram,
        )
        rec = {
            "condition": cond,
            "architecture": "C1",
            "prompt_level": args.prompt,
            "model": VICUNA,
            "conversation_no": cno,
            "topic": topic,
            "sb_prompt": sb_prompt,
            "persona_a": vars(a),
            "persona_b": vars(b),
            "raw_output": text,
        }
        (out_dir / f"{cno}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  saved {cno}  ({len(text.split())} words)")


if __name__ == "__main__":
    main()
