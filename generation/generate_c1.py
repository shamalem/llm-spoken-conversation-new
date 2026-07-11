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
    # C1 generates the WHOLE ~30-turn conversation in one generate() call (up to
    # max-new-tokens), unlike C2/C3/C4 which call chat() once per short turn. chat()'s
    # defaults (repetition_penalty=1.2, no_repeat_ngram_size=3) were tuned for those short
    # per-turn calls; a hard "never repeat any 3-gram" ban over ~2000 tokens of natural
    # dialogue is nearly unsatisfiable (turns reuse words/phrases constantly) and was
    # forcing the model to hit EOS almost immediately, producing 1-2 line fragments instead
    # of full conversations. Use the same no-repeat-ngram=6 the C2/C3/C4 family settled on
    # after their own repetition-drift testing: loose enough to allow normal short-phrase
    # reuse ("I think that", "do you think"), but still bans a genuine verbatim-sentence
    # loop (the pre-fix C1 pathology repeated whole clauses, well over 6 tokens). The soft
    # repetition_penalty stays on top as a second line of defense against loops that stay
    # just under the n-gram ban's radar.
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=6,
                    help="bans exact n-gram repeats; 0 disables (risks 3-gram-style breakage "
                         "if set too low over a whole conversation)")
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
