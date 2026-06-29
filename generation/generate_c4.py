"""
C4 generation: two independent first-person agents using different models.

By default ParticipantA uses Vicuna-13B and ParticipantB uses Mistral-7B-Instruct. Each
participant sees the call only from its own point of view, matching C3's independence
structure while adding model identity separation. Writes each conversation immediately to
data/generated/C4-<prompt>/<id>.json.

Run in tmux on the VM:
    python generation/generate_c4.py --prompt P0 --n 50 --max-turns 50
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from prompts.templates import build_agent                                  # noqa: E402
from analysis.swda import (                                                # noqa: E402
    load_metadata, make_personas, iter_conversation_files, conversation_no_of,
)
from generation.model_utils import (                                       # noqa: E402
    load_model, chat, clean_single_turn, VICUNA, MISTRAL,
)

OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "generated"
LABELS = ("ParticipantA", "ParticipantB")


def target_conversations(n: int, meta: dict) -> list[int]:
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
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--model-a", default=VICUNA)
    ap.add_argument("--model-b", default=MISTRAL)
    ap.add_argument("--stop-at-sentence", action="store_true")
    ap.add_argument("--min-new-tokens", type=int, default=8)
    args = ap.parse_args()

    cond = f"C4-{args.prompt}"
    out_dir = OUT_ROOT / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    ids = target_conversations(args.n, meta)
    todo = [c for c in ids if not (out_dir / f"{c}.json").exists()]
    print(f"[{cond}] target={len(ids)} todo={len(todo)} done={len(ids) - len(todo)}")
    if not todo:
        return

    print(f"loading {args.model_a} for {LABELS[0]}")
    model_a, tok_a = load_model(args.model_a)
    if args.model_b == args.model_a:
        model_b, tok_b = model_a, tok_a
    else:
        print(f"loading {args.model_b} for {LABELS[1]}")
        model_b, tok_b = load_model(args.model_b)

    loaded = {
        LABELS[0]: (model_a, tok_a, args.model_a),
        LABELS[1]: (model_b, tok_b, args.model_b),
    }

    for cno in todo:
        a, b, topic, sb_prompt = make_personas(meta[cno])
        personas = {a.label: a, b.label: b}
        # Peer-greeting seed (matches the paper's GPT4-1 setup): start with a mutual "Hello!"
        # so the conversation opens as two equals, not "is this the ... service?".
        history: list[tuple[str, str]] = [("ParticipantA", "Hello!"), ("ParticipantB", "Hello!")]
        multi = 0
        for i in range(args.max_turns):
            spk = LABELS[i % 2]
            me = personas[spk]
            partner = personas[LABELS[(i + 1) % 2]]
            model, tok, _model_name = loaded[spk]
            messages = build_agent(args.prompt, me, partner, topic, sb_prompt, history)
            raw = chat(
                model, tok, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_at_sentence=args.stop_at_sentence,
                min_new_tokens=args.min_new_tokens,
            )
            turn, ran_past = clean_single_turn(raw, LABELS)
            multi += int(ran_past)
            if not turn:
                break
            history.append((spk, turn))

        rec = {
            "condition": cond,
            "architecture": "C4",
            "prompt_level": args.prompt,
            "models": {a.label: args.model_a, b.label: args.model_b},
            "conversation_no": cno,
            "topic": topic,
            "sb_prompt": sb_prompt,
            "persona_a": vars(a),
            "persona_b": vars(b),
            "turns": history,
            "n_turns": len(history),
            "multi_turn_emissions": multi,
        }
        (out_dir / f"{cno}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  saved {cno}  turns={len(history)}  multi_turn_emissions={multi}")


if __name__ == "__main__":
    main()
