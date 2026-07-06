"""
Model loading + chat-generation helpers for the VM (Vicuna / Mistral).

4-bit quantized load to fit the V100 (16 GB). Uses each model's chat template so prompt
formatting is correct, with a Vicuna-v1.5 fallback if no template is shipped.

Requires (VM only): torch, transformers, accelerate, bitsandbytes.
"""

from __future__ import annotations

import re

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria,
    StoppingCriteriaList,
)

VICUNA = "lmsys/vicuna-13b-v1.5-16k"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.2"


def load_model(name: str):
    """Load a 4-bit quantized causal LM + tokenizer onto the GPU."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, device_map="auto", use_safetensors=True
    )
    # Some chat checkpoints ship sampling fields with do_sample=False, which triggers
    # transformers warnings. Keep model defaults neutral; pass sampling choices per call.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()
    return model, tok


class SentenceEndStoppingCriteria(StoppingCriteria):
    """Stop once a short generated turn reaches a sentence boundary."""

    def __init__(self, tok, prompt_len: int, min_new_tokens: int = 8):
        self.tok = tok
        self.prompt_len = prompt_len
        self.min_new_tokens = min_new_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_ids = input_ids[0][self.prompt_len:]
        if new_ids.shape[-1] < self.min_new_tokens:
            return False
        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        if not text:
            return False
        return text.endswith((".", "?", "!"))


@torch.inference_mode()
def chat(model, tok, messages, max_new_tokens=512, temperature=0.8, top_p=0.95,
         do_sample=True, stop_at_sentence=False, min_new_tokens=8,
         repetition_penalty=1.2, no_repeat_ngram_size=3) -> str:
    """messages: list of {role, content}. Returns the assistant's text completion."""
    try:
        encoded = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception:
        # Vicuna v1.5 may not ship a chat template — use its USER/ASSISTANT format.
        encoded = tok(_vicuna_format(messages), return_tensors="pt")

    if isinstance(encoded, torch.Tensor):
        model_inputs = {"input_ids": encoded.to(model.device)}
    else:
        model_inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in encoded.items()
        }
    input_len = model_inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        # min_new_tokens forbids the end-of-sequence token before this many tokens are
        # generated — this is what actually prevents 1-word "fragment" turns. It used to be
        # passed into this function but only fed the sentence-stop criteria, never generate(),
        # so there was no real floor on turn length. Now it is enforced.
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tok.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    if stop_at_sentence:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
            SentenceEndStoppingCriteria(tok, input_len, min_new_tokens=min_new_tokens)
        ])

    out = model.generate(**model_inputs, **gen_kwargs)
    new_tokens = out[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def clean_single_turn(text: str, labels=("ParticipantA", "ParticipantB")) -> tuple[str, bool]:
    """Return the first utterance and whether the model ran on past a single turn.

    Truncates at the first speaker/role marker. Besides the participant labels, this also
    catches chat-role residue the agent path leaks when Vicuna rambles into a whole fake
    dialogue (line-initial USER:, ASSISTANT:, and degraded 4-bit variants ASSISTATIVE: /
    ASSISTY:, plus HUMAN:/AI:/SYSTEM:/BOT:). A True flag here means the model did NOT keep
    to one turn — which we count as a multi-turn emission.
    """
    label_alt = "|".join(re.escape(label) for label in labels)
    # Tolerant participant label: catches degraded 4-bit variants like "ParticipantsA:"
    # (stray 's') and "Participant A:" (space) that the exact label misses. The C2
    # single-model path — where one model writes both speakers — leaks these often.
    fuzzy_label = r"Participants?\s*[AB]"
    marker_re = re.compile(
        rf"(?:\b(?:{label_alt}|{fuzzy_label})\s*:)"
        rf"|(?:(?:^|\n)\s*(?:USER|ASSIST\w*|HUMAN|AI|SYSTEM|BOT)\s*:)",
        re.I,
    )
    t = text.strip()
    m = re.match(rf"\s*(?:{label_alt}|{fuzzy_label})\s*:\s*", t, re.I)
    if m:
        t = t[m.end():]
    nxt = marker_re.search(t)
    ran_past = nxt is not None
    if ran_past:
        t = t[:nxt.start()]
    return t.strip().strip('"'), ran_past


# Farewell / sign-off cues used to end a conversation naturally (see generate_c3.py loop).
_CLOSING_RE = re.compile(
    r"\b(?:good-?bye|bye-?bye|bye|take care|farewell|see you(?: around| soon| later| next time)?|"
    r"talk (?:to you )?(?:soon|later)|catch you later|until next time|happy chatting|"
    r"(?:nice|great|lovely|a pleasure) (?:talking|chatting|speaking)(?: (?:to|with) you)?|"
    r"enjoy (?:the rest of )?your day|"
    r"have a (?:great|good|nice|wonderful|lovely|fantastic) (?:day|one|time|evening|weekend))\b",
    re.I,
)

# Assistant / template / end-of-session residue the model emits once it drops out of the
# conversation (observed in C3 tails: "[End of Response]", "*Session closed.*", code fences,
# "Here's a summary", stray role tokens, and garbage like "** | **" / "-> |" / "V V V").
_META_RE = re.compile(
    r"(?:"
    r"\[(?:end of|turn|tur|t\b|this|do you|assist|closed|/)"
    r"|\*{1,}\s*(?:conversation|chat|session|closed|ended|assistance|connection|end of)"
    r"|here'?s (?:the |a )?(?:quick )?(?:summary|recap)"
    r"|this (?:concludes|conversation (?:covered|concludes|ends))"
    r"|```"
    r"|(?:^|\n)\s*(?:USER|ASSISTANT|ASSISTMENT|SYSTEM|BOT)\b"
    r"|\*\*\s*\||\|\s*->|->\s*\||\bV\s+V\s+V\b"
    r"|(?:^|\n)\s*---\s*(?:$|\n)"
    r")",
    re.I | re.M,
)


def strip_meta_artifacts(text: str) -> str:
    """Cut a turn at the first assistant/template/end-of-conversation artifact.

    Keeps the clean leading part so chatbot residue never enters the transcript; if the whole
    turn was such residue this returns "" (the caller then ends the conversation).
    """
    m = _META_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip().strip('"').strip()


def looks_like_closing(text: str) -> bool:
    """True if a turn contains a farewell / sign-off (used to stop the conversation)."""
    return bool(_CLOSING_RE.search(text))


def _vicuna_format(messages) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    parts = [system] if system else []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"ASSISTANT: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

