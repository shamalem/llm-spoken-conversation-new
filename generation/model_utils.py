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
        # Vicuna v1.5 may not ship a chat template -- use its USER/ASSISTANT format.
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
    to one turn -- which we count as a multi-turn emission.
    """
    label_alt = "|".join(re.escape(label) for label in labels)
    marker_re = re.compile(
        rf"(?:\b(?:{label_alt})\s*:)"
        rf"|(?:(?:^|\n)\s*(?:USER|ASSIST\w*|HUMAN|AI|SYSTEM|BOT)\s*:)",
        re.I,
    )
    t = text.strip()
    m = re.match(rf"\s*(?:{label_alt})\s*:\s*", t, re.I)
    if m:
        t = t[m.end():]
    nxt = marker_re.search(t)
    ran_past = nxt is not None
    if ran_past:
        t = t[:nxt.start()]
    return t.strip().strip('"'), ran_past


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
