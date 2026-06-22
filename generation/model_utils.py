"""
Model loading + chat-generation helpers for the VM (Vicuna / Mistral).

4-bit quantized load to fit the V100 (16 GB). Uses each model's chat template so prompt
formatting is correct, with a Vicuna-v1.5 fallback if no template is shipped.

Requires (VM only): torch, transformers, accelerate, bitsandbytes.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


@torch.inference_mode()
def chat(model, tok, messages, max_new_tokens=512, temperature=0.8, top_p=0.95,
         do_sample=True) -> str:
    """messages: list of {role, content}. Returns the assistant's text completion."""
    try:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except Exception:
        # Vicuna v1.5 may not ship a chat template — use its USER/ASSISTANT format.
        input_ids = tok(_vicuna_format(messages), return_tensors="pt").input_ids.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tok.eos_token_id,
    }
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    out = model.generate(input_ids, **gen_kwargs)
    new_tokens = out[0][input_ids.shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def clean_single_turn(text: str, labels=("ParticipantA", "ParticipantB")) -> tuple[str, bool]:
    """Return the first utterance and whether the model started another labeled turn."""
    label_alt = "|".join(re.escape(label) for label in labels)
    label_re = re.compile(rf"\b(?:{label_alt})\s*:", re.I)
    t = text.strip()
    m = re.match(rf"\s*(?:{label_alt})\s*:\s*", t, re.I)
    if m:
        t = t[m.end():]
    nxt = label_re.search(t)
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
