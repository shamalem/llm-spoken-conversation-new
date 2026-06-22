"""
Prompt and message builders for the conversation-generation experiment.

Architectures
-------------
C1  all-at-once           : one model writes the entire dialogue in a single call.
C2  turn-by-turn single   : one model sees the shared transcript and writes the next
                            turn for whichever speaker is named ("scriptwriter" — it
                            knows it authors both sides).
C3  two agents, same      : two independent first-person sessions of the SAME model.
C4  two agents, different  : two independent first-person sessions of DIFFERENT models.

In C3/C4 each agent only ever sees the conversation from its own point of view: its own
past turns are `assistant` messages, the partner's turns arrive as `user` messages.
Neither agent sees a god's-eye "script" of both sides. This is the key difference from C2
and is how we isolate the single-author effect.

Prompt levels
-------------
P0  basic   : replicates the paper's basic prompt (~50-turn target, no early end).
P1  spoken  : short turns, natural ending, NO coordination-marker forcing.

All builders return a list of {"role", "content"} dicts, to be rendered with
`tokenizer.apply_chat_template(...)` so formatting matches each model.

NOTE: opening cues and strict user/assistant alternation may need light tuning after the
Phase-1 pilot, once we see how Vicuna/Mistral actually behave with these templates.
"""

from dataclasses import dataclass


@dataclass
class Persona:
    label: str        # "ParticipantA" / "ParticipantB"
    gender: str       # "man" / "woman"
    age: int
    education: str    # e.g. "a college degree"

    def describe(self) -> str:
        return f"a {self.age}-year-old {self.gender} with {self.education}"


def _style(prompt_level: str) -> str:
    """Prompt-level style instruction shared across architectures."""
    if prompt_level == "P0":
        return (
            "Have the conversation as this person would on the telephone. "
            "The conversation should last about 50 turns; do not end it too early."
        )
    if prompt_level == "P1":
        return (
            "Speak naturally and informally, the way people actually talk out loud on the "
            "phone. Keep each turn short — usually 1 to 3 sentences. Let the conversation "
            "end naturally when you both feel it is finished; do not pad or stretch it out."
        )
    raise ValueError(f"unknown prompt level: {prompt_level!r}")


def _task(topic: str, sb_prompt: str | None = None) -> str:
    """Render the Switchboard task without turning the topic into a service call."""
    if sb_prompt:
        return (
            f"Topic title: {topic}\n"
            f"Verbatim instruction given to the callers: {sb_prompt}"
        )
    return f"Topic title: {topic}"


def _genre_guard() -> str:
    return (
        "Treat this as an ordinary conversation between two assigned telephone callers. "
        "Do not turn the topic into a business, help desk, interview, survey, or customer "
        "service call unless the instruction explicitly says so."
    )


def render_transcript(history: list[tuple[str, str]]) -> str:
    """history: list of (speaker_label, text) -> 'ParticipantA: ...\\nParticipantB: ...'."""
    return "\n".join(f"{spk}: {txt}" for spk, txt in history)


# --- C1: all at once -----------------------------------------------------------------

def build_c1(prompt_level: str, a: Persona, b: Persona, topic: str,
             sb_prompt: str | None = None) -> list[dict]:
    system = (
        "You write realistic transcripts of telephone conversations between two people "
        f"who do not know each other. {_style(prompt_level)} {_genre_guard()}"
    )
    user = (
        f"Write a complete telephone conversation for this Switchboard calling task:\n"
        f"{_task(topic, sb_prompt)}\n"
        f"{a.label} is {a.describe()}. {b.label} is {b.describe()}.\n"
        f"Format every turn on its own line as '{a.label}: ...' or '{b.label}: ...'."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --- C2: turn-by-turn, single model sees the whole script ----------------------------

def build_c2(prompt_level: str, a: Persona, b: Persona, topic: str,
             sb_prompt: str | None,
             history: list[tuple[str, str]], next_speaker: str) -> list[dict]:
    system = (
        "You write realistic transcripts of telephone conversations between two people who "
        f"do not know each other. {a.label} is {a.describe()}. {b.label} is {b.describe()}. "
        f"{_style(prompt_level)} {_genre_guard()}\n"
        f"Switchboard calling task:\n{_task(topic, sb_prompt)}"
    )
    transcript = render_transcript(history) if history else "(the conversation has not started yet)"
    user = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Write ONLY the next single turn, spoken by {next_speaker}. "
        "Reply with just the utterance — no speaker label, no quotation marks."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --- C3 / C4: independent first-person agents ----------------------------------------

def build_agent(prompt_level: str, me: Persona, partner: Persona, topic: str,
                sb_prompt: str | None,
                history: list[tuple[str, str]]) -> list[dict]:
    """Build the message list from `me`'s point of view (used for both C3 and C4).

    `me`'s own past turns become `assistant` messages; `partner`'s turns become `user`
    messages. The model continues by producing `me`'s next turn. Because speakers strictly
    alternate, the rendered history alternates user/assistant cleanly.
    """
    system = (
        f"You are {me.describe()}. You are on a telephone call with someone you have just "
        f"met and do not know. {_style(prompt_level)} {_genre_guard()}\n"
        f"Switchboard calling task:\n{_task(topic, sb_prompt)}\n"
        "Reply with only what you say next, as a single spoken turn — no speaker label."
    )
    messages = [{"role": "system", "content": system}]
    if not history:
        # `me` is opening the call.
        messages.append({"role": "user", "content": "(The phone connects — your partner is on the line.)"})
        return messages
    if history[0][0] == me.label:
        messages.append({"role": "user", "content": "(The call is already underway.)"})
    for spk, txt in history:
        role = "assistant" if spk == me.label else "user"
        messages.append({"role": role, "content": txt})
    if messages[-1]["role"] == "assistant":
        # Last turn was mine; nudge a continuation so the model still has a user cue.
        messages.append({"role": "user", "content": "(Your partner is quiet — continue if you wish.)"})
    return messages
