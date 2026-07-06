"""
Prompt and message builders for the conversation-generation experiment.

Architectures
-------------
C1  all-at-once          : one model writes the entire dialogue in a single call.
C2  turn-by-turn single  : one model sees the whole transcript and writes the next turn for
                           the named speaker — replicates the paper's GPT4-1 setup.
C3  two agents, same     : two independent first-person sessions of the SAME model.
C4  two agents, different : two independent first-person sessions of DIFFERENT models.

Prompt levels
-------------
P0  basic / replication : faithfully follows the paper's BASIC prompt (GPT4-1 wording —
                          "act like a {persona}", topic = the verbatim SB instruction, ~50
                          turns). No marker forcing, no scripted openings/closings, no guard.
P1  spoken intervention : OUR prompt — short natural turns, natural ending, peer/anti-assistant
                          guard. No marker forcing (keeps marker metrics non-circular).
P2  few-shot            : P1 + ONE real Switchboard excerpt (from a DIFFERENT topic) shown as
                          a style example. NOTE: for P2, turn-length and conceptual-alignment
                          effects are the valid story; lexical alignment and specific marker
                          rates are partly example-driven and must be read as confounded.

We deliberately do NOT replicate the paper's *enhanced* prompts (which order "use okay/oh/
uh-huh" and script openings/closings, then measure exactly those). The generators seed every
turn-by-turn dialogue with a mutual "Hello!" opening, as the paper did.
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


def render_transcript(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {txt}" for spk, txt in history)


# --- shared prompt pieces ------------------------------------------------------------

def _length_clause() -> str:
    return "The conversation will have about 30 turns of talk; do not end it too early."


def _p1_style() -> str:
    return (
        "Speak naturally and informally, the way people actually talk out loud on the phone. "
        "Keep each turn short — usually 1 to 3 sentences. Let the conversation end naturally "
        "when you both feel it is finished; do not pad or stretch it out."
    )


def _peer_guard() -> str:
    return (
        "You are an ordinary member of the public — NOT an AI assistant, agent, operator, or a "
        "representative of any company or service. The other caller is also an ordinary person; "
        "the two of you are equals who were each asked to phone a stranger and chat. Never say "
        "things like 'how can I help you' or 'is this the ... service', and do not offer help "
        "or recommendations as if it were your job — just share your own views and ask about "
        "theirs, the way two regular people would."
    )


def _topic_clause(topic: str, sb_prompt: str | None, level: str) -> str:
    """How the Switchboard task is presented (plain for P0, reframed as peer goal otherwise)."""
    instruction = sb_prompt or topic
    if level == "P0":
        return f"The topic of the conversation is: {instruction}"
    return (
        f"You were both asked to chat about this topic: {topic}. Shared discussion goal (talk "
        f"it over together as equals, comparing your own experiences and opinions): {instruction}"
    )


_FEWSHOT_EXAMPLE: str | None = None


def _fewshot_block(level: str) -> str:
    """For P2 only: a real Switchboard excerpt (different topic) shown as a style example."""
    global _FEWSHOT_EXAMPLE
    if level != "P2":
        return ""
    if _FEWSHOT_EXAMPLE is None:
        try:
            from analysis.swda import fewshot_example
            _FEWSHOT_EXAMPLE = fewshot_example()
        except Exception:
            _FEWSHOT_EXAMPLE = ""
    if not _FEWSHOT_EXAMPLE:
        return ""
    return (
        "\n\nHere is an example of a real, natural telephone conversation between two strangers "
        "on a DIFFERENT topic, to show the spoken style (short, casual back-and-forth):\n"
        f"{_FEWSHOT_EXAMPLE}\n(End of example.)\n"
    )


# --- C1: all at once -----------------------------------------------------------------

def build_c1(prompt_level: str, a: Persona, b: Persona, topic: str,
             sb_prompt: str | None = None) -> list[dict]:
    if prompt_level == "P0":
        prompt = (
            "Write the log of a telephone conversation between two people who do not know each "
            f"other and have equal roles in the discussion. {a.label} is {a.describe()}. "
            f"{b.label} is {b.describe()}. {_topic_clause(topic, sb_prompt, 'P0')} "
            f"{_length_clause()} The log starts with:\n{a.label}: Hello!\n{b.label}: Hello!\n"
            "Each line is one turn beginning with the speaker's label and a colon. Write the "
            "complete conversation, continuing from those greetings."
        )
    else:  # P1 / P2
        prompt = (
            "Write a realistic telephone conversation between two ordinary people who do not "
            f"know each other. {a.label} is {a.describe()}. {b.label} is {b.describe()}. "
            f"{_topic_clause(topic, sb_prompt, 'P1')} {_p1_style()} {_peer_guard()}"
            f"{_fewshot_block(prompt_level)}\n"
            f"It opens with:\n{a.label}: Hello!\n{b.label}: Hello!\n"
            f"Write the full conversation, one turn per line as '{a.label}: ...' / '{b.label}: ...'."
        )
    return [{"role": "user", "content": prompt}]


# --- C2: turn-by-turn, single model sees the whole script (replicates GPT4-1) --------

def build_c2(prompt_level: str, a: Persona, b: Persona, topic: str,
             sb_prompt: str | None,
             history: list[tuple[str, str]], next_speaker: str) -> list[dict]:
    transcript = render_transcript(history) if history else f"{a.label}: Hello!\n{b.label}: Hello!"
    me = {a.label: a, b.label: b}[next_speaker]
    if prompt_level == "P0":
        prompt = (
            f"Act like {me.describe()} in a phone conversation with someone you do not know. "
            f"{_topic_clause(topic, sb_prompt, 'P0')} {_length_clause()} "
            f"The conversation log so far is:\n'''{transcript}'''\n"
            f"Each line is one turn; the speaker label precedes the colon. Your label is "
            f"{next_speaker}. Your response is the next turn — respond to the last line but take "
            "the whole log into account. Do not include more than one turn, and do not write a "
            "speaker label."
        )
    else:  # P1 / P2
        prompt = (
            "You are writing a realistic phone conversation between two ordinary people who do "
            f"not know each other. {a.label} is {a.describe()}; {b.label} is {b.describe()}. "
            f"{_topic_clause(topic, sb_prompt, 'P1')} {_p1_style()} {_peer_guard()}"
            f"{_fewshot_block(prompt_level)}\n"
            f"Conversation so far:\n{transcript}\n\n"
            f"Write ONLY {next_speaker}'s next single turn — just the utterance, no label."
        )
    return [{"role": "user", "content": prompt}]


# --- C3 / C4: independent first-person agents ----------------------------------------

def build_agent(prompt_level: str, me: Persona, partner: Persona, topic: str,
                sb_prompt: str | None,
                history: list[tuple[str, str]]) -> list[dict]:
    """Build the message list from `me`'s point of view (used for both C3 and C4)."""
    if prompt_level == "P0":
        system = (
            f"Act like {me.describe()} in a telephone conversation with someone you do not know. "
            f"{_topic_clause(topic, sb_prompt, 'P0')} {_length_clause()} "
            "Reply with only your next single turn of talk — no speaker label, one turn only."
        )
    else:  # P1 / P2
        system = (
            f"You are {me.describe()} on a telephone call with an ordinary stranger you just met. "
            f"{_topic_clause(topic, sb_prompt, 'P1')} {_p1_style()} {_peer_guard()}"
            f"{_fewshot_block(prompt_level)} "
            "Reply with only what you say next, as a single spoken turn — no speaker label."
        )
    messages = [{"role": "system", "content": system}]
    if not history:
        messages.append({"role": "user", "content": "(The phone connects — your partner is on the line.)"})
        return messages
    if history[0][0] == me.label:
        messages.append({"role": "user", "content": "(The call is already underway.)"})
    for spk, txt in history:
        role = "assistant" if spk == me.label else "user"
        messages.append({"role": role, "content": txt})
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "(Your partner is quiet — continue if you wish.)"})
    return messages
