"""K3.1 chat I/O at the token level: prompt IDs (segment-wise, as the model was trained) and
parsing of generated token IDs into reasoning / answer, plus reasoning-health metrics.

Generation output layout (after the prompt, which ends in <|open|>think<|sep|>):
  REASONING <|close|>think<|sep|><|open|>response<|sep|> ANSWER <|close|>response<|sep|>
  <|close|>message<|sep|><|end_of_msg|>
Non-thinking mode (prompt_ids(thinking=False), prompt ends in <|open|>response<|sep|>):
  ANSWER <|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def prompt_ids(tok, question: str | list[dict], effort: str = "max", system: str | None = None,
               tools: list[dict] | None = None, thinking: bool = True) -> list[int]:
    """question: a user string, or a full message list. tools: OpenAI-style tool specs (the K3
    template renders them as a tool-declare system message). thinking=False: the template's non-thinking mode
    (no effort message, no think channel; the prompt ends in <|open|>response<|sep|> and the first generated
    token is the answer); `effort` is ignored."""
    msgs = question if isinstance(question, list) else [{"role": "user", "content": question}]
    msgs = ([{"role": "system", "content": system}] if system else []) + msgs
    kw = {"thinking_effort": effort} if thinking else {"thinking": False}
    ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=True, **kw)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


class Markers:
    def __init__(self, tok):
        c = tok.convert_tokens_to_ids
        self.open, self.close, self.sep, self.eom = c("<|open|>"), c("<|close|>"), c("<|sep|>"), c("<|end_of_msg|>")
        self.think = tok.encode("think", allow_special_tokens=False)
        self.response = tok.encode("response", allow_special_tokens=False)

    def close_think(self):
        return [self.close, *self.think, self.sep]

    def open_response(self):
        return [self.open, *self.response, self.sep]

    def close_response(self):
        return [self.close, *self.response, self.sep]


def _find(seq, sub, start=0):
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return -1


@dataclass
class Parsed:
    reasoning_ids: list[int]
    answer_ids: list[int]
    reasoning_closed: bool
    answer_closed: bool


def parse(ids: list[int], m: Markers, thinking: bool = True) -> Parsed:
    if not thinking:  # generation starts inside the response channel
        cr = _find(ids, m.close_response())
        return Parsed([], ids[:cr] if cr >= 0 else ids, True, cr >= 0)
    ct = _find(ids, m.close_think())
    if ct < 0:
        return Parsed(ids, [], False, False)
    orr = _find(ids, m.open_response(), ct)
    a0 = orr + len(m.open_response()) if orr >= 0 else ct + len(m.close_think())
    cr = _find(ids, m.close_response(), a0)
    return Parsed(ids[:ct], ids[a0:cr] if cr >= 0 else ids[a0:], True, cr >= 0)


def repetition_rate(ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams that repeat an earlier n-gram (0 = none; loops push it toward 1)."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def off_language_fraction(text: str, lang: str) -> float:
    """Share of letters in the 'other' script: CJK in an EN answer, Latin in a ZH answer."""
    cjk, lat = len(CJK.findall(text)), len(LATIN.findall(text))
    tot = cjk + lat
    if tot == 0:
        return 0.0
    return (cjk if lang == "en" else lat) / tot


def health(parsed: Parsed, finish_reason: str, answer_text: str, lang: str) -> dict:
    return {
        "reasoning_closed": parsed.reasoning_closed,
        "answer_closed": parsed.answer_closed,
        "hit_max_tokens": finish_reason == "length",
        "reasoning_tokens": len(parsed.reasoning_ids),
        "answer_tokens": len(parsed.answer_ids),
        "repetition": repetition_rate(parsed.reasoning_ids + parsed.answer_ids),
        "off_language": off_language_fraction(answer_text, lang),
    }
