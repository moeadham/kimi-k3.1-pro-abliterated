"""Rebuild hosted K3.1 transcripts as exact token sequences for replay-prefill capture.

K3.1 renders chats with encoding_k3.py (XTML). Each segment is tokenized separately, so a
transcript (question, reasoning, answer, effort) maps to token IDs as:

  [thinking-effort system msg][user msg]<|open|>message role="assistant"<|sep|><|open|>think<|sep|>
  REASONING <|close|>think<|sep|><|open|>response<|sep|> ANSWER <|close|>response<|sep|>
  <|close|>message<|sep|><|end_of_msg|>

The prompt (add_generation_prompt=True) is an exact prefix of the full rendering, which the
builder asserts. The default effort is "max", the same as the tokenizer's own default.

Positions: the residual state at absolute index t predicts token t+1, so the state that
PRODUCES the token at index i is i-1. Capture-position sets (absolute indices):
  prompt_end      last K prompt tokens (ends with <|open|>think<|sep|>; the last one produces
                  reasoning token 0)
  reasoning_head  states producing the first N reasoning tokens
  onset           states producing the Jev onset sentence (a char span of the reasoning)
  answer_head     states producing the first N answer tokens (after <|open|>response<|sep|>)
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field

REPO = "moonshotai/Kimi-K3.1-Pro"


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)


def _ids(tok, messages, add_generation_prompt, effort, thinking: bool = True):
    kw = {"thinking_effort": effort} if thinking else {"thinking": False}
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


@dataclass
class Replay:
    ids: list[int]
    prompt_len: int
    reasoning: tuple[int, int]  # absolute [start, end) token span of the reasoning text
    answer: tuple[int, int]  # absolute [start, end) token span of the answer text
    reasoning_char_starts: list[int] = field(default_factory=list)  # char offset of each reasoning token
    answer_char_starts: list[int] = field(default_factory=list)  # char offset of each answer token

    def produced_by(self, start: int, end: int) -> list[int]:
        """States that produce tokens [start, end)."""
        return list(range(max(start - 1, 0), end - 1))

    def reasoning_tokens_for_chars(self, c0: int, c1: int) -> tuple[int, int]:
        """Absolute token span covering reasoning characters [c0, c1)."""
        s = bisect.bisect_right(self.reasoning_char_starts, c0) - 1
        e = bisect.bisect_left(self.reasoning_char_starts, c1)
        s = max(s, 0)
        return self.reasoning[0] + s, self.reasoning[0] + max(e, s + 1)

    def answer_token_for_char(self, c: int) -> int:
        """Absolute index of the answer token containing answer character c."""
        i = bisect.bisect_right(self.answer_char_starts, c) - 1
        return self.answer[0] + max(i, 0)

    def positions(self, *, prompt_k: int = 4, head_n: int = 16, onset_chars: tuple[int, int] | None = None,
                  comply_chars: tuple[int, int] | None = None) -> dict:
        r0, r1 = self.reasoning
        a0, a1 = self.answer
        out = {
            "prompt_end": list(range(self.prompt_len - prompt_k, self.prompt_len)),
            "reasoning_head": self.produced_by(r0, min(r0 + head_n, r1)) if r1 > r0 else [],
            "answer_head": self.produced_by(a0, min(a0 + head_n, a1)) if a1 > a0 else [],
        }
        if onset_chars is not None and r1 > r0:
            out["onset"] = self.produced_by(*self.reasoning_tokens_for_chars(*onset_chars))
        if comply_chars is not None and r1 > r0:  # compliance onset (answered responses)
            out["comply_onset"] = self.produced_by(*self.reasoning_tokens_for_chars(*comply_chars))
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _char_starts(tok, ids: list[int], text: str) -> list[int]:
    """Char offset at which each token starts (tokens can split multi-byte chars; such a
    token is assigned the char containing its first byte)."""
    enc = tok.model  # tiktoken.Encoding
    byte_starts, b = [], 0
    for t in ids:
        byte_starts.append(b)
        b += len(enc.decode_single_token_bytes(t))
    raw = text.encode("utf-8")
    if b != len(raw):
        raise ValueError(f"token bytes {b} != text bytes {len(raw)}")
    # map byte offset -> char offset
    char_of_byte, c = [], 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        char_of_byte.extend([c] * n)
        c += 1
    char_of_byte.append(c)
    return [char_of_byte[x] for x in byte_starts]


def build(tok, question: str, reasoning: str, answer: str, effort: str | None = "max",
          thinking: bool = True) -> Replay:
    user = [{"role": "user", "content": question}]
    if not thinking:  # no think channel: the prompt ends in <|open|>response<|sep|>, the answer follows
        prompt = _ids(tok, user, True, effort, thinking=False)
        full = _ids(tok, user + [{"role": "assistant", "content": answer}], False, effort, thinking=False)
        if full[: len(prompt)] != prompt:
            raise ValueError("prompt is not a prefix of the full rendering")
        a_ids = tok.encode(answer, allow_special_tokens=False) if answer else []
        P = len(prompt)
        if full[P:P + len(a_ids)] != a_ids:
            raise ValueError("answer tokens not found where expected")
        return Replay(ids=full, prompt_len=P, reasoning=(P, P), answer=(P, P + len(a_ids)),
                      answer_char_starts=_char_starts(tok, a_ids, answer))
    prompt = _ids(tok, user, True, effort)
    full = _ids(tok, user + [{"role": "assistant", "reasoning_content": reasoning, "content": answer}], False, effort)
    if full[: len(prompt)] != prompt:
        raise ValueError("prompt is not a prefix of the full rendering")
    r_ids = tok.encode(reasoning, allow_special_tokens=False) if reasoning else []
    a_ids = tok.encode(answer, allow_special_tokens=False) if answer else []
    P = len(prompt)
    r0, r1 = P, P + len(r_ids)
    if full[r0:r1] != r_ids:
        raise ValueError("reasoning tokens not found where expected")
    # between reasoning and answer: <|close|>think<|sep|><|open|>response<|sep|>
    a0 = full.index(tok.convert_tokens_to_ids("<|sep|>"), full.index(tok.convert_tokens_to_ids("<|open|>"), r1)) + 1
    a1 = a0 + len(a_ids)
    if full[a0:a1] != a_ids:
        raise ValueError("answer tokens not found where expected")
    return Replay(ids=full, prompt_len=P, reasoning=(r0, r1), answer=(a0, a1),
                  reasoning_char_starts=_char_starts(tok, r_ids, reasoning),
                  answer_char_starts=_char_starts(tok, a_ids, answer))
