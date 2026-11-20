"""Generate -> chunk -> judge -> flagged token positions (step 1.7).

Responses are generated in a batch (deterministic, in-process vLLM), split into token-span
chunks with exact offsets, and every chunk is judged concurrently. Judging completed
responses yields the same per-span verdicts as live streaming; a streaming driver (for early
stopping) can reuse `chunk_tokens` + the judges unchanged.

Flagged positions are absolute token indices into prompt_ids + response_ids, ready for
harness.capture.capture(). For a chunk covering response tokens [s, e) we record the
positions that PRODUCED those tokens: prompt_len + s - 1 ... prompt_len + e - 2
(the residual state at position t is what predicts token t+1).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SENT_END = re.compile(r"[.!?。！？\n]\s*$")


@dataclass
class Chunk:
    start: int  # response-token offset (inclusive)
    end: int  # exclusive
    text: str
    verdict: dict | None = None


@dataclass
class Record:
    prompt_id: str
    question: str
    prompt_ids: list[int]
    response_ids: list[int]
    response: str
    chunks: list[Chunk] = field(default_factory=list)

    def flagged_positions(self) -> list[int]:
        return self._positions(lambda v: v and v["flagged"])

    def clean_positions(self) -> list[int]:
        return self._positions(lambda v: v and not v["flagged"] and not v.get("error"))

    def _positions(self, pred) -> list[int]:
        P = len(self.prompt_ids)
        out = []
        for c in self.chunks:
            if pred(c.verdict):
                out.extend(range(max(P + c.start - 1, 0), P + c.end - 1))
        return out

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["chunks"] = [c.__dict__ for c in self.chunks]
        d["flagged_positions"] = self.flagged_positions()
        return d


def chunk_tokens(tokenizer, ids: list[int], target: int = 24, max_len: int = 48) -> list[Chunk]:
    """Split response ids into spans of ~target tokens, preferring sentence boundaries."""
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + max_len, len(ids))
        for e in range(start + target, end + 1):
            if SENT_END.search(tokenizer.decode(ids[start:e])):
                end = e
                break
        chunks.append(Chunk(start, end, tokenizer.decode(ids[start:end])))
        start = end
    return chunks


def generate(llm, prompts: list[dict], max_tokens: int = 256, chat: bool = True) -> list[Record]:
    """prompts: [{"id", "question"}]. Greedy decoding; chat template applied if chat=True."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()

    def encode(q: str) -> list[int]:
        if not chat:
            return list(tok.encode(q))
        # Render to text then encode: apply_chat_template(tokenize=True) returns a dict for
        # some tokenizers under transformers 5 (Qwen) but a list for others (K3).
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
        return list(tok.encode(text, add_special_tokens=False))

    ids = [encode(p["question"]) for p in prompts]
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids],
                        SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
    return [Record(p["id"], p["question"], list(i), list(o.outputs[0].token_ids), o.outputs[0].text)
            for p, i, o in zip(prompts, ids, outs)]


async def judge_records(records: list[Record], judge, tokenizer, target: int = 24) -> dict:
    jobs = []
    for r in records:
        r.chunks = chunk_tokens(tokenizer, r.response_ids, target=target)
        context = ""
        for c in r.chunks:
            jobs.append((c, judge.judge(r.question, c.text, context)))
            context += c.text
    verdicts = await asyncio.gather(*(j for _, j in jobs))
    for (c, _), v in zip(jobs, verdicts):
        c.verdict = v.to_dict()
    lat = sorted(v.latency_s for v in verdicts if v.error is None and v.latency_s)
    return {
        "chunks": len(verdicts),
        "errors": sum(v.error is not None for v in verdicts),
        "flagged": sum(v.flagged for v in verdicts),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[int(len(lat) * 0.95)] if lat else None,
        "cost": sum(v.cost for v in verdicts),
    }


def save_records(path: Path, records: list[Record]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
