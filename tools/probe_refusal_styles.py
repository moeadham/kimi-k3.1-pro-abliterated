"""Catalogue how K3.1 Pro refuses with reasoning OFF: full greedy answers (T=0, up to --max-tokens) with token
logprobs, a Jev verdict on the whole answer, and the decision sentence (refusal onset if flagged, compliance onset
if answered), located in tokens so the logprobs around it can be read.

Prompt sets (ids sampled with a fixed seed):
  harmbench      HarmBench prompts hosted K3.1 refused 5/5 with reasoning on
  pairs          harmful side of data/refusal_v1/pairs_*
  border_mixed   v1 borderline prompts whose 4 reasoning-on samples were mixed (out/border/mixed.json)
  border_always  v1 borderline prompts refused 4/4 with reasoning on (out/border/judged.jsonl)
  harmless       XSTest-safe answered 5/5 + harmless side of the pairs

Usage: uv run --env-file .env python tools/probe_refusal_styles.py [out_dir] [--max-tokens 512]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import refusal  # noqa: E402
from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import (COMPLY_Q, COMPLY_THRESHOLD, find_onset, is_flagged,  # noqa: E402
                                    judge_response)
from tools.probe_greedy_logprobs import ask  # noqa: E402
from tools.probe_no_reasoning import consistent  # noqa: E402


def border_counts() -> dict:
    by = collections.defaultdict(list)
    for line in open("out/border/judged.jsonl"):
        r = json.loads(line)
        by[r["item_id"]].append(r["flagged"])
    return {k: sum(v) for k, v in by.items()}


def items(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    border = {x["id"]: x for f, sp in (("data/refusal_v1/borderline_harmful.jsonl", "harmful"),
                                       ("data/refusal_v1/borderline_harmless.jsonl", "harmless"))
              for x in refusal.load_prompts(f, sp)}
    counts = border_counts()
    pairs_h = refusal.load_prompts("data/refusal_v1/pairs_harmful.jsonl", "harmful")
    pairs_s = refusal.load_prompts("data/refusal_v1/pairs_harmless.jsonl", "harmless")
    mixed = sorted(k for k, v in counts.items() if 0 < v < 4)
    always = sorted(k for k, v in counts.items() if v == 4)
    out = [{**x, "set": "harmbench"} for x in consistent("results/harmbench_standard/calib.jsonl", True, 50)]
    out += [{"id": x["id"], "question": x["question"], "set": "pairs"} for x in rng.sample(pairs_h, 50)]
    out += [{"id": k, "question": border[k]["question"], "set": "border_mixed", "refused_of_4": counts[k],
             "source": border[k]["source_split"]} for k in mixed]
    out += [{"id": k, "question": border[k]["question"], "set": "border_always", "refused_of_4": 4,
             "source": border[k]["source_split"]} for k in rng.sample(always, 30)]
    out += [{**x, "set": "harmless"} for x in consistent("results/xstest_safe/calib.jsonl", False, 10)]
    out += [{"id": x["id"], "question": x["question"], "set": "harmless"} for x in rng.sample(pairs_s, 10)]
    return out


def locate(tokens: list[dict], content: str, char: int) -> int | None:
    """Index into `tokens` of the token where answer character `char` starts (after the channel tag)."""
    start = next((i + 1 for i, t in enumerate(tokens) if t["t"] == "<|sep|>"), 0)
    pos = 0
    for i in range(start, len(tokens)):
        if pos >= char:
            return i
        pos += len(tokens[i]["t"])
    return None


async def main(out_dir: Path, max_tokens: int):
    its = items()
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=600, headers={"Authorization": f"Bearer {key}"}) as c:
        async def go(it):
            async with sem:
                return await ask(c, it, max_tokens, 0)
        rows = await asyncio.gather(*(go(it) for it in its))
    async with JevJudge(concurrency=48) as j:
        async def jd(r):
            if not r.get("content"):
                return r
            v = await judge_response(j, r["question"], "", r["content"])
            fl = is_flagged(v)
            on = (await find_onset(j, r["question"], r["content"]) if fl else
                  await find_onset(j, r["question"], r["content"], questions=COMPLY_Q, threshold=COMPLY_THRESHOLD))
            s = on["onset_sentence"]
            r.update(verdict=v, flagged=fl, onset_kind="refusal" if fl else "compliance", onset_sentence=s,
                     n_sentences=len(on["spans"]), onset_scores=on["scores"])
            if s is not None:
                a, b = on["spans"][s]
                r.update(onset_text=r["content"][a:b].strip(), onset_char=a,
                         onset_token=locate(r.get("tokens") or [], r["content"], a))
            return r
        rows = await asyncio.gather(*(jd(r) for r in rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} calls, errors {sum(bool(r.get('error')) for r in rows)}, "
          f"cost ~${sum(float(r.get('cost') or 0) for r in rows):.3f}, "
          f"hit max_tokens {sum(r.get('finish') == 'length' for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=Path("results/refusal_styles_probe"))
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    asyncio.run(main(a.out, a.max_tokens))
