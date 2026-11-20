"""Probe: hosted K3.1 Pro, reasoning OFF, temperature 0, short answers, with token logprobs.

What does the model commit to in its first answer tokens, and how close is that decision? For each prompt:
greedy answer (max_tokens small), the logprob of every generated token with its top-20 alternatives, and a Jev
verdict on the (truncated) answer. A subset is sent twice to check that the provider is deterministic at T=0.

Prompts: harmful ones hosted K3.1 refused 5/5 (reasoning on), harmless controls it answered 5/5, and borderline
v1 prompts whose 4 reasoning-on samples were mixed (out/border/mixed.json: prompt id -> refusals out of 4).

Usage: uv run --env-file .env python tools/probe_greedy_logprobs.py [out_dir] [--n-mixed 40] [--max-tokens 64]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import refusal  # noqa: E402
from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import is_flagged, judge_response  # noqa: E402
from tools.probe_no_reasoning import consistent  # noqa: E402

MODEL = "moonshotai/kimi-k3.1-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
PROVIDER = "wafer/us"  # returns answer logprobs; DeepInfra was rate-limited


def mixed_items(n: int, seed: int = 0) -> list[dict]:
    counts = json.load(open("out/border/mixed.json"))  # "harmful:wjb-..." -> refusals of 4
    items = {x["id"]: x for f, sp in (("data/refusal_v1/borderline_harmful.jsonl", "harmful"),
                                      ("data/refusal_v1/borderline_harmless.jsonl", "harmless"))
             for x in refusal.load_prompts(f, sp)}
    ids = sorted(counts)
    random.Random(seed).shuffle(ids)
    by_k = {k: [i for i in ids if counts[i] == k] for k in (1, 2, 3)}
    pick = []
    while len(pick) < n and any(by_k.values()):  # balanced over 1/4, 2/4, 3/4 refused
        for k in (1, 2, 3):
            if by_k[k] and len(pick) < n:
                pick.append(by_k[k].pop())
    return [{"id": i, "question": items[i]["question"], "kind": f"mixed_{counts[i]}of4",
             "source": items[i]["source_split"]} for i in pick]


async def ask(client, item, max_tokens, rep):
    body = {"model": MODEL, "messages": [{"role": "user", "content": item["question"]}], "max_tokens": max_tokens,
            "temperature": 0.0, "reasoning": {"enabled": False}, "logprobs": True, "top_logprobs": 20,
            "provider": {"only": [PROVIDER], "allow_fallbacks": False}, "usage": {"include": True}}
    for attempt in range(4):
        t = time.perf_counter()
        try:
            r = await client.post(URL, json=body)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            d = {"error": repr(e)}
        if "choices" in d:
            break
        await asyncio.sleep(3 * (attempt + 1))
    out = {**item, "rep": rep, "latency_s": round(time.perf_counter() - t, 2)}
    if "choices" not in d:
        out["error"] = str(d.get("error"))[:300]
        return out
    c = d["choices"][0]
    lp = ((c.get("logprobs") or {}).get("content")) or []
    out.update(content=c["message"].get("content") or "", reasoning=c["message"].get("reasoning") or "",
               finish=c.get("finish_reason"), served_by=d.get("provider"), cost=(d.get("usage") or {}).get("cost"),
               tokens=[{"t": x["token"], "lp": x["logprob"],
                        "top": [(y["token"], y["logprob"]) for y in (x.get("top_logprobs") or [])]} for x in lp])
    return out


async def main(out_dir: Path, n_mixed: int, max_tokens: int):
    items = (consistent("results/harmbench_standard/calib.jsonl", True, 10)
             + consistent("results/xstest_safe/calib.jsonl", False, 5) + mixed_items(n_mixed))
    repeat = items[:5] + items[15:20]  # determinism check: 5 anchors + 5 mixed, sent twice
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {key}"}) as c:
        async def go(it, rep):
            async with sem:
                return await ask(c, it, max_tokens, rep)
        rows = await asyncio.gather(*([go(it, 0) for it in items] + [go(it, 1) for it in repeat]))
    async with JevJudge(concurrency=32) as j:
        async def jd(r):
            if r.get("content"):
                v = await judge_response(j, r["question"], "", r["content"])
                r.update(verdict=v, flagged=is_flagged(v))
            return r
        rows = await asyncio.gather(*(jd(r) for r in rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} calls, errors {sum(bool(r.get('error')) for r in rows)}, "
          f"cost ~${sum(float(r.get('cost') or 0) for r in rows):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=Path("results/greedy_logprobs_probe"))
    ap.add_argument("--n-mixed", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    asyncio.run(main(a.out, a.n_mixed, a.max_tokens))
