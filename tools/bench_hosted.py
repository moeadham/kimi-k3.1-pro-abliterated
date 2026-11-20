"""Run capability benchmarks against hosted K3.1 Pro on OpenRouter (pilot / baseline trial).

Measures per item: reasoning + answer tokens, cost, latency, score. Uses the frozen sets in
data/evals and the same prompts/scorers as the on-GPU runner (harness/scorers.py), so the
hosted numbers are directly comparable in method (not in exact value: hosted quantization
and runtime differ from ours).

LiveCodeBench answers are NOT executed here (model-written code is only run in the pod/Modal
sandbox); they are saved for later scoring.

Usage: uv run --env-file .env --extra tok --with lm-eval --with langdetect --with immutabledict \
         --with nltk --with math-verify python tools/bench_hosted.py \
         --sets mmlu_pro,math500,ifeval,cmmlu,livecodebench --n 6 --efforts low,max [--out out/bench_pilot]
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
from harness import scorers  # noqa: E402
from harness.evals import TASKS, load_set  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k3.1-pro"


async def one(client, sem, name, item, effort, provider, seed, max_tokens):
    prompt_fn, score_fn = TASKS[name]
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt_fn(item)}], "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "seed": seed, "reasoning": {"effort": effort},
            "provider": {"only": [provider], "allow_fallbacks": False}}
    row = {"set": name, "item_id": item["id"], "effort": effort, "seed": seed}
    for attempt in range(4):
        async with sem:
            t = time.perf_counter()
            try:
                d = (await client.post(URL, json=body)).json()
            except Exception as e:  # noqa: BLE001
                d = {"error": {"message": repr(e)}}
            row["latency_s"] = round(time.perf_counter() - t, 1)
        if "choices" in d:
            m, u = d["choices"][0]["message"], d.get("usage") or {}
            answer = m.get("content") or ""
            row.update(finish=d["choices"][0].get("finish_reason"), answer=answer, reasoning=m.get("reasoning") or "",
                       prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       cost=float(u.get("cost") or 0), error=None)
            if name != "livecodebench":
                try:
                    row["score"] = score_fn(item, answer)
                except Exception as e:  # noqa: BLE001
                    row["score"] = {"correct": False, "error": repr(e)[:200]}
            return row
        row["error"] = d.get("error")
        await asyncio.sleep(3 * (attempt + 1))
    return row


async def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in a.sets.split(","):
        items = load_set(name)
        items = random.Random(0).sample(items, min(a.n, len(items))) if a.n else items
        for eff in a.efforts.split(","):
            for it in items:
                for s in range(a.samples):
                    jobs.append((name, it, eff, s))
    print(f"{len(jobs)} requests", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(timeout=1800, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}) as c:
        rows = await asyncio.gather(*(one(c, sem, n, it, e, a.provider, s, a.max_tokens) for n, it, e, s in jobs))
    with open(out / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wall {time.time() - t0:.0f}s, cost ${sum(r.get('cost', 0) or 0 for r in rows):.2f}, "
          f"errors {sum(r.get('error') is not None for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="mmlu_pro,math500,ifeval,cmmlu,livecodebench")
    ap.add_argument("--n", type=int, default=6, help="items per set (0 = all)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--efforts", default="low,max")
    ap.add_argument("--provider", default="wafer/us")
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default="out/bench_pilot")
    asyncio.run(main(ap.parse_args()))
