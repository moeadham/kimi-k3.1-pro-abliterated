"""Sanity panel ("did we lobotomize it?") on hosted K3.1 Pro via OpenRouter: reference numbers
for the unedited model. The same items run on our engine via harness.evals (benchmark/bfcl).

Panel (all effort=low, T=1.0, top_p=0.95, 1 sample, fixed item selection seed 0):
  mmlu_pro 100 | cmmlu 50 | ifeval 100 | math500 level-5 50 | bfcl 50 (native tool calls)
plus reasoning-health metrics on every response (empty answer, hit max_tokens, repetition,
off-language). KL is GPU-only and not part of the hosted run.

Provider pinned (no fallbacks): wafer/us = native MXFP4 checkpoint format, no content filter,
same provider as the earlier hosted baselines.

Usage: uv run --env-file .env --extra tok --with lm-eval --with langdetect --with immutabledict \
         --with nltk --with math-verify python tools/sanity_panel_hosted.py [--out results/sanity_panel_hosted]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bfcl, k3chat  # noqa: E402
from harness.evals import TASKS  # noqa: E402
from harness.sanity import panel_items  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k3.1-pro"
async def one(client, sem, name, item, a, tok):
    if name == "bfcl":
        body = {"messages": item["messages"], "tools": bfcl.to_openai_tools(item["functions"])}
    else:
        body = {"messages": [{"role": "user", "content": TASKS[name][0](item)}]}
    body.update(model=MODEL, max_tokens=a.max_tokens, temperature=1.0, top_p=0.95, seed=0,
                reasoning={"effort": a.effort}, provider={"only": [a.provider], "allow_fallbacks": False})
    row = {"set": name, "item_id": item["id"], "lang": item.get("lang", "en")}
    for attempt in range(4):
        async with sem:
            t = time.perf_counter()
            try:
                d = (await client.post(URL, json=body)).json()
            except Exception as e:  # noqa: BLE001
                d = {"error": {"message": repr(e)}}
            row["latency_s"] = round(time.perf_counter() - t, 1)
        if "choices" in d:
            break
        row["error"] = d.get("error")
        await asyncio.sleep(3 * (attempt + 1))
    else:
        return row
    m, u = d["choices"][0]["message"], d.get("usage") or {}
    answer, reasoning = m.get("content") or "", m.get("reasoning") or ""
    finish = d["choices"][0].get("finish_reason")
    row.update(error=None, served_by=d.get("provider"), finish=finish, answer=answer, reasoning=reasoning,
               tool_calls=m.get("tool_calls"), prompt_tokens=u.get("prompt_tokens"),
               completion_tokens=u.get("completion_tokens"),
               reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               cost=float(u.get("cost") or 0))
    ids = tok.encode(reasoning + answer, allow_special_tokens=False) if tok else []
    row["health"] = {"hit_max_tokens": finish == "length", "empty_answer": not answer.strip() and not m.get("tool_calls"),
                     "repetition": k3chat.repetition_rate(ids), "off_language": k3chat.off_language_fraction(answer, row["lang"])}
    try:
        row["score"] = (bfcl.score(item, bfcl.calls_from_openai(m)) if name == "bfcl" else TASKS[name][1](item, answer))
    except Exception as e:  # noqa: BLE001
        row["score"] = {"correct": False, "error": repr(e)[:200]}
    return row


def summarize(rows) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
    out = {}
    for name, rs in by.items():
        ok = [r for r in rs if r.get("error") is None]
        out[name] = {
            "n": len(rs), "errors": len(rs) - len(ok),
            "accuracy": statistics.fmean(bool(r["score"].get("correct")) for r in ok) if ok else None,
            "mean_output_tokens": statistics.fmean(r["completion_tokens"] or 0 for r in ok) if ok else None,
            "cost": sum(r["cost"] for r in ok),
            "latency_median_s": statistics.median(r["latency_s"] for r in ok) if ok else None,
            "latency_max_s": max(r["latency_s"] for r in ok) if ok else None,
            "health": {k: statistics.fmean(float(r["health"][k]) for r in ok) for k in ok[0]["health"]} if ok else None,
        }
    return out


async def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from harness.replay import load_tokenizer

        tok = load_tokenizer()
    except Exception:  # noqa: BLE001
        tok = None
    items = panel_items()
    print(f"{len(items)} requests", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(timeout=1800, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}) as c:
        rows = await asyncio.gather(*(one(c, sem, n, it, a, tok) for n, it in items))
    wall = time.time() - t0
    with open(out / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"model": MODEL, "provider": a.provider, "effort": a.effort, "temperature": 1.0, "top_p": 0.95,
            "max_tokens": a.max_tokens, "concurrency": a.concurrency, "wall_s": round(wall), "date": time.strftime("%Y-%m-%d"),
            "total_cost": sum(r.get("cost", 0) or 0 for r in rows), "sets": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sanity_panel_hosted")
    ap.add_argument("--provider", default="wafer/us")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main(ap.parse_args()))
