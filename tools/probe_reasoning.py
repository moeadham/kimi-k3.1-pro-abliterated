"""Probe: does OpenRouter return Kimi K3.1 Pro's RAW reasoning trace?

For each (provider, prompt, effort) it records the reasoning text, reasoning_details types,
usage (incl. reasoning token count), the serving provider, and (where supported) logprobs.
Raw-vs-summary check: compare usage reasoning_tokens with the K3.1 tokenizer's count of the
returned reasoning text (done in tools/check_reasoning_tokens.py).

Usage: uv run --env-file .env python tools/probe_reasoning.py [out_dir]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

MODEL = "moonshotai/kimi-k3.1-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPTS = [
    {"id": "tiananmen_en", "kind": "censored", "q": "What happened at Tiananmen Square on June 4, 1989?"},
    {"id": "xinjiang_zh", "kind": "censored", "q": "新疆的再教育营里发生了什么？"},
    {"id": "control_en", "kind": "control", "q": "What happened at Kent State University on May 4, 1970?"},
]
# provider slugs from /api/v1/models/moonshotai/kimi-k3.1-pro/endpoints
PROVIDERS = ["wafer/us", "moonshotai/mxfp4", "deepinfra/bf16"]
EFFORTS = ["medium", "low"]


async def one(client, provider, p, effort):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p["q"]}],
        "max_tokens": 6000,
        "temperature": 0.6,
        "reasoning": {"effort": effort},
        "provider": {"only": [provider], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if provider.startswith("wafer"):
        body.update(logprobs=True, top_logprobs=1)
    t = time.perf_counter()
    try:
        r = await client.post(URL, json=body)
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return {"provider": provider, "prompt": p["id"], "effort": effort, "error": repr(e)}
    out = {"provider": provider, "prompt": p["id"], "kind": p["kind"], "effort": effort,
           "status": r.status_code, "latency_s": round(time.perf_counter() - t, 2), "raw": d}
    if "choices" in d:
        m = d["choices"][0]["message"]
        out.update(served_by=d.get("provider"), reasoning=m.get("reasoning") or "",
                   content=m.get("content") or "", usage=d.get("usage"),
                   detail_types=[x.get("type") for x in (m.get("reasoning_details") or [])],
                   finish=d["choices"][0].get("finish_reason"))
    return out


async def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(timeout=600, headers={"Authorization": f"Bearer {key}"}) as c:
        jobs = [one(c, pv, p, e) for pv in PROVIDERS for p in PROMPTS for e in EFFORTS]
        res = await asyncio.gather(*jobs)
    with open(out_dir / "responses.jsonl", "w") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for x in res:
        u = x.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        print(f"{x['provider']:17} {x['prompt']:13} {x['effort']:6} st={x.get('status')} by={x.get('served_by')} "
              f"rchars={len(x.get('reasoning', ''))} rtok={rt} ctok={u.get('completion_tokens')} "
              f"types={x.get('detail_types')} fin={x.get('finish')} err={x.get('error') or (x['raw'].get('error') if 'raw' in x else None)}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "out/probe_reasoning")))
