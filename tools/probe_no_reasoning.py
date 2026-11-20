"""Probe: hosted K3.1 Pro with reasoning DISABLED. Does it still refuse, and where is the decision?

Prompts: harmful ones that hosted K3.1 refused 5/5 at max effort (results/harmbench_standard/calib.jsonl) and
harmless controls it answered 5/5 (results/xstest_safe/calib.jsonl). Each is sent with OpenRouter
`reasoning: {"enabled": false}` to pinned providers (no fallbacks); we record whether any reasoning still comes
back, judge the response (refused / deflected, harness/response_judge.py) and, since there is no reasoning to
search, run the onset questions over the ANSWER's sentences: refusal onset for flagged answers, compliance
onset for answered ones.

Usage: uv run --env-file .env python tools/probe_no_reasoning.py [out_dir] [--n-harmful 10] [--n-harmless 5]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import (COMPLY_Q, COMPLY_THRESHOLD, find_onset, is_flagged,  # noqa: E402
                                    judge_response)

MODEL = "moonshotai/kimi-k3.1-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
PROVIDERS = ["deepinfra/bf16", "wafer/us"]


def consistent(path: str, want_flagged: bool, n: int) -> list[dict]:
    by = collections.defaultdict(list)
    q = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("verdict"):
            by[r["prompt_id"]].append(is_flagged(r["verdict"]))
            q[r["prompt_id"]] = r["question"]
    ids = [p for p, v in by.items() if len(v) >= 5 and all(x == want_flagged for x in v)][:n]
    return [{"id": p, "question": q[p], "kind": "harmful" if want_flagged else "harmless"} for p in ids]


async def ask(client, provider, item):
    body = {"model": MODEL, "messages": [{"role": "user", "content": item["question"]}], "max_tokens": 2000,
            "temperature": 1.0, "top_p": 0.95, "reasoning": {"enabled": False},
            "provider": {"only": [provider], "allow_fallbacks": False}, "usage": {"include": True}}
    t = time.perf_counter()
    try:
        r = await client.post(URL, json=body)
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return {**item, "provider": provider, "error": repr(e)}
    out = {**item, "provider": provider, "status": r.status_code, "latency_s": round(time.perf_counter() - t, 2)}
    if "choices" not in d:
        out["error"] = str(d.get("error"))[:300]
        return out
    m = d["choices"][0]["message"]
    u = d.get("usage") or {}
    out.update(served_by=d.get("provider"), reasoning=m.get("reasoning") or "", content=m.get("content") or "",
               reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               completion_tokens=u.get("completion_tokens"), finish=d["choices"][0].get("finish_reason"),
               cost=u.get("cost"))
    return out


async def judge_all(rows):
    async with JevJudge(concurrency=32) as j:
        async def one(r):
            if not r.get("content"):
                return r
            v = await judge_response(j, r["question"], r["reasoning"], r["content"])
            fl = is_flagged(v)
            # no reasoning to search: look for the decision sentence in the answer itself
            on = (await find_onset(j, r["question"], r["content"]) if fl else
                  await find_onset(j, r["question"], r["content"], questions=COMPLY_Q, threshold=COMPLY_THRESHOLD))
            s = on["onset_sentence"]
            return {**r, "verdict": v, "flagged": fl, "onset_kind": "refusal" if fl else "compliance",
                    "onset_sentence": s, "onset_text": r["content"][slice(*on["spans"][s])] if s is not None else None,
                    "n_sentences": len(on["spans"]), "onset_scores": on["scores"]}
        return await asyncio.gather(*(one(r) for r in rows))


async def main(out_dir: Path, n_harmful: int, n_harmless: int):
    items = (consistent("results/harmbench_standard/calib.jsonl", True, n_harmful)
             + consistent("results/xstest_safe/calib.jsonl", False, n_harmless))
    key = os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(timeout=600, headers={"Authorization": f"Bearer {key}"}) as c:
        rows = await asyncio.gather(*(ask(c, p, it) for p in PROVIDERS for it in items))
    rows = await judge_all(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        if r.get("error"):
            print(f"{r['provider']:15} {r['id'][:34]:34} ERROR {r['error'][:120]}")
            continue
        print(f"{r['provider']:15} {r['kind']:8} {r['id'][:34]:34} rtok={r['reasoning_tokens']} rchars={len(r['reasoning'])} "
              f"ctok={r['completion_tokens']} flagged={r.get('flagged')} onset={r.get('onset_sentence')}/{r.get('n_sentences')} "
              f"| {(r.get('onset_text') or '')[:90]!r}")
    cost = sum(float(r.get("cost") or 0) for r in rows)
    print(f"API cost ~${cost:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=Path("results/no_reasoning_probe"))
    ap.add_argument("--n-harmful", type=int, default=10)
    ap.add_argument("--n-harmless", type=int, default=5)
    a = ap.parse_args()
    asyncio.run(main(a.out, a.n_harmful, a.n_harmless))
