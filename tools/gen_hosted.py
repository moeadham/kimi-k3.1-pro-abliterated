"""Generate Kimi K3.1 Pro transcripts (reasoning + answer) on deccp via OpenRouter.

Native settings per the model card: reasoning_effort in {low, high, max} (default max),
temperature 1.0, top_p 0.95. The effort is rendered into the prompt as an internal
"thinking-effort" system message (encoding_k3.py), so replay must use the same effort.

Runs (per prompt):
  wafer/us (MXFP4, native expert format): 4 samples @ T=1.0 effort=max, 1 greedy @ effort=max,
                                          1 sample @ effort=low
  deepinfra/bf16:                          1 sample @ T=1.0 effort=max (cross-provider check)
Moonshot's own endpoint is excluded: it content-filters censored prompts before the model.

Resumable: rows already in the output file (by run key) are skipped.
Usage: uv run --env-file .env --with pyarrow --with huggingface_hub python tools/gen_hosted.py \
         [--out out.jsonl] [--prompts prompts.jsonl] [--runs full|core]
  --prompts: jsonl with {id, split, question} (default: deccp)
  --runs core: only the Wafer max-effort runs (4 samples + greedy)
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

RUNS = (
    [("wafer/us", "max", 1.0, s) for s in range(4)]
    + [("wafer/us", "max", 0.0, 0), ("wafer/us", "low", 1.0, 0), ("deepinfra/bf16", "max", 1.0, 0)]
)


def load_deccp():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = []
    for split in ("censored", "uncensored"):
        f = hf_hub_download("augmxnt/deccp", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        for i, t in enumerate(pq.read_table(f).column("text").to_pylist()):
            out.append({"id": f"{split}-{i}", "split": split, "question": t})
    return out


def run_key(pid, provider, effort, temp, seed):
    return f"{pid}|{provider}|{effort}|{temp}|{seed}"


async def one(client, sem, p, provider, effort, temp, seed):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p["question"]}],
        "max_tokens": 16000,
        "temperature": temp,
        "top_p": 0.95 if temp > 0 else 1.0,
        "seed": seed,
        "reasoning": {"effort": effort},
        "provider": {"only": [provider], "allow_fallbacks": False},
    }
    row = {"key": run_key(p["id"], provider, effort, temp, seed), "prompt_id": p["id"], "split": p["split"],
           "question": p["question"], "provider": provider, "effort": effort, "temperature": temp, "seed": seed}
    for attempt in range(4):
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.post(URL, json=body)
                d = r.json()
            except Exception as e:  # noqa: BLE001
                d = {"error": {"message": repr(e)}}
            row["latency_s"] = round(time.perf_counter() - t, 2)
        if "choices" in d:
            m = d["choices"][0]["message"]
            row.update(served_by=d.get("provider"), reasoning=m.get("reasoning") or "", content=m.get("content") or "",
                       detail_types=sorted({x.get("type") for x in (m.get("reasoning_details") or [])}),
                       finish=d["choices"][0].get("finish_reason"), usage=d.get("usage"), error=None)
            return row
        row["error"] = d.get("error")
        await asyncio.sleep(2 * (attempt + 1))
    return row


async def main(out: Path, prompts_path: str | None, runs_name: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in open(out) if json.loads(l).get("error") is None}
    prompts = [json.loads(l) for l in open(prompts_path)] if prompts_path else load_deccp()
    runs = RUNS if runs_name == "full" else [r for r in RUNS if r[0] == "wafer/us" and r[1] == "max"]
    todo = [(p, *r) for p in prompts for r in runs if run_key(p["id"], *r) not in done]
    print(f"{len(prompts)} prompts, {len(todo)} runs to do ({len(done)} done)", flush=True)
    sem = asyncio.Semaphore(48)
    key = os.environ["OPENROUTER_API_KEY"]
    n_ok = n_err = 0
    cost = 0.0
    async with httpx.AsyncClient(timeout=900, headers={"Authorization": f"Bearer {key}"}) as c:
        with open(out, "a") as f:
            for fut in asyncio.as_completed([one(c, sem, *t) for t in todo]):
                row = await fut
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if row.get("error") is None:
                    n_ok += 1
                    cost += float((row.get("usage") or {}).get("cost") or 0)
                else:
                    n_err += 1
                if (n_ok + n_err) % 50 == 0:
                    print(f"{n_ok + n_err}/{len(todo)} ok={n_ok} err={n_err} cost=${cost:.2f}", flush=True)
    print(f"done ok={n_ok} err={n_err} cost=${cost:.2f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/hosted_deccp/transcripts.jsonl")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--runs", default="full", choices=["full", "core"])
    a = ap.parse_args()
    asyncio.run(main(Path(a.out), a.prompts, a.runs))
