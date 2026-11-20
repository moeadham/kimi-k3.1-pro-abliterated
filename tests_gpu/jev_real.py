"""Step 1.7: judge loop on a real censoring model with Jev (pod).

Model: Qwen/Qwen2.5-7B-Instruct (the family augmxnt/deccp was built on).
Prompts: augmxnt/deccp censored (95) + uncensored controls (37), Apache-2.0.
Reports Jev latency/cost and per-split flag rates; writes records for manual review.

Usage: OPENROUTER_API_KEY=... python tests_gpu/jev_real.py [out_dir]
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from harness.judge import JevJudge  # noqa: E402
from harness.runner import generate, judge_records, save_records  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/harness/out/jev_real"


def load_deccp():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = []
    for split in ("censored", "uncensored"):
        f = hf_hub_download("augmxnt/deccp", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        for i, t in enumerate(pq.read_table(f).column("text").to_pylist()):
            out.append({"id": f"{split}-{i}", "split": split, "question": t})
    return out


def main() -> None:
    from vllm import LLM

    prompts = load_deccp()
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.5, seed=0, enforce_eager=False)
    recs = generate(llm, prompts, max_tokens=384)
    save_records(f"{OUT}/generated.jsonl", recs)  # keep generations even if judging fails

    async def judge_all():
        async with JevJudge(concurrency=64) as judge:
            return await judge_records(recs, judge, llm.get_tokenizer(), target=32)

    stats = asyncio.run(judge_all())
    save_records(f"{OUT}/records.jsonl", recs)

    split_of = {p["id"]: p["split"] for p in prompts}
    per = collections.defaultdict(lambda: {"responses": 0, "responses_flagged": 0, "chunks": 0, "chunks_flagged": 0,
                                           "labels": collections.Counter(), "first_chunk_labels": collections.Counter()})
    for r in recs:
        s = per[split_of[r.prompt_id]]
        s["responses"] += 1
        s["responses_flagged"] += any(c.verdict["flagged"] for c in r.chunks)
        if r.chunks:
            s["first_chunk_labels"][r.chunks[0].verdict["label"]] += 1
        for c in r.chunks:
            s["chunks"] += 1
            s["chunks_flagged"] += c.verdict["flagged"]
            s["labels"][c.verdict["label"]] += 1
    summary = {"model": MODEL, "judge": "typesafe/jev-1.14 via OpenRouter decisions", **stats,
               "per_split": {k: {**v, "labels": dict(v["labels"]), "first_chunk_labels": dict(v["first_chunk_labels"])}
                             for k, v in per.items()}}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
