"""Is a single long prefill deterministic in vLLM itself (no capture hooks)? K3.1's DSA indexer keeps the
top-2048 tokens, so prompts longer than 2048 exercise sparse top-k selection.

Usage: python tests_gpu/long_prefill_determinism.py /vol/tiny-k3/mxfp4 [--tp 1] [--eager] --lengths 1024 2048 2486 4110
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 2486, 4110])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--topk-backend", default=None, help="kernel_config.sparse_indexer_topk_backend")
    ap.add_argument("--attention-backend", default=None, help="e.g. FLASHMLA_SPARSE")
    a = ap.parse_args()
    os.environ["K3H_STEERING"] = "1"
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from harness.session import replay_rows
    from harness.vllm_launch import make_llm

    llm = make_llm(a.model, eager=a.eager, tensor_parallel_size=a.tp, max_num_seqs=1, max_model_len=16384,
                   max_num_batched_tokens=16384, gpu_memory_utilization=0.85,
                   **({"kernel_config": {"sparse_indexer_topk_backend": a.topk_backend}} if a.topk_backend else {}),
                   **({"attention_backend": a.attention_backend} if a.attention_backend else {}))
    rows = replay_rows()
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    for L in a.lengths:
        i = min((i for i, r in enumerate(rows) if len(r["ids"]) >= L), key=lambda i: len(rows[i]["ids"]))
        ids = rows[i]["ids"]
        runs = []
        for _ in range(a.reps):
            o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
            runs.append(torch.tensor([next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]]))
        diffs = [(runs[0] - r).abs() for r in runs[1:]]
        first = [int((d > 0).nonzero()[0]) + 1 if (d > 0).any() else None for d in diffs]
        print(json.dumps({"topk_backend": a.topk_backend, "attn": a.attention_backend, "len": len(ids), "row": i, "max_diff": max(float(d.max()) for d in diffs),
                          "first_diff_pos": first, "n_diff": [int((d > 0).sum()) for d in diffs]}), flush=True)


if __name__ == "__main__":
    main()
