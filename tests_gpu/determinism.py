"""Diagnose run-to-run nondeterminism: compare prefill (prompt) logprobs bitwise across
repeated identical runs, and for decode find the first divergent step and the top-1/top-2
logprob margin there (near-ties on random weights flip greedy argmax under tiny noise).

Usage (pod): python tests_gpu/determinism.py /dev/shm/tiny-k3/mxfp4 [--graphs] [--runs 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.vllm_launch import make_llm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--same-len", action="store_true", help="all prompts the same length")
    ap.add_argument("--warmup", type=int, default=0, help="untimed generate calls before measuring")
    args = ap.parse_args()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm = make_llm(args.path, eager=not args.graphs, max_model_len=1024, max_num_seqs=8)
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(1000 + 37 * i, 1000 + 37 * i + 16 + (0 if args.same_len else 5 * i))))
        for i in range(args.n_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=24, logprobs=2, prompt_logprobs=1)
    for _ in range(args.warmup):
        llm.generate(prompts, sp, use_tqdm=False)
    runs = [llm.generate(prompts, sp, use_tqdm=False) for _ in range(args.runs)]

    def prompt_lps(run):
        return [[next(iter(d.values())).logprob if d else None for d in o.prompt_logprobs] for o in run]

    base_p = prompt_lps(runs[0])
    prefill_max_diff = 0.0
    for run in runs[1:]:
        for a, b in zip(base_p, prompt_lps(run)):
            for x, y in zip(a, b):
                if x is not None:
                    prefill_max_diff = max(prefill_max_diff, abs(x - y))

    divergences = []
    for p in range(len(prompts)):
        t0 = list(runs[0][p].outputs[0].token_ids)
        for r in range(1, args.runs):
            t = list(runs[r][p].outputs[0].token_ids)
            if t != t0:
                step = next(i for i, (a, b) in enumerate(zip(t0, t)) if a != b)
                lp = runs[0][p].outputs[0].logprobs[step]
                top = sorted((v.logprob for v in lp.values()), reverse=True)
                margin = top[0] - top[1] if len(top) > 1 else None
                divergences.append({"prompt": p, "run": r, "step": step, "top1_top2_margin": margin})
                break

    print(json.dumps({
        "mode": "cudagraph" if args.graphs else "eager",
        "n_prompts": args.n_prompts, "same_len": args.same_len, "warmup": args.warmup,
        "multiproc": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1"),
        "prefill_logprob_max_abs_diff": prefill_max_diff,
        "decode_divergences": divergences,
    }))


if __name__ == "__main__":
    main()
