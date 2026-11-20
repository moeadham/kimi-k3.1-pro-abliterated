"""Step 1.2 acceptance: tiny K3.1 loads in vLLM and generates deterministically without NaNs.

Usage (pod): python tests_gpu/smoke_load.py /dev/shm/tiny-k3/mxfp4 [--graphs]
Prints one JSON line per check; exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.vllm_launch import make_llm as _make_llm  # noqa: E402


def make_llm(path: str, graphs: bool):
    return _make_llm(path, eager=not graphs, max_model_len=1024, max_num_seqs=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--graphs", action="store_true")
    args = ap.parse_args()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm = make_llm(args.path, args.graphs)
    cfg = llm.llm_engine.model_config.hf_config
    backend = (llm.llm_engine.vllm_config.additional_config or {}).get("decode_backend", "auto")
    assert backend != "latency_specialized", "specialized decode bypasses hooks/edits"

    prompts = [TokensPrompt(prompt_token_ids=list(range(1000 + 37 * i, 1000 + 37 * i + 16 + 5 * i))) for i in range(4)]
    sp = SamplingParams(temperature=0.0, max_tokens=24, logprobs=1)
    runs = [llm.generate(prompts, sp, use_tqdm=False) for _ in range(2)]

    ok = True
    toks = [[list(o.outputs[0].token_ids) for o in run] for run in runs]
    lps = [o.outputs[0].cumulative_logprob for o in runs[0]]
    finite = all(lp is not None and math.isfinite(lp) for lp in lps)
    deterministic = toks[0] == toks[1]
    nonempty = all(len(t) == 24 for t in toks[0])
    distinct = len({tuple(t) for t in toks[0]}) > 1
    for name, val in [("finite_logprobs", finite), ("deterministic", deterministic),
                      ("full_length", nonempty), ("prompt_dependent", distinct)]:
        print(json.dumps({"check": name, "pass": bool(val)}))
        ok &= bool(val)
    print(json.dumps({
        "mode": "cudagraph" if args.graphs else "eager",
        "arch": cfg.architectures, "decode_backend": backend,
        "quant": llm.llm_engine.model_config.quantization,
        "sample_tokens": toks[0][0][:12], "cum_logprobs": [round(x, 2) for x in lps],
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
