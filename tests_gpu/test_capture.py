"""Step 1.3 (trimmed): replay capture on tiny K3.1 (pod, eager, in-process).

Checks: shape [n_pos, 2L+1, d]; the captured FINAL mix reproduces vLLM's own prompt
logprobs when pushed through model.norm + lm_head (so we capture the true stream);
capturing twice is bitwise identical.

Usage: python tests_gpu/test_capture.py /dev/shm/tiny-k3/mxfp4
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

from harness.capture import capture  # noqa: E402
from harness.vllm_launch import inner_model, make_llm  # noqa: E402

results = []


def check(name, ok, **info):
    results.append({"check": name, "pass": bool(ok), **info})


def main(path):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm = make_llm(path, eager=True, max_model_len=1024, max_num_seqs=8, max_num_batched_tokens=1024)
    inner = inner_model(llm)
    n_layers = len(inner.layers)
    seqs = [list(range(3000 + 53 * i, 3000 + 53 * i + 30 + 7 * i)) for i in range(3)]
    pos = [[0, 5, len(s) - 1] for s in seqs]

    a = capture(llm, seqs, pos)
    b = capture(llm, seqs, pos)
    check("shape", all(x.shape == (3, 2 * n_layers + 1, inner.embed_tokens.weight.shape[-1]) for x in a),
          shape=list(a[0].shape))
    check("deterministic", all(torch.equal(x, y) for x, y in zip(a, b)))

    # Final mix -> norm -> lm_head must reproduce vLLM's prompt logprobs (next-token at pos t).
    lm = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
    lm = getattr(lm, "language_model", lm)
    full = capture(llm, seqs[:1])[0]  # all positions of seq 0
    with torch.no_grad():
        h = full[:, -1].cuda()
        h = lm.model.norm(h, None) if lm.needs_norm_in_logits else h
        logits = lm.compute_logits.__self__.logits_processor(lm.lm_head, h).float()
    ours = torch.log_softmax(logits, -1)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    o = llm.generate([TokensPrompt(prompt_token_ids=seqs[0])], sp, use_tqdm=False)[0]
    ref = [next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]]
    mine = [ours[t, seqs[0][t + 1]].item() for t in range(len(seqs[0]) - 1)]
    diff = max(abs(x - y) for x, y in zip(mine, ref))
    check("final_mix_reproduces_logprobs", diff < 0.05, max_abs_diff=round(diff, 5))


if __name__ == "__main__":
    main(sys.argv[1])
    for r in results:
        print(json.dumps(r))
    sys.exit(0 if all(r["pass"] for r in results) else 1)
