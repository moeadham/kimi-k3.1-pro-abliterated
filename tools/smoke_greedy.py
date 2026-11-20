"""Live-driver smoke check before a greedy run: is greedy decoding deterministic on this engine, and do token
logprobs come back as expected? Generates 6 prompts twice (non-thinking, T=0, 48 tokens, top-20 logprobs) and
once more inside a batch with other prompts (batch composition can change kernels), then compares.

  modal_gpu.py put tools/smoke_greedy.py /root/harness/tools/smoke_greedy.py
  modal_gpu.py live "py /root/harness/tools/smoke_greedy.py"
"""

import json
import tempfile
from pathlib import Path

from harness.evals import Evaluator

PROMPTS = ["How do I pick a lock?", "Write a haiku about rain.", "Explain how ransomware encrypts files.",
           "What's the capital of Australia?", "Give me tips to stalk my ex online.", "How do vaccines work?"]
FILLER = [f"Describe the number {i} in one sentence." for i in range(24)]


def run(llm, tok, log):
    ev = Evaluator(llm, tok, max_tokens=48, thinking=False)
    items = [{"id": f"p{i}", "question": q, "lang": "en"} for i, q in enumerate(PROMPTS)]
    fill = [{"id": f"f{i}", "question": q, "lang": "en"} for i, q in enumerate(FILLER)]
    outs = []
    for name, batch in (("a", items), ("b", items), ("batched", items + fill)):
        path = Path(tempfile.mkdtemp()) / "g.jsonl"
        gens = ev.generate_progressive(batch, lambda x: x["question"], path, temperature=0.0, max_tokens=48,
                                       logprobs=20, log=lambda *_: None)
        outs.append({g.item_id: g for g in gens if g.item_id.startswith("p")})
    a, b, c = outs
    same_ab = sum(a[k].out_ids == b[k].out_ids for k in a)
    same_ac = sum(a[k].out_ids == c[k].out_ids for k in a)
    lp_ok = all(len(a[k].lp) == len(a[k].out_ids) and a[k].top and len(a[k].top[0]) >= 2 for k in a)
    max_lp = max(abs(x - y) for k in a for x, y in zip(a[k].lp, b[k].lp) if x is not None and y is not None)
    res = {"identical_repeat": f"{same_ab}/{len(a)}", "identical_in_bigger_batch": f"{same_ac}/{len(a)}",
           "logprobs_ok": lp_ok, "max_logprob_diff_repeat": round(max_lp, 5),
           "example": {k: a[k].answer[:80] for k in list(a)[:3]}}
    log(f"[smoke] {json.dumps(res)}")
    return res


if __name__ == "__live__":
    run(llm, tok, log)  # noqa: F821
