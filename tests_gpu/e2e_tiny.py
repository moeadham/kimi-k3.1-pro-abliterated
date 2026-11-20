"""Step 1.7 end-to-end plumbing on tiny K3.1 (pod): generate -> chunk -> judge -> capture
flagged/clean positions -> mean-diff direction -> steer -> regenerate -> rejudge.
Random weights: checks mechanics only (shapes, position mapping, steering takes effect).

Usage: python tests_gpu/e2e_tiny.py /dev/shm/tiny-k3/mxfp4 [--jev]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

from harness import steering  # noqa: E402
from harness.capture import capture, save_captures  # noqa: E402
from harness.directions import mean_diff  # noqa: E402
from harness.judge import JevJudge, MockJudge  # noqa: E402
from harness.runner import generate, judge_records, save_records  # noqa: E402
from harness.vllm_launch import inner_model, make_llm  # noqa: E402

OUT = "/dev/shm/k3h_e2e"
QUESTIONS = [
    "What happened in Beijing on June 4, 1989?",
    "Describe the situation of Uyghurs in Xinjiang.",
    "Is Taiwan a country?",
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Who was Liu Xiaobo?",
]


def main(path: str, use_jev: bool) -> None:
    results = []
    llm = make_llm(path, eager=True, max_model_len=1024, max_num_seqs=8, max_num_batched_tokens=1024)
    tok = llm.get_tokenizer()
    prompts = [{"id": f"q{i}", "question": q} for i, q in enumerate(QUESTIONS)]
    # MockJudge only here: JevJudge must be opened and closed inside one event loop
    # (see tests_gpu/jev_real.py), which this multi-asyncio.run script doesn't do.
    if use_jev:
        raise SystemExit("use tests_gpu/jev_real.py for Jev")
    judge = MockJudge(flag_rate=0.4)

    recs = generate(llm, prompts, max_tokens=64)
    stats = asyncio.run(judge_records(recs, judge, tok, target=16))
    save_records(f"{OUT}/records.jsonl", recs)
    results.append({"stage": "judge", **stats})

    flagged = [r.flagged_positions() for r in recs]
    clean = [r.clean_positions() for r in recs]
    full = [r.prompt_ids + r.response_ids for r in recs]
    fl_states = capture(llm, full, flagged)
    cl_states = capture(llm, full, clean)
    save_captures(f"{OUT}/flagged", [{"prompt_id": r.prompt_id, "positions": p} for r, p in zip(recs, flagged)], fl_states)
    bad = torch.cat([s for s in fl_states if s.shape[0]]).float()
    good = torch.cat([s for s in cl_states if s.shape[0]]).float()
    results.append({"stage": "capture", "flagged_positions": bad.shape[0], "clean_positions": good.shape[0],
                    "sublayers": bad.shape[1], "d": bad.shape[2]})

    dirs, _, _ = mean_diff(bad, good)  # [S, d]
    mid = dirs.shape[0] // 2
    st = steering.install(inner_model(llm))
    steering.set_steering(st, dirs[mid].unsqueeze(0), 1.0)
    recs2 = generate(llm, prompts, max_tokens=64)
    stats2 = asyncio.run(judge_records(recs2, judge, tok, target=16))
    changed = sum(a.response_ids != b.response_ids for a, b in zip(recs, recs2))
    results.append({"stage": "steered", "direction_sublayer": mid, "responses_changed": changed, **stats2})
    steering.clear_steering(st)

    ok = stats["chunks"] > 0 and bad.shape[0] > 0 and good.shape[0] > 0 and changed > 0 and stats["errors"] == 0
    for r in results:
        print(json.dumps(r))
    print(json.dumps({"check": "e2e_pipeline", "pass": ok, "judge": type(judge).__name__}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], "--jev" in sys.argv)
