"""Mean-difference directions straight from refusal captures, for the probe (no PCA, no val AUC).

  python tools/quick_bank.py CAPTURE_DIR OUT.pt FAMILY [FAMILY ...]
  families: prompt_end/all, reasoning_head/all, onset_vs_comply/all, mixed/onset_vs_comply, mixed/reasoning_head

Same definition as the `directions` stage's mean_diff: normalize(mean over bad-class position vectors - mean over
good-class position vectors) per stream index, bad = judge-flagged rows, good = answered rows, train fold for the
pooled families and train+val for mixed/* (as in stage_directions), up to 4 evenly spaced positions per row.
Streams the files with a thread pool and keeps only running sums, so it runs anywhere the captures are visible
(e.g. inside the GPU sandbox that wrote them, whose Volume reads are much faster than a CPU sandbox's).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file

FAMS = {  # family -> (bad position type, good position type, mixed only)
    "prompt_end/all": ("prompt_end", "prompt_end", False),
    "reasoning_head/all": ("reasoning_head", "reasoning_head", False),
    "onset_vs_comply/all": ("onset", "comply_onset", False),
    "mixed/onset_vs_comply": ("onset", "comply_onset", True),
    "mixed/reasoning_head": ("reasoning_head", "reasoning_head", True),
}


def evenly(xs, k=4):
    if len(xs) <= k:
        return xs
    return [xs[round(i * (len(xs) - 1) / (k - 1))] for i in range(k)]


def main(cap: str, out: str, *fams: str) -> None:
    cap = Path(cap)
    index = [json.loads(l) for l in open(cap / "index.jsonl")]
    sums: dict = {}

    def one(x):
        st = load_file(str(cap / "states" / x["file"]))["states"]
        where = {p: j for j, p in enumerate(x["positions"])}
        got = []
        for f in fams:
            tb, tg, mixed_only = FAMS[f]
            if mixed_only and not x.get("mixed_prompt"):
                continue
            if not mixed_only and x["fold"] != "train":
                continue
            t, cls = (tb, "bad") if x["censored"] else (tg, "good")
            ps = evenly(x["position_sets"].get(t) or [])
            if ps:
                v = st[[where[p] for p in ps]].double()
                got.append((f, cls, v.sum(0), len(ps)))
        return got

    t0 = time.time()
    with ThreadPoolExecutor(32) as pool:
        for k, got in enumerate(pool.map(one, index)):
            for f, cls, s, n in got:
                acc = sums.setdefault((f, cls), [0, 0])
                acc[0] = acc[0] + s
                acc[1] += n
            if k % 500 == 0:
                print(f"{k}/{len(index)} ({time.time() - t0:.0f}s)", flush=True)
    bank = {}
    for f in fams:
        if (f, "bad") not in sums or (f, "good") not in sums:
            print(f"{f}: missing a class, skipped", flush=True)
            continue
        (sb, nb), (sg, ng) = sums[(f, "bad")], sums[(f, "good")]
        d = sb / nb - sg / ng
        bank[f] = {"mean_diff": torch.nn.functional.normalize(d, dim=-1).float()}
        print(f"{f}: bad {nb} / good {ng} position vectors", flush=True)
    torch.save(bank, out)
    print(f"saved {out} {sorted(bank)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
