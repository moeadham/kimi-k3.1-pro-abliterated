"""Token-by-token internal onset from the `forks` stage captures (OUT/forks/index.jsonl + states/).

For each offset k relative to an anchor and each captured stream index s: 5-fold cross-validated AUC of the
mean-difference direction (refused vs answered) applied to the held-out rows' state at that offset. The internal
onset is the earliest offset where the AUC becomes reliable.

Anchors (state index of offset 0):
  fork        the state that produces the fork token (fork_abs - 1); refused rows fork at the refusal word,
              answered rows at the start of the compliance sentence (usually the first answer token)
  prompt_end  the last prompt token (prompt_len - 1), which produces the first answer token

Two streaming passes over the files (pass 1: per-fold class sums; pass 2: held-out projections), so memory stays
at a few GB however many rows there are.

  python tools/analyze_forks.py /vol/RUN/forks /vol/RUN/forks_analysis [--anchor fork] [--before 64] [--after 16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.session import _auc  # noqa: E402  tie-aware AUC


def fold_of(key: str, k: int = 5) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % k


def anchor_state(x: dict, anchor: str) -> int | None:
    if anchor == "prompt_end":
        return x["prompt_len"] - 1
    return x["fork_abs"] - 1 if x.get("fork_abs") is not None else None


def rows_at_offsets(x: dict, cap: Path, anchor: str, offsets: list[int]):
    """{offset: [S', d] float32} for the offsets this row has captured."""
    a = anchor_state(x, anchor)
    if a is None:
        return {}
    st = load_file(str(cap / "states" / x["file"]))["states"]
    where = {p: j for j, p in enumerate(x["positions"])}
    return {k: st[where[a + k]].float() for k in offsets if (a + k) in where}


def main(cap: str, out: str, anchor: str = "fork", before: int = 64, after: int = 16, folds: int = 5,
         confirmed_only: bool = False, workers: int = 16) -> dict:
    cap, out = Path(cap), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = [json.loads(l) for l in open(cap / "index.jsonl")]
    if confirmed_only:
        index = [x for x in index if x.get("fork_confirmed") or not x.get("flagged")]
    offsets = list(range(-before, after + 1))
    streams = index[0]["streams"]
    # pass 1: per (fold, class) sums at every (offset, stream)
    sums: dict = {}
    counts: dict = {}

    def load(x):
        return x, rows_at_offsets(x, cap, anchor, offsets)

    with ThreadPoolExecutor(workers) as pool:
        for x, vs in pool.map(load, index):
            f, c = fold_of(x["key"], folds), bool(x["flagged"])
            for k, v in vs.items():
                key = (f, c, k)
                sums[key] = sums.get(key, 0) + v.double()
                counts[key] = counts.get(key, 0) + 1

    def direction(f, k):  # mean difference from all folds except f
        tot = {c: (sum(sums[(g, c, k)] for g in range(folds) if g != f and (g, c, k) in sums),
                   sum(counts.get((g, c, k), 0) for g in range(folds) if g != f)) for c in (True, False)}
        if tot[True][1] == 0 or tot[False][1] == 0:
            return None
        d = tot[True][0] / tot[True][1] - tot[False][0] / tot[False][1]
        return torch.nn.functional.normalize(d.float(), dim=-1)  # [S', d]

    dirs = {(f, k): direction(f, k) for f in range(folds) for k in offsets}
    # pass 2: held-out projections
    scores: dict = {k: {True: [], False: []} for k in offsets}
    with ThreadPoolExecutor(workers) as pool:
        for x, vs in pool.map(load, index):
            f, c = fold_of(x["key"], folds), bool(x["flagged"])
            for k, v in vs.items():
                r = dirs[(f, k)]
                if r is not None:
                    scores[k][c].append((v * r).sum(-1))  # [S']
    res = {"anchor": anchor, "streams": streams, "offsets": offsets, "auc": {}, "n": {}}
    for k in offsets:
        pos, neg = scores[k][True], scores[k][False]
        res["n"][k] = [len(pos), len(neg)]
        if len(pos) >= 5 and len(neg) >= 5:
            P, N = torch.stack(pos), torch.stack(neg)
            res["auc"][k] = [round(_auc(P[:, j], N[:, j]), 4) for j in range(P.shape[1])]
    (out / f"tokens_{anchor}.json").write_text(json.dumps(res))
    lines = [f"anchor={anchor}  rows: refused/answered per offset; AUC per stream index (layer = s // 2)",
             "offset  n      " + " ".join(f"s{s:>3}" for s in streams[::max(1, len(streams) // 12)])]
    for k in offsets:
        if k in res["auc"]:
            a = res["auc"][k][::max(1, len(streams) // 12)]
            lines.append(f"{k:+5d}  {res['n'][k][0]:>3}/{res['n'][k][1]:<3} " + " ".join(f"{v:.2f}" for v in a))
    (out / f"tokens_{anchor}.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("out")
    ap.add_argument("--anchor", choices=["fork", "prompt_end"], default="fork")
    ap.add_argument("--before", type=int, default=64)
    ap.add_argument("--after", type=int, default=16)
    ap.add_argument("--confirmed-only", action="store_true")
    a = ap.parse_args()
    main(a.cap, a.out, a.anchor, a.before, a.after, confirmed_only=a.confirmed_only)
