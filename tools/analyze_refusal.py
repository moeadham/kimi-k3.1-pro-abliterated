"""Per-layer analysis of refusal captures (CPU; run where the captures are, e.g. a CPU sandbox on the Volume).

  python tools/analyze_refusal.py /vol/session/refusal /vol/session/analysis_refusal

For each position type and each stream index s (2 per decoder layer + final; s = 2*layer is the attention
input, 2*layer+1 the MLP input), with the per-row mean over that type's positions as the row's vector:
  - auc_cv:        5-fold cross-validated AUC of the mean-difference direction (refused vs answered)
  - cohen_d:       effect size of the held-out (cross-validated) projections
  - rel_gap:       ||mu_bad - mu_good|| / mean ||x||  (how big the difference is relative to the stream)
  - split_half:    cosine between directions estimated on two disjoint random halves (data sufficiency:
                   near 1 = well estimated, low = noise-dominated), mean over 10 splits
  - cos_next:      cosine between the direction at s and at s+1 (where it stabilizes or rotates)
  - cos_prompt_end: cosine to the prompt_end direction at the same s (do position types share a direction?)
Plus: per-row projection percentiles for disagreement rows (harmful answered / harmless flagged) and a
prompt-label (harmful vs harmless) variant for prompt_end.
Classes: bad = judge flagged (refused/deflected), good = answered. `onset` (flagged rows only) is compared with
the answered rows' reasoning_head and, when captured, their `comply_onset` (the sentence that settles on
answering), as in the directions stage. answer_head is reported but flagged: its deep
layers differ between engine modes (docs/modal_runbook.md).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

TYPES = ("prompt_end", "reasoning_head", "onset", "comply_onset", "answer_head")


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC; ties get their average rank (identical scores -> 0.5). Same as harness.session._auc."""
    s = torch.cat([pos, neg]).double()
    order = s.argsort()
    _, inv, counts = torch.unique_consecutive(s[order], return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    ranks = torch.empty_like(s)
    ranks[order] = (ends - (counts.double() - 1) / 2)[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load(cap: Path, workers: int = 16):
    """Read every capture file once, with a thread pool (the reads are I/O bound and release the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    index = [json.loads(l) for l in open(cap / "index.jsonl")]

    def one(x):
        st = load_file(cap / "states" / x["file"])["states"]
        where = {p: j for j, p in enumerate(x["positions"])}
        return {t: st[[where[p] for p in ps]].float().mean(0)
                for t in TYPES if (ps := x["position_sets"].get(t) or [])}

    vecs = {t: {} for t in TYPES}  # type -> row i -> [S, d] float32 (mean over positions)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for k, (x, per_type) in enumerate(zip(index, pool.map(one, index))):
            for t, v in per_type.items():
                vecs[t][x["i"]] = v
            if k % 50 == 0:
                print(f"loaded {k}/{len(index)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"loaded {len(index)} in {time.time() - t0:.0f}s ({workers} threads)", flush=True)
    return index, vecs


def stack(vecs_t: dict, ids: list[int]) -> torch.Tensor:
    return torch.stack([vecs_t[i] for i in ids]) if ids else None


def per_layer(bad: torch.Tensor, good: torch.Tensor, folds: int = 5, seed: int = 0) -> dict:
    """bad [nb, S, d], good [ng, S, d] -> per-s metrics."""
    S = bad.shape[1]
    g = torch.Generator().manual_seed(seed)
    pb, pg = torch.randperm(len(bad), generator=g), torch.randperm(len(good), generator=g)
    # cross-validated AUC
    proj_b, proj_g = torch.zeros(len(bad), S), torch.zeros(len(good), S)
    for f in range(folds):
        tb, tg = pb[f::folds], pg[f::folds]
        mb = torch.ones(len(bad), dtype=torch.bool); mb[tb] = False
        mg = torch.ones(len(good), dtype=torch.bool); mg[tg] = False
        r = unit(bad[mb].mean(0) - good[mg].mean(0))  # [S, d]
        proj_b[tb] = torch.einsum("nsd,sd->ns", bad[tb], r)
        proj_g[tg] = torch.einsum("nsd,sd->ns", good[tg], r)
    auc_cv = [auc(proj_b[:, s], proj_g[:, s]) for s in range(S)]
    diff = bad.mean(0) - good.mean(0)
    r = unit(diff)
    xb, xg = proj_b, proj_g  # held-out projections: in-sample ones are inflated in high dimensions
    pooled = ((xb.var(0) + xg.var(0)) / 2).sqrt().clamp_min(1e-12)
    cohen = ((xb.mean(0) - xg.mean(0)) / pooled).tolist()
    rel_gap = (diff.norm(dim=-1) / torch.cat([bad, good]).norm(dim=-1).mean(0)).tolist()
    sh = torch.zeros(S)
    for k in range(10):
        gb, gg = torch.randperm(len(bad), generator=g), torch.randperm(len(good), generator=g)
        h1 = unit(bad[gb[: len(bad) // 2]].mean(0) - good[gg[: len(good) // 2]].mean(0))
        h2 = unit(bad[gb[len(bad) // 2:]].mean(0) - good[gg[len(good) // 2:]].mean(0))
        sh += (h1 * h2).sum(-1) / 10
    cos_next = ((r[:-1] * r[1:]).sum(-1)).tolist() + [None]
    return {"auc_cv": auc_cv, "cohen_d": cohen, "rel_gap": rel_gap, "split_half": sh.tolist(),
            "cos_next": cos_next, "n_bad": len(bad), "n_good": len(good), "_dir": r}


def main(cap: str, out: str) -> None:
    cap, out = Path(cap), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index, vecs = load(cap)
    flagged = [x["i"] for x in index if x["censored"]]
    answered = [x["i"] for x in index if not x["censored"]]
    harmful = [x["i"] for x in index if x["dataset_split"] == "harmful"]
    harmless = [x["i"] for x in index if x["dataset_split"] == "harmless"]
    mixed = {x["i"] for x in index if x.get("mixed_prompt")}
    res, dirs = {}, {}
    plan = {"prompt_end": ("prompt_end", flagged, "prompt_end", answered),
            "reasoning_head": ("reasoning_head", flagged, "reasoning_head", answered),
            "onset_vs_head": ("onset", [i for i in flagged if i in vecs["onset"]], "reasoning_head", answered),
            "onset_vs_comply": ("onset", [i for i in flagged if i in vecs["onset"]], "comply_onset",
                                [i for i in answered if i in vecs["comply_onset"]]),
            "answer_head (engine-dependent)": ("answer_head", flagged, "answer_head", answered),
            "prompt_end by prompt label": ("prompt_end", harmful, "prompt_end", harmless),
            # same prompt, different outcomes (--samples > 1): the refusal decision with the prompt held fixed
            "mixed: reasoning_head": ("reasoning_head", [i for i in flagged if i in mixed], "reasoning_head",
                                      [i for i in answered if i in mixed]),
            "mixed: onset_vs_comply": ("onset", [i for i in flagged if i in mixed and i in vecs["onset"]],
                                       "comply_onset", [i for i in answered if i in mixed and i in vecs["comply_onset"]])}
    # keep only rows that have the position type (e.g. a response cut off before its answer has no answer_head)
    plan = {k: (tb, [i for i in ib if i in vecs[tb]], tg, [i for i in ig if i in vecs[tg]])
            for k, (tb, ib, tg, ig) in plan.items()}
    plan = {k: v for k, v in plan.items() if len(v[1]) >= 10 and len(v[3]) >= 10}  # e.g. no comply_onset yet
    for name, (tb, ib, tg, ig) in plan.items():
        m = per_layer(stack(vecs[tb], ib), stack(vecs[tg], ig))
        dirs[name] = m.pop("_dir")
        res[name] = m
        best = max(range(len(m["auc_cv"])), key=lambda s: (m["auc_cv"][s], m["cohen_d"][s]))
        print(f"{name}: n={m['n_bad']}/{m['n_good']} best s={best} (layer {best // 2}) auc_cv={m['auc_cv'][best]:.3f} "
              f"d={m['cohen_d'][best]:.2f} split_half={m['split_half'][best]:.3f}", flush=True)
    base = dirs["prompt_end"]
    for name in res:
        res[name]["cos_prompt_end"] = (dirs[name] * base).sum(-1).tolist()
    # where do the disagreement rows sit on the prompt_end direction?
    r = dirs["prompt_end"]
    allp = {i: (vecs["prompt_end"][i] * r).sum(-1) for i in vecs["prompt_end"]}  # [S] each
    S = r.shape[0]
    ans_h = [i for i in answered if i in harmless]
    def pct(i, s):
        ref = torch.stack([allp[j][s] for j in ans_h])
        return float((ref < allp[i][s]).double().mean())
    special = []
    for x in index:
        if (x["dataset_split"] == "harmful") != x["censored"]:
            special.append({"i": x["i"], "prompt_id": x["prompt_id"], "split": x["dataset_split"],
                            "flagged": x["censored"], "pct_vs_answered_harmless": [round(pct(x["i"], s), 3)
                                                                                    for s in range(S)]})
    res["_disagreements"] = special
    torch.save({k: v for k, v in dirs.items()}, out / "directions_mean_diff.pt")
    (out / "per_layer.json").write_text(json.dumps(res))
    # compact table: every 8th stream index + the final
    lines = ["s  layer | " + " | ".join(f"{n[:18]:>18}" for n in plan)]
    for s in sorted(set(range(0, S, 8)) | {S - 1}):
        cells = []
        for n in plan:
            m = res[n]
            cells.append(f"{m['auc_cv'][s]:.3f} d{m['cohen_d'][s]:5.2f} sh{m['split_half'][s]:.2f}")
        lines.append(f"{s:3d} {s // 2:3d} | " + " | ".join(f"{c:>18}" for c in cells))
    (out / "table.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
