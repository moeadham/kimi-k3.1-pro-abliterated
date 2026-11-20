"""Direction bank from a prompt-end capture (tools/pe_capture.py), for tools/quick_probe.py edits.

Classes (refusing vs answering), by label source:
  first_token  p_refuse_start >= --hi refusing, <= --lo answering, in between dropped (pe_capture pass 1)
  jev          the judged flag (pe_capture with labels_from: run 3's 512-token label pass, or residual refusals under
               an edit); cut-off traces by heading score, >= 0.8 refusing, <= 0.3 answering, in between dropped
  prompt       the prompt's split (harmful vs harmless), whatever the model did
Rows of the --ref-sets tags are never in a class: they are the harmless reference for projection.
Positions: offsets from the prompt end within the captured last k (e.g. "-1", "-4:-1" inclusive), averaged per
row; one family per --positions argument.

Per family (unit vectors per stream index, zero outside the captured streams):
  mean_diff     mean(refusing) - mean(answering)
  projected     minus its component along the answering mean (run 1/2's "projected")
  projected_ref minus its component along the reference mean
  pc{k}         minus its components in the top-k (uncentered) principal directions of the reference rows
Also a report (OUT.json): class counts, split-half cosine of mean_diff per stream (rows hashed into two halves),
and cosine between the variants.

  python tools/pe_bank.py CAPDIR OUT.pt --label first_token --positions -1 -4:-1 --ref-sets reference \
      --pcs 2 4 8 --prefix h3
Streams are read in chunks (--chunk) with partial safetensors reads, so memory stays bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

S_TOTAL = 187


def parse_positions(spec: str, k: int) -> list[int]:
    """'-1' or '-4:-1' (inclusive) -> indices into the captured last-k positions."""
    a, b = (spec.split(":") + [None])[:2] if ":" in spec else (spec, spec)
    a, b = -abs(int(a)), -abs(int(b))  # "4:1" (CLI-safe) == "-4:-1": the last 4 through the last
    if not (-k <= a <= b <= -1):
        raise ValueError(f"positions {spec!r} outside the captured last {k}")
    return [k + i for i in range(a, b + 1)]


CUT_HI, CUT_LO = 0.8, 0.3  # tools/heading_check.py: on 512-token cut traces >= 0.8 is 97.5% refusing, <= 0.3 99.3%
#                           answering; 0.3-0.8 (~10% of cut traces) holds most misread deflections, so it is dropped


def classify(x: dict, label: str, hi: float, lo: float, cut_hi: float = CUT_HI, cut_lo: float = CUT_LO) -> bool | None:
    if label == "prompt":
        return x["split"] == "harmful"
    if label == "jev":
        if "flagged" not in x:
            return None
        if x.get("cutoff") and x.get("score") is not None:  # judged on where the cut reasoning was heading
            return True if x["score"] >= cut_hi else False if x["score"] <= cut_lo else None
        return bool(x["flagged"])
    p = x.get("p_refuse_start")
    if p is None:
        return None
    return True if p >= hi else False if p <= lo else None


def half_of(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 2


def remove_span(r: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """r [c, d] minus its projection on the orthonormal rows of basis [c, m, d]; normalized."""
    coef = torch.einsum("cd,cmd->cm", r, basis)
    return F.normalize(r - torch.einsum("cm,cmd->cd", coef, basis), dim=-1)


def v1_rows(cap: Path, labels: dict | None = None) -> list[dict]:
    """Run-1 `refusal` captures (index with position_sets) as prompt-end rows: the 4 prompt_end positions, label =
    the judge flag (or a corrected flag by key), set = fold, pair = the prompt id (matched twins share it)."""
    out = []
    for x in map(json.loads, open(cap / "index.jsonl")):
        pe = x.get("position_sets", {}).get("prompt_end") or []
        if len(pe) != 4:
            continue
        flag = labels.get(x["key"], x["censored"]) if labels else x["censored"]
        out.append({"id": x["key"], "file": x["file"], "_dir": str(cap), "_pick": [x["positions"].index(q) for q in pe],
                    "positions": pe, "streams": [0, S_TOTAL], "flagged": bool(flag), "split": x["dataset_split"],
                    "set": x["fold"], "pair": x["prompt_id"].split(":", 1)[1]})
    return out


def build(cap, out: str, label: str = "first_token", positions: list[str] = ("-1",), ref_sets=("reference",),
          pcs=(2, 4, 8), prefix: str = "h3", hi: float = 0.5, lo: float = 0.1, chunk: int = 24,
          workers: int = 16, class_sets=None, v1_labels: dict | None = None) -> dict:
    """cap: a capture dir or a list of them (pe_capture or run-1 `refusal` format). class_sets: only rows of
    these sets in the classes (e.g. ["train"] for run 1's folds). With no reference rows, the pcK variants use the
    answering rows as the reference."""
    caps = [Path(c) for c in ([cap] if isinstance(cap, (str, Path)) else cap)]
    index = []
    for c in caps:
        first = json.loads(open(c / "index.jsonl").readline())
        index += (v1_rows(c, v1_labels) if "position_sets" in first
                  else [{**x, "_dir": str(c)} for x in map(json.loads, open(c / "index.jsonl"))])
    s0, s1 = index[0]["streams"]
    k = len(index[0]["positions"])
    pos_sets = {p: parse_positions(p, k) for p in positions}
    ref_rows = [x for x in index if x.get("set") in set(ref_sets)]
    cls_rows = [(x, c) for x in index if x.get("set") not in set(ref_sets)
                and (class_sets is None or x.get("set") in set(class_sets))
                for c in [classify(x, label, hi, lo)] if c is not None]
    if not ref_rows and pcs:
        ref_rows = [x for x, c in cls_rows if not c]
        ref_from = "answering rows"
    else:
        ref_from = f"sets {sorted(ref_sets)}"
    if not cls_rows or all(c for _, c in cls_rows) or not any(c for _, c in cls_rows):
        raise ValueError(f"need both classes; got {sum(c for _, c in cls_rows)} refusing of {len(cls_rows)}")
    d = None
    bank = {f"{prefix}/{p}": {} for p in positions}
    report = {"label": label, "hi": hi, "lo": lo, "rows": len(index), "ref_rows": len(ref_rows), "ref_from": ref_from,
              "refusing": sum(c for _, c in cls_rows), "answering": sum(not c for _, c in cls_rows),
              "streams": [s0, s1], "positions": {p: [i - k for i in v] for p, v in pos_sets.items()},
              "tokens_at_positions": index[0].get("tokens"), "split_half_cos": {}, "variant_cos": {}}

    def read(x, c0, c1):
        pick = x.get("_pick") or list(range(k))
        with safe_open(str(Path(x["_dir"]) / "states" / x["file"]), "pt") as f:
            sl = f.get_slice("states")
            st = torch.cat([sl[i:i + 1, c0:c1] for i in pick])  # [k, c, d]: only the needed positions are read
        return {p: st[ix].float().mean(0) for p, ix in pos_sets.items()}

    for c0 in range(0, s1 - s0, chunk):
        c1 = min(c0 + chunk, s1 - s0)
        sums = {}  # (pos, class, half) -> (sum [c, d] float64, n)
        with ThreadPoolExecutor(workers) as pool:
            for (x, c), v in zip(cls_rows, pool.map(lambda xc: read(xc[0], c0, c1), cls_rows)):
                h = half_of(x.get("pair") or x["id"])
                for p, t in v.items():
                    s, n = sums.get((p, c, h), (0, 0))
                    sums[(p, c, h)] = (s + t.double(), n + 1)
            ref = list(pool.map(lambda x: read(x, c0, c1), ref_rows)) if ref_rows else []
        for p in positions:
            fam = bank[f"{prefix}/{p}"]

            def mean(c, halves=(0, 1)):
                s = sum(sums[(p, c, h)][0] for h in halves if (p, c, h) in sums)
                n = sum(sums[(p, c, h)][1] for h in halves if (p, c, h) in sums)
                return (s / n).float()
            good = mean(False)
            r = F.normalize(mean(True) - good, dim=-1)  # [c, d]
            d = r.shape[-1]
            variants = {"mean_diff": r, "projected": remove_span(r, F.normalize(good, dim=-1)[:, None])}
            halves = [F.normalize(mean(True, (h,)) - mean(False, (h,)), dim=-1) for h in (0, 1)]
            report["split_half_cos"].setdefault(p, []).extend((halves[0] * halves[1]).sum(-1).tolist())
            if ref:
                R = torch.stack([v[p] for v in ref], 1)  # [c, n_ref, d]
                variants["projected_ref"] = remove_span(r, F.normalize(R.mean(1), dim=-1)[:, None])
                if pcs:  # top principal directions (uncentered, so the first is ~the mean), randomized low rank
                    q = min(max(pcs) + 8, R.shape[1], R.shape[2])
                    torch.manual_seed(0)
                    _, _, V = torch.svd_lowrank(R, q=q, niter=4)  # V [c, d, q]
                    V = V.transpose(1, 2)
                    for kk in pcs:
                        if kk <= V.shape[1]:
                            variants[f"pc{kk}"] = remove_span(r, V[:, :kk])
            for name, t in variants.items():
                full = fam.setdefault(name, torch.zeros(S_TOTAL, d))
                full[s0 + c0: s0 + c1] = t
    for p in positions:
        fam = bank[f"{prefix}/{p}"]
        md = fam["mean_diff"][s0:s1]
        report["variant_cos"][p] = {name: round(float((md * t[s0:s1]).sum(-1).mean()), 4)
                                    for name, t in fam.items() if name != "mean_diff"}
    torch.save(bank, out)
    Path(out).with_suffix(".json").write_text(json.dumps(report))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cap", help="capture dir, or several joined by commas")
    ap.add_argument("out")
    ap.add_argument("--label", choices=["first_token", "jev", "prompt"], default="first_token")
    ap.add_argument("--positions", nargs="+", default=["-1"])
    ap.add_argument("--ref-sets", nargs="*", default=["reference"])
    ap.add_argument("--pcs", nargs="*", type=int, default=[2, 4, 8])
    ap.add_argument("--prefix", default="h3")
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--class-sets", nargs="*", default=None)
    ap.add_argument("--v1-labels", default=None, help="jsonl with key + corrected_censored (run-2 cut-off review)")
    a = ap.parse_args()
    v1l = ({x["key"]: x["corrected_censored"] for x in map(json.loads, open(a.v1_labels))} if a.v1_labels else None)
    r = build(a.cap.split(","), a.out, a.label, a.positions, a.ref_sets, a.pcs, a.prefix, a.hi, a.lo, a.chunk,
              class_sets=a.class_sets, v1_labels=v1l)
    print(json.dumps({k: v for k, v in r.items() if k != "split_half_cos"}, indent=1))
    for p, v in r["split_half_cos"].items():
        print(f"split-half cos {p}: min {min(v):.3f} median {sorted(v)[len(v) // 2]:.3f}")
