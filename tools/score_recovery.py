"""Score candidate refusal directions on the recovery captures (tools/recovery_capture.py).

For each answer position where a token swap either let the model recover its original outcome or flipped it, the
two continuations share every token up to the swap. Per direction and stream index: does the projection track the
eventual outcome after the swap? Reported per offset bucket after the swap state:
  AUC     refused-outcome vs answered-outcome continuations, unpaired
  paired  share of positions where the refusing continuation projects higher than the answering one (same prompt,
          same position: the cleanest check of a candidate refusal vector without editing)
Offset 0 is the swap state itself, identical in both continuations: AUC 0.5 is the sanity check (paired shows ties
as 0). Run on the GPU host (fast volume reads), e.g. `python tools/score_recovery.py` from /root/harness.

Run 2 (docs/refusal_run2_results.md): 600 positions; all four directions track the outcome from 1-2 tokens after the
swap (paired 0.70-0.80), decision strongest (0.80 at layer 44, 9-32 tokens after).
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.session import _auc

import os
R = Path(os.environ.get("K3_RECOVERY_OUT", "/vol/session_nt/recovery"))
NT_BANK = os.environ.get("K3_NT_BANK", "/root/banks/nt.pt")
nt = torch.load(NT_BANK, map_location="cpu", weights_only=False)
border = torch.load("/vol/session_v1_border/directions_refusal/directions.pt", map_location="cpu", weights_only=False)
pairs = torch.load("/vol/session_v1_pairs/directions_refusal/directions.pt", map_location="cpu", weights_only=False)
dirs = {"D(decision)": border["onset_vs_comply/all"]["mean_diff"], "H(harm)": pairs["prompt_end/all"]["mean_diff"],
        "nt_pe": nt["nt_prompt_end/all"]["mean_diff"], "nt_fork": nt["nt_fork/all"]["mean_diff"]}
idx = [json.loads(l) for l in open(R / "index.jsonl")]
streams = idx[0]["streams"]
names = list(dirs)
W = torch.stack([torch.nn.functional.normalize(dirs[n].float()[streams], dim=-1) for n in names])  # [D,S',d]
OFF = list(range(0, 33))

def load(x):
    st = load_file(str(R / "states" / x["file"]))["states"].float()  # [P,S',d]
    proj = torch.einsum("psd,ksd->pks", st, W)  # [P,D,S']
    out = {}
    for j, p in enumerate(x["positions"]):
        out[p - x["swap_state"]] = proj[j]
    return x, out

groups = {}
with ThreadPoolExecutor(48) as pool:
    for x, out in pool.map(load, idx):
        groups.setdefault((x["key"], x["i"]), {})[x["kind"]] = (bool(x["flagged"]), out)
print(f"{len(idx)} sequences, {len(groups)} positions", flush=True)
# A) unpaired AUC over swapped continuations (recovered + flipped): refused-outcome vs answered-outcome
# B) paired: within a position that has both, P(refused-outcome seq projects higher than answered-outcome seq)
res = {}
buckets = {"k=0 (swap state)": [0], "k=1-2": [1, 2], "k=3-8": list(range(3, 9)), "k=9-32": list(range(9, 33))}
for bname, ks in buckets.items():
    pos, neg, wins, tot = [], [], 0, 0
    pw = []
    for g in groups.values():
        sw = {k: v for k, v in g.items() if k != "original"}
        for kind, (fl, out) in sw.items():
            vs = [out[k] for k in ks if k in out]
            if vs:
                (pos if fl else neg).append(torch.stack(vs).mean(0))
        if "recovered" in g and "flipped" in g:
            a, b = g["recovered"], g["flipped"]
            ref, ans = (a, b) if a[0] else (b, a)
            vr = [ref[1][k] for k in ks if k in ref[1]]; va = [ans[1][k] for k in ks if k in ans[1]]
            if vr and va:
                pw.append((torch.stack(vr).mean(0) > torch.stack(va).mean(0)).float())
    P, N = torch.stack(pos), torch.stack(neg)
    auc = torch.tensor([[_auc(P[:, d, s], N[:, d, s]) for s in range(P.shape[2])] for d in range(P.shape[1])])
    pa = torch.stack(pw).mean(0) if pw else None
    res[bname] = {"auc": auc, "paired": pa, "n": (len(pos), len(neg), len(pw))}
show = [s for s in streams if s in (40, 80, 100, 112, 116, 120, 128, 140, 160, 184, 186)]
cols = [streams.index(s) for s in show]
for bname, r in res.items():
    print(f"\n== {bname}  refused-outcome/answered-outcome = {r['n'][0]}/{r['n'][1]}, paired positions = {r['n'][2]}")
    print("direction       metric  " + " ".join(f"L{s//2:>3}" for s in show) + "   best(layer)")
    for d, n in enumerate(names):
        a = r["auc"][d]; j = int(a.argmax())
        print(f"{n:<15} AUC     " + " ".join(f"{a[c]:.2f}".rjust(4) for c in cols) + f"   {a[j]:.2f}(L{streams[j]//2})")
        if r["paired"] is not None:
            p = r["paired"][d]; j = int(p.argmax())
            print(f"{'':<15} paired  " + " ".join(f"{p[c]:.2f}".rjust(4) for c in cols) + f"   {p[j]:.2f}(L{streams[j]//2})")
