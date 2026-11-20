"""Step 1.6 GPU tests (pod). Each mode runs in its own process.

  bake   : bake r (from test_steering) at lam=1 into <ckpt> -> <ckpt>-baked
  steer  : <ckpt> + steering hooks (lam=1, r), eager -> /dev/shm/k3h_bake_steer.json
  baked  : <ckpt>-baked, no hooks, eager; compares with `steer`:
           - stream orthogonality (attn_res recorder) as strong as steering
           - prompt-logprob distance steer<->baked much smaller than steer<->baseline
  edit   : edit verification, eager or --graphs: large in-place edit of each live writer
           weight must change outputs; restoring must give bitwise-identical outputs.

Usage: python tests_gpu/test_bake.py {bake|steer|baked|edit} /dev/shm/tiny-k3/mxfp4 [--graphs]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODE, CKPT = sys.argv[1], Path(sys.argv[2])
GRAPHS = "--graphs" in sys.argv
BAKED = Path(str(CKPT) + "-baked")
R_PATH = "/dev/shm/k3h_steer_r.pt"
STEER_OUT = "/dev/shm/k3h_bake_steer.json"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

results: list[dict] = []


def check(name, ok, **info):
    results.append({"check": name, "pass": bool(ok), **info})


def llm_for(path, graphs=False):
    from harness.vllm_launch import make_llm

    return make_llm(str(path), eager=not graphs, max_model_len=1024, max_num_seqs=8)


def bake() -> None:
    from harness.bake import bake_checkpoint

    r = torch.load(R_PATH)
    man = bake_checkpoint(CKPT, BAKED, r.unsqueeze(0), 1.0)
    check("bake_edited_all_writers", len(man["edited_tensors"]) == man["n_writer_tensors"],
          edited=len(man["edited_tensors"]))


def steer() -> None:
    from harness import steering
    from harness.vllm_launch import inner_model
    from test_steering import max_cos, record_attnres, run

    llm = llm_for(CKPT)
    base = run(llm)
    st = steering.install(inner_model(llm))
    r = torch.load(R_PATH)
    steering.set_steering(st, r.unsqueeze(0), 1.0)
    states, restore = record_attnres()
    out = run(llm)
    restore()
    json.dump({"base": base, "steer": out, "steer_max_cos": max_cos(states, r.cuda())}, open(STEER_OUT, "w"))
    check("steer_ran", True)


def baked() -> None:
    from test_steering import max_cos, max_diff, record_attnres, run

    ref = json.load(open(STEER_OUT))
    llm = llm_for(BAKED)
    r = torch.load(R_PATH).cuda()
    states, restore = record_attnres()
    toks, plp = run(llm)
    restore()
    cos = max_cos(states, r)
    d_sb = max_diff(plp, ref["steer"][1])
    d_s0 = max_diff(ref["steer"][1], ref["base"][1])
    check("baked_stream_orthogonal", cos < 0.02, baked_max_cos=round(cos, 5),
          steer_max_cos=round(ref["steer_max_cos"], 5))
    check("baked_close_to_steered", d_sb < 0.25 * d_s0,
          prefill_max_diff_baked_vs_steer=round(d_sb, 4), prefill_max_diff_steer_vs_base=round(d_s0, 4),
          tokens_equal=toks == ref["steer"][0])


TARGETS = {  # label -> (layer, name suffix)
    "embed": (None, "embed_tokens.weight"),
    "kda_o_proj": (1, "self_attn.o_proj.weight"),
    "dsa_o_proj": (3, "self_attn.o_proj.weight"),
    "dense_down_proj": (0, "mlp.down_proj.weight"),
    "routed_expert_up_proj": (1, "routed_expert_up_proj.weight"),
    "shared_down_proj": (1, "shared_experts.down_proj.weight"),
}


def edit() -> None:
    from harness.vllm_launch import inner_model
    from test_steering import max_diff, run

    llm = llm_for(CKPT, graphs=GRAPHS)
    params = dict(inner_model(llm).named_parameters())
    _, base = run(llm)

    def find(layer, suffix):
        for n, p in params.items():
            if n.endswith(suffix) and (layer is None or n.startswith(f"layers.{layer}.")):
                return n, p
        raise KeyError(suffix)

    targets = {k: find(*v) for k, v in TARGETS.items()}
    # Control: a routed-expert tensor (not a writer; checks whether the MoE kernel reads it live).
    exp = [(n, p) for n, p in params.items() if n.startswith("layers.1.") and ".experts." in n and "w2" in n]
    if exp:
        targets["CONTROL_routed_expert_w2"] = exp[0]

    g = torch.Generator(device="cuda").manual_seed(0)
    for label, (name, p) in targets.items():
        orig = p.data.clone()
        if p.dtype.is_floating_point:
            p.data.add_(torch.randn(p.shape, generator=g, device=p.device, dtype=torch.float32).to(p.dtype)
                        * (orig.float().std() * 20).to(p.dtype))
        else:
            p.data.zero_()
        _, edited = run(llm)
        p.data.copy_(orig)
        _, restored = run(llm)
        changed = max_diff(edited, base)
        res = {"param": name, "dtype": str(p.dtype), "shape": list(p.shape),
               "edit_effect": round(changed, 4), "restore_diff": max_diff(restored, base)}
        if label.startswith("CONTROL"):
            results.append({"info": label, **res})
        else:
            check(f"edit_reaches_{label}", changed > 0.05 and res["restore_diff"] == 0.0, **res)


if __name__ == "__main__":
    {"bake": bake, "steer": steer, "baked": baked, "edit": edit}[MODE]()
    for r in results:
        print(json.dumps({**r, "mode": MODE + ("/graphs" if GRAPHS else "")}))
    sys.exit(0 if all(r.get("pass", True) for r in results) else 1)
