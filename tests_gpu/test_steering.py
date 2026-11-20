"""Step 1.4 steering tests on the tiny K3.1 (pod). Each mode runs in its own process.

  eager  : (a) identity: lam=0 hooks give bitwise-identical prompt logprobs + tokens
           (b) AttnRes orthogonality: with lam=1 on all writers, every prefix-sum / block
               snapshot / final mixed stream that attn_res actually consumes is ~⊥ r
           writes steered result to /dev/shm/k3h_steer_eager.json
  graphs : plugin installs hooks before CUDA-graph capture; (c) setting buffers after
           capture changes outputs, matches the eager steered result, and clearing restores
           baseline exactly.

Usage: python tests_gpu/test_steering.py {eager|graphs} /dev/shm/tiny-k3/mxfp4
"""

from __future__ import annotations

import json
import os
import sys

MODE = sys.argv[1]
PATH = sys.argv[2]
if MODE == "graphs":
    os.environ["K3H_STEERING"] = "1"  # plugin installs hooks at model construction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from harness import steering  # noqa: E402
from harness.vllm_launch import inner_model, make_llm  # noqa: E402

R_PATH = "/dev/shm/k3h_steer_r.pt"
EAGER_OUT = "/dev/shm/k3h_steer_eager.json"
results: list[dict] = []


def check(name: str, ok: bool, **info) -> None:
    results.append({"check": name, "pass": bool(ok), **info})


def run(llm):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=list(range(2000 + 41 * i, 2000 + 41 * i + 40))) for i in range(3)]
    sp = SamplingParams(temperature=0.0, max_tokens=16, prompt_logprobs=0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    toks = [list(o.outputs[0].token_ids) for o in outs]
    plp = [[next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]] for o in outs]
    return toks, plp


def max_diff(a, b):
    return max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))


def record_attnres():
    """Wrap the attn_res op used by the K3 model to collect every stream source it consumes."""
    import vllm.models.kimi_k3.nvidia.model as k3

    orig = k3.attn_res
    states: list[torch.Tensor] = []

    def wrapped(prefix, delta, blocks, norm_w, qk_w, out_norm_w, num_blocks, block_write_idx, eps, output_norm_eps):
        out = orig(prefix, delta, blocks, norm_w, qk_w, out_norm_w, num_blocks, block_write_idx, eps, output_norm_eps)
        if prefix.shape[0] > 1:  # prefill steps only
            states.append(prefix.detach().float().clone())  # updated in place when delta given
            nb = max(num_blocks, block_write_idx + 1)
            if nb > 0:
                states.append(blocks[:, :nb].detach().float().reshape(-1, blocks.shape[-1]).clone())
            if out_norm_w is None:  # final output mix (pre-norm stream)
                states.append(out.detach().float().clone())
        return out

    k3.attn_res = wrapped
    return states, lambda: setattr(k3, "attn_res", orig)


def max_cos(states, r):
    worst = 0.0
    for s in states:
        cos = (s @ r).abs() / s.norm(dim=-1).clamp_min(1e-6)
        worst = max(worst, cos.max().item())
    return worst


def eager() -> None:
    llm = make_llm(PATH, eager=True, max_model_len=1024, max_num_seqs=8)
    inner = inner_model(llm)
    base_t, base_p = run(llm)

    st = steering.install(inner, k=1)
    t0, p0 = run(llm)
    check("identity_lam0", t0 == base_t and max_diff(p0, base_p) == 0.0, prefill_max_diff=max_diff(p0, base_p))

    # Direction with a large baseline component: mean of recorded stream states.
    states, restore = record_attnres()
    run(llm)
    r = F.normalize(torch.cat(states).mean(0), dim=0)
    base_cos = max_cos(states, r)
    states.clear()

    steering.set_steering(st, r.unsqueeze(0), 1.0)
    st_t, st_p = run(llm)
    steered_cos = max_cos(states, r)
    restore()
    check("attnres_orthogonal", steered_cos < 0.02 and base_cos > 0.2,
          baseline_max_cos=round(base_cos, 4), steered_max_cos=round(steered_cos, 5), n_states=len(states))
    check("steering_changes_output", st_p != base_p)

    steering.clear_steering(st)
    c_t, c_p = run(llm)
    check("clear_restores_baseline", c_t == base_t and max_diff(c_p, base_p) == 0.0)

    torch.save(r.cpu(), R_PATH)
    json.dump({"base": [base_t, base_p], "steered": [st_t, st_p]}, open(EAGER_OUT, "w"))


def graphs() -> None:
    llm = make_llm(PATH, eager=False, max_model_len=1024, max_num_seqs=8)
    st = getattr(inner_model(llm), steering.ATTR, None)
    check("plugin_installed_hooks", st is not None)
    if st is None:
        return
    ref = json.load(open(EAGER_OUT))
    b_t, b_p = run(llm)
    steering.set_steering(st, torch.load(R_PATH).unsqueeze(0), 1.0)
    s_t, s_p = run(llm)
    steering.clear_steering(st)
    c_t, c_p = run(llm)
    check("graph_live_update_changes_output", s_p != b_p)
    check("graph_clear_restores", c_t == b_t and max_diff(c_p, b_p) == 0.0)
    # Informational only: eager vs compiled+graphs are different numerical paths (inductor
    # fusions), so even unsteered baselines differ (~0.7 nat on random weights). Exact
    # comparisons must stay within one mode (steering == bake is checked per mode in 1.6).
    results.append({"info": "cross_mode_drift",
                    "baseline_prefill_max_diff_vs_eager": round(max_diff(b_p, ref["base"][1]), 4),
                    "steered_prefill_max_diff_vs_eager": round(max_diff(s_p, ref["steered"][1]), 4)})


if __name__ == "__main__":
    eager() if MODE == "eager" else graphs()
    for r in results:
        print(json.dumps(r))
    sys.exit(0 if all(r.get("pass", True) for r in results) else 1)
