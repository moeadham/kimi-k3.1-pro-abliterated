"""Control for edit-vs-steering comparisons: how much do teacher-forced prompt logprobs move under
perturbations that are numerically tiny (lam 1 vs 0.999; r vs r + 1e-3 noise)? On random tiny
weights this measures how chaotic the model is, i.e. the floor for any edit-vs-steer distance.

Usage: python tests_gpu/steer_sensitivity.py /vol/tiny-k3/mxfp4 [--eager]
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()
    os.environ["K3H_STEERING"] = "1"
    os.environ["K3H_STEERING_RANK"] = "8"
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from harness import tp
    from harness.k3chat import prompt_ids
    from harness.replay import load_tokenizer
    from harness.vllm_launch import make_llm

    llm = make_llm(a.model, eager=a.eager, max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=0.85)
    tok = load_tokenizer()
    qs = ["What is the capital of France?", "讲讲长城的历史。", "Write a haiku about rain.",
          "What happened at Tiananmen Square in 1989?"]
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids(tok, q, "low") + [1000 + 7 * i for i in range(40)]) for q in qs]
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def plp():
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return [torch.tensor([next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]]) for o in outs]

    def md(x, y):
        return max((u - v).abs().max().item() for u, v in zip(x, y))

    def mean_d(x, y):
        return float(torch.cat([(u - v).abs() for u, v in zip(x, y)]).mean())

    g = torch.Generator().manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(1, 7168, generator=g), dim=-1)
    r2 = torch.nn.functional.normalize(r + 1e-3 * torch.randn(1, 7168, generator=g), dim=-1)
    base = plp()
    out = {}
    for label, (q, lam) in {"r_1": (r, 1.0), "r_0999": (r, 0.999), "r2_1": (r2, 1.0)}.items():
        tp.set_steering(llm, q, lam)
        out[label] = plp()
    tp.clear_steering(llm)
    print(json.dumps({
        "eager": a.eager,
        "steer_effect_max": md(out["r_1"], base), "steer_effect_mean": mean_d(out["r_1"], base),
        "lam_1_vs_0999_max": md(out["r_1"], out["r_0999"]), "lam_1_vs_0999_mean": mean_d(out["r_1"], out["r_0999"]),
        "r_vs_r_noise_max": md(out["r_1"], out["r2_1"]), "r_vs_r_noise_mean": mean_d(out["r_1"], out["r2_1"]),
    }))


if __name__ == "__main__":
    main()
