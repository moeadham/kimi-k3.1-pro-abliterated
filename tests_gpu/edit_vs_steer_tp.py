"""Steering vs in-memory edit at TP>1, without decode divergence (pod/Modal).

harness.session `checks` compares 48-token greedy generations; on random tiny weights a single
near-tie argmax flip makes the sequences diverge, so a large max |diff| there does not by itself
mean the edit is wrong. Here everything is teacher-forced: prompt logprobs of fixed token
sequences (no sampling), for baseline / steered (lam=1) / edited / restored, same batch.

Usage: python tests_gpu/edit_vs_steer_tp.py /vol/tiny-k3/mxfp4 --tp 2 [--eager]
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--groups", action="store_true", help="also compare per writer group (lam only on its slots)")
    a = ap.parse_args()
    os.environ["K3H_STEERING"] = "1"
    os.environ["K3H_STEERING_RANK"] = "8"
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from harness import tp
    from harness.k3chat import prompt_ids
    from harness.replay import load_tokenizer
    from harness.vllm_launch import make_llm

    llm = make_llm(a.model, eager=a.eager, tensor_parallel_size=a.tp, max_model_len=4096, max_num_seqs=16,
                   gpu_memory_utilization=0.85)
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

    g = torch.Generator().manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(1, 7168, generator=g), dim=-1)
    base, base2 = plp(), plp()

    def compare(lam, label):
        tp.set_steering(llm, r, lam)
        steered = plp()
        tp.clear_steering(llm)
        er = tp.edit_writers(llm, r, lam)
        edited = plp()
        tp.restore_writers(llm)
        restored = plp()
        res = {
            "group": label, "tp": a.tp, "eager": a.eager, "edited_per_rank": [x["edited"] for x in er],
            "base_repeat": md(base, base2), "steer_effect": md(steered, base), "edit_effect": md(edited, base),
            "edit_vs_steer": md(edited, steered), "restore": md(restored, base),
        }
        res["ratio"] = round(res["edit_vs_steer"] / max(res["steer_effect"], 1e-9), 4)
        print(json.dumps(res), flush=True)

    compare(1.0, "all")
    if a.groups:
        # slots: 0 embed; 2i+1 attn of layer i; 2i+2 mlp of layer i (tiny: DSA layers 3,7; dense MLP layer 0)
        n_slots = 2 * 8 + 1
        groups = {"embed": [0], "attn_kda": [2 * i + 1 for i in (0, 1, 2, 4, 5, 6)], "attn_dsa": [7, 15],
                  "mlp_dense0": [2], "mlp_moe": [2 * i + 2 for i in range(1, 8)]}
        for label, slots in groups.items():
            lam = torch.zeros(n_slots)
            lam[slots] = 1.0
            compare(lam, label)


if __name__ == "__main__":
    main()
