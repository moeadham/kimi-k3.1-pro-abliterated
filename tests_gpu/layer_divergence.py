"""Localize run-to-run nondeterminism: hook every writer module (embed, layers.i.self_attn,
layers.i.mlp) plus the MoE router logits, run the same prefill twice, and report per-module
max |diff| and whether routed expert selection changed.

Runs the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so hooks can be attached to
the live model directly. Eager only.

Usage (pod): python tests_gpu/layer_divergence.py /dev/shm/tiny-k3/bf16
"""

from __future__ import annotations

import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

from harness.vllm_launch import make_llm  # noqa: E402


def find_inner_model(llm):
    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    model = runner.model
    lm = getattr(model, "language_model", model)
    return lm.model  # KimiLinearModel


def main() -> None:
    path = sys.argv[1]
    min_len = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # >0: use the shortest replay row with len >= min_len
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    mml = 16384 if min_len else 1024
    llm = make_llm(path, eager=True, max_model_len=mml, max_num_seqs=8, max_num_batched_tokens=mml)
    inner = find_inner_model(llm)

    records: dict[str, list[torch.Tensor]] = {}

    def hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] > 1:  # prefill only
                records.setdefault(name, []).append(t.detach().float().cpu().clone())
        return fn

    handles = [inner.embed_tokens.register_forward_hook(hook("embed"))]
    for i, layer in enumerate(inner.layers):
        handles.append(layer.self_attn.register_forward_hook(hook(f"L{i}.attn")))
        handles.append(layer.mlp.register_forward_hook(hook(f"L{i}.mlp")))
        if hasattr(layer.mlp, "gate"):
            handles.append(layer.mlp.gate.register_forward_hook(hook(f"L{i}.router")))

    ids = list(range(1000, 1048))
    if min_len:
        from harness.session import replay_rows
        ids = min((r["ids"] for r in replay_rows() if len(r["ids"]) >= min_len), key=len)
    print(json.dumps({"prompt_len": len(ids)}))
    prompt = [TokensPrompt(prompt_token_ids=ids)]
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    for _ in range(2):
        llm.generate(prompt, sp, use_tqdm=False)
    for h in handles:
        h.remove()

    rows = []
    for name, (a, b) in ((k, v[:2]) for k, v in records.items() if len(v) >= 2):
        diff = (a - b).abs().max().item()
        row = {"module": name, "max_abs_diff": diff, "scale": a.abs().max().item()}
        tok_diff = (a - b).abs().reshape(a.shape[0], -1).amax(-1) > 0
        row["first_diff_token"] = int(tok_diff.nonzero()[0]) if tok_diff.any() else None
        row["n_diff_tokens"] = int(tok_diff.sum())
        if name.endswith("router"):
            k = 4
            row["topk_changed_tokens"] = int((a.topk(k).indices.sort().values != b.topk(k).indices.sort().values).any(-1).sum())
        rows.append(row)
    order = ["embed"] + [f"L{i}.{s}" for i in range(len(inner.layers)) for s in ("attn", "router", "mlp")]
    rows.sort(key=lambda r: order.index(r["module"]) if r["module"] in order else 999)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
