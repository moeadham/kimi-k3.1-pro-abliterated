"""Residual-stream capture by replay prefill (step 1.3), in-process vLLM, eager.

What is captured: for every sublayer s (2 per decoder layer + the final output mix), the
PRE-NORM AttnRes mix that sublayer actually reads, i.e. softmax_mix(blocks, prefix)
without the RMSNorm gain (docs/k3_residual_map.md §1, §5). Index s = 2*layer (attention
input), 2*layer+1 (MLP input), 2*L (final mix fed to norm + lm_head).

How: wrap the model's `attn_res` op. After each real call (which updates prefix / writes
blocks in place), call it again read-only (delta=None, block_write_idx=-1, no output norm)
to get the pre-norm mix. Only during capture; zero overhead otherwise.

Replay: resubmit prompt+response token IDs as a prompt with max_tokens=1, one sequence
per step (so rows map 1:1 to token positions), no prefix caching, no chunking.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor


@contextmanager
def record_stream():
    import vllm.models.kimi_k3.nvidia.model as k3

    orig = k3.attn_res
    calls: list[Tensor] = []

    def wrapped(prefix, delta, blocks, norm_w, qk_w, out_norm_w, num_blocks, block_write_idx, eps, output_norm_eps):
        out = orig(prefix, delta, blocks, norm_w, qk_w, out_norm_w, num_blocks, block_write_idx, eps, output_norm_eps)
        if out_norm_w is None:
            mixed = out  # final output mix is already pre-norm
        else:
            mixed = orig(prefix, None, blocks, norm_w, qk_w, None, num_blocks, -1, eps, 0.0)
        calls.append(mixed.detach().clone())
        return out

    k3.attn_res = wrapped
    try:
        yield calls
    finally:
        k3.attn_res = orig


def capture(llm, token_ids: list[list[int]], positions: list[list[int]] | None = None,
            dtype: torch.dtype = torch.bfloat16) -> list[Tensor]:
    """Return, per sequence, a tensor [n_pos, S, d] of pre-norm stream states.

    positions[i] selects token positions of sequence i (default: all). Requires the
    in-process engine in eager mode (harness.vllm_launch.make_llm(..., eager=True)).
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    out: list[Tensor] = []
    for i, ids in enumerate(token_ids):
        with record_stream() as calls:
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        prefill = [c for c in calls if c.shape[0] == len(ids)]
        if not prefill:
            raise RuntimeError("no prefill-sized attn_res calls captured (chunked prefill or cache hit?)")
        states = torch.stack(prefill, dim=1)  # [T, S, d]
        pos = positions[i] if positions is not None else list(range(len(ids)))
        out.append(states[pos].to(dtype).cpu())
    return out


def save_captures(out_dir: Path, records: list[dict], states: list[Tensor]) -> None:
    """records[i]: JSON-serializable metadata (prompt id, positions, labels, ...)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({f"seq{i}": s.contiguous() for i, s in enumerate(states)}, out_dir / "states.safetensors")
    with open(out_dir / "index.jsonl", "w") as f:
        for i, r in enumerate(records):
            f.write(json.dumps({"key": f"seq{i}", "shape": list(states[i].shape), **r}) + "\n")
