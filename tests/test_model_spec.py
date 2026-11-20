"""Model-spec checks against the real K3.1 config/index, plus a CPU reference of the
AttnRes recurrence (docs/k3_residual_map.md §1) to confirm the orthogonality argument."""

import gzip
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from harness.ablation import project_activation
from harness.model_spec.kimi_k3 import NON_WRITER_SUFFIXES, K3Layout

FIX = Path(__file__).parent / "fixtures"


def layout() -> K3Layout:
    return K3Layout.from_config(json.loads((FIX / "k31_config.json").read_text()))


def test_layout_matches_known_k31_structure():
    lay = layout()
    assert lay.num_layers == 93 and lay.attn_res_block_size == 12
    types = [lay.attn_type(i) for i in range(93)]
    assert types.count("kda") == 69 and types.count("dsa") == 24
    assert lay.attn_type(3) == "dsa" and lay.attn_type(92) == "dsa" and lay.attn_type(0) == "kda"
    assert not lay.is_moe(0) and lay.is_moe(1)
    assert [i for i in range(93) if lay.is_block_write(i)] == list(range(0, 93, 12))


def test_all_writer_weights_exist_in_checkpoint_and_exclude_score_vectors():
    names = set(gzip.open(FIX / "k31_nonexpert_weight_names.txt.gz", "rt").read().split("\n"))
    writers = layout().all_writer_weights()
    missing = [w for w in writers if w not in names]
    assert not missing, missing[:5]
    assert len(writers) == 1 + 93 + 1 + 92 * 2  # embed + o_proj + dense down + (up, shared) per MoE layer
    assert not any(w.endswith(NON_WRITER_SUFFIXES) for w in writers)


# --- AttnRes reference (mirrors ops/attn_res.py _attn_res_kernel, fp32) ---------------

def _mix(prefix, blocks, nb, norm_w, proj_w, eps=1e-6):
    sources = torch.cat([blocks[:, :nb], prefix[:, None]], dim=1)  # [T, nb+1, d]
    rstd = torch.rsqrt(sources.pow(2).mean(-1) + eps)
    logits = (sources * (norm_w * proj_w)).sum(-1) * rstd
    return (logits.softmax(-1)[..., None] * sources).sum(1)


def run_attnres(n_layers, block, d, T, attn_fns, mlp_fns, embed, writer_proj=None, seed=0):
    """Returns the list of pre-norm mixed stream states read by each sublayer."""
    g = torch.Generator().manual_seed(seed)
    nw = [torch.randn(d, generator=g) for _ in range(2 * n_layers + 1)]
    pw = [torch.randn(d, generator=g) for _ in range(2 * n_layers + 1)]
    wp = writer_proj or (lambda y: y)
    blocks = torch.zeros(T, (n_layers + block - 1) // block, d)
    prefix, pending, states = wp(embed), None, []
    for i in range(n_layers):
        if pending is not None:
            prefix = prefix + pending
        if i % block == 0:
            blocks[:, i // block] = prefix
        prev_valid = (i + block - 1) // block
        x = _mix(prefix, blocks, prev_valid, nw[2 * i], pw[2 * i])
        states.append(x)
        a = wp(attn_fns[i](F.normalize(x, dim=-1)))
        if i % block == 0:
            prefix = a
        else:
            prefix = prefix + a
        valid = prev_valid + (i % block == 0)
        x = _mix(prefix, blocks, valid, nw[2 * i + 1], pw[2 * i + 1])
        states.append(x)
        pending = wp(mlp_fns[i](F.normalize(x, dim=-1)))
    final = _mix(prefix + pending, blocks, blocks.shape[1], nw[-1], pw[-1])
    states.append(final)
    return states


def test_writer_orthogonality_propagates_through_attnres():
    torch.manual_seed(0)
    L, B, d, T = 7, 3, 32, 5
    attn = [torch.nn.Linear(d, d, bias=False) for _ in range(L)]
    mlp = [torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.SiLU(), torch.nn.Linear(2 * d, d)) for _ in range(L)]
    embed = torch.randn(T, d)
    r = F.normalize(torch.randn(d), dim=0)
    with torch.no_grad():
        base = run_attnres(L, B, d, T, attn, mlp, embed)
        abl = run_attnres(L, B, d, T, attn, mlp, embed, writer_proj=lambda y: project_activation(y, r))
    assert max((s @ r).abs().max().item() for s in base) > 1e-2  # baseline has an r component
    assert max((s @ r).abs().max().item() for s in abl) < 1e-5  # every mixed stream state is ⊥ r
