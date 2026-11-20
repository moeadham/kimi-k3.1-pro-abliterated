"""CPU tests for harness.bake on a synthetic mini K3-style checkpoint."""

import hashlib
import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from harness.bake import bake_checkpoint, writer_slots
from harness.model_spec.kimi_k3 import K3Layout

D, V, H = 32, 50, 16


def make_ckpt(tmp_path):
    cfg = {"text_config": {
        "num_hidden_layers": 2, "first_k_dense_replace": 1, "attn_res_block_size": 2,
        "linear_attn_config": {"kda_layers": [1], "full_attn_layers": [2]},
        "dsa_config": {"dsa_layers": [2]},
    }}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)  # noqa: E731
    p = "language_model.model."
    shard1 = {  # writers + non-writers mixed
        p + "embed_tokens.weight": r(V, D),
        p + "layers.0.self_attn.o_proj.weight": r(D, H),
        p + "layers.0.mlp.down_proj.weight": r(D, H),
        p + "layers.0.self_attention_res_proj.weight": r(1, D),
    }
    shard2 = {
        p + "layers.1.self_attn.o_proj.weight": r(D, H),
        p + "layers.1.block_sparse_moe.routed_expert_up_proj.weight": r(D, H),
        p + "layers.1.block_sparse_moe.shared_experts.down_proj.weight": r(D, H),
        p + "layers.1.block_sparse_moe.gate.weight": r(4, D),
    }
    shard3 = {"language_model.lm_head.weight": r(V, D)}  # no writers
    wm = {}
    for i, sh in enumerate((shard1, shard2, shard3), 1):
        name = f"model-{i:05d}-of-00003.safetensors"
        save_file(sh, tmp_path / name, metadata={"format": "pt"})
        wm.update({k: name for k in sh})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return cfg


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_bake_edits_only_writers_and_makes_them_orthogonal(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    make_ckpt(src)
    r = F.normalize(torch.randn(D), dim=0)
    man = bake_checkpoint(src, dst, r.unsqueeze(0), 1.0)
    assert len(man["edited_tensors"]) == man["n_writer_tensors"] == 6

    # untouched shard is byte-identical
    assert sha(src / "model-00003-of-00003.safetensors") == sha(dst / "model-00003-of-00003.safetensors")
    a = {**load_file(src / "model-00001-of-00003.safetensors"), **load_file(src / "model-00002-of-00003.safetensors")}
    b = {**load_file(dst / "model-00001-of-00003.safetensors"), **load_file(dst / "model-00002-of-00003.safetensors")}
    slots = writer_slots(K3Layout.from_config(json.loads((src / "config.json").read_text())))
    for name in a:
        if name in slots:
            w = b[name].float()
            resid = (w @ r).abs().max() if "embed" in name else (r @ w).abs().max()
            assert resid < 0.05 * w.abs().max(), name  # bf16 rounding residue only
            assert b[name].dtype == torch.bfloat16
        else:
            assert torch.equal(a[name], b[name]), name  # non-writers untouched


def test_lambda_zero_is_byte_identical(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    make_ckpt(src)
    man = bake_checkpoint(src, dst, F.normalize(torch.randn(1, D), dim=-1), 0.0)
    assert man["edited_tensors"] == []
    for i in (1, 2, 3):
        a = load_file(src / f"model-{i:05d}-of-00003.safetensors")
        b = load_file(dst / f"model-{i:05d}-of-00003.safetensors")
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_slot_mapping_matches_steering_convention(tmp_path):
    cfg = make_ckpt(tmp_path)
    s = writer_slots(K3Layout.from_config(cfg))
    p = "language_model.model."
    assert s[p + "embed_tokens.weight"] == 0
    assert s[p + "layers.0.self_attn.o_proj.weight"] == 1
    assert s[p + "layers.0.mlp.down_proj.weight"] == 2
    assert s[p + "layers.1.self_attn.o_proj.weight"] == 3
    assert s[p + "layers.1.block_sparse_moe.routed_expert_up_proj.weight"] == 4
    assert s[p + "layers.1.block_sparse_moe.shared_experts.down_proj.weight"] == 4
