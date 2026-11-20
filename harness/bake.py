"""Bake directional ablation into a K3.x safetensors checkpoint (step 1.6).

Edits only the residual-stream writer weights (docs/k3_residual_map.md §2), all BF16:
  embed_tokens                               E <- E - lam (E Q^T) Q
  layers.i.self_attn.o_proj                  W <- W - lam Q^T (Q W)
  layers.0.mlp.down_proj (dense)             "
  layers.i.block_sparse_moe.routed_expert_up_proj + shared_experts.down_proj   "
Every other tensor, and every shard that holds no writer, is copied byte-identical.
Slot convention matches harness/steering.py: 0 = embed, 1+2i = attn i, 2+2i = mlp i, so a
bake with (Q, lam) is the weight-space equivalent of steering with the same (Q, lam).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from harness.ablation import ablate_embedding, ablate_output_weight
from harness.model_spec.kimi_k3 import K3Layout, Writer
from harness.steering import slot_of


def writer_slots(layout: K3Layout) -> dict[str, int]:
    """Checkpoint tensor name -> steering slot."""
    p = layout.checkpoint_prefix
    out = {p + "model.embed_tokens.weight": 0}
    for i in range(layout.num_layers):
        ws = layout.writer_weights(i)
        for name in ws[Writer.ATTN]:
            out[name] = slot_of("attn", i)
        for name in ws[Writer.MLP]:
            out[name] = slot_of("mlp", i)
    return out


def _basis_for_slot(Q: Tensor, slot: int) -> Tensor:
    return Q[slot] if Q.dim() == 3 else Q


def _lam_for_slot(lam: Tensor | float, slot: int) -> float:
    return float(lam[slot]) if isinstance(lam, Tensor) and lam.dim() > 0 else float(lam)


def bake_checkpoint(src: Path, dst: Path, Q: Tensor, lam: Tensor | float,
                    preserve_row_norms: bool = False) -> dict:
    """Write an ablated copy of the checkpoint at `src` into `dst`. Returns the manifest.

    Q: [k, d] (same basis for all slots) or [n_slots, k, d]; lam: scalar or [n_slots].
    """
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    layout = K3Layout.from_config(cfg)
    slots = writer_slots(layout)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]

    missing = [n for n in slots if n not in weight_map]
    if missing:
        raise KeyError(f"writer tensors missing from checkpoint: {missing[:5]}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, dst / f.name)

    shards_with_writers = {weight_map[n] for n in slots}
    edited: list[str] = []
    for shard in sorted(set(weight_map.values())):
        if shard not in shards_with_writers:
            shutil.copy2(src / shard, dst / shard)  # byte-identical
            continue
        tensors: dict[str, Tensor] = {}
        with safe_open(src / shard, framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if name in slots:
                    s = slots[name]
                    lam_s = _lam_for_slot(lam, s)
                    if lam_s != 0.0:
                        Qs = _basis_for_slot(Q, s)
                        if s == 0:
                            t = ablate_embedding(t, Qs, lam_s)
                        else:
                            t = ablate_output_weight(t, Qs, lam_s, preserve_row_norms)
                        edited.append(name)
                tensors[name] = t.contiguous()
        save_file(tensors, dst / shard, metadata=meta)

    q_bytes = Q.detach().float().cpu().numpy().tobytes()
    manifest = {
        "source": str(src),
        "direction_sha256": hashlib.sha256(q_bytes).hexdigest(),
        "basis_shape": list(Q.shape),
        "lam": lam.tolist() if isinstance(lam, Tensor) else lam,
        "preserve_row_norms": preserve_row_norms,
        "edited_tensors": edited,
        "n_writer_tensors": len(slots),
    }
    torch.save({"Q": Q.detach().float().cpu(), "lam": lam}, dst / "bake_direction.pt")
    (dst / "bake_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
