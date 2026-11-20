"""Inference-time directional ablation (steering) for Kimi K3 in vLLM (step 1.4).

Projects a direction basis out of every residual-stream writer output:
    y <- y - lam_slot * (y Q_slot^T) Q_slot
Slots: 0 = embed_tokens, 1 + 2i = layers.i.self_attn, 2 + 2i = layers.i.mlp
(see docs/k3_residual_map.md §2-3). Projection only: it is linear, so it is exact even on
per-rank partial sums under TP/SP; additive steering is deliberately unsupported.

Buffers live on the model's device and are updated IN PLACE, so hooks installed before
torch.compile / CUDA-graph capture keep working without recapture. With all lam = 0 the
hooks are an exact identity (y - 0 * ... == y bitwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

ATTR = "_k3h_steering"


@dataclass
class SteeringState:
    Q: Tensor  # [n_slots, k, d] orthonormal rows per slot
    lam: Tensor  # [n_slots] float32
    handles: list

    @property
    def n_slots(self) -> int:
        return self.lam.shape[0]


def slot_of(kind: str, layer: int) -> int:
    if kind == "embed":
        return 0
    return 1 + 2 * layer + (0 if kind == "attn" else 1)


def _make_hook(state: SteeringState, slot: int):
    def hook(_mod, _inp, out):
        y = out[0] if isinstance(out, tuple) else out
        Q = state.Q[slot].to(y.dtype)  # [k, d]
        coef = (y @ Q.T) * state.lam[slot].to(y.dtype)  # [T, k]
        y2 = y - coef @ Q
        if isinstance(out, tuple):
            return (y2, *out[1:])
        return y2

    return hook


def install(inner_model: nn.Module, k: int = 1) -> SteeringState:
    """Attach identity-initialized projection hooks to a KimiLinearModel (`.model`)."""
    if hasattr(inner_model, ATTR):
        return getattr(inner_model, ATTR)
    layers = inner_model.layers
    d = inner_model.embed_tokens.weight.shape[-1]
    dev = inner_model.embed_tokens.weight.device
    n_slots = 1 + 2 * len(layers)
    state = SteeringState(
        Q=torch.zeros(n_slots, k, d, device=dev, dtype=torch.float32),
        lam=torch.zeros(n_slots, device=dev, dtype=torch.float32),
        handles=[],
    )
    state.handles.append(inner_model.embed_tokens.register_forward_hook(_make_hook(state, 0)))
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Module) and hasattr(layer, "self_attn"):
            state.handles.append(layer.self_attn.register_forward_hook(_make_hook(state, slot_of("attn", i))))
            state.handles.append(layer.mlp.register_forward_hook(_make_hook(state, slot_of("mlp", i))))
    setattr(inner_model, ATTR, state)
    return state


def set_steering(state: SteeringState, Q: Tensor, lam: Tensor) -> None:
    """Q: [n_slots, k', d] or [k', d] (broadcast to all slots), k' <= k; lam: [n_slots] or scalar."""
    Q = Q.to(state.Q.device, torch.float32)
    if Q.dim() == 2:
        Q = Q.unsqueeze(0).expand(state.n_slots, -1, -1)
    if Q.shape[1] > state.Q.shape[1]:
        raise ValueError(f"basis rank {Q.shape[1]} exceeds installed k={state.Q.shape[1]}")
    # orthonormalize each slot's basis
    Qn = torch.linalg.qr(Q.transpose(1, 2))[0].transpose(1, 2)
    state.Q.zero_()
    state.Q[:, : Qn.shape[1]].copy_(Qn)
    state.lam.copy_(torch.as_tensor(lam, dtype=torch.float32, device=state.lam.device).expand_as(state.lam))


def clear_steering(state: SteeringState) -> None:
    state.lam.zero_()
