"""Weight-space directional ablation for residual-stream writers.

Conventions (PyTorch nn.Linear): W has shape [d_out, d_in] and y = W x, so for a
writer into the residual stream d_out == d_model and the direction r lives in d_out.

Ablating basis Q (orthonormal rows, [k, d_model]) with strength lam:
    W' = W - lam * Q^T (Q W)
lam = 1 makes every output of W orthogonal to span(Q).

Embedding matrices are [vocab, d_model] (rows are written into the stream), so
    E' = E - lam * (E Q^T) Q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _basis(r: Tensor) -> Tensor:
    """Accept a single direction [d] or a basis [k, d]; return an orthonormal basis of its span, [k', d] float32,
    k' <= k. Zero rows and linearly dependent rows are dropped: plain QR would turn them into arbitrary unit
    vectors (a zero direction at a degenerate layer, or two parallel directions in a combined edit, would ablate a
    random direction). An all-zero input gives [0, d], i.e. no edit."""
    q = r.float()
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = q[q.norm(dim=1) > 1e-8]
    if q.shape[0] == 0:
        return q.contiguous()
    qt, rr = torch.linalg.qr(q.T)  # keeps the span, fixes drift
    diag = rr.diagonal().abs()
    keep = diag > 1e-6 * diag.max()
    return qt.T[keep].contiguous()


def ablate_output_weight(W: Tensor, r: Tensor, lam: float = 1.0, preserve_row_norms: bool = False) -> Tensor:
    """Return the ablated copy of a writer weight W [d_model, d_in] (dtype preserved).

    preserve_row_norms implements MPOA-style magnitude preservation: each row of W
    (one output coordinate) is rescaled back to its original norm after ablation.
    """
    if lam == 0.0:
        return W.clone()
    Q = _basis(r).to(W.device)
    W32 = W.float()
    W_new = W32 - lam * Q.T @ (Q @ W32)
    if preserve_row_norms:
        norms = W32.norm(dim=1, keepdim=True)
        W_new = F.normalize(W_new, dim=1) * norms
    return W_new.to(W.dtype)


def ablate_embedding(E: Tensor, r: Tensor, lam: float = 1.0) -> Tensor:
    """Return the ablated copy of an embedding matrix E [vocab, d_model]."""
    if lam == 0.0:
        return E.clone()
    Q = _basis(r).to(E.device)
    E32 = E.float()
    return (E32 - lam * (E32 @ Q.T) @ Q).to(E.dtype)


def project_activation(y: Tensor, r: Tensor, lam: float | Tensor = 1.0) -> Tensor:
    """Inference-time equivalent on activations y [..., d_model]: y - lam * (y Q^T) Q.

    This is what the steering plugin applies to writer outputs; for lam/r fixed it is
    mathematically identical to ablate_output_weight applied to the writer's weight
    (bias-free writers).
    """
    Q = _basis(r).to(y.device)
    y32 = y.float()
    return (y32 - lam * (y32 @ Q.T) @ Q).to(y.dtype)
