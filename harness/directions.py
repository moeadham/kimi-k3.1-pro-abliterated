"""Direction extraction from captured residual activations.

All inputs are float tensors shaped [n, L, d] (samples, layers, model dim) unless noted.
Outputs are unit vectors (or orthonormal bases) per layer, in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def winsorize(x: Tensor, quantile: float) -> Tensor:
    """Symmetric per-(sample, layer) clamp of components to the given |x| quantile.

    quantile >= 1 disables it. Tames 'massive activation' dimensions.
    """
    if quantile >= 1.0:
        return x
    thr = torch.quantile(x.abs().float(), quantile, dim=-1, keepdim=True)
    return torch.clamp(x, -thr, thr)


def mean_diff(bad: Tensor, good: Tensor, winsorization_quantile: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    """Difference-of-means direction per layer.

    Returns (direction [L, d] unit, bad_mean [L, d], good_mean [L, d]).
    """
    bad = winsorize(bad.float(), winsorization_quantile)
    good = winsorize(good.float(), winsorization_quantile)
    bad_mean = bad.mean(0, dtype=torch.float64).float()
    good_mean = good.mean(0, dtype=torch.float64).float()
    return F.normalize(bad_mean - good_mean, dim=-1), bad_mean, good_mean


def project_out_good(direction: Tensor, good_mean: Tensor) -> Tensor:
    """'Projected abliteration': remove the component of `direction` along the good mean.

    Only the part of the bad-vs-good difference orthogonal to normal activations is ablated.
    """
    g = F.normalize(good_mean, dim=-1)
    r = direction - (direction * g).sum(-1, keepdim=True) * g
    return F.normalize(r, dim=-1)


def pca_directions(bad: Tensor, good: Tensor, k: int, winsorization_quantile: float = 1.0) -> Tensor:
    """Multi-direction basis per layer: top-k principal directions of bad samples
    centered on the good mean (captures several 'failure modes' rather than one mean).

    Returns an orthonormal basis [L, k, d].
    """
    bad = winsorize(bad.float(), winsorization_quantile)
    good_mean = winsorize(good.float(), winsorization_quantile).mean(0)
    n, L, d = bad.shape
    # centered one layer at a time: no full [n, L, d] copy
    return torch.stack([top_k_directions(bad[:, layer, :] - good_mean[layer], k, seed=layer) for layer in range(L)])


def top_k_directions(x: Tensor, k: int, seed: int = 0) -> Tensor:
    """Top-k right singular vectors of x [n, d] -> [k, d] orthonormal rows (zero rows if rank < k).
    Small inputs: exact SVD. Otherwise a seeded randomized SVD: same span to numerical precision; a full SVD
    of [~5k, 7168] at each of 187 stream indices took ~7-13 min per family on 32 CPUs."""
    n, d = x.shape
    if float(x.abs().max()) == 0.0:  # degenerate (e.g. identical template tokens at layer 0): no direction
        return torch.zeros(k, d, dtype=x.dtype)
    if min(n, d) <= 4 * k:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        top = vh[:k]
    else:
        _, _, v = _svd_lowrank(x, k + 8, torch.Generator().manual_seed(seed))
        top = v[:, :k].T
    if top.shape[0] < k:
        top = torch.cat([top, torch.zeros(k - top.shape[0], d, dtype=top.dtype)])
    return top.contiguous()


def _svd_lowrank(x: Tensor, q: int, g: torch.Generator, niter: int = 6):
    """Seeded randomized SVD (Halko et al.): x [n, d] -> U [n, q], S [q], V [d, q]."""
    omega = torch.randn(x.shape[1], q, generator=g, dtype=x.dtype)
    y = x @ omega
    for _ in range(niter):  # power iterations sharpen the top of the spectrum
        y, _ = torch.linalg.qr(y)
        y = x @ (x.T @ y)
    qm, _ = torch.linalg.qr(y)
    ub, s, vh = torch.linalg.svd(qm.T @ x, full_matrices=False)
    return qm @ ub, s, vh.T


def interpolate_layer_direction(directions: Tensor, index: float) -> Tensor:
    """Global direction at fractional layer index (linear interpolation, renormalized)."""
    lo = int(index)
    hi = min(lo + 1, directions.shape[0] - 1)
    w = index - lo
    return F.normalize(torch.lerp(directions[lo], directions[hi], w), dim=-1)
