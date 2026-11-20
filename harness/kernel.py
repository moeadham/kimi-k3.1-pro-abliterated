"""Per-layer ablation strength kernel (Heretic-style, reimplemented).

Strength peaks at `max_weight` at layer `peak`, decays linearly to `min_weight` at
distance `falloff`, and is 0 beyond that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class WeightKernel:
    max_weight: float
    peak: float
    min_weight: float
    falloff: float

    def __post_init__(self) -> None:
        if self.falloff <= 0:
            raise ValueError("falloff must be > 0")

    def weights(self, n_layers: int) -> torch.Tensor:
        idx = torch.arange(n_layers, dtype=torch.float32)
        dist = (idx - self.peak).abs()
        w = self.max_weight + (dist / self.falloff) * (self.min_weight - self.max_weight)
        return torch.where(dist > self.falloff, torch.zeros_like(w), w)

    def to_dict(self) -> dict:
        return asdict(self)
