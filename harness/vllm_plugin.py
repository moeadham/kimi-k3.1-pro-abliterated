"""vLLM general plugin: install steering hooks at model construction (before compile and
CUDA-graph capture). Enabled only when K3H_STEERING=1; K3H_STEERING_RANK sets max basis rank.

Registered via pyproject entry point `vllm.general_plugins` -> `k3h = harness.vllm_plugin:register`.
"""

from __future__ import annotations

import os


def register() -> None:
    if os.environ.get("K3H_STEERING", "0") != "1":
        return
    import vllm.models.kimi_k3.nvidia.model as k3

    from harness import steering

    cls = k3.KimiLinearModel
    if getattr(cls, "_k3h_patched", False):
        return
    orig_init = cls.__init__
    rank = int(os.environ.get("K3H_STEERING_RANK", "1"))

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        steering.install(self, k=rank)

    cls.__init__ = __init__
    cls._k3h_patched = True
