"""Single source of truth for how the harness launches vLLM on Kimi K3.x.

Mirrors the production Blackwell recipe (vllm-project/recipes models/moonshotai/Kimi-K3.yaml,
`blackwell` overrides) so kernels/paths under test match the 8xB300 run, minus things that
would bypass hooks/edits (see docs/k3_residual_map.md §4):
  - decode_backend stays "auto" (never "latency_specialized")
  - no speculative decoding (DSpark) except the capture-only extract_hidden_states path
  - prefix caching OFF by default: KV and KDA recurrent state depend on steering
"""

from __future__ import annotations

import os
from typing import Any

RECIPE_ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM": "1",  # capture the mixed AttnRes stream
    # In-process engine core. Measured (docs/runtime_notes.md "Determinism"): with the
    # multiprocess core, identical multi-prompt batches are scheduled into different step
    # compositions run-to-run, and K3.1's kernels are batch-variant (VLLM_BATCH_INVARIANT is
    # unsupported by both sparse-MLA backends) -> up to ~0.5 nat prefill logprob drift on the
    # tiny model. In-process: bitwise identical. Also lets hooks attach to the live model.
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


# NOTE: K3.1 (KDA + DSA) needs the packed-hybrid block-size clamp from
# vllm-project/vllm#67734; tools/setup_pod.sh applies it as a vendored patch.


def apply_env(extra: dict[str, str] | None = None) -> None:
    for k, v in {**RECIPE_ENV, **(extra or {})}.items():
        os.environ.setdefault(k, v)


def llm_kwargs(
    model: str,
    *,
    eager: bool = False,
    max_model_len: int = 4096,
    max_num_seqs: int = 16,
    gpu_memory_utilization: float = 0.6,
    tensor_parallel_size: int = 1,
    prefix_caching: bool = False,
    seed: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        model=model,
        trust_remote_code=True,
        language_model_only=True,
        kv_cache_dtype="fp8",
        # K3 recipe pins TOKENSPEED_MLA, but it rejects K3.1's sparse (DSA) layers
        # ("sparse not supported"), so let vLLM auto-select per layer type.
        enable_flashinfer_autotune=False,
        enable_prefix_caching=prefix_caching,
        enforce_eager=eager,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        additional_config={"decode_backend": "auto"},
    )
    kw.update(overrides)
    if (kw.get("additional_config") or {}).get("decode_backend") == "latency_specialized":
        raise ValueError("latency_specialized decode bypasses hooks and weight edits")
    return kw


def inner_model(llm):
    """The live KimiLinearModel (`.model` of the text LM). Requires the in-process engine."""
    core = llm.llm_engine.engine_core.engine_core
    model = core.model_executor.driver_worker.worker.model_runner.model
    lm = getattr(model, "language_model", model)
    return lm.model


def make_llm(model: str, **kwargs: Any):
    apply_env()
    from vllm import LLM

    return LLM(**llm_kwargs(model, **kwargs))
