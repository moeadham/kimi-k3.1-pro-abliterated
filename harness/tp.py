"""Multi-GPU (TP) control of steering, in-memory writer edits and stream capture.

At TP>1 the model lives in worker processes; everything here runs through
`llm.collective_rpc(fn, args=...)`, where fn(worker, *args) executes on every rank.

Correctness arguments (docs/k3_residual_map.md §3):
- Steering is a projection on writer OUTPUTS, linear, so it is exact on per-rank partial
  sums (gemm_rs_ar, SP reduce-scatter, fused all-reduce). Each rank holds its own buffers.
- In-memory edits act on each rank's weight shard. Valid only when the shard keeps the full
  d_model OUTPUT dimension (row-parallel / replicated linears; vocab-parallel embedding rows).
  `w_edit_writers` asserts that rather than assuming it.
- Capture: the pre-norm AttnRes mix must be full on the capturing rank. `w_capture_take`
  returns per-rank L1 + element samples; the driver requires ranks to agree within a loose tolerance (stream replicated) and
  fails loudly otherwise (sequence-parallel sharding would need a gather).
"""

from __future__ import annotations

import torch

from harness import ablation, steering

_ORIG: dict[str, torch.Tensor] = {}  # per-worker originals of edited writer weights (CPU)
_CAP: dict = {}


def _inner(worker):
    m = worker.model_runner.model
    m = getattr(m, "language_model", m)
    return m.model


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


# ---------- steering ----------

def w_steering_info(worker) -> dict:
    st = getattr(_inner(worker), steering.ATTR, None)
    return {"rank": _rank(), "installed": st is not None,
            "k": None if st is None else st.Q.shape[1], "n_slots": None if st is None else st.n_slots}


def w_set_steering(worker, Q: torch.Tensor, lam) -> int:
    steering.set_steering(getattr(_inner(worker), steering.ATTR), Q, lam)
    return _rank()


def w_clear_steering(worker) -> int:
    steering.clear_steering(getattr(_inner(worker), steering.ATTR))
    return _rank()


def set_steering(llm, Q, lam) -> None:
    llm.collective_rpc(w_set_steering, args=(Q.cpu(), lam.cpu() if isinstance(lam, torch.Tensor) else lam))
    llm.reset_prefix_cache()


def clear_steering(llm) -> None:
    llm.collective_rpc(w_clear_steering)
    llm.reset_prefix_cache()


# ---------- in-memory writer edits ----------

WRITER_SUFFIXES = ("self_attn.o_proj.weight", "mlp.down_proj.weight", "routed_expert_up_proj.weight",
                   "shared_experts.down_proj.weight")


def writer_params(inner) -> dict[str, tuple[int, torch.nn.Parameter]]:
    """name -> (steering slot, parameter) for every residual writer on this rank."""
    out = {"embed_tokens.weight": (0, inner.embed_tokens.weight)}
    for name, p in inner.named_parameters():
        if not name.startswith("layers."):
            continue
        layer = int(name.split(".")[1])
        if name.endswith("self_attn.o_proj.weight"):
            out[name] = (steering.slot_of("attn", layer), p)
        elif name.endswith(WRITER_SUFFIXES[1:]) and ".experts." not in name:
            if name.endswith("mlp.down_proj.weight") and "shared_experts" not in name and layer != 0:
                continue  # only layer 0 has a dense MLP writer
            out[name] = (steering.slot_of("mlp", layer), p)
    return out


def w_edit_writers(worker, Q: torch.Tensor, lam, preserve_row_norms: bool = False) -> dict:
    """Apply W <- (I - lam Q^T Q) W to every writer shard in place (originals kept on CPU)."""
    inner = _inner(worker)
    d = inner.embed_tokens.weight.shape[-1]
    Q = Q.float()
    lam_t = torch.as_tensor(lam, dtype=torch.float32)
    n = 0
    for name, (slot, p) in writer_params(inner).items():
        Qs = (Q[slot] if Q.dim() == 3 else Q).to(p.device)
        ls = float(lam_t[slot]) if lam_t.dim() > 0 else float(lam_t)
        if ls == 0.0:
            continue
        if slot == 0:
            if p.shape[-1] != d:
                raise RuntimeError(f"rank {_rank()}: {name} shape {tuple(p.shape)} lacks full d_model on dim -1")
            new = ablation.ablate_embedding(p.data, Qs, ls)
        else:
            if p.shape[0] != d:
                raise RuntimeError(f"rank {_rank()}: {name} shape {tuple(p.shape)} lacks full d_model on dim 0 "
                                   "(sharded or transposed output dim; in-place edit would be wrong)")
            new = ablation.ablate_output_weight(p.data, Qs, ls, preserve_row_norms)
        if name not in _ORIG:
            _ORIG[name] = p.data.detach().to("cpu", copy=True)
        p.data.copy_(new.to(p.dtype))
        n += 1
    return {"rank": _rank(), "edited": n}


def w_restore_writers(worker) -> dict:
    params = dict(writer_params(_inner(worker)))
    for name, orig in _ORIG.items():
        params[name][1].data.copy_(orig.to(params[name][1].device))
    n = len(_ORIG)
    _ORIG.clear()
    return {"rank": _rank(), "restored": n}


def edit_writers(llm, Q, lam, preserve_row_norms=False) -> list[dict]:
    r = llm.collective_rpc(w_edit_writers, args=(Q.cpu(), lam.cpu() if isinstance(lam, torch.Tensor) else lam,
                                                   preserve_row_norms))
    llm.reset_prefix_cache()
    return r


def restore_writers(llm) -> list[dict]:
    r = llm.collective_rpc(w_restore_writers)
    llm.reset_prefix_cache()
    return r


# ---------- capture ----------

def w_capture_reset(worker) -> int:
    """Drop any recording left by a failed capture and free its GPU memory (a leftover recorder would keep
    recording every later forward pass)."""
    import gc

    cm = _CAP.pop("cm", None)
    _CAP.clear()
    if cm is not None:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 -- best effort
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _rank()


def w_capture_start(worker) -> int:
    from harness.capture import record_stream

    if _CAP:
        w_capture_reset(worker)
    cm = record_stream()
    _CAP["cm"], _CAP["calls"] = cm, cm.__enter__()
    return _rank()


def w_capture_take(worker, n_tokens: int, positions: list[int], out_path: str | None,
                   streams: list[int] | None = None) -> dict:
    """Stop recording; keep prefill-sized calls -> [T, S, d]; select positions; rank 0 writes.
    Never raises: an error (e.g. CUDA OOM) is returned as ok=False after the recording is dropped and the cache
    emptied. Raising would let the worker keep the traceback, and with it the whole recording (~21 GB for a long
    row), so every later capture would run out of memory."""
    err = None
    try:
        return _capture_take(n_tokens, positions, out_path, streams)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:300]}"
    w_capture_reset(worker)  # outside the except block: the traceback (and its tensors) is gone by now
    return {"rank": _rank(), "ok": False, "error": err}


def _capture_take(n_tokens: int, positions: list[int], out_path: str | None, streams: list[int] | None = None) -> dict:
    from safetensors.torch import save_file

    calls = _CAP.pop("calls")
    _CAP.pop("cm").__exit__(None, None, None)
    prefill = [c for c in calls if c.shape[0] == n_tokens]
    shapes = sorted({tuple(c.shape) for c in calls})
    if not prefill:
        return {"rank": _rank(), "ok": False, "shapes": shapes[:5]}
    # select the positions from each [T, d] call BEFORE stacking: stacking all T first would allocate a second full
    # copy of the recording ([T, S, d], ~21 GB for a 7.7k-token row on K3.1) and OOM on long rows
    if streams is not None:  # keep a subset of stream indices (e.g. every 4th) to bound storage
        prefill = [prefill[s] for s in streams]
    idx = torch.as_tensor(list(positions), dtype=torch.long, device=prefill[0].device)
    states = torch.stack([c.index_select(0, idx) for c in prefill], dim=1)  # [n_pos, S or len(streams), d]
    del calls, prefill
    s32 = states.float()
    checksum = float(s32.sum().item())
    # L1 and a fixed strided sample, so a cross-rank mismatch can be sized element-wise (signed sums cancel)
    l1 = float(s32.abs().sum().item())
    sample = s32.flatten()[:: max(1, s32.numel() // 4096)][:4096].cpu()
    if _rank() == 0 and out_path:
        import json

        # positions in the file, in order: lets a rerun tell whether an existing file covers a row's positions
        meta = {"positions": json.dumps(list(positions))}
        if streams is not None:
            meta["streams"] = json.dumps(list(streams))
        import os

        tmp = f"{out_path}.tmp"  # write then rename: a crash never leaves a half-written file under the real name
        save_file({"states": states.to(torch.bfloat16).contiguous().cpu()}, tmp, metadata=meta)
        os.replace(tmp, out_path)
    return {"rank": _rank(), "ok": True, "checksum": checksum, "l1": l1, "sample": sample, "shape": list(states.shape)}


def rank_agreement(res: list[dict]) -> dict:
    """Cross-rank agreement of a captured stream: relative L1-norm gap and relative L1 distance of a fixed
    element sample, each vs rank 0. Replicated streams give ~0 (bitwise below 2048 tokens; tiny noise past it,
    from DSA long-prefill nondeterminism, docs/runtime_notes.md); a sharded stream gives O(1)."""
    l1_0, s0 = res[0]["l1"], res[0]["sample"]
    l1_gap = max(abs(r["l1"] - l1_0) for r in res) / max(l1_0, 1e-12)
    sample_dist = max(float((r["sample"] - s0).abs().sum()) for r in res) / max(float(s0.abs().sum()), 1e-12)
    return {"l1_rel_gap": l1_gap, "sample_rel_dist": sample_dist,
            "bitwise_checksums": len({r["checksum"] for r in res}) == 1}


def capture_one(llm, ids: list[int], positions: list[int], out_path: str | None, tol: float = 0.05,
                streams: list[int] | None = None) -> dict:
    """Capture on all ranks; rank 0 writes. Fails only on a gross cross-rank mismatch (> tol relative), which is
    what sequence-parallel sharding would produce. Small numerical noise is recorded in the result, not fatal."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm.collective_rpc(w_capture_start)
    try:
        llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)
    except BaseException:
        llm.collective_rpc(w_capture_reset)  # never leave a recorder running on the workers
        raise
    res = llm.collective_rpc(w_capture_take, args=(len(ids), positions, out_path, streams))
    if not all(r["ok"] for r in res):
        raise RuntimeError(f"capture failed (chunked prefill / prefix cache hit / sharded stream?): {res}")
    shapes = {tuple(r["shape"]) for r in res}
    agree = rank_agreement(res)
    if len(shapes) > 1 or agree["l1_rel_gap"] > tol or agree["sample_rel_dist"] > tol:
        raise RuntimeError(f"stream differs across ranks (sequence-parallel sharding?): shapes {shapes}, {agree}, "
                           f"checksums {[r['checksum'] for r in res]}")
    return {**{k: v for k, v in res[0].items() if k != "sample"}, "rank_agreement": agree}
