# Runtime notes: running K3.1 in vLLM (learned in Step 1)

These carry over to Steps 2–3. Each entry says what we observed, why it matters and what to do.

## vLLM version
- **Use `vllm==0.34.0rc1.dev673+g2440ffc38`** (nightly from main, 2026-11-17), installed by `tools/setup_pod.sh`. The **0.34.0 release** has the K3 model code, but its branch predates the K3.1 packed hybrid KV-cache plumbing on main (`Platform._align_hybrid_block_size` → `get_packed_hybrid_block_size`). On 0.34.0, any K3.1 config with DSA layers fails KV-cache init:
  `DEEPSEEK_V32_INDEXER needs 64-token kernel blocks, which packed N-token KV cache blocks cannot be split into`.
  `--block-size 64` and `VLLM_KV_CACHE_LAYOUT=LBNHC` do **not** work around it: the platform overrides the block size, and KDA only accepts BLHNC.
- vllm-project/vllm#67734 (open) clamps the packed block size further. It's not needed on the pinned nightly at TP1. **Re-check at TP8/PP**, where that PR's author saw the same error on the full K3.1-Pro.
- `setup_pod.sh` installs vLLM and its extras **in one resolver call** and fails if torch/transformers/flashinfer drift from vLLM's pins. A separate `-U` install once upgraded torch underneath vLLM's compiled ops.
- Vendored patches (`tools/patches/*.diff`) are applied **all-or-nothing**. An earlier partial apply (helper added, main hunk rejected) went unnoticed until checked.

## Launch settings (`harness/vllm_launch.py`)
- Adapted from the Blackwell recipe for K3: `kv_cache_dtype=fp8`, `VLLM_USE_V2_MODEL_RUNNER=1`, `language_model_only=True`, flashinfer autotune off.
- **Not** `attention_backend=TOKENSPEED_MLA` from the K3 recipe: it rejects K3.1's DSA layers ("sparse not supported"). Auto-selection is used instead.
- Prefix caching is off by default (steering changes KV/KDA state). `decode_backend` is asserted `auto`.

## Kernels vLLM selects on B200 (SM100), nightly 2440ffc38, TP1
| Component | Backend |
|---|---|
| KDA decode / prefill | native fused KDA decode / FlashKDA prefill |
| DSA (sparse MLA) | `FLASHINFER_MLA_SPARSE`, standard fp8 KV (`FLASHMLA_SPARSE` would use `fp8_ds_mla`) |
| MLA prefill | `FLASH_ATTN` |
| Routed experts (MXFP4) | `FLASHINFER_TRTLLM_MXFP4_MXFP8`. This is the repacking backend from vllm#40536; harmless for us (experts are never edited) but covered by the 1.6 edit-verification test |
| Hybrid KV block | 9216 tokens (tiny model, TP1) |

The 0.34.0 release chose `CutlassExpertsMxfp4` for the same checkpoint. Backend choice varies by version, which is another reason to pin.

## Determinism (measured on the tiny BF16 model, eager, nightly 2440ffc38)
| Setup | Prefill logprob max \|Δ\| over 3 runs | Decode divergences |
|---|---|---|
| 1 prompt, multiprocess engine | 0.0 | 0/1 |
| 4 prompts (same lengths), multiprocess | 0.48 | 4/4 |
| 4 prompts, after a warmup run, multiprocess | 0.29 | 2/4 |
| 4 prompts, **in-process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | **0.0** | **0/4** |

- `tests_gpu/layer_divergence.py` hooked every writer and router. A single prefill gives **bitwise identical** outputs at every module and identical expert selection. **The model forward pass is deterministic; the drift comes from batch composition.** The multiprocess engine core schedules identical request sets into different step batches from run to run, and the kernels are batch-variant.
- `VLLM_BATCH_INVARIANT=1` is **unavailable for K3.1**: neither `FLASHINFER_MLA_SPARSE` nor `FLASHMLA_SPARSE` supports batch invariance.
- **Harness rules:**
  1. Run the engine in-process (now the default in `vllm_launch.py`).
  2. Exact comparisons (steering ≡ bake, capture replay vs. generation) must use **identical batch composition**, e.g. a single sequence or the same fixed batch.
  3. Comparisons across different batchings need a tolerance taken from a measured batch-variance floor, not from zero.
- On random weights a small numerical change can flip router top-k and greedy argmax (near-ties), which amplifies the drift. The real model's margins should be larger; re-measure the floor on the proxy in Step 2.

## First-launch cost
FlashInfer downloads cubins and JIT-builds its sampling module on first use: about 10 minutes on a fresh pod, cached in `/root/.cache` (container disk, lost on pod stop).

## Tiny model (`tools/make_tiny_k3.py`)
- 8 layers: KDA, KDA, KDA, DSA, repeated twice, matching the real 3:1 pattern. Layer 0 has the dense MLP. `attn_res_block_size=4`, 16 experts top-4, real d_model/vocab/tokenizer. Shapes come from the real checkpoint's safetensors headers (no weights downloaded).
- Two variants:
  - `mxfp4`: production compressed-tensors expert format, about 17 GB
  - `bf16`: unquantized experts
- Writes to `/dev/shm/tiny-k3/<variant>`.

## B300 on Modal (2026-11-18, 2×B300 smoke test, tiny K3.1 mxfp4)
Image: `tools/modal_image.py` (CUDA 13.2.1 devel base; same pins as env.lock; FlashInfer `flashinfer-cubin` +
`flashinfer-jit-cache-sm103a`). Volume `kimi-k31-pro-ab`. Logs: `/vol/session/smoke2/`.

- **Node:** B300 SXM6 AC, 275 GB, cc 10.3, driver 610.57.04 (CUDA UMD 13.3). gVisor kernel. 2-GPU container:
  32 CPUs visible, `/dev/shm` 4.0 TB, `/` 512 GB. `nvidia-smi topo -m` fails under gVisor (harmless).
- **Interconnect under gVisor works:** peer access on, NCCL `P2P/CUMEM`, NVLS multicast available, torch
  symmetric memory + one-shot all-reduce OK, all-reduce busbw 574 GB/s @1 GiB (2 GPUs).
  FlashInfer MNNVL FABRIC handles fail (`CUDA_ERROR_NOT_PERMITTED`, no IMEX channel) and fall back to POSIX
  fds; the allreduce-norm fusion workspace still initializes. vLLM TP2 all-reduce order: FLASHINFER, CUSTOM,
  SYMM_MEM, PYNCCL; GEMM-AR enabled.
- **Kernels:** same backends as B200 (FlashKDA prefill, native KDA decode, FLASHINFER_MLA_SPARSE,
  FLASHINFER_TRTLLM_MXFP4_MXFP8). Step-1 GPU suite passes at TP1 (smoke eager/graphs, steering eager/graphs,
  capture, edit reach/restore, short-prompt determinism).
- **Long-prefill nondeterminism (blocks bitwise checks):** a single prefill is bitwise reproducible up to
  2047 tokens, but every module from the first DSA layer on differs run-to-run **for positions ≥ 2048**
  (`tests_gpu/layer_divergence.py <ckpt> 2486`: L3.attn first diff at token 2048). Not fixed by
  `sparse_indexer_topk_backend` = per_row or persistent; `FLASHMLA_SPARSE` can't start (packed-block error,
  #67734). Consequences: `capture` cross-rank checksum fails on long rows at TP2 (row 329, 2486 tokens);
  determinism checks must use prompts < 2048 tokens or a measured tolerance. Fixed in `tp.capture_one`: the
  cross-rank check now fails only on a gross mismatch (shape, or > 5% relative L1 gap / sample distance, which
  is what sequence-parallel sharding gives); smaller noise is recorded per row in the capture index.
  Only `answer_head` capture positions (2,581 of 58,467) lie at >= 2048 tokens.
- **Edit vs steering on the tiny model is dominated by chaos:** a 1e-3 perturbation of r moves prompt
  logprobs by up to 3.5 nats (mean 0.22) vs a steering effect of 4.3 (mean 0.31), so `checks`
  `edit_matches_steering` (greedy max-diff < 0.25) and `test_bake` `baked_close_to_steered` fail on random
  weights at TP1 and TP2. Structural equivalence holds (baked |cos| 0.00091 vs steered 0.00113).
  Re-judge on real weights.
