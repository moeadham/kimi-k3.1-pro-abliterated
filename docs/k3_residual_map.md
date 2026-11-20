# Kimi K3 residual stream in vLLM 0.34.0 (step 1.1)

Source: `vllm/models/kimi_k3/nvidia/model.py` and `ops/attn_res.py` plus `kda.py`, `mla.py`, `dsa.py` and `low_latency_gemm.py` in the installed vLLM 0.34.0 (see `env.lock.json` on the pod). Line numbers refer to that version.

## 1. There are two state variables, not one residual stream

`KimiLinearModel.forward` (model.py ~L1197) carries, per token:

- **`prefix_sum` [d]**: the running sum of sublayer outputs since the current *block* began.
- **`blocks` [n_blocks ≤ 8, d]**: snapshots ("block residuals"). `n_blocks = cdiv(93, 12) = 8`.

With `attn_res_block_size = 12`, layer `i` is a *block-write layer* iff `i % 12 == 0`.

Per decoder layer (`KimiDecoderLayer.forward`, L923), with `pending` = the previous layer's MLP output:

1. **pre-attn** `attn_res(prefix, pending, blocks, …, block_write_idx)`:
   - `prefix += pending`
   - on a block-write layer, `blocks[i/12] = prefix`
   - attention input = `RMSNorm_inputLN( softmax_mix(blocks[0:prev_valid], prefix) )`
2. **attention**: `a = self_attn(x)`. Its output ends in `o_proj`.
3. **post-attn**:
   - on a block-write layer, `prefix = a` (reset); otherwise `prefix += a`
   - MLP input = `RMSNorm_postLN( softmax_mix(blocks[0:valid], prefix) )`
4. **MLP / MoE**: `m = mlp(x)`, which becomes the next layer's `pending`.

The final output is `softmax_mix(blocks, prefix + m_last)`, using `output_attn_res_*`. The final `norm` and `lm_head` run in `compute_logits`.

At layer 0 the prefix is the embedding, so **`blocks[0]` = the token embedding alone**. Every later snapshot is the sum of all attention and MLP outputs of one 12-layer block.

**`softmax_mix`** (Triton `_attn_res_kernel`, plus a native SM100 op `kimi_k3_attn_res` when d=7168 and ≤8 blocks):
- score per source = `<v * rsqrt(mean(v²)), res_norm.w ⊙ res_proj.w>`
- mixed = `Σ softmax(score)_j · v_j`, a **convex combination of the sources**
- an optional RMSNorm comes after

`self_attention_res_proj`, `mlp_res_proj` and `output_attn_res_proj` are `[1, 7168]` **score vectors**. They read the stream; they don't write to it. The `*_res_norm` weights only feed the scores.

## 2. Residual-stream writers (complete list)

Everything in `prefix_sum` and `blocks` is a sum of exactly these outputs:

| Writer | Module (per layer) | Weight(s) to bake | Notes |
|---|---|---|---|
| Embedding | `model.embed_tokens` | `embed_tokens.weight` [163840, 7168] | becomes `blocks[0]` and the initial prefix |
| Attention, KDA (69 layers) | `layers.i.self_attn` (`KimiK3DeltaAttention`) | `self_attn.o_proj.weight` | `kda.py` L990–992 |
| Attention, MLA (non-DSA full layers, if any) | `layers.i.self_attn` (`MultiHeadLatentAttention`) | `self_attn.o_proj.weight` | `mla.py` L636–639 |
| Attention, DSA (24 full layers) | `layers.i.self_attn` (`KimiK3DSAAttention`) | `self_attn.o_proj.weight` | `dsa.py` L758, gated input |
| Dense MLP (layer 0 only, `first_k_dense_replace=1`) | `layers.0.mlp` (`KimiMLP`) | `mlp.down_proj.weight` | |
| MoE (layers 1–92) | `layers.i.mlp` (= `block_sparse_moe`, `KimiMoE`) | `block_sparse_moe.routed_expert_up_proj.weight` [7168, 3584] **and** `block_sparse_moe.shared_experts.down_proj.weight` | output = up_proj(norm(routed latent)) + shared |

All of these weights are BF16; the quantization ignore list covers them. Routed experts (`experts.*`, MXFP4) write into the 3584-dim latent and **are never edited**.

**Theorem we rely on (to be verified empirically in 1.4b).** If every writer output is orthogonal to r, then every source (`blocks[j]`, `prefix`) is orthogonal to r. A convex mix of orthogonal vectors is orthogonal, so every *pre-norm* stream state is orthogonal to r. Ablation also changes the softmax scores, but the result stays a convex combination.

Caveats:
- The RMSNorm gain γ (elementwise) is applied *after* mixing. The normalized sublayer input is not orthogonal to r, but it isn't part of the stream. This is the same as standard abliteration.
- BF16 rounding in `prefix += delta` (the kernel rounds to BF16 on purpose) leaves a small residue.

## 3. Where to hook (steering plugin, 1.4)

Hook the **outputs of `layers.i.self_attn`, `layers.i.mlp`, and `model.embed_tokens`**, not the inner linears:
- `KimiRoutedOutputTransform` can fuse the shared-expert add into `residual.addmm_(h, up_proj.weight.t())` (model.py L336–339), and MegaMoE folds it into a GEMM epilogue. A hook on `routed_expert_up_proj` would be skipped on those paths.
- With TP>1, `gemm_rs_ar.apply(x, self.o_proj)` (kda.py L990, mla.py L636, KimiMLP L301) fuses GEMM and reduce-scatter/all-reduce **without calling `o_proj.forward`**.
- Under sequence parallelism, `o_proj.reduce_results=False` and the reduce-scatter happens in the decoder layer (L947–949), so a hook on `self_attn` sees **per-rank partial sums**. That's fine for projection, because `x ↦ x − (x·r)r` is linear and commutes with the sum. **Additive** steering (`+αr`) would be multiplied by the TP size there, so it isn't supported.

## 4. Things that can silently bypass hooks or weight edits

| Mechanism | Effect | Harness rule |
|---|---|---|
| `additional_config.decode_backend = "latency_specialized"` (`specialized_decode/runtime/__init__.py`) | Decode steps skip `self.model` entirely, and the executor keeps **its own copies derived from the weights** | **Forbidden.** The default `auto` gives the stock path. The plugin asserts this. |
| `enable_kimi_k3_low_latency_gemm` (low_latency_gemm.py L1042) | Swaps `quant_method` on unquantized linears matching an SM100 (N, K) table | Safe: it reads `layer.weight` live (L993–999) |
| `KimiK3DSAAttention.pack_projections` | Packs q_a/kv_a/indexer and q_b/wq_index into one buffer; the linears become views | Safe: none of these are writers, and edits through the views still land |
| CUDA graphs / torch.compile | Captured kernels read fixed pointers | Edits must be **in place** (`.copy_()`); hooks must be installed before compile/capture and read device buffers |
| FusedMoE backends (`process_weights_after_loading` repack) | Expert weights repacked (vllm#40536 trap) | Not relevant: we never edit `experts.*`. The edit-verification test (1.6) still checks every writer. |
| DSpark drafter / MTP | Separate draft model | Not used (`num_nextn_predict_layers=0`, no spec decode except the capture-only `extract_hidden_states`) |

## 5. Hidden-state capture (`extract_hidden_states`, 1.3)

`_capture_aux_hidden_stream` (L1157) supports two modes, chosen by the env var `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM`:

- **0 (default, "prefix_only")**: returns `prefix_sum + pending_mlp_out`, i.e. **only the current block's partial sum**. It excludes the embedding and earlier blocks, so it's a poor "residual stream".
- **1 ("attn_res_stream")**: returns `softmax_mix(blocks, prefix + m)` scored by the *next* layer's `self_attention_res_*` (pre-norm, no γ). That's exactly what layer i+1 reads, and it's **our default for direction finding**.

Indexing: aux layer id `k` = the state after layer `k-1`; id 0 = the embedding. The capture is full width (`sp_all_gather` when SP is on).

## 6. Layer types (0-indexed in code; the config lists are 1-indexed: `is_kda_layer(i) := (i+1) in kda_layers`)

- K3.1: 93 layers.
  - full-attention/DSA: code layers `{3, 7, …, 91, 92}` (config `{4, 8, …, 92, 93}`)
  - everything else KDA
- Layer 0: KDA attention + dense `KimiMLP` (`intermediate_size=33792`). Layers 1–92: MoE.
- Block-write layers: 0, 12, 24, …, 84.

## 7. Open questions for later steps

- With the stock decode path and our hooks, are CUDA graphs still captured with hooks inside (no graph breaks)? Verified in 1.4c.
- How large is the BF16 residue in the orthogonality test? Measured in 1.4b; sets the tolerance.
- TP>1 paths (`gemm_rs_ar`, SP reduce-scatter, LatentMoERunner fused all-reduce): Step 2.
