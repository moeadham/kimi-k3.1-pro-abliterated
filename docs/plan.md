# Kimi K3.1 Pro censorship-ablation harness: plan

## Context

Goal: remove CCP-propaganda / censorship behavior from Kimi K3.1 Pro by directional ablation ("abliteration"). We find censorship direction(s) in the residual stream and project them out of the weights. Before that, we need a **measurement harness** that can:
- generate at scale and stream output chunks to a fast judge (TypeSafe **Jev**) for real-time per-chunk verdicts
- capture the activations behind flagged tokens
- compute directions (single, multi-direction, or distribution)
- sweep ablation strength via inference-time steering, without reloading weights
- bake the result into the weights and evaluate it

What we've established so far:
- **Heretic** can't run this model: HF-only, no K3 modeling code, needs BF16 weights fully resident (~5.6 TB). It's also AGPL. We reimplement its math (projected abliteration, MPOA, per-layer weight kernel, Optuna search, KL scorer) rather than its code.
- **vLLM** (≥0.34 / `kimi-k31` image) is the inference framework. It has first-class hidden-state capture (`extract_hidden_states`), and we steer with our own plugin.
- **K3.1 architecture** (verified from config and weight index):
  - 93 layers, d_model 7168, 896 routed experts (16 active) + 2 shared
  - 69 KDA + 24 MLA/DSA layers
  - **Attention Residuals** (`attn_res_block_size=12`) instead of a single running residual stream
  - **latent MoE** (3584-dim latent → `routed_expert_up_proj`)
  - MXFP4 only on the routed experts
- **Every residual-stream writer is BF16**: `self_attn.o_proj`, `block_sparse_moe.routed_expert_up_proj`, `shared_experts.down_proj`, layer-0 `mlp.down_proj`, `embed_tokens`. Plus, to be checked, the AttnRes `*_res_proj` modules. The bake never touches the MXFP4 experts.

Repo: `/Users/m/workspace/kimi-k3.1-pro-ab` (currently empty). Dev pod: RunPod 1×B200 (SM100, same kernel family as the production B300s). Volume disk is 50 GB at `/workspace`; the container disk is separate.

## The three steps

| Step | Model | Hardware | Purpose |
|---|---|---|---|
| **1** | Tiny random-weight K3.1 (real config shape, 2–4 layers, few experts) + a small real model for the judge loop | 1×B200 | Build and prove every piece of harness plumbing against the real vLLM K3 code |
| **2** | Pruned K3.1 proxy (~64 experts/layer, all 93 layers, router sliced to the kept experts) | 1×B300 (TP1), then 2×B300 (TP2) | Run the full method end to end on real weights; exercise the multi-GPU code paths |
| **3** | Full K3.1 Pro | 8×B300 | Real run: directions, sweeps, bake, final eval |

Step 1 is detailed below. **At the end of Step 1 we write a detailed Step 2 plan**, informed by what Step 1 teaches us. Steps 2–3 are outlines only.

---

## Step 1: harness plumbing on a tiny random K3.1 (detailed)

### 1.0 Pod bootstrap
- **SSH:** add a `Host kimi-b200` alias to `~/.ssh/config` using the pod's "SSH over exposed TCP" details and `~/.ssh/id_ed25519`. Verify with `ssh kimi-b200 nvidia-smi`.
- **Inventory:** `df -h`, GPU, driver/CUDA, RAM. Decide disk placement:
  - venv on `/workspace/venv` (persists across stops)
  - HF and uv caches plus tiny checkpoints on the container disk
  - resize the volume via Edit Pod if it's tight
- **`tools/setup_pod.sh`** (idempotent):
  - uv venv, then install the vLLM nightly
  - confirm K3/K3.1 model support exists (`vllm/model_executor/models/kimi_k3*`); if not, fall back to a pinned version matching the `kimi-k31` image
  - install `compressed-tensors`, `typesafe-sdk`, pytest, safetensors
  - write `env.lock` (vLLM commit, torch, CUDA, FlashInfer versions)
- **Code sync:** rsync from the Mac to `/root/harness` (`tools/sync.sh`). Code lives in this repo; the pod is disposable.
- **Cost hygiene:** stop the pod whenever no job is running.

### 1.1 Map K3.1's residual stream in vLLM (gate for everything after)
Read the installed vLLM K3 model code on the pod and write `docs/k3_residual_map.md`:
- the exact module path of every residual-stream writer, per layer type (dense layer 0, KDA, MLA/DSA)
- the AttnRes math: block snapshots, the intra-block prefix sum, the softmax mixture (`_pre_attn_norm` / `_post_attn_norm`), and what `self_attention_res_proj` / `mlp_res_proj` / `output_attn_res_proj` do (writers, or mixing weights?)
- where TP all-reduce / reduce-scatter and fused residual+norm epilogues happen, i.e. where a forward hook actually sees the full value
- which weights `process_weights_after_loading` repacks, fuses or copies (the repack trap from vllm#40536), especially if shared experts are fused into the MoE kernel
- which hidden state `extract_hidden_states` returns by default vs. with `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1`

This doc defines `harness/model_spec/kimi_k3.py`: the writer list, layer types and stream definitions, used by all later code.

### 1.2 Tiny random K3.1 checkpoints — `tools/make_tiny_k3.py`
- Start from the real K3.1 `config.json`, tokenizer and custom config code; no real weights.
- Shrink while keeping every feature:
  - 4 layers: layer 0 dense; at least one KDA MoE layer; at least one full-attention/DSA MoE layer
  - `attn_res_block_size=2`, so block boundaries and snapshot mixing actually happen
  - ~16–32 experts with top-k ≥ 2, plus 2 shared experts
  - keep the latent MoE and real d_model 7168 and vocab 163,840
  - vision tower dropped if the loader allows; otherwise tiny
- Random init (std ~0.02, norms = 1) with a fixed seed.
- Variants:
  - **(a) BF16**, no quantization
  - **(b) MXFP4** compressed-tensors on the routed experts with the same ignore list as the real model, so the production MoE path runs
- Each is ~5–8 GB. Write them to the container disk.
- **Acceptance:** vLLM loads both, generates without NaNs in eager and CUDA-graph modes, and greedy outputs are deterministic across runs.

### 1.3 Activation capture — `harness/capture.py`
- **Primary:** replay-prefill capture. Resubmit prompt+response token IDs with `max_tokens=1` to a vLLM instance configured with `extract_hidden_states` (all layer IDs, `/dev/shm` output, chunked prefill off), then slice the requested token positions into `[n_pos, L, d]` float tensors stored as safetensors plus a JSONL index.
- **Cross-check:** a second capture path through our own plugin's forward hooks.
- **Tests:**
  - shapes and layer indexing are correct
  - the two capture paths agree for each stream definition
  - capturing the same tokens twice gives the same results, including at different batch sizes (try `VLLM_BATCH_INVARIANT`)
  - activations captured during generation vs. via replay agree within tolerance

### 1.4 Steering plugin — `harness/vllm_plugin/`
- A vLLM general plugin, registered via entry point, that wraps each writer module from the model spec at load time, before compile/graph capture.
- Applies `y ← y − λ_l · (y·r_l) r_l` on writer outputs.
- Reads persistent device buffers `R[L, d]` and `λ[L, n_writer_types]`. The buffers are updated in place through `collective_rpc` (`set_steering`, `clear_steering`), so there is no reload and no graph recapture. `reset_prefix_cache()` runs on every change.
- **Tests:**
  - **a. Identity:** λ=0 gives logits identical to no plugin
  - **b. Orthogonality under AttnRes:** with λ=1 on all writers including embeddings, both stream definitions have ~0 component along r at every layer. This is the key check that "writers orthogonal ⇒ stream orthogonal" holds through AttnRes mixing and norms. If the `*_res_proj` modules are writers, they're covered here.
  - **c. Live update:** changing buffers via `collective_rpc` changes the output immediately; eager and CUDA-graph runs give equal results
  - **d. Prefix cache:** a stale cache doesn't leak old steering

### 1.5 Ablation math (CPU, unit-tested on the Mac) — `harness/directions.py`, `harness/ablation.py`, `harness/kernel.py`
- **Directions:**
  - difference of means (flagged vs. clean, or censored-prompt vs. control)
  - projected variant (orthogonalized against the good mean)
  - multi-direction (top-k SVD/PCA of flagged-minus-control activations)
  - per-layer and global options
  - optional winsorization
- **Weight edit:** `W ← W − λ r (rᵀW)` for output writers; embedding rows `E ← E − (E r) rᵀ`. Optional MPOA (row-norm-preserving). Rank-k generalization for multiple directions.
- **Per-layer λ kernel:** Heretic-style max weight, peak position, min weight, falloff distance.
- **Tests:** projection idempotence; the edited weight has zero output along r; MPOA preserves row norms; λ=0 is a no-op.

### 1.6 Bake + edit verification — `harness/bake.py`
- Apply the 1.5 weight edits to the BF16 writer tensors and rewrite only the affected safetensors shards. Every other shard stays byte-identical. Update the index, keep the quantization ignore list valid, write a manifest with a hash, direction and λ.
- **Steering ≡ bake test:** load the baked checkpoint, compare logits to the steering-plugin run with the same r and λ; they should agree within BF16 tolerance.
- **Edit-verification test:** for each writer type, make a large random in-place edit through `collective_rpc` and confirm the logits change; restore and confirm the baseline exactly. Catches repacking and stale-graph traps.

### 1.7 Judge loop — `harness/judge/`, `harness/runner.py`
- **Judge interface:** `async judge(prompt, chunk, context) -> {labels, probs}`
  - `MockJudge` (regex/keyword, for the tiny model)
  - `JevJudge` (`typesafe-sdk`, `TYPESAFE_API_KEY` from the pod env). Questions:
    - Noul: "deflects/denies/whitewashes"
    - Choice: {factual, evasive, party-line, refusal, off-topic}
    - Score: factuality
- **Runner:**
  - streaming generation (vLLM AsyncLLM) → chunker (every N tokens or sentence boundary, keeping token offsets) → async judge calls with concurrency limits and timeouts
  - records `(prompt_id, chunk token span, verdicts)` to JSONL/parquet
  - flagged spans → capture queue (1.3)
- **Tests:**
  - end to end on the tiny model with MockJudge: flagged positions → captured activations → directions → steering → re-judge
  - then a **small real model** on the same B200 to measure Jev's latency, throughput and label quality on actual censored vs. factual text. Candidate: a small Chinese-origin instruct model known to censor; chosen at 1.7 time.

### 1.8 Step 1 exit criteria → write the Step 2 plan
All of these hold:
- the residual map is documented
- both tiny variants load and generate
- the capture paths agree
- the orthogonality-under-AttnRes test passes
- steering ≡ bake
- the edit-verification test passes
- the judge loop works end to end with Jev latency measured
- the CPU and GPU test suites are green

Then write `docs/plan_step2.md` in detail.

### Repo layout (new files)
```
pyproject.toml                  # uv project
harness/model_spec/kimi_k3.py   # writers, layer types, stream defs (from 1.1)
harness/capture.py  directions.py  ablation.py  kernel.py  bake.py  runner.py
harness/judge/{base,mock,jev}.py
harness/vllm_plugin/{__init__,steering}.py   # entry-point plugin
tools/{setup_pod.sh,sync.sh,make_tiny_k3.py}
tests/        # CPU tests (Mac)
tests_gpu/    # pod tests (vLLM)
docs/k3_residual_map.md  docs/plan_step2.md
```

### Verification
- On the Mac: `uv run pytest tests/`
- On the pod: `ssh kimi-b200 'cd /root/harness && /workspace/venv/bin/pytest tests_gpu/ -x'`, covering 1.2–1.6 and the 1.7 end-to-end test
- Jev: a manual latency/quality report on the small real model in `docs/`

---

## Step 2: pruned proxy (outline; detailed plan written at the end of Step 1)
- **Calibration:** a short run on the full 8×B300 (~1 h) to log per-layer router selections on censorship + control prompts (EPLB load stats or router hooks).
- **`tools/make_proxy.py`:**
  - keep the top-64 experts per layer by router-weighted usage
  - slice `gate.weight` / `e_score_correction_bias`, renumber the experts, set `num_experts=64`
  - ~218 GB checkpoint, kept on a global or network volume
- **Gate:** the proxy must be coherent **and** still censor. If it fails, adjust the expert count or selection.
- **Datasets v1:**
  - censored-topic prompts + matched controls
  - a harmful-request set (refusals must be retained)
  - a general capability set
- Run the whole harness: baseline metrics → flagged captures → directions → λ sweeps (Optuna; objectives censorship ↓, KL ↓, harmful-refusal retained) → bake → eval.
- Repeat the key hook and steering tests at **TP2** to exercise the collective, sharded-AttnRes and EP code paths.

## Step 3: full K3.1 Pro on 8×B300 (outline)
- Re-run the edit-verification and steering tests on the production backends.
- Baseline evals → capture at scale → directions computed on the full model (they don't transfer from the proxy) → steering sweeps → bake into the BF16 writer shards → final eval:
  - censorship rate (Jev)
  - harmful-request refusal retention
  - KL vs. base
  - capability benchmarks
- Package the checkpoint plus a manifest (directions, λ kernel, versions).

## Open items
- Jev API key, rate limits and latency: needed by 1.7.
- Whether the HF repo is gated (`HF_TOKEN` as a RunPod secret).
- Whether the `*_res_proj` modules are residual writers: resolved in 1.1.
