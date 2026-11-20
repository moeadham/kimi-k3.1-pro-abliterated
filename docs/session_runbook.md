# 8×B300 session runbook (Step 2)

> **The session runs on Modal** (the only 8×B300 source). Node setup, Volume, image, costs and current status are in
> `docs/modal_runbook.md`; the ssh/`setup_pod.sh` steps below are the original RunPod flow. The stages and
> go/no-go table below apply unchanged; on Modal run them as
> `python -m harness.session <stage> --model /vol/Kimi-K3.1-Pro --out /vol/session --tp 8` inside a GPU sandbox.

## Before the session
- **Node:** 8×B300 with **~3.5–4 TB** local NVMe. Optionally a same-DC network volume with the weights pre-staged (see below).
- **GPQA-Diamond** (optional): accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa with the account behind `HF_TOKEN`, then run `uv run --env-file .env --extra tok python tools/fetch_evals.py`. **HarmBench** works the same way (`walledai/HarmBench`). Without it, `mlabonne/harmful_behaviors` is used.
- **Pre-staging weights on a volume** (cheap CPU pod in the same DC): `python tools/download_weights.py /workspace/Kimi-K3.1-Pro --max-minutes 600`.

## On the node
1. Update `tools/ssh_config` (HostName/Port), then `tools/sync.sh` and `ssh kimi-b200 'bash /root/harness/tools/setup_pod.sh'`.
2. `data/evals/livecodebench.jsonl` (140 MB) is not in git. `tools/sync.sh` pushes it from the Mac copy; or regenerate it on the pod with `tools/fetch_evals.py`.
3. Start the session in tmux:
   ```
   read -rs OPENROUTER_API_KEY; export OPENROUTER_API_KEY
   MODEL=/local/Kimi-K3.1-Pro OUT=/local/session [WEIGHTS_FROM=/workspace/Kimi-K3.1-Pro] bash tools/session_8xb300.sh
   ```
4. **Stages** (each resumable; rerun the script to continue):

   | Stage | What it does | Go/no-go |
   |---|---|---|
   | download | speed test, then download or copy from the volume, verify sizes | aborts if the projected time > 60 min |
   | `checks` | steering on all ranks; λ=0 identity; steer/clear; in-memory edit ≈ steering; edit restore is bitwise; writer count | all must pass |
   | `capture` | determinism; stream orthogonal under λ=1 (\|cos\| < 0.02); capture all 1,224 replay rows | all must pass |
   | `fidelity` | logprob of hosted tokens with/without the effort message; greedy prefix agreement | look: is our runtime ≈ the hosted model? |
   | `refusal` | refusal-direction data: HarmBench 200 + XSTest-safe 200 → generate (max effort, T=1) → Jev judge → onset pass → capture prompt_end / reasoning_head / onset / answer_head (`harness/refusal.py`); then `directions --capture-from refusal` | look: harmful flagged ≈ 100%, harmless flagged ≈ 0%, no prompt-id mismatches |
   | `directions` | mean-diff / projected / PCA per position type × language; val AUC | look: which family separates best |
   | `baseline` | unedited KL refs, Tier 1, Tier 2 | reference numbers |
   | `sweep` | Optuna, 60 trials; constraints: KL ≤ 0.1, control flagged rate rise ≤ 10 pts, harmful refusal ≥ 95%, health | feasible trials > 0 |
   | `tier2` | top 3 trials as in-memory edits → Tier 2, paired bootstrap vs baseline | pick a winner → write `$OUT/tier2/winner.json` |
   | `bake` | `BAKED=/local/Kimi-K3.1-Pro-abl bash tools/session_8xb300.sh bake` | then Tier 3 |

## Things that can only be checked on the node (expected failure points)
- Whether vLLM `#67734` is needed at TP8 (packed-block error on startup).
- Whether every writer shard keeps the full 7168 output dim at TP8. `tp.w_edit_writers` raises if not.
- Whether the AttnRes stream is replicated across ranks in the eager capture engine. `tp.capture_one` raises on a checksum mismatch, which would mean sequence-parallel sharding.
- Whether vLLM's worker module path is `worker.model_runner.model` under the V2 runner at TP8.
- The edit-vs-steering tolerance (`--edit-tol` 0.25) on real weights.
