# GPU run 2 plan: token-level fork (reasoning off) + edit follow-ups

8×B300 on Modal, live driver, `decode-graphs` engine. Hard cap 200 min. Estimated 2–2.5 h, **~$150–185**.
Method: `docs/plan_nonthinking.md`. Lessons applied from run 1 (`docs/run_log.md`): failure Monitor on the live log
from the start; capture frees memory on failure; restarts use `NCCL_NVLS_ENABLE=0`; direction banks computed on the
GPU host (CPU sandboxes only for the long analysis); `oneshot` no longer hijacks the sandbox id.

| # | Step | Command / config | Est. |
|---|---|---|---|
| 0 | Start sandbox (`--minutes 200 --cpu 32 --mem-gib 128`), live driver in `decode-graphs` | `modal_gpu.py start` / `live-start` | 25–30 min (cold load) |
| 1 | Smoke: greedy determinism (repeat and bigger batch) + logprob format | `py tools/smoke_greedy.py` | 2 min |
| 2 | Smoke: whole forks stage on 5+5 prompts incl. capture | `stage forks --n-per-class 5 --out /vol/session_nt_smoke` | 3 min |
| 3 | **Forks run, all v1 English prompts** (3,122): borderline 600 + 400, pairs 1,061 + 1,061; answers, judging and branch test pipelined, then capture | `stage forks --out /vol/session_nt --harmful data/refusal_v1/borderline_harmful.jsonl --harmless data/refusal_v1/borderline_harmless.jsonl --extra-harmful data/refusal_v1/pairs_harmful.jsonl --extra-harmless data/refusal_v1/pairs_harmless.jsonl` | pipeline 25–35 min, capture 25–35 min |
| 4 | Fork directions on the GPU host | `tools/fork_bank.py /vol/session_nt/forks /root/banks/nt.pt` | 3 min |
| 5 | **Edit probe** (greedy, T = 0, paired vs baselines), below | `py probes/probe_run2.py` | 30 min |
| 6 | Stop the sandbox; token × layer analysis on a 32-CPU sandbox | `tools/analyze_forks.py` (anchors `fork`, `prompt_end`) | 10–20 min CPU |

## Edit probe conditions (priority order; each judged on 50 HarmBench + 50 XSTest held-out prompts)
Banks: `B` = `/vol/session_v1_border/directions_refusal/directions.pt` (decision `onset_vs_comply`, with projected +
PCA), `P` = `/vol/session_v1_pairs/directions_refusal/directions.pt` (harmfulness `prompt_end`), `N` = `/root/banks/nt.pt`
(new non-thinking `nt_fork`, `nt_prompt_end`). Reasoning-mode steps: T = 0, ≤4,096 tokens (~2 min each);
non-thinking: ≤512 tokens (~30 s).

| # | Mode | Edit | Layers | Why |
|---|---|---|---|---|
| 0 | both | check + baselines `base_think`, `base_nt` | — | paired reference |
| 1 | think | decision `onset_vs_comply` (B) | 52–76 | run 1's best (60% at T=1), now greedy and paired |
| 2 | think | decision + harmfulness together (B + P) | 52–76 | does a 2-direction edit beat either alone? |
| 3 | think | decision, PCA rank 3 (B) | 52–76 | several refusal modes |
| 4 | think | decision (B) | 40–92 | wider band (run 1: 20–92 barely beat 52–76 for harmfulness) |
| 5 | think | decision, strength λ = 1.5 (B) | 52–76 | over-removal |
| 6 | think | decision, projected variant (B) | 52–76 | less collateral damage? |
| 7 | nt | `nt_fork` (N) | 52–76 | the new direction in its own mode |
| 8 | think | `nt_fork` (N) | 52–76 | **does the non-thinking direction carry over to reasoning mode?** |
| 9 | nt | decision `onset_vs_comply` (B) | 52–76 | the reverse transfer |
| 10 | nt | `nt_prompt_end` (N) | 52–76 | prompt end vs fork in non-thinking mode |

Reported per condition: harmful refusal, over-refusal, health, and paired flips vs its baseline
(refused → answered, answered → refused). Bands may shift after step 3's quick look at the fork directions.
