# Is the edited model dumber? Benchmark options, cost and time (2026-11-18)

**Method:** run the same items with the same sampling seeds on the original and the edited model, then compare **per item** (paired bootstrap, `harness/evals.paired_bootstrap`). "Dumber" means a drop larger than run-to-run noise. So run the original **twice** first to measure that noise.

## Pilot (measured)
- Hosted K3.1 Pro on OpenRouter (Wafer, MXFP4), 8 items per set, `low` and `max` effort, T=1.0, top-p 0.95.
- Totals: **$2.28**, 742 s wall (dominated by LiveCodeBench tails), 0 errors.
- Raw data: `results/bench_pilot/responses.jsonl`. Script: `tools/bench_hosted.py`.

| Set | Effort | Accuracy (n=8) | Mean output tokens | Max output tokens | $/item | Median latency | Max latency |
|---|---|---|---|---|---|---|---|
| MMLU-Pro | low | 8/8 | 649 | 2,751 | 0.0058 | 14 s | 18 s |
| MMLU-Pro | max | 8/8 | 991 | 4,402 | 0.0088 | 14 s | 35 s |
| MATH-500 | low | 8/8 | 420 | 1,430 | 0.0037 | 14 s | 14 s |
| MATH-500 | max | 8/8 | 1,324 | 7,888 | 0.0117 | 14 s | 56 s |
| IFEval | low | 5/8 | 1,284 | 2,389 | 0.0113 | 14 s | 19 s |
| IFEval | max | 6/8 | 1,780 | 4,317 | 0.0157 | 16 s | 34 s |
| CMMLU | low | 8/8 | 359 | 732 | 0.0032 | 3 s | 18 s |
| CMMLU | max | 8/8 | 626 | 2,538 | 0.0055 | 5 s | 35 s |
| LiveCodeBench | low | (not executed) | 9,662 | 63,765 | 0.085 | 10 s | 727 s |
| LiveCodeBench | max | (not executed) | 15,244 | 64,000 (hit the cap) | 0.134 | 64 s | 589 s |

**Takeaways**
- K3.1's reasoning is short (median 150–360 reasoning tokens outside code), so most sets are cheap.
- **MMLU-Pro, MATH-500 and CMMLU look near the ceiling** (8/8; n is tiny). A benchmark the model nearly maxes out can't show damage well. **MATH-500 is saturated: replace it with AIME.**
- **LiveCodeBench has long tails**: some responses run to the 64k-token cap. Cap `max_tokens` (e.g. 32k) and count a capped answer as wrong.

## Recommended "dumbness panel"
✱ = no pilot data; tokens estimated from comparable sets. Hosted cost is per full pass on OpenRouter (~$0.25/M input, ~$9/M output).

| Benchmark | What it catches | Items × samples | Effort | Hosted cost / pass | Wall time / pass (concurrency 48) | Data access |
|---|---|---|---|---|---|---|
| **KL on harmless prompts** (EN+ZH) | any distribution shift; the most sensitive signal | 200 × 1 | low | GPU only (needs our logprobs) | minutes | in repo |
| **GPQA-Diamond** | hard science reasoning | 198 × 4 | max | ✱ ~$15–30 | ~15–25 min | **gated: accept the terms on HF** |
| **AIME 2025** | hard math; replaces saturated MATH-500 | 30 × 8 | max | ✱ ~$20–45 | ~20–30 min (long traces) | public (`MathArena/aime_2025`) |
| **LiveCodeBench** (test6, stdin) | coding | 112 × 1 | max | ~$15 | ~15 min (tails) | in repo (scored on Modal only) |
| **IFEval** | instruction following | 541 × 1 | low | ~$6 | ~5 min | in repo |
| **SimpleQA** (subset) | factual recall and hallucination, which ablation can damage | 1,000 × 1 | low | ✱ ~$3–5 (+ cents for a Jev grader) | ~5 min | public (`basicv8vc/SimpleQA`) |
| **MMLU-Pro** | broad knowledge canary (near the ceiling) | 500 × 1 | low | ~$3 | ~5 min | in repo |
| **CMMLU** | Chinese knowledge canary (near the ceiling) | 500 × 1 | low | ~$1.6 | ~3 min | in repo |
| ~~MATH-500~~ | saturated | — | — | — | — | — |
| *Optional:* Humanity's Last Exam (text subset) | frontier-hard reasoning, no ceiling | ~500 × 1 | max | ✱ ~$20–40 | ~20 min | gated (`cais/hle`) |

- **Hosted panel total (one pass):** roughly **$65–110** and about **30–40 min** when sets run in parallel (longest tail ≈ AIME/LCB).
- **Run the original twice** to measure noise: about **$130–220**.
- **On Modal (our model, both arms):** about **10M output tokens per pass** (AIME ~3.6M, GPQA ~2.4M, LCB ~1.7M, rest ~2M). At an assumed 3–6k tok/s aggregate decode for K3.1 on 8×B300 (**not measured**), that is **~30–60 min per model per pass**.

## How to read results
- With n items, the paired difference has SE ≈ √(discordant fraction / n). For example GPQA (198 items × 4 samples) can resolve about ±3–4 points, AIME (30 × 8) only about ±6–8.
- **Pass:** within ~1–2 points of the original, judged against the measured run-to-run noise, on every benchmark. **Plus:** KL below threshold, and no change in reasoning health (closed `</think>`, capped outputs, repetition).
- The hosted trial gives token budgets, ceiling and noise estimates. The comparison itself (original vs edited) must run on our own runtime, because hosted quantization and runtime differ.

## Sanity panel: the cheap "did we lobotomize it?" check (2026-11-18)
- **Definition:** `harness/sanity.py`. Identical items on hosted and on our engine (`run_on_engine`).
- **Settings:** all items at effort `low`, T=1.0, top-p 0.95, 1 sample.
- **Coarse by design:** it catches large drops, not 1–2 point shifts.

**Reference: unedited K3.1 Pro, hosted** (`wafer/us`, native MXFP4)
- Data: `results/sanity_panel_hosted/`
- Totals: **$2.31**, **181 s** wall at concurrency 32, 0 errors.

| Set | n | Accuracy | Mean output tokens | Cost | Median / max latency | Repetition | Off-language |
|---|---|---|---|---|---|---|---|
| MMLU-Pro | 100 | **0.92** | 526 | $0.47 | 2.9 / 65 s | 0.06 | 0.00 |
| CMMLU | 50 | **0.98** | 322 | $0.14 | 3.3 / 28 s | 0.07 | 0.09 (option letters) |
| IFEval (strict) | 100 | **0.89** | 1,391 | $1.23 | 9.9 / 154 s | 0.14 | 0.00 |
| MATH-500, level 5 | 50 | **0.98** | 873 | $0.39 | 4.5 / 48 s | 0.10 | 0.00 |
| BFCL v3 tool calls (simple/multiple/parallel) | 50 | **0.88** | 186 | $0.09 | 1.6 / 7 s | 0.01 | 0.00 |

- No response hit `max_tokens`, and none came back empty.
- **BFCL misses are BFCL's strict ground truth.** For example `"Luxury room"` vs the allowed `"luxury"`, and `"violent crime"` vs `"violent"`. These rules apply equally to base and edited, so they don't distort a comparison.
- **On our engine:** add KL on `data/evals/kl_prompts.jsonl` (`Evaluator.kl_reference` / `kl`).

**Pass rule for an edited model**
- every set within ~5 points of the unedited model *run on our own engine* (the hosted numbers are a cross-check, not the reference)
- KL below threshold
- health metrics unchanged: no new empty answers or max-token hits; repetition and off-language within noise
