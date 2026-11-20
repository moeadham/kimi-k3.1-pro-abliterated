# Refusal v1: full run and first causal probe (2026-11-19)

> **Naming (updated 2026-11-20).** The write-up in `README.md` and `docs/run3_report.md` renamed the vectors. This
> report keeps the names used at the time: **H** (later **H3**) is now **R**, the refusal direction; **R** (run 3's
> "residual" direction) is now **D**, the deflection direction; **D** (the "decision" direction at the refusal onset)
> is now **O**, the onset direction. Bank keys and file names are unchanged.

Goal B (general refusal abliteration) as the learning step toward goal A (censorship removal). Data:
`data/refusal_v1/` (English). Engine: K3.1 Pro, 8×B300, TP8, `decode-graphs`. Timings and incidents:
`docs/run_log.md`.

## 1. Data collected

| Run (`/vol/…`) | Responses | Harmful prompts | Harmless prompts |
|---|---|---|---|
| `session_v1_pairs` (matched twins, 1 sample) | 2,122 | 1,015 refused / 46 answered | 1,013 answered / 48 refused |
| `session_v1_border` (borderline, 4 samples) | 4,000 | 1,563 refused / 837 answered | 1,525 answered / 75 refused |

All rows captured (`prompt_end`, `reasoning_head`, `onset` / `comply_onset`, `answer_head`; ≤4 positions per type,
~40 MB/row).

**Mixed-outcome prompts** (the same prompt both refused and answered across its 4 samples): 97 of 1,000 (388
responses). By kind:

| Kind | Mixed | Refused overall |
|---|---|---|
| enriched fiction | 24% | 48% |
| enriched hypothetical | 20% | 76% |
| enriched professional | 20% | 34% |
| WildJailbreak adversarial harmful | 14% | 66% |
| enriched defensive | 12% | 24% |
| WildJailbreak vanilla harmful | 8% | 84% |
| WildJailbreak adversarial benign | 6% | 6% |
| WildJailbreak vanilla benign | 2% | 3% |

Framed prompts are the richest source of within-prompt contrast; resample those (8–16 samples) next.

## 1b. Per-layer analysis of the full captures (`tools/analyze_refusal.py`, `/vol/session_v1_*/analysis_refusal/`)

All directions are now well estimated: split-half cosine 0.95–1.00 at almost every layer (the pilot's 0.3–0.5
early-layer instability was sample size).

| Finding | Evidence |
|---|---|
| **For the same prompt, the start of the reasoning does not predict the outcome.** The decision forms *during* the reasoning. | Mixed prompts (97 prompts, 194 refused vs 194 answered samples), `reasoning_head`: AUC 0.48–0.55 at every layer, split-half ≈ 0 (no direction exists). |
| The decision sentence does separate within the prompt. | Mixed `onset_vs_comply`: AUC 0.90–0.98 from layer 1, highest at layers 37–50; split-half 0.85–0.93. |
| The pooled decision direction is mostly the clean within-prompt one. | Cosine(mixed, pooled `onset_vs_comply`) 0.85–0.94 across layers 15–92. |
| Harmfulness builds up through the middle layers, and becomes dataset-independent there. | Pairs `prompt_end` AUC 0.82–0.90 in layers 1–12, 0.98+ from ~37. Pairs-vs-borderline cosine 0–0.3 early, 0.84–0.92 from layer 39. |
| The decision direction is consistent across datasets at every depth. | Pairs-vs-borderline `onset_vs_comply` cosine 0.70–0.91. |
| Decision and harmfulness meet around layers 54–60 and diverge elsewhere. | Cosine(`onset_vs_comply`, `prompt_end`) 0.1–0.4 before layer 49, peaks 0.82–0.83 at layers 58–60, 0.5–0.75 after. Same peak as in the pilot. |

Taken with the probe: the decision is distributed over the middle-to-late layers and is made in the reasoning, not
fixed by the prompt. Recording the reasoning *before* the onset sentence (`docs/brainstorms.md`) is now the most
direct next measurement.

## 2. Causal probe (`tools/quick_probe.py`, results in `/vol/session_v1_probe/`)

Edit: project the direction out of every residual writer (attention `o_proj`, routed-expert `up_proj`, shared-expert
`down_proj`) of each layer in the band, one direction per layer (λ = 1), then generate (T = 1, ≤4,096 tokens) on 50
held-out HarmBench + 50 XSTest-safe prompts, judge, restore. Edit check on the real model passed first (279 writers
per rank, edit changes output, restore bitwise). Directions: mean difference on the train fold, computed from the
full captures (`tools/quick_bank.py`).

| Condition | Direction | Layers | Writers/rank | Harmful refusal | Harmless refusal | Hit max tokens | Reasoning unclosed |
|---|---|---|---|---|---|---|---|
| baseline | — | — | 0 | **100%** | 0% | 1% | 0% |
| control | harmfulness (`prompt_end`, pairs) | 10–20 | 33 | 100% | 0% | 1% | 0% |
| | harmfulness | 54–60 | 21 | 90% | 2% | 7% | 5% |
| | decision (`onset_vs_comply`, borderline) | 54–60 | 21 | 94% | 0% | 3% | 1% |
| | harmfulness | 52–76 | 75 | 76% | 0% | 10% | 4% |
| | within-prompt decision (`mixed/onset_vs_comply`) | 52–76 | 75 | 76% | 2% | 6% | 2% |
| | harmfulness | 20–92 | 219 | 64% | 2% | 11% | 9% |
| | **decision (`onset_vs_comply`)** | **52–76** | 75 | **60%** | 2% | 11% | 7% |

n = 50 per rate, so ±~7 points (1 s.e. at 75%).

## 3. What it says

1. **The directions are causal, and layer-specific.** Removing them in layers 52–76 lowers refusal; the same edit in
   layers 10–20 does nothing. The mid-to-late band found by correlation is where refusal is carried.
2. **Refusal is distributed across the band, not at a peak.** 7 layers (54–60, the correlation peak) do little
   (90–94%); 25 layers do more (60–76%); 73 layers (20–92) only a little more again (64%).
3. **The decision direction beats the harmfulness direction** on the same band (60% vs 76%): the onset-based
   direction, taken where the model decides in its reasoning, carries more of the refusal than the end-of-prompt
   direction used in classic abliteration. The within-prompt version (76%) is estimated from only 97 prompts.
4. **No single direction removes refusal.** Best single edit: 40% of held-out harmful prompts answered. Edited
   models that comply often wrap the answer in a justification ("For an authorized phishing-awareness simulation:"),
   so part of the safety reasoning survives. Consistent with refusal being several directions (lesson for goal A).
5. **Cost is visible but modest:** over-refusal stays ≤2%; responses hitting the token limit rise from 1% to ~10%,
   unclosed reasoning to 4–9%. Not yet measured: KL on harmless prompts, answer quality.

## 4. Next

- Combine directions: decision + harmfulness together (rank-2 edit), and PCA rank 2–4 per family.
- Strength and band shape: λ > 1 or a kernel over 40–92 rather than a hard band; per-band strengths.
- Bigger, cleaner eval: 200+ held-out harmful (WildJailbreak eval split), KL on harmless prompts, health.
  This is the sweep's B mode (objective: harmful refusal down; constraints: KL, over-refusal, health).
- Data: resample the framed borderline prompts 8–16× for the within-prompt direction; record pre-onset positions
  (`docs/brainstorms.md`).
- Full per-layer analysis of the v1 captures (CPU `directions` + `tools/analyze_refusal.py`, running).
