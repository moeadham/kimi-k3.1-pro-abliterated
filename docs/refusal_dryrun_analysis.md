# Refusal dry run: per-layer analysis (2026-11-18)

Data: 400 K3.1 responses on 8×B300 (`/vol/session/refusal/`): HarmBench standard 200 + XSTest-safe 200, effort max,
T=1, one sample each. Judge flags: harmful 195/200, harmless 11/200 → 206 flagged vs 194 answered.
Analysis: `tools/analyze_refusal.py` (CPU sandbox, ~16 min with the `directions` stage), outputs in
`/vol/session/analysis_refusal/` (`per_layer.json`, `table.txt`, `directions_mean_diff.pt`) and
`/vol/session/directions_refusal{,_prompt}/` (the session stage's bank). Stream index s: 2 per decoder layer
(s = 2·layer attention input, 2·layer+1 MLP input), s = 186 the final mix. Per-row vector = mean over the
positions of each type. AUC and effect size are 5-fold cross-validated (in-sample effect sizes are inflated at
d = 7168).

## Findings
1. **The directions are well estimated.** Split-half cosine is 0.92–0.99 at almost every layer for every
   position type, so ~200 rows per class is enough to estimate a mean-difference direction per layer.
2. **Separation does not localize anything.** Cross-validated AUC is 0.95–0.99 from layer ~2 to the end at
   `prompt_end` and 0.91–0.99 at `reasoning_head`. Harmful vs harmless *prompt label* separates perfectly
   (AUC 1.000) from layer 1. There is no single layer where the signal appears.
3. **At `prompt_end`, the refusal direction is the harmfulness direction.** The direction from judge flags and
   the one from prompt labels have cosine 0.99+ at every layer. The 5 HarmBench prompts the model *answered*
   project like refused ones at every layer (above ~99–100% of answered harmless rows). So what `prompt_end`
   carries is "this request looks harmful", not "I will refuse"; with 96% label/flag agreement this data cannot
   separate the two there.
4. **Borderline cases are resolved around layer 40.** The 11 harmless prompts the model refused (9 fictional-
   character privacy requests) sit mid-distribution up to layer ~30 (median 0.43–0.58) and jump to 0.89–0.94 from
   layer ~40 on. Flag-label AUC at `prompt_end` also steps up there (0.96 → 0.98 at layer 40, peak 0.99 at
   layers 64–87). That is the only refusal-specific structure visible, and it rests on 11 rows.
5. **`reasoning_head` carries a different direction.** Its cosine to the `prompt_end` direction is 0.2–0.85
   (highest at layers 51–56); its AUC climbs steadily to a peak at layers ~60–64. `onset` vs answered reasoning
   separates trivially (AUC ≈ 1 everywhere: "I must refuse" vs task text) and says little about location.
6. **The direction rotates.** Cosine between adjacent stream indices is 0.7–0.98: a single global direction
   will not fit all layers; per-layer (or per-band) directions are needed.
7. **Artifacts.** s = 0 at `prompt_end` is degenerate (the last 4 prompt tokens are fixed template tokens, so
   the layer-0 input is identical for every row): the session `directions` stage picked `best_s = 0` there
   because its AUC broke ties by argsort order. Fixed: AUC now gives ties their average rank (identical
   scores → 0.5) and `best_s` skips zero-norm directions; the sweep never used s = 0 (`direction_index` ≥ 1). At s = 186 (final mix) reasoning/answer separation
   collapses (AUC 0.62–0.74). `answer_head` is excluded (engine-dependent, `docs/modal_runbook.md`).

## Is there enough data?
- For estimating per-layer mean-difference directions: yes (finding 1).
- For locating *where refusal is decided* or separating refusal from harmfulness: no. Only 16 rows disagree
  with their prompt label (5 harmful-answered, 11 harmless-refused), and the label and flag directions coincide.

## Next
- **Get within-prompt variation:** `--samples 4` at T=1 on borderline prompts (HarmBench categories the model
  sometimes answers, XSTest privacy/contrast sets), so the same prompt yields refusals and compliances; the
  contrast between those samples isolates the decision from the prompt's harmfulness. With `decode-graphs`,
  1,600 generations take ~15 min.
- **Causal test on GPU:** ablate the per-layer `prompt_end` / `reasoning_head` directions (bands around layers
  40–64) by steering, measure refusal on held-out HarmBench and on XSTest, plus KL on harmless prompts.
- Rerun the CPU `directions` stage to refresh `/vol/session/directions_refusal*/summary.json` (old `best_s` for
  `prompt_end` is 0).
