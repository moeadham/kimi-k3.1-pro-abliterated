# Sanity panel: "did the edit lobotomize the model?"

A cheap, fast check that an edited Kimi K3.1 Pro still works: about 350 fixed questions plus a KL measurement. It is **coarse by design**. It catches a broken model, not a 1–2 point regression. For fine-grained capability numbers use the larger benchmarks in `docs/capability_benchmarks.md`.

## What it runs
- **Code:** `harness/sanity.py`. One definition is shared by every run, so the questions are always identical.

| Check | Items | What a broken model looks like |
|---|---|---|
| MMLU-Pro | 100 | knowledge collapse |
| CMMLU (Chinese) | 50 | Chinese broken; matters because the edit targets a China-linked direction |
| IFEval (strict) | 100 | can't follow simple formatting instructions |
| MATH-500, level 5 only | 50 | reasoning collapse |
| BFCL v3 tool calling (20 simple / 15 multiple / 15 parallel) | 50 | broken tool-call format, i.e. agentic breakage |
| KL on harmless prompts (EN+ZH) | 100 | output distribution shifted wholesale (our engine only) |
| Reasoning health on every answer | all | thinking never closes, max-token hits, empty answers, repetition loops, language mixing |

- **Settings:** effort `low`, temperature 1.0, top-p 0.95, 1 sample per item, fixed item selection (seed 0). Frozen data with hashes: `data/evals/` (`manifest.json`).

## Pass rule
- **Code:** `harness/sanity.py: compare`. Thresholds are in `THRESHOLDS`, and each can be overridden from the CLI.
- **Accuracy:** each set may drop at most **5 points** vs the unedited model.
- **KL:** per-token KL vs the unedited model at most **0.10 nats**.
- **Health:** max-token hits and empty answers may rise by at most 2 points; repetition and off-language by at most 0.05. Reasoning that closes properly may drop by at most 2 points.

**Always compare against the unedited model run on the same engine.** Hosted numbers differ in quantization and runtime, so they are a cross-check, not the reference.

## How to run
**1. Our engine (Modal / 8×B300): unedited and edited in one go, same engine, same items**
```
python -m harness.session sanity --model /local/Kimi-K3.1-Pro --out /local/session --sanity-edit winner
```
- `--sanity-edit` takes `none`, `winner` (`$OUT/tier2/winner.json`) or `trial:N` (a sweep trial).
- The edit is applied in memory and restored afterwards.
- Writes `$OUT/sanity/{base,edited,verdict}.json`. The stage summary says PASS or FAIL and names the failed checks.

**2. Hosted reference (OpenRouter, unedited only)**
```
uv run --env-file .env --extra tok --with lm-eval --with langdetect --with immutabledict \
  --with nltk --with math-verify python tools/sanity_panel_hosted.py --out results/sanity_panel_hosted
```
- Provider pinned to `wafer/us`: native MXFP4, no content filter, no fallbacks.
- About **$2.30 and 3 minutes** at concurrency 32.

**3. Compare any two results**
```
python tools/compare_sanity.py BASE.json CANDIDATE.json        # exit 0 = PASS, 1 = FAIL
```
Accepts engine results (`base.json`, `edited.json`) and hosted summaries (`summary.json`).

## Reference: unedited K3.1 Pro, hosted (2026-11-18)
`results/sanity_panel_hosted/` ($2.31, 181 s, 0 errors)

| Set | Accuracy |
|---|---|
| MMLU-Pro | 0.92 |
| CMMLU | 0.98 |
| IFEval | 0.89 |
| MATH-500 L5 | 0.98 |
| BFCL | 0.88 |

- No empty answers or max-token hits.
- BFCL misses come from BFCL's strict answer key (e.g. `"Luxury room"` vs the allowed `"luxury"`). They apply equally to base and edited.
