# Refusal dataset v1 (English), 2026-11-18

Built by `tools/build_refusal_v1.py` into `data/refusal_v1/`. Sources are pinned; `manifest.json` has the revision, sha256,
counts per kind and every filter count. A rebuild is byte-identical.

## Why
The dry run (`docs/refusal_dryrun_analysis.md`) could not separate "I will refuse" from "this request looks harmful": HarmBench
was always refused and XSTest-safe almost always answered, so the two directions coincided (cos 0.99+). v1 adds matched
twins (harmfulness with phrasing held fixed) and borderline prompts that K3.1 should refuse only *sometimes*, so the
refusal decision can be contrasted within one prompt.

## Files
| File | n | Contents | Use |
|---|---|---|---|
| `pairs_harmful` / `pairs_harmless` | 1,061 each, same ids, same order | OBLITERATUS built-ins (840 pairs, template- or topic-matched) + heretic-org/Semantic-Harmful pairs with similarity ≥ 0.7 (221) | harmfulness direction with phrasing fixed; `--samples 1` |
| `borderline_harmful` | 600 | WildJailbreak vanilla_harmful 200, adversarial_harmful 200; enriched framed prompts 4 × 50 (fiction, professional, hypothetical, defensive) | `--samples 4`, T=1; label by judge |
| `borderline_harmless` | 400 | WildJailbreak vanilla_benign 200, adversarial_benign 200 | same |
| `heldout_wjb_harmful` / `heldout_wjb_benign` | 1,834 / 205 | WildJailbreak eval split | held out: never for direction finding |

`split` holds the kind (e.g. `wjb_adversarial_benign`, `enriched_defensive`) and reaches `rows.jsonl` as `source_split`.
OBLITERATUS rows carry their section `category`; Semantic rows their similarity `score`.

## Rules for using it
- **Label by the judge's verdict, not by file.** Some WildJailbreak "harmful" items are mild ("reveal YouTube's
  recommendation algorithm"), and the `enriched_defensive` prompts are mostly legitimate questions.
- **Contrast within a kind.** Adversarial prompts are ~130 words, vanilla ~16. Within each kind, the harmful and benign
  sides are length-matched (same per-bin word counts: vanilla median 16/16, adversarial 130/130).
- **Held out everywhere** (decontaminated against, never sampled): HarmBench standard 200, XSTest safe 200,
  `data/evals/harmful.jsonl` (and its source, mlabonne/harmful_behaviors test), `data/evals/kl_prompts.jsonl`,
  the WildJailbreak eval split.

## Filters
- English only: any character outside ASCII (except typographic quotes and dashes) drops the prompt.
- Decontamination: normalized exact match, or word-5-gram containment ≥ 0.5 against any held-out prompt. This is
  conservative: most of the 44 dropped pairs are real paraphrases (mlabonne's train and test splits paraphrase each other),
  a few are stock phrases ("a step-by-step guide on how to …"). A pair is dropped if either side fails.
- Enriched: drop the 520 rows copied from mlabonne/harmful_behaviors (includes all 104 of `data/evals/harmful.jsonl`), drop
  DAN/"no restrictions" rows (WildJailbreak covers jailbreak wrappers), keep only rows with a recognizable framing, and allow
  at most 2 rows per opening 6 words (the set is heavily templated).
- WildJailbreak adversarial prompts capped at 300 words.

## Running it
`--n-per-class N` takes the first N rows; files are shuffled so any prefix is a representative mix. The stage writes to
`OUT/refusal`, so use a separate `--out` per run:
```
python -m harness.session refusal --model /vol/Kimi-K3.1-Pro --tp 8 --out /vol/session_v1_pairs \
  --harmful data/refusal_v1/pairs_harmful.jsonl --harmless data/refusal_v1/pairs_harmless.jsonl
python -m harness.session refusal --model /vol/Kimi-K3.1-Pro --tp 8 --out /vol/session_v1_border --samples 4 \
  --harmful data/refusal_v1/borderline_harmful.jsonl --harmless data/refusal_v1/borderline_harmless.jsonl
```

## Size and cost
- Full v1 = 2,122 pair generations + 4,000 borderline generations ≈ 6,100 rows. From the dry run, 1,600 generations take
  ~15 min with `decode-graphs`, so generation is ~1 h.
- **Captures dominate storage:** the dry run stored every position (~107 MB per row). Captures are now trimmed to
  `--capture-per-type 4` evenly spaced positions per type (prompt_end, reasoning_head, answer_head, and onset or
  comply_onset): ≤16 positions ≈ ≤43 MB per row → **~260 GB** for everything, ~130 GB with `--capture-per-type 2`.
  The first pass can cap the pairs (`--n-per-class 400` → 800 rows, ~35 GB), since 200 rows per class already gives
  stable directions.
- **Mixed prompts:** with `--samples 4`, a prompt whose samples were both refused and answered is marked
  `mixed_prompt`; `directions` and `tools/analyze_refusal.py` contrast those samples within the prompt
  (`mixed/reasoning_head`, `mixed/onset_vs_comply`). `prompt_end` is identical across a prompt's samples, so the
  within-prompt contrast starts at the reasoning.

## Known issues
- ~~Train/val folds split twins.~~ Fixed: folds hash the `pair` field when present (`refusal.fold_key`), so both
  twins land in the same fold; prompts without a pair (borderline, held-out, the dry run) keep their own hash.
  Samples of one prompt share a fold.
- English only. Chinese is deferred.
- OBLITERATUS prompts are AGPL-3.0: fine for internal direction-finding; check before redistributing a derived set.
