# Data sources and licences

The code in this repository is MIT-licensed (`LICENSE` at the root). The prompt files are derived from the public
datasets below, and each keeps the licence of its source; this file is the attribution. Exact revisions, filters and
counts are in each folder's `manifest.json`.

**AGPL-3.0 notice.** Prompts derived from the OBLITERATUS prompt lists are licensed under the GNU Affero General
Public License v3.0, whose full text is in [`LICENSE-AGPL-3.0`](LICENSE-AGPL-3.0). They are in
`refusal_v1/pairs_harmful.jsonl` and `refusal_v1/pairs_harmless.jsonl`, and in the `v1_pairs_obliteratus` rows of
`refusal_v3/build_harmful.jsonl` and `refusal_v3/build_harmless.jsonl`. Those four files are therefore distributed
under AGPL-3.0; the rows they contain from the other sources below keep their attribution.

**Non-commercial notice.** `refusal_v3/reference_alpaca.jsonl`, the English half of `evals/kl_prompts.jsonl`, and
`evals/cmmlu.jsonl` are CC BY-NC 4.0: non-commercial use only.

| Source | Licence | Used in |
|---|---|---|
| [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) (Jiang et al., 2024, Allen Institute for AI; gated, terms accepted on Hugging Face) | ODC-BY | `refusal_v3/build_*.jsonl` (adversarial and vanilla blocks, benign twins of framed and residual-theme prompts), `refusal_v3/reference_harmless.jsonl`, `refusal_v1/heldout_wjb_*.jsonl`, `refusal_v1/borderline_*.jsonl` |
| [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) prompt lists (`obliteratus/prompts.py`) | **AGPL-3.0** | `refusal_v1/pairs_*.jsonl`, `refusal_v3/build_*.jsonl` (`v1_pairs_obliteratus`) |
| [Semantic-Harmful](https://huggingface.co/datasets/heretic-org/Semantic-Harmful) (heretic-org) | CC BY 4.0 | `refusal_v1/pairs_*.jsonl`, `refusal_v3/build_*.jsonl` (`v1_pairs_semantic`), `refusal_v3/reference_harmless.jsonl` |
| [abliteration-harmful-enriched](https://huggingface.co/datasets/Bahushruth/abliteration-harmful-enriched) (Bahushruth) | Apache-2.0 | `refusal_v1/borderline_harmful.jsonl`, `refusal_v3/build_harmful.jsonl` (framed and residual-theme blocks) |
| [harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) (mlabonne), from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | Alpaca data: **CC BY-NC 4.0** | `refusal_v3/reference_alpaca.jsonl`, `evals/kl_prompts.jsonl` |
| [alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) (shibing624) | CC BY 4.0 | `evals/kl_prompts.jsonl` (Chinese half) |
| [harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) (mlabonne), from [AdvBench](https://github.com/llm-attacks/llm-attacks) (Zou et al., 2023) | AdvBench: MIT | `evals/harmful.jsonl` |
| [HarmBench](https://huggingface.co/datasets/walledai/HarmBench) (Mazeika et al., 2024), standard behaviours | MIT | `../results/harmbench_standard/prompts.jsonl` |
| [XSTest](https://huggingface.co/datasets/walledai/XSTest) (Röttger et al., 2024), safe prompts | CC BY 4.0 | `../results/xstest_safe/prompts.jsonl` |
| [deccp](https://huggingface.co/datasets/augmxnt/deccp) (augmxnt), CCP-sensitive questions | Apache-2.0 | `step2/replay_set.jsonl`, `../results/hosted_deccp/`, `../results/generic_judge_v*/deccp.jsonl` (the control questions in `step2/controls_v0.jsonl` were written for this project) |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (TIGER-Lab) | MIT | `evals/mmlu_pro.jsonl` |
| [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu) (Li et al., 2023) | **CC BY-NC 4.0** | `evals/cmmlu.jsonl` |
| [IFEval](https://huggingface.co/datasets/google/IFEval) (Zhou et al., 2023, Google) | Apache-2.0 | `evals/ifeval.jsonl` |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (HuggingFaceH4), from [MATH](https://github.com/hendrycks/math) (Hendrycks et al., 2021) | MATH: MIT | `evals/math500.jsonl` |
| [Berkeley Function-Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (Gorilla) | Apache-2.0 | `evals/bfcl.jsonl` |

WildJailbreak is distributed on Hugging Face behind a terms-acceptance gate; its licence (ODC-BY) permits
redistribution with attribution, given above.

Licences were read from each dataset's Hugging Face page or repository on 2026-11-20. The harmless_alpaca,
harmful_behaviors and MATH-500 pages declare none, so the licence of their upstream source is given.
