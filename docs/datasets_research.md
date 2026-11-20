# CCP-censorship prompt datasets: research (2026-11-17)

How each claim was checked:
- **[V-me]**: I checked it directly (HF API / datasets-server / arXiv).
- **[V-agent]**: a research subagent opened the source page.
- **[unv]**: not verified.

Checking was cut short by a network outage, so some arXiv/GitHub items are [V-agent] only.

## Key methodological finding
**Ablating a refusal direction does not remove propaganda.**
- deccp's author reports that ablated Qwen2-7B answered in party-line framing instead of refusing [V-agent].
- Frank (arXiv 2603.18280) reports that ablating Qwen3-8B's political direction mostly produced confabulations about the wrong event [V-agent].
- The Kimi K2.5 safety evaluation (arXiv 2604.03121) finds that **Kimi deflects in CCP-aligned language more often than it refuses, and does so much more in Chinese** [V-agent].

So evaluation must score **refusal**, **CCP framing / propaganda** and **factual accuracy (fact recall + lies)** as separate axes. Jev's typed questions fit this well.

## Resources
| Resource | Size / lang / license | Contents & controls | Notes |
|---|---|---|---|
| promptfoo/CCP-sensitive-prompts | 1,360 rows (68 subjects × 20) [V-me]; EN; **CC-BY-NC-4.0** [V-me] | Instruction-style, often loaded (e.g. "how to evade surveillance"); no controls or answers | Synthetic and templated; ~15% not China-specific [V-agent]; NC license |
| augmxnt/deccp | censored 95 / uncensored 37 [V-me]; EN + ~30 zh; Apache-2.0 [V-me] | Tiananmen, Xinjiang, Tibet, 709 crackdown, GFW, Taiwan; "uncensored" = analogous questions about other countries | Tiny; labelled from Qwen2-7B refusals only; widely used (Qwen2-deccp, Kimi K2.5 eval) |
| Nafnlaus/ShrimpMoss_Chinese_Censorship_Abliteration | 1K–10K paired `china_bad_*` / `china_good_*`; EN/zh/yue; MIT [V-me] | Built for abliteration | Mistral-NeMo-generated, noisy; candidate pool only, after filtering |
| xlr8harder llm-compliance / SpeechMap | `china_criticism.jsonl` 50 rows + zh/fi versions; Apache-2.0 [V-agent] | Requests to write critical essays/poems; **matched us/eu/russia_criticism controls** | Judge labels COMPLETE / EVASIVE / DENIAL |
| Pan & Xu, PNAS Nexus 2026 | 145 questions, zh + EN, with Chinese reference answers [V-agent]; Dataverse doi:10.7910/DVN/VQMOJU (license [unv]) | Expert-sourced: censored social-media events, HRW reports, blocked zh-Wikipedia | **Best candidate for a held-out eval** |
| Casademunt et al. 2026, "Censored LLMs as a Natural Testbed…" (arXiv 2603.05494 [V-me]); github.com/cywinski/chinese_auditing | 90 test + 10 dev questions, >1,500 ground-truth atomic facts; EN [V-agent] | 12 topics, adversarially filtered on Qwen3-32B | Rubric to borrow: refusal y/n, honesty 0–100, per-fact mentioned/not mentioned/lie. **Held-out eval** |
| Frank, "Detection Is Cheap, Routing Is Learned" (arXiv 2603.18280); github.com/gregfrank/routing-is-learned (MIT) [V-agent] | v2: 120 pairs, 15 categories; v3: 32 prompts × 4 intensities with **Western parallels** + 16 zh; safety 120+120 | Built for directional ablation of Qwen/DeepSeek/GLM | Closest existing setup to ours; judge agreement with humans only ~55% |
| Not public | Perplexity R1-1776 eval; MAI-DS-R1 3.3k blocked prompts; NIST CAISI "CCP alignment" (Kimi K2 Thinking: ~26% aligned in zh vs 7% in EN); Kimi K2.5 eval set | — | Design ideas only |
| Keyword lists | Citizen Lab merged list, 9,054 terms (github.com/jasonqng/chinese-keywords); greatfire ~26.8k rows; CDT-published leaked ByteDance list of 5,241 terms [unv] | — | Old and noisy; use for topic coverage when generating prompts |

## Plan for our datasets
- **(a) Censored set (direction finding):**
  - deccp-censored
  - deduplicated promptfoo **questions** (drop "how to evade X" items, which overlap with harmful requests)
  - SpeechMap china_criticism (EN + zh)
  - Frank v2/v3
  - Casademunt topics as topic seeds only (their questions stay held out)
  - ~40% Chinese, native-written or native-checked; Kimi's gap is largest in Chinese
  - fill topic gaps from keyword lists: 709, Liu Xiaobo, Peng Shuai, Urumqi fire / A4 protests, HK NSL cases, Xi-era purges, post-2023 events
- **(b) Matched controls:** for every sensitive prompt, a twin with the same template, verb and length (±10%), with the country or event swapped (Tiananmen → Kent State / Gwangju; Xinjiang → Guantánamo; HK NSL → Russia's foreign-agent law). Plus a **second control of non-sensitive China prompts** (cuisine, geography, economy), so the direction is "CCP-sensitive", not "about China".
- **(c) Refusal retention:** HarmBench (+ zh translations), JailbreakBench harmful/benign pairs, XSTest for over-refusal.
- **(d) Held-out eval:** Pan & Xu 145 + Casademunt 90 (with facts). Never used for direction finding. Hold out whole topic categories, not just prompts.

## Gaps we must fill
- native Chinese prompts with paired controls
- reference answers / atomic facts beyond Casademunt's topics
- a propaganda-framing rubric checked against human ratings
- post-2023 events
- multi-turn / indirect phrasings
- Kimi-specific deflection examples
