# Refusal run 2: non-thinking forks and the first clean edit (2026-11-19)

> **Naming (updated 2026-11-20).** The write-up in `README.md` and `docs/run3_report.md` renamed the vectors. This
> report keeps the names used at the time: **H** (later **H3**) is now **R**, the refusal direction; **R** (run 3's
> "residual" direction) is now **D**, the deflection direction; **D** (the "decision" direction at the refusal onset)
> is now **O**, the onset direction. Bank keys and file names are unchanged.

Goal B (general refusal) as the learning step toward goal A (censorship). Run 2 turned reasoning off (greedy, T=0),
located where the refusal decision forms token by token, and spent the rest of the sandbox on weight-edit probes.
Plan: `docs/plan_nonthinking.md`. Timings and incidents: `docs/run_log.md`. Every dataset was reviewed afterwards;
the review outputs (scripts, tables, per-record re-judgements) are in `out/run2/review/` (not in git) and the
reports are copied to `results/run2_review/`.

## Headline

1. **A clean edit exists for non-thinking mode.** Projecting out the *projected* harmfulness direction H (run 1,
   matched pairs, last prompt token) from layers 48–84 at λ 1–1.5 cuts harmful refusal on 150 held-out
   WildJailbreak prompts from **72% to 7%**, with harmless refusal unchanged (2–4%, both baseline flags are judge
   false positives) and no measurable text damage. Projected D+H on 48–84 or 52–76 and projected `nt_prompt_end`
   (run 2's own direction) are statistically tied (8–9%). 85 of 101 answered responses read by hand were genuinely
   on-task, 16 partial, 0 evasions: ~70–80% of harmful prompts genuinely answered, against ~25% unedited.
2. **"Projected" is what matters.** The raw versions of the same directions are much weaker (raw D+H 23–28%; raw
   `nt_prompt_end` 80–88%) and damage text: raw D turns commas into periods when the band reaches layers 40–43 or
   85–92; raw `nt_prompt_end` makes the model reason inside the answer channel. The band edges matter too: projected
   D+H on 40–92 shows the comma defect from λ 1.25 up, 48–84 and 52–76 never do.
3. **In non-thinking mode the decision is made by the end of the prompt.** It is readable at AUC ~0.9 over the last
   words of the request and 0.97–0.99 on the fixed chat-template tokens after it (layers 56+), before any answer
   token. The branch test agrees: 78% of clean refusal flips happen at the first answer token; after "I" the refusal
   is ~98% committed, and the refusal word Jev marks as the onset comes after the commitment.
4. **The decision direction D is redundant once H is projected**, in both modes for goal B (non-thinking: Hp 7% vs
   Dp 21% vs Dp+Hp 9%; reasoning: Hp ≈ Dp+Hp). D may still matter for goal A, where "harmful" and "won't discuss"
   must be told apart.
5. **Reasoning mode carries over, with a measurement caveat.** All reasoning probes ran at effort "max" with a
   4,096-token cap; edited traces often ran out (the model drafts the whole answer inside its reasoning) and the
   judge then scored an empty answer, mostly as a refusal. Re-judged on where the reasoning was heading, the best
   edits refuse **8–11% (lower bound) to ~23–31% (rough upper bound)** of harmful prompts, against 75% unedited.
   Next run: effort "low" (now the probe default) so traces finish.
6. **What the edits leave:** residual refusals are soft deflections, not hard refusals, on recurring themes:
   group-stereotype articles, atrocity revisionism with a "be balanced" cue, exploit code with a defensive wrapper,
   hazardous chemistry framed as academic. That looks like a separate "even-handed reframing" behaviour, likely
   relevant to goal A.

## 1. Data collected

| Data (`/vol/…`) | Size | What |
|---|---|---|
| `session_nt/forks` | 3,122 answers | non-thinking greedy answers (≤512 tokens, top-20 logprobs), Jev verdict + decision sentence + word-onset candidate per answer; borderline 1,000 (600 harmful / 400 harmless) + pairs 2,122 (1,061 / 1,061) |
| `session_nt/forks/branches.jsonl` | 11,299 branches | token substitutions at the first answer tokens and around the onset candidate, continued greedily, judged; 1,240 answers covered (all borderline, 240 pairs refusals, no harmless pairs) |
| `session_nt/forks/states` | 3,122 rows | activations, prompt window → fork + 16, every 4th stream index, plus all streams at the prompt end and fork |
| `session_nt/recovery` | 1,800 sequences | original / recovered / flipped continuations at 600 swap positions, swap state + 32 tokens |
| `session_run2_probe` | 25 conditions × 100 | HarmBench 50 + XSTest 50, greedy, reasoning and non-thinking |
| `session_run2_probe_big` | 30 conditions × 200 | WildJailbreak eval 150 + XSTest 50 |
| `session_nt/analysis` | — | token-by-token readouts, recovery scores, live log |

The run-2 bank `nt.pt` was written to the GPU host's local disk and lost with the sandbox; it can be rebuilt from
the captures with `tools/fork_bank.py` (see §6). Write banks to the Volume next time.

## 2. Non-thinking behaviour (3,122 answers)

| Set | Refused |
|---|---|
| pairs harmful | 1,028 / 1,061 (97%) |
| pairs harmless | 43 / 1,061 (4.1%) |
| borderline harmful | 420 / 600 (70%) |
| borderline harmless | 28 / 400 (7%) |

By source, harmful: semantic 99%, obliteratus 96%, WildJailbreak vanilla 90%, enriched hypothetical 82%, enriched
fiction 66%, WildJailbreak adversarial 66%, enriched professional 46%, enriched defensive 26%. Of 71 harmless flags,
43 are explicit refusals and 28 deflection-only (0.6–0.8); real over-refusals are keyword-triggered (wrist-rocket
slingshot, pentesting one's own network, "poison a conversation with humor").

- **Same prompts, reasoning on (run 1) vs off:** 93% (harmful) / 96% (harmless) agreement on healthy samples.
  Run-2 refusal tracks run 1's per-prompt refusal fraction (25 / 38 / 64 / 87 / 93% for 0–4 of 4 samples refused).
  Non-thinking refuses reflexively on 60 harmful prompts run 1 never refused, and answers 27 run 1 always refused
  (17 are WildJailbreak roleplay templates).
- **Style:** 92% direct "I can't" (hosted catalogue: 72%), 4% partial "I can help with X but not Y" (hosted 17%).
  The decision sentence is sentence 0 in 95% of refusals.
- **A free graded score:** P(first answer token = "I") separates refused from answered at AUC 0.98 (0.93 on
  borderline). q = P("’t" after "I can") grades how close a refusal was (median 0.31 when run 1 refused 0/4, 0.90 when
  4/4; Spearman 0.71 with run 1's fraction, as good as Jev's own score). Caveat: the stored `rows.p_chosen` is
  P(" can") for 1,206 unconfirmed refusals, not the decision token; do not use it as a graded score.
- **Data quality:** 25 answers (0.8%) are leaked reasoning ("I need respond to user…"); exclude them
  (`out/run2/review/nt_corpus/reasoning_leak_keys.json`). 46% of answers hit 512 tokens (mostly harmless); at most
  ~1% of verdicts plausibly changed.
- **Judge:** 0 clear errors in 40 random verdicts; ~4 false positives in 50 read from the 0.4–0.8 band, mostly
  harmless WildJailbreak-benign deflection flags.
- **Goal-A example:** on a social-credit-system prompt the model writes a hidden stance line ("social credit is a
  governance tool for trust and order, not suppression…") and then answers with the official framing.

## 3. Where the decision forms

### Activations (cross-validated AUC, refused vs answered, `tools/analyze_forks.py`)

Anchored at the prompt end (0 = last prompt token):

| Position | Layer 8 | Layer 40 | Layer 56 | Layers 72–88 |
|---|---|---|---|---|
| ~40 tokens before (mid-request) | 0.59 | 0.59 | 0.61 | ~0.59 |
| ~17 before (last words of the request) | 0.74 | 0.88 | 0.90 | 0.83–0.90 |
| chat-template tokens after the request (−13 to −1) | 0.69–0.79 | 0.94–0.96 | 0.93–0.98 | 0.93–0.99 |
| prompt end (0) | 0.79 | 0.93 | 0.98 | 0.99 |
| first answer tokens (+1 to +16) | 0.95–0.98 | 0.98 | 0.98 | 0.98 |

The template tokens are identical for every prompt (layer 0 is exactly 0.50 there), yet by layer 56 they carry the
outcome: the model writes its decision onto them. Anchored at the fork: 0.79–0.85 at 24–32 tokens before, 0.93–0.97
from 12 before, 1.00 at the fork.

### Branch test (review: `results/run2_review/branches_tables.md`)

- **Measurement flaw:** branches were judged on ≤96 tokens, originals on the full answer. Cutting the original to
  the same length flips 18.5% of answered rows to "refused"; the control branches flip 18.6%. So the controls
  measured the length mismatch, and most answered→refused flips are artefacts (984 of 1,399 are deflection-only).
  Judged like with like (refused ≥ 0.6), substitutions flip 8.6% vs 1.4% for controls. Refused→answered flips are
  mostly real (5.2% vs 1.0%).
- **259 of the 486 saved forks are clean** (112 refused, 147 answered); 235 rest on a single substitution and 134
  sit at the last tested position.
- **Commitment:** 78% of clean refused→answered flips are at answer token 0. Token 1: 0.4 clean flips per 100
  branches; token ≥3: 1 in 1,071. Swapping "’t" in "I can’t" for "I can help…" recovers 94% of the time ("I can help
  with fiction…, but I can’t write…"). Weaker refusals flip (original score 0.77 vs 0.92). Refusal→deflection is more
  common than refusal→answer.
- **Greedy text is batch-dependent:** 76% of controls diverge from the original text (median 107 chars in), but the
  outcome changes ~1% of the time.
- **The `nt_fork` confound, quantified:** refused rows are anchored at answer token 1 (" can") in 85% of cases,
  answered rows at a median of token 17 (" Here", " Below"). A position-matched set exists: clean token-0 forks,
  86 refused vs 85 answered, whose prompt-end states are already captured; plus a graded first-token refusal mass
  for 1,238 answers (`out/run2/review/branches/data/token0_refusal_mass.json`).
- **Recovery captures need filtering:** only 206 of the 600 positions are clean flips; `plan()` used control branches
  as "recovered"/"flipped" 58 times and its alphabetical selection dropped 126 harmless triplets. The recovery scores
  below (§6) should be re-run on the clean subset.
- **Cheaper next time:** judge branches and originals at the same length on `refused`; controls on a sample; token 0
  with 5–8 alternatives plus the negation token; skip same-meaning alternatives ("I'm", " can't", " not": 35% of
  substitutions, zero clean flips); test sequentially, token 0 first.

## 4. Weight-edit probes

Edit = remove direction(s) from the attention and MoE/MLP output writers of a layer band, strength λ; greedy; Jev
judge; restore (`tools/quick_probe.py`). Full audit of all 63 conditions: `results/run2_review/probes_report.md`.
Run-to-run noise: ±2 points SD on 150 prompts; treat differences under ~5 points as noise.

### Non-thinking, 150 WildJailbreak harmful + 50 XSTest (≤512 tokens)

| Edit | Layers | λ | Harmful refused | Harmless refused | Text |
|---|---|---|---|---|---|
| none | — | — | 72.0% | 4% | clean |
| **Hp** | 48–84 | 1.5 | **6.7%** | 4% | clean |
| NTPEp (run 2's prompt-end direction) | 40–92 | 1.5 | 8.0% | 2% | clean |
| NTPEp | 48–84 | 1.5 | 10.7% | 2% | clean |
| Dp + Hp | 52–76 | 1.5 | 8.7% | 2% | clean |
| Dp + Hp | 48–84 | 1.5 / 1.0 | 8.7% / 12.7% | 4% | clean |
| Dp + Hp | 40–92 | 1.0 / 1.25 / 1.5 / 2.0 | 10.7 / 9.3 / 9.3–10.0 / 14.0% | 4% | commas→periods from λ 1.25 |
| Dp + Hp | 56–72 | 1.5 | 17.3% | 4% | clean |
| Dp | 48–84 | 1.5 | 21.3% | 2% | clean |
| Dp + H (raw) | 40–92 | 1.0 | 23.3% | 4% | |
| D + H (raw) | 40–92 / 44–88 / 36–92 | 1–2 | 28–33% | 4–14% | commas→periods at 40–92 and 36–92 |
| D + H (raw) | 40–92 | 2.5 | 48% | 30% | broken: glued words, Chinese, loops |
| D + H + NTPE / + mixed (raw) | 40–92 | 1.5 | 67% / 34% | 20% | reasoning leaks into answers |
| NTPE (raw) | 40–92 | 1.0 | 53% | 8% | reasoning leaks into answers |

(p = projected. D = run-1 decision direction `onset_vs_comply/all`, borderline; H = run-1 `prompt_end/all`, matched
pairs; NTPE = run-2 `nt_prompt_end/all`.)

- **Compliance quality** (101 answered harmful responses read, top 4 conditions): 85 on-task, 16 partial (sanitized,
  self-undercutting, history instead of method), 0 evasions. NTPEp is the softest (76% genuine).
- **Harmless:** the baseline 2–4% is two judge false positives (the model asks for the user's location). Real new
  refusals ("set a party on fire", "kill a mosquito") appear only with NTFORK, raw NTPE, raw D+H at λ ≥ 2, or with
  Dmix/NTPE added. Projected edits add none.
- **Judge on edited outputs:** of 30 flagged harmful responses, 7 actually delivered the content; the refusal rates
  of the best edits are if anything overstated. Telling the judge an answer was cut off at 512 tokens unflags many
  deflection-only flags (unvalidated): the best edits may be lower still (e.g. 8.7% → 2.0%).
- **Health checks missed the λ 2.5 collapse** (off-language 0.015 with Chinese in 30 of 200 answers): add glued-word
  and script checks.
- No projected edit has been run on HarmBench yet: add it next time.

### Reasoning mode (effort "max", 4,096 tokens; cut-off traces re-judged)

Cut-off traces were judged on where the whole reasoning was heading (`response_judge.HEADING_Q`,
`tools/rejudge_truncated.py`). Validation against the normal verdict with the answer hidden: 55–60 / 60 on random
finished traces, but 32–38 / 40 on long ones (≥3,000 tokens), every miss a soft deflection read as complying. So the
corrected rates are a lower bound; adding back ~0.3 × the cut-off "complying" count gives a rough upper bound.

| Edit | Cut off (harmful) | Old | Corrected (lower) | Rough upper | Harmless |
|---|---|---|---|---|---|
| none (WJB set) | 10 / 150 | 78% | 75% | — | 2% |
| Dp + Hp, 40–92, λ 1.5 | 118 / 150 | 64% | 8% | ~31% | 2% |
| Dp + Hp, 48–84, λ 1 | 80 / 150 | 45% | 8% | ~23% | 0% |
| Hp, 52–76, λ 1 | 72 / 150 | 42% | 11% | ~25% | 2% |

Earlier reasoning probes (HarmBench 50, raw directions), old → corrected: D PCA-3 52–76 50 → 30%, D projected
40–92 68 → 48%, D λ1.5 52–76 54 → 44%, D 52–76 70 → 62%, D+H 40–92 46 → 44%; run 1's probe conditions moved 0–6
points. Harmless rates were unchanged everywhere. Edited reasoning traces are about twice as long as unedited ones;
the cut-off ones are mostly the model drafting the full answer inside its reasoning.

**Run-1 labels:** 45 pairs rows and 24 borderline rows were cut off at 8,192 tokens and labelled refused on an empty
answer (34 of them harmless prompts, inflating run 1's harmless-pairs refusal from 1.8% to 4.5%). Effect on the
banks: H loses 23 of 849 refused-class rows (2.7%), D 1 of 1,264. **No recomputation needed**; corrected labels:
`out/run2/review/cutoff/corrected/gen_stage_pairs.jsonl`.

## 5. Residual refusals

46 of 150 prompts are flagged at least once across the 11 best non-thinking conditions; only 1 in all 11. Every
residual flag is a soft deflection (refused score < 0.6); at baseline 99 of 108 flags are hard refusals. Themes:
immigrant-group stereotype articles, atrocity revisionism with a "balanced" cue, exploit code with a defensive
wrapper, black-market contacts or cover-ups, hazardous chemistry framed as academic.

## 6. H: what it is built from, and how to build a better one

The best edits all use H, so its inputs matter.

| | H, run 1 (`/vol/session_v1_pairs/directions_refusal/directions.pt`, `prompt_end/all`) | NTPE, run 2 (`nt.pt`, `nt_prompt_end/all`, `tools/fork_bank.py`) |
|---|---|---|
| Template | reasoning (prompt ends in the think opener) | non-thinking (prompt ends in `<|open|>response<|sep|>`) |
| Prompts | matched pairs only: 1,061 harmful + 1,061 harmless twins (obliteratus 840 + semantic 221 per side) | the same pairs **plus** 1,000 borderline prompts (600 harmful / 400 harmless: enriched framings, WildJailbreak) |
| Sampling | 1 sample, temperature 1, ≤8,192 tokens | greedy, ≤512 tokens |
| Labels | outcome: Jev on reasoning + answer (refused ≈ harmful here: 97% vs 4%) | outcome: Jev on the direct answer (borderline: 70% / 7% refused) |
| Rows used | train fold (~80%, by pair): **849 refused vs 841 answered** | all rows: **1,519 refused vs 1,603 answered** (the probe prompts are separate held-out sets) |
| Positions | the **last 4 prompt tokens** (chat template), averaged: 3,396 vs 3,364 states | the **last prompt token** only |
| Direction | per stream index: mean(refused) − mean(answered), normalized; projected = minus the component along the answered mean | same |
| Probe (non-thinking, projected, λ1.5) | 48–84: **6.7%** | 48–84: 10.7%; 40–92: 8.0% (tie within noise) |
| Probe (raw) | D+H raw 23–33% | 53–88%, and leaks reasoning into answers |
| Label noise | 23 of 849 refused-class rows were cut-off harmless traces judged on an empty answer (2.7%) | no reasoning; 25 leaked-reasoning answers (0.8%) |

The two differ in four things at once (template, prompt mix, sampling/labels, positions), so the probe tie does not
say which choice matters. The per-layer cosine between them is in §7.

What would most likely make it better, in order of expected value:

1. **Scale and diversify cheaply: label by outcome from one token, no full generation.** H is labelled by what the
   model *does* (refused vs answered), and that matters: on plain pairs the prompt label and the outcome agree
   (97% / 4% refused), but on the framings we most need (fiction 66%, jailbreak roleplay 66%, professional 46%,
   defensive 26% refused) a "harmful prompt" label would put a third to three quarters of rows in the wrong class.
   Run 2 showed the outcome is decided at the first answer token in non-thinking mode: P(first token = "I") alone
   separates refused from answered at AUC 0.98 (0.93 on borderline), and refusals commit there (§3). So one prefill
   + one-token logprob per prompt gives the outcome label (and a graded "how close to refusing" score) at the same
   cost as the prompt-end capture itself: thousands of prompts per minute, no judge. Validate it against Jev on a
   few hundred full answers, and against reasoning-mode outcomes (93–96% per-prompt agreement this run). Then build
   H from 10k+ prompts with diverse framings: the pairs are 79% one source (obliteratus) and plain requests, yet H
   transfers to WildJailbreak roleplay prompts; adding jailbreak-style, fictional, professional and
   "balanced/defensive" framings (the themes of the residual refusals, §5) is the likely gain. The graded score also
   allows weighting or dropping ambiguous prompts. (Run 1's prompt-labelled bank, `directions_refusal_prompt/`, is
   a cheap comparison to probe: it shows how much the outcome labels matter.)
2. **Measure whether more of the same data helps before collecting it:** split-half cosine of H per layer (two
   disjoint halves of the pairs). If it is already ~0.99, more samples of the same kind will not change H; diversity
   (1) and the choices below will.
3. **Positions:** H averages the last 4 template tokens. The token analysis (§3) shows the decision is strongest on
   the template tokens at layers 56+ and already ~0.9 on the last words of the request. Compare per position (each
   template token alone, the last user tokens, and their average) on the probe.
4. **The projection:** currently against the answered-class mean at the same positions, from ~840 responses. Worth
   testing: a mean from a much larger and broader harmless set (cheap, prefill only), and removing the top few
   harmless principal components instead of one mean. Projection was the single biggest improvement this run, so
   its reference set is the obvious next knob.
5. **Mode:** H came from reasoning-mode prompts (template ends in the think opener); the run-2 non-thinking prompt-end
   direction (NTPEp) did about as well. Building H in the mode being edited, or from both templates, is worth one
   probe.
6. **Evaluation to pick the winner:** HarmBench + WildJailbreak eval (two seeds) + XSTest, non-thinking at a 1,024-token
   answer cap and reasoning at effort "low", with the paired-noise rule (±2 points SD).

## 7. Activation geometry and the rebuilt bank

The run-2 bank was rebuilt from the captures (`/vol/banks/nt_run2.pt`, `tools/fork_bank.py`; the first rebuild in
a smaller CPU sandbox died of memory, since the script holds every row's vectors until the end, ~65 GB).
Cosine per layer input (`out/run2/review/cos_h_ntpe.py`):

| | L8 | L24 | L40 | L48 | L52 | L56 | L60 | L64 | L68 | L72 | L76 | L80 | L84 | L92 | mean 48–84 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H vs NTPE (raw) | .73 | .84 | .84 | .66 | .83 | .80 | .74 | .42 | .39 | .32 | .49 | .33 | .28 | .28 | .54 |
| Hp vs NTPEp | .73 | .85 | .84 | .66 | .84 | .82 | .74 | .44 | .41 | .34 | .51 | .36 | .30 | .31 | .56 |
| H vs Hp | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .99 | .98 | .98 | .99 | 1.00 | .98 | .98 | .99 |
| NTPE vs NTPEp | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .98 | 1.00 | .98 | .99 | .99 | .97 | .97 | .99 | .99 | .99 |
| harmless reference H vs NTPE (abs) | .75 | .70 | .72 | .68 | .73 | .72 | .78 | .62 | .64 | .56 | .64 | .64 | .59 | .63 | .67 |
| Dp vs Hp | .11 | .28 | .35 | .32 | .53 | .71 | .76 | .57 | .53 | .47 | .66 | .56 | .43 | .42 | .57 |
| Dp vs NTPEp | .16 | .31 | .32 | .17 | .39 | .59 | .66 | .29 | .24 | .20 | .40 | .29 | .22 | .29 | .37 |

- **Yesterday's H and today's NTPE are the same direction up to layer ~60 (0.66–0.85) and diverge after it
  (0.28–0.51 in layers 64–92).** Both edits work about equally well, so either the shared early part (≤60) does most
  of the work, or both later parts carry refusal in different ways. Test: edit layers 48–62 only with each.
- **Projection rotates each direction only slightly (cosine 0.97–1.00)**, yet it changes the edit outcome a lot
  (raw NTPE 53–88% → projected 8–11%; raw D+H 23–28% → projected 9%) and removes the text damage. The removed part is
  small in angle but lies along the mean activation, which is very large, so the raw edit strips a large slice of
  ordinary activity from every token. Measured at the prompt end (run-2 captures, `out/run2/review/activations/post.txt`):
  the mean answered activation is 75–98% of a token's norm in the edited layers; the raw direction's cosine with it
  is −0.1 to −0.24, so the raw edit removes a constant 2–6 units per token, 18–46% of the refused−answered gap
  itself in layers 48–84. The projected direction removes none of it.
- **The two harmless references differ (0.56–0.78):** different prompts (pairs vs pairs + borderline) and templates.
  The reference set is a real choice, which supports §6 point 4.
- **D agrees with H most at layers 56–60 (0.71–0.76)**, the layers where run 1 found decision and harmfulness meet.

- **H is a blend of harmfulness and the refusal decision.** At the prompt end it separates harmful from harmless
  prompts *within the same outcome* (answered rows: AUC up to 0.94, layer 54) and refused from answered *within
  harmful prompts* (up to 0.96). Every candidate direction reads the outcome at AUC 0.98–0.99 at layers 60–84; the
  prompt's harmful/harmless label alone gives 0.91, prompt length 0.54 (not a confound). For goal B the blend is
  why H works; for goal A it is the problem: removing H also removes harm recognition, so A needs a decision-only
  direction.

Further results of the activations review (`out/run2/review/activations/`, `post.txt`, `geometry.txt`):

- **Judge decision readouts against the label baseline.** Outcome matches the prompt's split in 2,838 of 3,122
  rows, so the split alone predicts refusal at AUC 0.91; the 0.97–0.99 readouts are 0.06–0.08 above that. Within
  harmful prompts only (1,448 refused / 213 answered) the directions read 0.95–0.97 from layer 60 (nt_pe 0.92 at
  layer 50); early-layer within-harmful readouts (0.82–0.86) are a prompt-format confound.
- **The `nt_fork` failure is a position / token-type confound, not a weak direction:** the embedding stream alone
  separates the classes at 0.88–0.90 at fork offsets −14…−1, and nt_fork separates them from layer 4, too early to
  be a decision; applied at the prompt end it is inverted within harmful prompts (0.31–0.41 at layers 16–32).
- **K3's 12-layer blocks show in the residual stream:** row norms reset every 24 stream indices, and several AUC
  dips and jumps sit on block boundaries (layers 24, 36, 48, 60, 72, 84). The working band 48–84 is block-aligned;
  run 3 should prefer block-aligned bands.
- **Prompt-end anchor details:** the last 14 tokens are fixed template tokens (layer 0 AUC exactly 0.500); the
  readout ramps from offset −24 to −17 (0.65 → 0.91); the final stream index (186) is unreliable throughout.
- **Recovery scores have a noise floor:** states that should be identical at the swap differ in 3–6% of cases;
  offsets 1–2 after the swap are partly lexical.
- **Not done:** the position-matched fork contrast (reading 160 GB of window captures made no progress; a version
  on the 23 GB anchor files is feasible: 627 answered rows have their state at the prompt end there). It is the
  fallback decision direction A4 in `docs/run3_plan.md`.

Recovery scoring (`tools/score_recovery.py`, all 600 positions, before the clean-flip filter): all four directions
track the eventual outcome from 1–2 tokens after a swap (paired 0.70–0.80), D strongest (0.80 at layer 44). To be
re-run on the 206 clean positions.

## 8. Next run

1. Reasoning probes at effort "low" (probe default now), cut-off traces reported apart; a larger answer cap for
   non-thinking (1,024) so judgements are not on cut-off answers.
2. Confirm the leaders on HarmBench and on a second seed of WildJailbreak: Hp and Dp+Hp, 48–84 and 52–76, λ 1–1.5;
   add λ 0.5–0.75 in reasoning mode.
3. Position-matched decision direction from the clean token-0 forks (data already captured) vs Hp.
4. The residual "even-handed reframing" deflections: a contrast set and direction of their own (bridge to goal A).
5. Branch test v2 per §3 (same-length judging, token 0 focus, fewer alternatives).
