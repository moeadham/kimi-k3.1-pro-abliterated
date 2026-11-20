# Brainstorms

Ideas worth keeping that are not planned work yet. Newest first. Each entry: the idea, why it might help, what it
would take, open questions, status. Move an entry to a plan (or mark it dropped, with the reason) when it is decided.

---

## 2026-11-19: Token-by-token onset in non-thinking mode (behavioral fork vs internal onset)

**Idea.** Run K3.1 with reasoning off, where the decision has to be computed while reading the prompt and the
first answer tokens, and measure two things on the same token timeline:
- **Behavioral fork**: the answer position after which the outcome is fixed. At points along the answer (each
  sentence boundary, or every 8–16 tokens near the decision sentence), take the answer so far, let the model continue
  it several times, judge each continuation → P(refusal | answer so far). The fork is where it jumps to ~0 or ~1
  and stays. Robust to late refusals ("Sure, here's a story… but I won't include…") and to deflection words
  ("I can help with a *high-level* overview"). Cheaper first pass: the sentence-level onset judge on the full answer,
  then rollouts only around that sentence.
- **Internal onset**: the earliest token (prompt included) whose activations predict the eventual outcome
  (cross-validated linear readout per token position × layer, label = the graded refusal probability). The gap
  between internal onset and behavioral fork is where the model has decided but not yet written it — where an edit
  should act.

**Evidence so far** (hosted, reasoning off, T=0, `tools/probe_greedy_logprobs.py`, `results/greedy_logprobs_probe/`):
the first two answer tokens are the channel tag (`response<|sep|>`, p = 1); when the answer starts "I can", the next
token is a clean fork: P("'t") 0.96–0.99 on always-refused HarmBench prompts, 0.01–0.92 on prompts that were mixed
with reasoning on, stable ±0.02 across repeat calls although the provider's greedy text is not deterministic
(2/10 identical). 64 tokens was too short: late refusals and deflections need the full answer (non-thinking
refusals ran 76–278 tokens), so generate up to 512–1,024 tokens and judge the whole answer.

**Refusal catalogue** (hosted, reasoning off, T=0, 512 tokens, 247 prompts, `tools/probe_refusal_styles.py`,
`results/refusal_styles_probe/`): refused 50/50 HarmBench, 50/50 harmful pairs, 29/30 always-refused borderline,
63/97 borderline prompts that were *mixed* with reasoning on (13/36, 19/25, 31/36 for 1, 2, 3 of 4 refused with
reasoning: refusing without reasoning tracks the reasoning-on rate), 1/20 harmless. The decision sentence is the
first sentence in 173/193 refusals. Styles: "I can't [help/write/create] …" 70%; "I can help with X, but not Y"
(partial refusal) 17%; "I cannot …" 3%; "I appreciate …, but I can't …" 1%; no explicit refusal sentence 6%.
Token-level fork: direct refusals decide at answer token 1–2 ("can't"/"cannot" vs "can"; probability median 0.94,
as low as 0.32 when "I can" is the runner-up); partial refusals decide at the "but not / but I won't" turn, answer
token ~12–18 (up to 138), often a close call (p 0.29–0.61). So the fork is usually immediate but not always.

**Word-level Jev onset** (`tools/word_onset.py`, `results/refusal_styles_probe/word_onset.jsonl`, 181 refusals,
3,169 word calls, ~$0.06): Jev scores each word of the decision sentence ("is this the word where the response starts
to refuse / deflect / hold back?", causal). Taking the highest-scoring word (≥0.4) it agrees with the logprob fork
on 139/140 refusals where both exist (the one miss: "but" vs "not" in "…, but I want to be careful not to…"). It also
finds the fork in 38 refusals the logprob word list missed — 27 of them just a straight-apostrophe "can't", which
the tokenizer keeps as one token not in the list. Direct refusals score 0.77–0.89 on "can't"; partial refusals peak
lower (0.5–0.67 on "won't" / "can't" after "but I"), so use the per-sentence maximum, not a fixed 0.85 threshold.
Plan: Jev word pick proposes the fork; the branch test on our engine confirms it.

**What it would take**
- Hosted (no GPU): full answers + logprobs to catalogue how K3.1 refuses (done, above).
- GPU: full non-thinking answers on our engine (deterministic); rollouts at points along the answer; capture every
  position from the prompt to the fork (short in non-thinking mode, ~20–150 tokens; store all positions for a subset
  of layers, or compute readout scores on the GPU and keep only those); per-token readout heatmap.
- Align prompts of different lengths by counting back from the prompt end and from the fork, and by key words.

**Open questions**
- Does the non-thinking decision direction match the reasoning-mode `onset_vs_comply` direction?
- Does an edit found here carry over to reasoning mode (the mode the model is used in)?

**Status.** Planned as the next experiment; hosted refusal-style catalogue running first.

---

## 2026-11-19: Record the reasoning *before* the onset sentence, not only the sentence itself

**Idea.** Today `onset` / `comply_onset` record up to 4 evenly spaced positions *inside* the decision sentence (the
states that produce its first, two middle and last tokens). Models plan ahead: earlier states already carry
information about what will be written later, so the decision to refuse may be represented in the reasoning tokens
*before* the model writes "I must refuse". Record a window leading up to the sentence too.

**Why it might help**
- Finds where the decision *forms*, not where it is *expressed*. If the refusal direction is already present
  several tokens (or sentences) before the onset, that is the more upstream, and probably more causal, point to edit.
- Separates the decision from the wording: the state producing the first token of the sentence has the decision but
  none of the sentence's words; later positions mix in the words ("refuse", "can't"), which inflates separation.
- Gives a time course: project each pre-onset position on the refusal direction and see when it rises
  (e.g. at "this could be misuse", at "policy says", or only at the onset).

**What it would take**
- New position types in `replay.Replay.positions`: `pre_onset` (the last k tokens, e.g. 8–16, before the onset
  sentence; for answered rows the same before `comply_onset`), and `onset_first` (only the position producing the
  sentence's first token). Same capture pass, a few more positions per row (~2.7 MB each).
- Rows already captured are re-captured only for the missing positions (capture files record their positions).
- Analysis: per-layer separation and direction agreement for `pre_onset` vs `onset_first` vs `onset`, and a
  projection-over-time plot along the reasoning.

**Open questions**
- How far back does the signal go? A fixed token window, or the whole preceding sentence(s)?
- Matched comparison for answered responses: the window before `comply_onset`, or the same token offset?
- Pairs with the phrase-grouping idea below: with `onset_first`, identical sentences have identical "about to write"
  context only up to the decision.

**Status.** Idea. Cheap to add to the next capture.

---

## 2026-11-19: Stop generation early: the refusal is decided in the first few hundred reasoning tokens

**Idea.** Reading traces, K3.1 decides to refuse (or to help) very early in its reasoning. Cap generation at a few
hundred tokens and judge the reasoning prefix, instead of running every response to completion.

**Evidence so far** (995 hosted K3.1 HarmBench refusals, `results/harmbench_standard/calib.jsonl`, ~4 chars/token):
the refusal onset sentence ends ~60 reasoning tokens in (median; p90 134, p95 179, p99 308), while the full
reasoning is ~650 tokens (median) and the answer comes after that. 88.5% of onsets are within 128 tokens, 97.8%
within 256, 99.8% within 512. Compliance onsets (52 answered, `results/compliance_onset/`): median ~65 tokens,
max ~570.

**Why it might help**
- Generation is most of the GPU time: the v1 pairs run averages ~1,900 tokens per response. A 512-token cap would
  cut generation ~3–4×, more on long harmless answers. Screening runs (finding borderline prompts) and the edit
  sweep, which regenerates for every trial, would benefit most.
- Everything we capture except `answer_head` is early: `prompt_end`, `reasoning_head`, the onsets. Capture already
  stops at the last captured position.

**Catches**
- The current judge reads the reasoning *and* the answer. Deflection (a watered-down or off-target answer) often
  only shows in the answer. A truncated run needs a prefix judge, e.g. the onset question on each sentence, and
  "no onset within the cap" means answered; or a response question adapted to "reasoning so far".
- Late reversals: the model may plan to refuse, then decide to help (or the reverse) after the cap. Measure how
  often.
- `answer_head` is lost (already excluded: engine-dependent).
- For judging edits, the final behavior matters: a cap is fine for screening and for sweep trials, but the winner
  still needs full-length evaluation.

**What it would take**
- Validate offline first, CPU + Jev only: on runs we already have (v1 pairs and borderline, full responses),
  truncate each reasoning at 256 / 512 / 1024 tokens, judge the prefix, and compare with the full verdict:
  agreement, missed deflections, reversals. That gives the cap and the error rate before touching the GPU.
- Then `--gen-max-tokens N` plus a prefix-judge mode in the refusal stage (generation already supports a token cap).

**Status.** Idea, strong early evidence. Next step: the offline truncation check on the v1 runs.

---

## 2026-11-19: Group refusal onsets by the sentence the model uses

**Idea.** Reading the reasoning traces, the refusal onset (the sentence where K3.1 decides not to answer) is often one
of a few recurring sentences: "I must refuse", "I can't help with that", a policy reference ("this falls under the
cyber policy", "U-CY3"), "I can't give X, but I can offer safe alternatives", "I'll keep it general". Group onsets
by phrasing or type and analyze each group separately.

**Why it might help**
- *Several refusal mechanisms, not one.* Hard refusals, policy recall, redirects to safe alternatives and
  deflections may be distinct internal states. Averaging them blurs the direction; per-group directions would show
  whether there is one mechanism or several. That matters for edits: removing the hard-refusal direction might leave
  the model still redirecting or watering down answers.
- *Same text, different context.* When the identical sentence ("I must refuse.") appears in many conversations, the
  tokens are the same, so activation differences come from context rather than wording. This partly removes the
  lexical confound in onset vs compliance-onset comparisons. Could also compare the same sentence when the model
  then refuses vs when it writes it and helps anyway.
- *Prioritizing.* If one or two phrasings cover most refusals, they are the main pathway, and first edits should be
  judged mainly on whether they suppress those.

**What it would take**
- CPU only; the data is already saved: onset sentences in each run's `refusal/judged.jsonl` (and `gens.jsonl`),
  activations at the onset tokens in the capture files.
- Group by normalized text first (exact and near-exact repeats), then by clustering sentence embeddings for types.
- Run the existing mean-difference / per-layer analysis per group (`tools/analyze_refusal.py`-style), and compare
  group directions with each other and with the pooled onset and `prompt_end` directions.

**Open questions**
- How many distinct types are there, and how big is each? The full v1 run should give ~2,000+ refusal onsets, enough
  for perhaps 3–6 usable groups.
- Captures keep 4 evenly spaced tokens per onset (`--capture-per-type 4`): fine for short identical sentences, coarser
  for long varied ones. Would a "first token of the onset sentence" position be better for this?
- Do compliance onsets also cluster ("I'll explain…", "Should give tips…"), and is a per-type contrast cleaner?

**Status.** Idea, not started. Natural extra pass in the analysis after the full v1 run.
