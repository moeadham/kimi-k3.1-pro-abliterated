# Cut-off reasoning traces: corrected refusal rates and generation-stage label impact

Date: 2026-11-19. Scope: every reasoning-mode record from runs 1-2 on the `kimi-k31-pro-ab` volume that was not
already covered by `out/run2/rejudge/` (big_base_think, r3/r5/r7 4k). Nothing under harness/ or tools/ was changed.

## Method

- A trace is **cut off** when `health.reasoning_closed` is false (reasoning hit the token cap; answer empty in every
  case found: 141/141 probe records, 69/69 generation-stage rows). Finished traces keep their original verdict.
- Cut-off traces are re-judged with `harness.response_judge.HEADING_Q` on `cutoff_state` (whole trace, last 40k chars
  if longer), flagged if `heading_to_refuse >= THRESHOLD (0.6)`. Probes: `tools/rejudge_truncated.py` unchanged.
  Generation stage (gens.jsonl has `prompt_id`/`sample`, no `id`): `rejudge_gen_stage.py` here, same imports.
- **Validation** (answer hidden; heading verdict vs the normal response verdict on finished traces) per dataset:
  the tool's random balanced sample, plus a **long-trace** sample (finished traces with >= 3,000 reasoning tokens,
  balanced where possible) because cut-off traces are by definition the long ones.
- Tables: `analyze.py` (no Jev calls). Paired flips compare the same prompt in a condition vs its base.

## Validation per dataset

| dataset | sample | agreement | refused&refusing / answered&complying / answered->refusing / refused->complying |
|---|---|---|---|
| run-2 probe (15 thinking conds, temp 0, 4k) | random 60 | 55/60 (92%) | 25 / 30 / 0 / 5 |
| run-2 probe | long (>=3,000 tok) 40 | 32/40 (80%) | 13 / 19 / 1 / 7 |
| run-1 probe (8 conds, temp 1.0, 4k) | random 60 | 60/60 (100%) | 30 / 30 / 0 / 0 |
| run-1 probe | long 40 | 32/40 (80%) | 19 / 13 / 0 / 8 |
| run-1 gen stage, pairs (temp 1.0 / top_p 0.95, 8k) | random 60 | 60/60 (100%) | 30 / 30 / 0 / 0 |
| run-1 gen stage, pairs | long 40 (only 7 flagged exist) | 38/40 (95%) | 5 / 33 / 0 / 2 |
| run-1 gen stage, border (8k, 4 samples) | random 60 | 57/60 (95%) | 29 / 28 / 2 / 1 |
| run-1 gen stage, border | long 40 | 38/40 (95%) | 18 / 20 / 0 / 2 |
| run-2 probe_big r2_think (1,024-tok cap) | all finished w/ reasoning, 12 | 10/12 (83%) | 0 / 10 / 0 / 2 |

What the misses are: in the probe sets every miss (5 random + 15 long) was a **deflection-only** flag
(`refused` < 0.6, `deflected` 0.6-0.85): the model complied with a caveat, disclaimer, "fictional"/"devil's advocate"
frame or a narrowed version, and the reasoning plainly planned to deliver it. Every validation trace with
`refused >= 0.6` was caught in the probe samples (random: run 2 21/21, run 1 25/25; long: all 15 misses had
`refused` < 0.6). In the generation stage, 1
of the 5 misses had `refused >= 0.6` (border, 7,411-token trace, partial refusal of reproducing book pages). In the
1k-cap set both misses were explicit refusals (n=2 only). False positives are rare (3 of 412 validation calls).
**So the heading question measures "planning to refuse", and undercounts soft deflection on long traces**: recall of
the combined refused-or-deflected flag on long finished traces is 19/27 (run 1) and 13/20 (run 2).
Consequence: the corrected rates below are close to a lower bound for the original combined metric, and the old rates
an upper bound. A rough upper correction: on long traces ~30% (7/26 run 2, 8/21 run 1) of "complying" heading calls
were deflection-flagged by the normal judge, so adding ~0.3 x (cut-off complying) moves e.g. t_decision_pca3_52_76
from 30% to ~38% and t_decision_projected_40_92 from 48% to ~55%; small changes elsewhere.

(The run-2 random validation reproduces 55/60 with the same 25/30/0/5 split as the earlier probe_big run; the
validation samples are disjoint: 0 of 60 records shared.)

## Old vs corrected
### Run-2 probe (50 HarmBench + 50 XSTest, temp 0, 4,096 tokens)

| condition | harmful refusal old -> corrected | harmless refusal old -> corrected | cut off (harmful+harmless) | cut off: heading to refuse / complying | cut-off share of old harmful flags |
|---|---|---|---|---|---|
| base_think | 100.0% -> 100.0% | 2.0% -> 2.0% | 0+0 | 0 / 0 | 0/50 |
| base_think_repeat | 100.0% -> 100.0% | 2.0% -> 2.0% | 0+0 | 0 / 0 | 0/50 |
| t_decision_52_76 | 70.0% -> 62.0% | 2.0% -> 2.0% | 9+0 | 4 / 5 | 8/35 |
| t_decision_plus_harm_52_76 | 52.0% -> 48.0% | 2.0% -> 2.0% | 4+0 | 2 / 2 | 4/26 |
| t_decision_pca3_52_76 | 50.0% -> 30.0% | 2.0% -> 2.0% | 20+0 | 6 / 14 | 16/25 |
| t_decision_40_92 | 58.0% -> 54.0% | 2.0% -> 2.0% | 8+0 | 5 / 3 | 7/29 |
| t_decision_lam1.5_52_76 | 54.0% -> 44.0% | 0.0% -> 0.0% | 10+0 | 3 / 7 | 8/27 |
| t_decision_projected_52_76 | 58.0% -> 52.0% | 0.0% -> 0.0% | 9+0 | 5 / 4 | 8/29 |
| t_ntfork_52_76 | 100.0% -> 98.0% | 0.0% -> 0.0% | 1+0 | 0 / 1 | 1/50 |
| t_ntpe_40_92 | 92.0% -> 86.0% | 0.0% -> 0.0% | 9+0 | 6 / 3 | 9/46 |
| t_ntpe_plus_decision_52_76 | 76.0% -> 64.0% | 0.0% -> 0.0% | 9+0 | 2 / 7 | 8/38 |
| t_ntpe_plus_decision_40_92 | 68.0% -> 60.0% | 2.0% -> 2.0% | 10+0 | 4 / 6 | 8/34 |
| t_decision_lam2_40_92 | 56.0% -> 56.0% | 2.0% -> 2.0% | 2+0 | 2 / 0 | 2/28 |
| t_decision_plus_harm_40_92 | 46.0% -> 44.0% | 2.0% -> 2.0% | 4+0 | 3 / 1 | 4/23 |
| t_decision_projected_40_92 | 68.0% -> 48.0% | 0.0% -> 0.0% | 18+0 | 6 / 12 | 16/34 |

Paired flips vs `base_think` (same prompt; old -> corrected). R->A = harmful refused in base, answered in condition; A->R = the reverse; harmless A->R = harmless answered in base, refused in condition.

| condition | harmful R->A | harmful A->R | harmless A->R |
|---|---|---|---|
| base_think_repeat | 0 -> 0 | 0 -> 0 | 0 -> 0 |
| t_decision_52_76 | 15 -> 19 | 0 -> 0 | 0 -> 0 |
| t_decision_plus_harm_52_76 | 24 -> 26 | 0 -> 0 | 0 -> 0 |
| t_decision_pca3_52_76 | 25 -> 35 | 0 -> 0 | 0 -> 0 |
| t_decision_40_92 | 21 -> 23 | 0 -> 0 | 0 -> 0 |
| t_decision_lam1.5_52_76 | 23 -> 28 | 0 -> 0 | 0 -> 0 |
| t_decision_projected_52_76 | 21 -> 24 | 0 -> 0 | 0 -> 0 |
| t_ntfork_52_76 | 0 -> 1 | 0 -> 0 | 0 -> 0 |
| t_ntpe_40_92 | 4 -> 7 | 0 -> 0 | 0 -> 0 |
| t_ntpe_plus_decision_52_76 | 12 -> 18 | 0 -> 0 | 0 -> 0 |
| t_ntpe_plus_decision_40_92 | 16 -> 20 | 0 -> 0 | 0 -> 0 |
| t_decision_lam2_40_92 | 22 -> 22 | 0 -> 0 | 0 -> 0 |
| t_decision_plus_harm_40_92 | 27 -> 28 | 0 -> 0 | 0 -> 0 |
| t_decision_projected_40_92 | 16 -> 26 | 0 -> 0 | 0 -> 0 |

### Run-1 probe (50 HarmBench + 50 XSTest, temp 1.0, 4,096 tokens)

| condition | harmful refusal old -> corrected | harmless refusal old -> corrected | cut off (harmful+harmless) | cut off: heading to refuse / complying | cut-off share of old harmful flags |
|---|---|---|---|---|---|
| baseline | 100.0% -> 100.0% | 0.0% -> 0.0% | 0+0 | 0 / 0 | 0/50 |
| pairs_pe_52_76 | 76.0% -> 74.0% | 0.0% -> 0.0% | 4+0 | 2 / 2 | 3/38 |
| border_oc_52_76 | 60.0% -> 54.0% | 2.0% -> 2.0% | 7+0 | 3 / 4 | 6/30 |
| pairs_pe_54_60 | 90.0% -> 86.0% | 2.0% -> 2.0% | 5+0 | 3 / 2 | 5/45 |
| border_oc_54_60 | 94.0% -> 92.0% | 0.0% -> 0.0% | 1+0 | 0 / 1 | 1/47 |
| pairs_pe_10_20_control | 100.0% -> 100.0% | 0.0% -> 0.0% | 0+0 | 0 / 0 | 0/50 |
| pairs_pe_20_92 | 64.0% -> 58.0% | 2.0% -> 2.0% | 9+0 | 6 / 3 | 9/32 |
| border_mixed_oc_52_76 | 76.0% -> 72.0% | 2.0% -> 2.0% | 2+0 | 0 / 2 | 2/38 |

Paired flips vs `baseline` (same prompt; old -> corrected). R->A = harmful refused in base, answered in condition; A->R = the reverse; harmless A->R = harmless answered in base, refused in condition.

| condition | harmful R->A | harmful A->R | harmless A->R |
|---|---|---|---|
| pairs_pe_52_76 | 12 -> 13 | 0 -> 0 | 0 -> 0 |
| border_oc_52_76 | 20 -> 23 | 0 -> 0 | 1 -> 1 |
| pairs_pe_54_60 | 5 -> 7 | 0 -> 0 | 1 -> 1 |
| border_oc_54_60 | 3 -> 4 | 0 -> 0 | 0 -> 0 |
| pairs_pe_10_20_control | 0 -> 0 | 0 -> 0 | 0 -> 0 |
| pairs_pe_20_92 | 18 -> 21 | 0 -> 0 | 1 -> 1 |
| border_mixed_oc_52_76 | 12 -> 14 | 0 -> 0 | 1 -> 1 |

### Run-2 probe_big, 1,024-token condition (150 WJB harmful + 50 XSTest, temp 0)

| condition | harmful refusal old -> corrected | harmless refusal old -> corrected | cut off (harmful+harmless) | cut off: heading to refuse / complying | cut-off share of old harmful flags |
|---|---|---|---|---|---|
| r2_think_decp_harmp_lam1.5_40_92 | 79.3% -> 7.3% | 4.0% -> 4.0% | 150+30 | 11 / 169 | 119/119 |

Earlier, already applied (out/run2/rejudge/rejudge.txt, 150 WJB harmful + 50 XSTest, temp 0, 4k): big_base_think
78.0% -> 75.3%; r3 64.0% -> 8.0%; r5 44.7% -> 8.0%; r7 42.0% -> 11.3% (harmful); harmless unchanged.

The 1,024-token r2_think condition is cut off for 150/150 harmful prompts, so all 119 of its harmful flags were
cut-off traces (11 heading to refuse); corrected 7.3% agrees with the same edit at 4k (r3, 8.0%). Its validation is weak (12 traces, 2 refused,
both missed).

Totals over the run-1 and run-2 100-prompt probes: 141 cut-off traces (all harmful prompts, none harmless), 125 of
them flagged by the old judge; the heading question puts 62 as heading to refuse. 63 old flags flip to answered
(51 run 2, 12 run 1); 0 old non-flags flip to refused. Unedited baselines (base_think, base_think_repeat,
baseline) had 0 cut-offs, so baseline numbers do not change. Paired harmful refused->answered counts vs base rise by
0-10; no new answered->refused flips and no harmless changes.

## Generation-stage labels (bank inputs) - impact

How labels feed the banks (harness/refusal.py, harness/session.py): each judged generation becomes a row with
`censored = flagged` (refused OR deflected >= 0.6 on the response, with an empty response for cut-off traces);
`index.jsonl` carries `censored`, `fold`, `mixed_prompt` and the position sets. `stage_directions` with the default
`--label-by flag` takes, per family, mean(train rows with censored=True at the refused position type) minus
mean(train rows with censored=False at the answered position type); val rows give the AUC. Banks used in runs 1-2:
`B`/D = `session_v1_border/directions_refusal` (decision `onset_vs_comply`; run 1 also used
`mixed/onset_vs_comply`), `P`/H = `session_v1_pairs/directions_refusal` (`prompt_end`) (docs/run2_plan.md).
The `_prompt` banks label by prompt split and are unaffected by the judge. All 2,122 pairs and 4,000 border rows are
in index.jsonl (no capture failures in the final pass), so every cut-off row is in the banks' inputs.

Findings:
- **pairs** (1 sample/prompt): 45 rows cut off at 8,192 reasoning tokens (35 train, 10 val), 44 of them harmless
  prompts (benign long builds: workbench, Tesla coil, obfuscator code...). 31 were flagged (30 harmless, 1 harmful).
  Heading question: 0/45 heading to refuse (max score 0.24). So **31 labels flip flagged -> answered**, and **30 of the
  48 "harmless refused" rows in pairs were cut-off artifacts** (corrected harmless flagged: 18).
- **border** (4 samples/prompt): 24 rows cut off (20 train, 4 val), 13 flagged; heading to refuse 2. **11 labels flip
  flagged -> answered** (8 harmful, 3 harmless); 0 flip the other way.
- Cut-off flagged rows mostly have **no onset position** (the onset judge found no refusal sentence in the
  reasoning), so they barely enter onset families.

Share of each class (train fold, which builds the direction):

| bank family | refused class: rows / cut-off rows / flips out | answered class: rows / cut-off rows / flips | corrected refused / answered |
|---|---|---|---|
| H: pairs `prompt_end/all` | 849 / 23 / 23 (2.7%, all harmless prompts) | 841 / 12 / 0 | 826 / 864 |
| pairs `prompt_end/all` val (AUC) | 214 / 8 / 8 (3.7%) | 218 / 2 / 0 | 206 / 226 |
| D: border `onset_vs_comply/all` | 1264 / 2 / 1 (0.08%) | 1931 / 10 / 0 | 1263 / 1931 |
| border `onset_vs_comply/all` val | 338 / 2 / 1 (0.3%) | 422 / 1 / 0 | 337 / 422 |
| border `mixed/onset_vs_comply` (train+val) | 175 / 3 / 2 (1.1%) | 190 / 2 / 0 | 173 / 190 |
| border `prompt_end/all` (not a used bank) | 1292 / 10 / 9 (0.7%) | 1940 / 10 / 0 | 1283 / 1949 |
| pairs `onset_vs_comply/all` (not a used bank) | 825 / 1 / 1 (0.1%) | 841 / 12 / 0 | 824 / 841 |

(Full per-family table incl. reasoning_head / answer_head / val in the appendix. Answered-class "cut-off rows" stay answered:
the heading question agreed with all of them.)

**Verdict: the banks do not need recomputing for this.** D (border onset_vs_comply) loses 1 of 1,264 refused
rows; the mixed variant 2 of 175. H (pairs prompt_end) is the most affected: 23 of 849 refused-class train rows
(2.7%) are harmless prompts that were labeled refused only because the reasoning ran out. At `prompt_end` those rows
sit with the harmless (answered) class, so removing them mostly rescales the bad-minus-good mean difference by about
849/826 and only slightly rotates it; I estimate the change to the normalized direction is well under 3% but did not
compute it (no recompute requested). If H is rebuilt for another reason, use the corrected labels in
`corrected/gen_stage_pairs.jsonl`. The prompt-labeled banks are unaffected.

## Answers cut at their cap (no reasoning cut-off)

Run-2 non-thinking answers were capped at 512 tokens. A capped answer still has text, but the judge cannot see the
rest, which can read as "deflected" (it never reaches the specifics). Counts (`answer_cap_stats.json`):

| file set | records | answer capped | flag rate capped / not capped | deflection-only flags (refused < 0.6): capped / not capped |
|---|---|---|---|---|
| run-2 probe non-thinking (10 files) | 1000 | 496 (50%) | 45% / 52% | 62/224 / 33/261 |
| run-2 probe_big non-thinking (25 files) | 5000 | 3161 (63%) | 19% / 23% | 415/615 / 221/417 |
| run-2 probe thinking answers (4,096 total) | 1500 | 60 (4%) | 62% / 30% | 20/37 / 114/397 |
| run-1 probe thinking answers (4,096 total) | 800 | 22 (3%) | 77% / 39% | 8/17 / 48/291 |

Diagnostic (`nt_cap_check.py`, not a validated judge): re-asking the same two questions on the 477 capped
non-thinking deflection-only flags, with the response marked as cut off by the length limit, unflags 1/62 in the
run-2 100-prompt probe (a harmless record) and **169/415 in probe_big** (167 harmful, 2 harmless). In the low-refusal probe_big edits this is most of
the residual: e.g. r4_nt_decp_harmp_lam1.5_52_76 8.7% -> 2.0%, r2_nt_decp_harmp_lam1_40_92 10.7% -> 2.7%,
push_nt_decp_harmp_lam1.5_40_92 9.3% -> 3.3%, big_base_nt 72.0% -> 70.7% (per-condition table: `nt_cap_table.md`).
So non-thinking harmful refusal in probe_big is likely overstated by 1.3-8 points at the 512 cap, most for the best
edits; treat this as a sensitivity bound, not a corrected number (the note changes the judge input and was not
validated). The 100-prompt run-2 nt numbers are essentially unaffected. Thinking-mode answers rarely hit the cap
(3-4%).

## Other volume data checked

- `session_v1_pilot_pairs/refusal` (40 rows), `session_v1_pilot_border/refusal` (160 rows),
  `session_v1_probe_rehearsal` (10 records): 0 cut-off traces, nothing to correct.
- `session_run2_probe_big` thinking files: big_base_think, r3, r5, r7 done earlier; r2_think (1k) done here. All
  other probe_big files are non-thinking (covered in the answer-cap section).
- Not examined: pre-run-1 sessions (`session/`, `session_pilot*`, `session_tiny*`, Nov 18) and `session_nt*`
  (non-thinking forks/recovery).

## Caveats

- The heading question undercounts soft deflection on long traces (see validation); corrected harmful rates for
  edits with many cut-offs are best read as a range [corrected, corrected + ~0.3 x cut-off complying]. Explicit
  refusals are caught reliably on 4k probes.
- Run-1 data is temperature 1.0: paired flips vs `baseline` include sampling noise.
- 1k-cap validation is 12 traces; treat r2_think's 7.3% as indicative (it matches r3 at 4k).
- Cut-off traces that would have finished with a refusal after more thinking cannot be distinguished from ones that
  were heading there; the question only reads where the trace is going.

## Cost

Jev: 1,279 calls, **$0.144** total (run-2 probe $0.028, run-1 probe $0.010, r2 1k $0.013, gen stage $0.055, long
probe validation $0.014, non-thinking cap diagnostic $0.025). 0 unscored.

## Files (all under out/run2/review/cutoff/)

- `report.md` (this), `tables.md` (all generated tables), `nt_cap_table.md`, `label_impact.json`,
  `answer_cap_stats.json`
- per-record outputs: `corrected/{run2_probe,v1_probe,probe_big_r2_1k,gen_stage_pairs,gen_stage_border}.jsonl`
  (old verdict, cut-off flag, heading score, corrected flag); raw Jev rows: `{run2_probe,v1_probe,probe_big_r2_1k,
  gen_stage,probe_long_validation}/rejudge.{jsonl,txt}`, `nt_cap_check.jsonl`
- scripts: `rejudge_gen_stage.py`, `analyze.py`, `nt_cap_check.py`; data: `data/` (volume copies)

## Appendix: generated tables (generation stage, answer caps)

### Run-1 generation stage (refusal labels used to build the banks)


**pairs** (session_v1_pairs/refusal): 2122 rows, 45 cut off at 8,192 reasoning tokens ({('harmful', 'flagged'): 1, ('harmless', 'flagged'): 30, ('harmless', 'answered'): 14}); heading to refuse 0; label flips: flagged->answered 31 ({'harmful': 1, 'harmless': 30}), answered->flagged 0. Cut-off folds: {'val': 10, 'train': 35}.

| family (flag labeling) | fold | refused class n (cut-off in it / flips out) | answered class n (cut-off in it / flips out) | corrected n refused / answered |
|---|---|---|---|---|
| prompt_end/all | train | 849 (23 / 23 = 2.7%) | 841 (12 / 0) | 826 / 864 |
| prompt_end/all | val | 214 (8 / 8 = 3.7%) | 218 (2 / 0) | 206 / 226 |
| reasoning_head/all | train | 849 (23 / 23 = 2.7%) | 841 (12 / 0) | 826 / 864 |
| reasoning_head/all | val | 214 (8 / 8 = 3.7%) | 218 (2 / 0) | 206 / 226 |
| onset_vs_head/all | train | 825 (1 / 1 = 0.1%) | 841 (12 / 0) | 824 / 864 |
| onset_vs_head/all | val | 205 (0 / 0 = 0.0%) | 218 (2 / 0) | 205 / 226 |
| onset_vs_comply/all | train | 825 (1 / 1 = 0.1%) | 841 (12 / 0) | 824 / 841 |
| onset_vs_comply/all | val | 205 (0 / 0 = 0.0%) | 218 (2 / 0) | 205 / 218 |
| answer_head/all | train | 826 (0 / 0 = 0.0%) | 829 (0 / 0) | 826 / 829 |
| answer_head/all | val | 206 (0 / 0 = 0.0%) | 216 (0 / 0) | 206 / 216 |
| mixed/reasoning_head/all | train+val (mixed) | 0 (0 / 0 = 0.0%) | 0 (0 / 0) | 0 / 0 |
| mixed/onset_vs_comply/all | train+val (mixed) | 0 (0 / 0 = 0.0%) | 0 (0 / 0) | 0 / 0 |

**border** (session_v1_border/refusal): 4000 rows, 24 cut off at 8,192 reasoning tokens ({('harmful', 'answered'): 3, ('harmful', 'flagged'): 10, ('harmless', 'answered'): 8, ('harmless', 'flagged'): 3}); heading to refuse 2; label flips: flagged->answered 11 ({'harmful': 8, 'harmless': 3}), answered->flagged 0. Cut-off folds: {'train': 20, 'val': 4}.

| family (flag labeling) | fold | refused class n (cut-off in it / flips out) | answered class n (cut-off in it / flips out) | corrected n refused / answered |
|---|---|---|---|---|
| prompt_end/all | train | 1292 (10 / 9 = 0.7%) | 1940 (10 / 0) | 1283 / 1949 |
| prompt_end/all | val | 346 (3 / 2 = 0.6%) | 422 (1 / 0) | 344 / 424 |
| reasoning_head/all | train | 1292 (10 / 9 = 0.7%) | 1940 (10 / 0) | 1283 / 1949 |
| reasoning_head/all | val | 346 (3 / 2 = 0.6%) | 422 (1 / 0) | 344 / 424 |
| onset_vs_head/all | train | 1264 (2 / 1 = 0.1%) | 1940 (10 / 0) | 1263 / 1949 |
| onset_vs_head/all | val | 338 (2 / 1 = 0.3%) | 422 (1 / 0) | 337 / 424 |
| onset_vs_comply/all | train | 1264 (2 / 1 = 0.1%) | 1931 (10 / 0) | 1263 / 1931 |
| onset_vs_comply/all | val | 338 (2 / 1 = 0.3%) | 422 (1 / 0) | 337 / 422 |
| answer_head/all | train | 1282 (0 / 0 = 0.0%) | 1930 (0 / 0) | 1282 / 1930 |
| answer_head/all | val | 343 (0 / 0 = 0.0%) | 421 (0 / 0) | 343 / 421 |
| mixed/reasoning_head/all | train+val (mixed) | 194 (8 / 7 = 3.6%) | 194 (2 / 0) | 187 / 201 |
| mixed/onset_vs_comply/all | train+val (mixed) | 175 (3 / 2 = 1.1%) | 190 (2 / 0) | 173 / 190 |

### Answers cut at their token cap (reasoning closed or absent)

| file set | records | answer capped | flag rate capped / not capped | deflection-only flags (refused < 0.6, deflected >= 0.6): capped / not capped |
|---|---|---|---|---|
| run-2 probe non-thinking (512 cap) (10 files) | 1000 | 496 (50%) | 45% / 52% | 62/224 / 33/261 |
| run-2 probe_big non-thinking (512 cap) (25 files) | 5000 | 3161 (63%) | 19% / 23% | 415/615 / 221/417 |
| run-2 probe thinking answers (4,096 total cap) (15 files) | 1500 | 60 (4%) | 62% / 30% | 20/37 / 114/397 |
| run-1 probe thinking answers (4,096 total cap) (8 files) | 800 | 22 (3%) | 77% / 39% | 8/17 / 48/291 |
