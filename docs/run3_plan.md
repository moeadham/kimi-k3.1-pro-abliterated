# Run 3 plan: below 5% refusal, ~0% over-refusal, model intact (tonight)

Goal B, in one GPU session: an edit that

| Target | Measured on (held out, never used to build directions) | Pass |
|---|---|---|
| Harmful refusal | WildJailbreak eval (400 new + the 150 from run 2) + HarmBench standard 200; non-thinking **and** reasoning at effort "low" | **< 5%** in both modes |
| Incorrect refusal | XSTest safe 200 + WildJailbreak eval benign 205 (+ OR-Bench-hard 300 if supplied) | hard refusals ≈ 0; each flag read |
| Coherence | the same runs | reasoning closes as often as unedited (−1 point at most); median reasoning length ≤ 1.3× unedited; no loops, glued words, script switches, comma→period, leaked reasoning |
| Intelligence | sanity panel (MMLU-Pro 100, CMMLU 50, IFEval 100, MATH-500 L5 50, BFCL 50, effort low) + KL on 200 harmless prompts | ≤ 2 points drop per set (panel threshold is 5; we aim tighter); KL ≤ 0.1 nats/token |

Background: `docs/refusal_run2_results.md` (what we know), `docs/run_log.md` (timings, incidents).

## 1. What runs 1–2 tell us (the basis for every choice below)

1. **We may already be close in non-thinking mode, but we have never measured the rest.** Projected H (run 1)
   on layers 48–84, λ 1.5: 72% → 6.7% harmful refusal (150 WildJailbreak prompts), no new harmless refusals. That
   number is probably *over*stated: 7 of 30 flagged "refusals" read by hand delivered the content, and many flags
   are deflection-only on answers cut at 512 tokens. What has **never** been measured: reasoning at low effort,
   capability (no sanity panel, no KL in either run), and HarmBench for the projected edits. Run 3 measures these
   first.
2. **Projection is essential, and the band edges matter.** Raw directions tilt toward the mean activation, which is
   75–98% of each token's vector in the edited layers; the raw edit strips 18–46% of the refusal signal's size from
   every token, which is what broke text (commas → periods, reasoning leaking into answers). Projected edits on
   48–84 or 52–76 are clean; 40–92 is not (layers 40–43 / 85–92). λ 1–1.5 is the plateau; λ 2 starts to lose;
   λ 2.5 breaks the model.
3. **The decision is made by the end of the prompt** (AUC ~0.9 on the last words of the request, 0.97–0.99 on the
   chat-template tokens at layers 56+), and refusals commit at the first answer token. So the prompt end is the
   right place to measure and edit, and the first-token probability is a near-free outcome label (AUC 0.98).
4. **H is a blend of "harmful" and "will refuse"** (it separates harmful from harmless among answered prompts,
   AUC 0.94, and refused from answered among harmful ones, 0.96). For goal B that is fine; for goal A it is the
   problem to solve later.
5. **D adds nothing on top of projected H in non-thinking mode** (Hp 7%, Dp 21%, Dp+Hp 9%), and in reasoning mode
   Hp ≈ Dp+Hp. **Decision: no D rebuild.** Dp+Hp stays as one comparison arm in reasoning-low, because raw D did
   carry reasoning-mode decisions in run 1.
6. **What is left after the edit is a different behaviour:** soft deflections, not refusals, on recurring themes
   (group-stereotype articles, atrocity revisionism with a "be balanced" cue, exploit code with a defensive
   wrapper, black-market contacts, academic hazardous chemistry). Pushing H harder does not remove them (λ 2 is
   worse than 1.5). This needs its own direction: measured **on the edited model** (§3.3).
7. **H's inputs are narrow:** 849 refused vs 841 answered responses, 79% from one source of plain requests, last 4
   template tokens, harmless reference = the answered mean of the same ~840 responses. H from run 2's non-thinking
   data agrees with it only up to layer ~60 (cosine 0.66–0.85, then 0.28–0.51), and the two harmless references
   differ (0.56–0.78). So the data and the reference set are real choices, and both are cheap to improve.
8. **Measurement is now the bottleneck at these rates:** at 5%, one judge error in twenty matters. Known failure
   modes: cut-off reasoning judged on an empty answer, answers cut at 512 tokens flagged as deflection, clarifying
   questions ("where are you?") flagged as deflection, short branch snippets judged against full answers.
9. **Every residual flag under the best edits is soft.** Hard refusals (refused ≥ 0.6) under Hp 48–84,
   Dp+Hp 52–76 / 48–84 and NTPEp: **0 of 150** harmful; harmless hard refusals 0 everywhere (the 1–2 harmless flags
   are clarifying questions). A delivery second pass (§2.2 A3) clears 13 of 48 of those deflection flags as content
   delivered, so the best non-thinking edits are at roughly 6% on the judge's final verdict and 0% hard.
10. **The residual stream has 12-layer blocks** (norm resets at layers 24, 36, 48, 60, 72, 84); 48–84 is three
   whole blocks. Bands and leave-out tests in run 3 are block-aligned.
11. **Coherence risk is real:** at effort "max", edited models reasoned about twice as long and often never
   finished (they drafted the whole answer inside the reasoning). Effort "low" may avoid it, or may not.

## 2. Before the GPU (today, CPU only)

### 2.1 Data needed (please prepare; English; jsonl with `id`, `prompt`, `split` harmful/harmless, `category`, `pair` when matched)

| Set | Size | Purpose | Notes |
|---|---|---|---|
| **H build corpus** | ~3,000 harmful + ~3,000 harmless | new H (§3.2) and the residual direction (§3.3) | must not overlap any eval set; mix below |
| – WildJailbreak **train** | 1,000 adversarial harmful + 1,000 adversarial benign + 500 vanilla harmful + 500 vanilla benign | jailbreak / roleplay framings | train split only (eval split is held out) |
| – residual themes | ~500 harmful + ~500 harmless twins in the same style | the deflections H misses | stereotype persuasive pieces, revisionism with "balanced" cue, exploit / malware with a defensive wrapper, black-market contacts / cover-ups, hazardous chemistry framed academically; twins: same framing, harmless subject (e.g. a balanced essay on a benign controversy) |
| – enriched framings | ~1,000 harmful + ~1,000 harmless | fiction, professional, hypothetical, defensive | as in `data/refusal_v1` borderline, new prompts |
| – existing pairs | 1,061 + 1,061 | continuity with run 1's H | `data/refusal_v1/pairs_*` |
| **Harmless reference** | ~2,000 ordinary prompts | the projection reference (§3.2) | everyday chat, coding, math, writing, advice; not adversarial |
| OR-Bench-hard (optional) | 300 | over-refusal eval | seemingly-toxic but safe prompts; strengthens the "≈0 incorrect refusals" claim |

Already in the repo: WildJailbreak eval harmful (1,834) / benign (205), HarmBench standard (200), XSTest safe (200),
capability panel sets and `kl_prompts` (200).

### 2.2 Analyses on existing captures (CPU sandbox, 32 CPUs, banks saved to `/vol/banks/`)

| # | What | Why |
|---|---|---|
| A1 | **done:** split-half cosine of H per layer (two disjoint halves of run-1 pairs): **0.987–0.998 in layers 48–84** (`out/run3/prebanks/`) | more plain pairs will not change H; diversity, positions and the reference will |
| A2 | **done** (`/vol/banks/r3pre_{pairs,all}.pt`): the rebuild reproduces run 1's H exactly (cosine 1.000); corrected labels change nothing (1.000); **each template position gives a different, equally stable direction** (cosine to the 4-token average 0.54–0.81, split-half ≥ 0.98); pairs + borderline ≈ H (0.96); removing 2 / 4 / 8 harmless principal components moves H to 0.97 / 0.79 / 0.70. Candidates for the screen: per-position, pc2/4/8, pairs + borderline. Built from: candidate banks from run-1 captures: H with corrected labels; H per template position (each of the 4 tokens); **H from pairs + borderline prompt ends** (6,122 rows, already captured, never probed); H projected against the top 2 / 4 / 8 harmless principal components instead of one mean | 5–8 ready-made candidates, no GPU time |
| A3 | **done:** delivery second pass (`response_judge.DELIVERY_Q`, threshold 0.7) for deflection-only flags, calibrated on run-2 generations (`tools/judge_calib.py`): clears 0 of 99 unedited hard refusals and 13 of 48 edited deflection flags, which read as genuine deliveries; the probe reports hard / final / raw | the judge must be trustworthy at 5% |
| A4 | (optional) position-matched decision direction from the 171 clean token-0 forks | a cleaner D, only if reasoning-low needs it |

### 2.3 Code (tests on CPU before launch)

| # | Change |
|---|---|
| C1 | **built:** **probe evaluation tiers**: `screen` (dev: 200 harmful + 100 harmless + 100 KL prompts, non-thinking, ~2 min) and `gate` (full harmful / harmless sets, both modes, sanity panel, KL); hard vs soft refusal reported apart; non-thinking answer cap 1,024; reasoning effort low with cut-off traces judged by heading (done); text-health checks for glued words, script switches, comma→period, leaked reasoning; reasoning length ratio vs unedited; KL vs unedited reference |
| C2 | **built:** **edit options**: per-layer λ profile; writer subset (attention only / MoE only / both); top-k PCA basis of a family |
| C3 | **built (`tools/pe_capture.py`):** **prefill-only capture**: prompt-end states (last 8 prompt tokens, all 187 stream indices, reasoning template) plus the non-thinking first-token distribution as the outcome label, in one pass; progressive, resumable; works with an edit applied |
| C4 | **built (`tools/pe_bank.py`, also reads run-1 captures):** **bank builder** for any capture set: label source (Jev / first token / prompt), positions, projection reference (class mean / external harmless mean / top-k PCs), split-half report; always writes to `/vol/banks/` (run 2 lost `nt.pt` on local disk) |
| C5 | **built (probe + `pe_capture` with `edit` + `labels_from` + `pe_bank --label jev`):** residual workflow: generate under an edit → judge → capture under the edit → residual bank |

## 3. The GPU session (revised 2026-11-20: trimmed to the four goals)

Goals, in order: **measure** a better H (the model's own refusals on a broader prompt set), **calculate** it,
**push** it below 5%, **refine** it to the smallest edit that holds, with intelligence checked against the unedited
model. Cut from the earlier draft: the 8 A2 candidates from run-1 captures and the D+H / NTPE leaders (the new
build tests the same choices on better data; D and NTPE did not beat H), and the residual direction R (now a fallback
only). One labelling method for all 8,120 build prompts.

8×B300, one sandbox, `--minutes 420` hard stop, live driver in `decode-graphs`, failure + silence monitor on
`/vol/logs/live.log` re-armed every 30 min. **All screens and gates are greedy (temperature 0)**, as in run 2, so
numbers compare with run 2 and edited vs unedited is a per-prompt comparison; the final gate adds one temperature 1
reasoning check. Reasoning always at effort **low**.

| # | Config (`runs/run3/`) | What | Mode / effort / temp / cap | Goal | Est. |
|---|---|---|---|---|---|
| 1 | | engine load, `NCCL_NVLS_ENABLE=0` | | | 25 min |
| 2 | `p1_base` | edit check; KL reference (200 harmless, T=1 samples); unedited gate 1,155 prompts both modes; capability panel | nt 1,024 · think low 4,096 · T 0 | refine (the reference) | ~35 min |
| 3 | `labels` | all 8,120 build prompts answered unedited and judged as each finishes | think low · T 0 · ≤512 | measure | ~45–55 min |
| 4 | `pe_build` | prefill capture: 8,120 build + 2,000 alpaca reference, last 6 positions, streams 72–186 (layers 36–92), labels from step 3 | think template, low | measure | ~45 min |
| 5 | `exec` pe_bank (sandbox CPUs) ‖ `p2_bar` | build H3 variants ‖ the bar: run-1 H 48–84 λ1.5, both modes, screen | nt / think low · T 0 | calculate | ~25 min |
| 6 | `p3_h3`, then `make_configs.py think` | 6 H3 variants, non-thinking screen; best 2 in reasoning | nt / think low · T 0 | calculate | ~25 min |
| 7 | `make_configs.py push_lam`, `push_band` | λ 1 / 2 / 2.5 at 48–84, then bands 48–71 / 60–84 / 36–84 at the chosen λ; best 2–3 in reasoning | nt + think low · T 0 | push | ~45 min |
| 8 | `make_configs.py precision` (rounds) | λ −0.25; drop each 12-layer block; attention-only / MoE-only | nt screen, accepted changes in think | refine | ~60 min |
| 9 | `make_configs.py final` | best 1–2 edits: full gate both modes + panel + KL; T=1 reasoning screen vs unedited T=1; read every harmless flag | both · T 0 (+ T 1) | refine | ~50 min |

Screen = 150 WildJailbreak harmful (dev) + 100 XSTest safe; gate = 750 harmful (WJB 550 + HarmBench 200) + 405
harmless (XSTest 200 + WJB benign 205). Edits are paired with the unedited baselines by prompt text.

### 3.1 Labels (step 3) and the 512-token cap

`tools/heading_check.py` replayed the cap on 1,615 finished max-effort traces (run 1 borderline T=1, run 2 probes
greedy): **96.7% agreement** with the full-trace label, refusal rate 36.8% vs 37.9%. Weak spot: soft deflections
(78%, 13/61 read as answering when cut). On cut traces the heading score is clean at the ends (≥ 0.8: 97.5%
refusing; ≤ 0.3: 99.3% answering) and the misreads sit in 0.3–0.8, so `pe_bank --label jev` **drops cut traces
scoring 0.3–0.8** (~10% of them). Low effort reasons shorter than max, so fewer traces are cut tonight.

### 3.2 Decision rules (numbers fixed before the run)

| Point | Rule |
|---|---|
| Step 2 | check must pass (else restart engine). Record unedited refusal, reasoning closed rate, median reasoning length, panel. |
| Step 3, first 500 | throughput (re-plan if > 60 min projected); unedited plain-pair harmful refusal ≥ 85% and harmless ≤ 10%, else stop and read outputs (judge or template problem). |
| Step 5 build | ≥ 800 refusing and ≥ 800 answering rows after the drop band; split-half cosine median ≥ 0.95 on layers 48–84. Below that: report, build with `--label prompt` as a second variant. |
| Step 6 | rank by screen harmful refusal among edits with: harmless hard refusals ≤ unedited + 1, KL ≤ 0.10, no health regression > 1 point. Keep the best 2 by refusal; an H3 variant is "better" if ≥ 2 points below the bar or equal with lower KL. If none is, push the bar (run-1 H) and say so. |
| Step 7 | per candidate, lowest λ whose non-thinking screen harmful refusal ≤ 3% (margin for the 5% gate) with health clean; then the band with the lowest KL at that λ that stays ≤ 3%. Reasoning screen on the best 2–3 points. A line stops at the first health failure. |
| Step 8 | accept a change if screen harmful refusal rises ≤ 2 prompts (of 150) **and** KL falls or writers/layers shrink, health no worse; re-screen accepted changes in reasoning every round. Stop when a round accepts nothing or at 5:00 elapsed. |
| Step 9 | pass = targets table (top): < 5% both modes on the gate, hard harmless refusals ≈ 0 (each flag read), reasoning closed ≥ unedited − 1 point, length ≤ 1.3×, panel ≤ 2 points per set, KL ≤ 0.1. |
| Fallback R | only if after step 7 no point is < 5% in the reasoning screen with clean health: residual labels under the best edit (`labels`-style config on the 3,000 non-pairs harmful, edit applied), capture under the edit, R = still refusing − answered, screen H + R. |
| Clock | the final gate starts by 5:50 elapsed whatever state the precision loop is in. |

## 4. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Label pass slower than ~55 min | stop at 6,000 rows (saved progressively); build from what is labelled |
| Capture slower than ~0.3 s/row | cap at what fits 50 min (rows saved one by one) |
| New H no better than run-1 H | push and refine run-1 H; the run still delivers a smaller, verified edit |
| Reasoning refusal stays above non-thinking | rank on the reasoning screen in step 7; fallback R from reasoning-mode residuals |
| Edits lengthen reasoning / break closing | precision loop leads with block drops and writer subsets; λ ≤ 1.5 |
| Judge noise near 5% | gate is 750 harmful (±1.6 points); read every harmless flag and a sample of harmful ones |
| Crash / sandbox loss | every response, verdict and capture row is saved as it finishes; rerunning a config resumes it |

## 5. Runbook

```
M="uv run --no-project --with modal --env-file ../../../.env python tools/modal_gpu.py"
$M start --gpu B300:8 --minutes 420 --cpu 32 --mem-gib 256
$M live-start --engine-mode decode-graphs --out /vol/session_r3   # a restart in the same sandbox: prefix the
#   harness.live command with NCCL_NVLS_ENABLE=0 (run 2 incident: NCCL init hang on restart)
export K3_LIVE_OUT=/vol/session_r3
$M live "py /root/harness/runs/run3/p1_base.py"      # then labels, pe_build, p2_bar ...
$M exec "cd /root/harness && python tools/pe_bank.py /vol/session_r3/pe_build /vol/banks/r3_h3.pt --label jev \
    --positions 6:1 1 4:1 --ref-sets reference --pcs 4 8 --prefix h3"   # while p2_bar runs
python runs/run3/make_configs.py think|push_lam|push_band|precision|final ...; $M put runs/run3/X.* /root/harness/runs/run3/
```
Every finished step: a row in `docs/run_log.md`, results copied to `out/run3/`.
