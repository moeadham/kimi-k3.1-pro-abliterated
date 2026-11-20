# GPU run log

One row per engine start and per stage run, so we know what is normal. Times come from the live driver log
(`/vol/logs/live.log`, copied next to each run's outputs) and each stage's `summary.json` `timing`.
Add a row after every run; note anything unusual in the last column.

## Engine starts

"Cold" = first model load on that sandbox (weights read from the Volume, no compile caches). "Reload" = engine
restarted on the same sandbox.

| Date | GPUs | Mode | Cold/reload | Weights (s) | JIT warmup (s) | init engine (s) | Ready after (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | 1×B300, tiny model | eager | cold | 31 | n/a | — | 466 | tiny-k3, TP1 |
| 2026-11-18 | 2×B300, tiny model | eager | cold | — | n/a | 120 | 184 | TP2 |
| 2026-11-18 | 8×B300 | eager | cold | 916 | n/a | 96 | 1,098 | first full-model load; Volume ~1.66 GB/s |
| 2026-11-18 | 8×B300 | decode-graphs | reload | 188 | 146 | 191 | 452 | same sandbox as above |
| 2026-11-18 | 8×B300 | decode-graphs | cold | 956 | 323 | 632 | 1,675 | v1 sandbox; warmup 2.2× and init 3.3× the reload's (no compile cache); CUDA graphs 29 at ~1.7/s |
| 2026-11-19 | 8×B300 | decode-graphs | restart, same sandbox | — | — | — | 875 | first restart hung in NCCL init (see incidents); retry with `NCCL_NVLS_ENABLE=0` |
| 2026-11-19 | 8×B300 | decode-graphs | cold (run 2) | 882 | — | 479 | 1,464 | run 2 sandbox `sb-GjWYhWw3KcXlAdurMwhgVO`; whole session ~3 h 10 min, engine never restarted |
| 2026-11-20 | 8×B300 | decode-graphs | cold (run 3) | ~1,080 (96 shards, ~11 s each) | — | — | 1,391 | run 3 sandbox `sb-mbGE1PMOgP74dqgwBrhMc8` (started 00:23 JST, 420 min); edit check OK in 8 s (279 writers per rank) |

Known, harmless log lines on Modal 8×B300 (seen on every run so far):
- `[MNNVL] FABRIC handles failed ... CUDA_ERROR_NOT_PERMITTED ... using POSIX file descriptors instead`: no IMEX
  channel in the container; single-host all-reduce falls back to file descriptors.
- `No available shared memory broadcast block found in 60 seconds` during JIT warmup: the driver waiting on workers.
- CUTLASS `AuxData ... is not supported` during warmup: a compiler warning.

## Stage runs

| Date | Run (`--out`) | Rows | Samples | Mode | Generate (s) | tok/s | Judge tail (s) | Capture (s) | GB | Failures | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | `/vol/session` refusal dry run (eager, stopped) | 400 | 1 | eager | — | ~3.9/seq | — | — | — | — | stopped; lost 374 responses (pre progressive saving) |
| 2026-11-18 | `/vol/session` refusal dry run | 400 | 1 | decode-graphs | 266 | ~1,600 | (serial judge) | — | 43 | 0 | all positions captured (~107 MB/row) |
| 2026-11-19 | `/vol/session_v1_pairs` (pairs, full) | 2,122 | 1 | decode-graphs | 2,113 | ~2,050 | 49 | 438 (stopped at 1,419) | — | 3 | harmful 1,015 refused / 46 answered; harmless 1,013 answered / 48 refused; 75 hit 8,192 tokens; capture OOM (see incidents) |
| 2026-11-19 | `/vol/session_v1_border` (borderline, full) | 4,000 | 4 | decode-graphs | ~3,900 | ~2,250 | — | (failed at row 0) | — | 3 | generation + judging complete; capture OOM (see incidents) |
| 2026-11-19 | `/vol/session_v1_pilot_border` (borderline, n=20/class) | 160 | 4 | decode-graphs | 236 | ~1,450 (160 in flight) | 2 | 51 | — | 0 | 159/160 judged during generation; harmful 68 refused / 12 answered, harmless 80/80 answered; 2/20 harmful prompts mixed |
| 2026-11-19 | `/vol/session_v1_pairs` capture rerun (`--phase capture`) | 706 | — | decode-graphs | — | — | — | 245 | — | 0 | after the OOM fix; long rows OK |
| 2026-11-19 | `/vol/session_v1_border` capture rerun | 4,000 | — | decode-graphs | — | — | — | 1,241 | ~160 | 0 | 0.31 s/row; 97 mixed prompts |
| 2026-11-19 | `/vol/session_v1_probe` (quick_probe: check, baseline, 7 edits × 100 prompts) | 900 | 1 | decode-graphs | ~100–130 per condition | — | — | — | — | 0 | check 13 s; whole probe 14 min (`docs/refusal_v1_results.md`) |
| 2026-11-19 | `/vol/session_nt_smoke` (`forks`, non-thinking, n=5/class) | 10 | 1 (greedy) | decode-graphs | 17 (pipeline) | — | — | 7 | — | 0 | greedy smoke check 3.5 s (repeats identical); 44 branches judged in the same 17 s |
| 2026-11-19 | `/vol/session_nt` `forks --phase generate` (no branches yet) | 3,122 | 1 (greedy, T=0, ≤512 tok) | decode-graphs | 995 | — | 0 (judged during generation) | — | — | 0 | borderline 1,000 (600 harmful + 400 harmless) + pairs 2,122 (1,061 + 1,061); 1,519 refused |
| 2026-11-19 | `/vol/session_nt` branch pass (alts 3, head 3, ±3) | 62k planned | — | decode-graphs | 263 + 848 (both interrupted) | — | — | — | — | 0 | far too many substitutions; tightened (alts 2, head 2, ±1) and stopped once borderline was done; `--no-branch` rebuilt the rows in 34 s. 11,299 branches, 486 forks confirmed |
| 2026-11-19 | `/vol/session_nt` `forks --phase capture` | 3,122 | — | decode-graphs | — | — | — | 2,087 | — | 0 | 0.67 s/row; prompt window → fork + 16, every 4th stream, + all streams at the prompt end and fork |
| 2026-11-19 | `/vol/session_nt/recovery` (`tools/recovery_capture.py`) | 1,800 seq | — | decode-graphs | — | — | — | 571 | — | 0 | 600 positions × original / recovered / flipped, swap state + 32 tokens |
| 2026-11-20 | `/vol/session_r3/probe` p1_base: check, KL ref, unedited gate both modes, panel | 1,155 ×2 + 200 KL | 1 (greedy) | decode-graphs | nt 351, think 611 | 1,467 / 1,839 | ~0 (judged during generation) | — | — | 0 | whole phase 1,221 s; nt harmful 78.8% / harmless 7.9%; think-low 80.7% / 5.2%, median reasoning 352 tok; panel MMLU-Pro .90 CMMLU .98 IFEval .93 MATH .98 BFCL .88 |
| 2026-11-20 | `/vol/session_r3/labels` (8,120 build prompts, think low, ≤512 tok) | 8,120 | 1 (greedy) | decode-graphs | 2,556 | 1,481 | ~9 | — | — | 0 | harmful 74.1% / harmless 6.0% refused (final); 29.6% reasoning cut at 512 (judged on heading); progress lines buffered in p1_base, flushed from here on |
| 2026-11-20 | `/vol/session_r3/pe_build` (prefill capture, last 6 positions, streams 72–186) | 10,120 | — | decode-graphs | — | — | — | 3,194 | ~96 | 0 | 0.31 s/row; 9.5 MB/row; labels from the label pass |
| 2026-11-20 | `pe_bank` on the sandbox CPUs → `/vol/banks/r3_h3.pt` | 10,120 | — | — | — | — | — | — | — | 0 | 339 s wall (32 CPUs); 3,130 refusing / 4,708 answering after the 0.3–0.8 cut-off drop; split-half cos median 0.997 |
| 2026-11-20 | `/vol/session_r3/probe` p2_bar + p3_h3 + p4_think (screens, 250 prompts) | 250 × 10 | 1 (greedy) | decode-graphs | nt ~91, think ~190–230 each | ~1,800–1,950 | ~5–10 | — | — | 0 | best: new H pc4 (6:1) 48–84 λ1.5: nt 1.3%, think 3.3%, KL 0.067, harmless reasoning 1.17× (run-1 H: 4.7% / 5.3%, KL 0.082, 1.40×) |
| 2026-11-20 | `/vol/session_r3/probe` p5–p6 (push/precision screens), p9–p10 (final gates λ 1.25 / 1.5 / 1.75) | 250 screens, 1,155 gates | 1 (greedy; T=1 check) | decode-graphs | gate nt ~440, think ~900 | ~1,700–2,100 | ~5–10 | — | — | 0 | λ 1.5: nt 4.9%, think 4.0%, KL 0.067, panel within 2 points; whole final phase 3,663 s (two edits) + 1,755 s (λ 1.75) |
| 2026-11-20 | residual R: `r_labels` (4,060, under the edit), `pe_r` (5,060 rows, under the edit), p11–p12 screens | | 1 (greedy) | decode-graphs | 1,142 | 1,787 | ~6 | 1,593 | ~48 | 0 | R dropped (KL 0.122 at λ 1.5, no gain at 1.25) |
| 2026-11-20 | `/vol/session_r3/loop` (max effort, T=1, 16k cap, 20 prompts × 2) | 40 | 1 | decode-graphs | ~45–68 per set | — | — | — | — | 0 | 20/20 both; all closed; sandbox stopped 06:14 JST (5 h 51 min, ~$440) |
| 2026-11-19 | run 2 probes (`quick_probe`, 9 configs, 41 conditions) | 100–200 per condition | 1 (greedy) | decode-graphs | non-thinking 50 s, reasoning (max effort, 4,096 tok) 200–300 s per condition | — | — | — | — | 0 | first 2 configs 100 prompts (HarmBench + XSTest), then 200 (WildJailbreak eval 150 + XSTest 50); results in `docs/refusal_run2_results.md` |
| 2026-11-18 | `/vol/session_v1_pilot_pairs` (pairs, n=20/class) | 40 | 1 | decode-graphs | 187 | ~320 (40 in flight) | 2 | 15 | 1.5 | 0 | 39/40 judged during generation; 40 MB/row (trimmed); 20/20 flagged harmful, 20/20 answered harmless |

## Incidents

| Date | Run | What happened | Fix |
|---|---|---|---|
| 2026-11-20 | run 3 | The background hand-off that was to start pe_bank when the capture finished ran `$M` (a zsh variable holding the whole command) and failed at once; the bar screens ran, then the GPU sat idle ~6 min until a STALL alert. Earlier, progress prints in p1_base were block-buffered in live.log and caused a false STALL. | Ran pe_bank directly; quick_probe progress prints flush; the watcher counts output-file growth as activity; never wrap the modal command in a variable in background scripts. |
| 2026-11-19 | v1 pairs + borderline (8×B300) | Capture OOM on ~8.3k-token rows: `w_capture_take` stacked the whole recording ([T, S, d], 20.7 GiB) before selecting positions; 3 in a row stopped the pairs capture at row 1419/2122 and the borderline capture at row 0. The OOM traceback kept the recording alive on the workers, so later captures and even the hot reload OOMed (workers left half-reloaded). Generation and judging were complete and saved. | Select positions before stacking; capture failures return ok=False after freeing the recording; failed prefill resets the recorder. Driver restarted on the same sandbox, then `--phase capture` reruns. |
| 2026-11-19 | same | Engine restart on the same sandbox hung 18 min in `ncclCommInitRank` (all 8 ranks spinning). | Killed it; restarted with `NCCL_NVLS_ENABLE=0` (stale NVLink multicast state from the killed workers): past NCCL init in 78 s. Do this by default for restarts on a used sandbox. |
| 2026-11-19 | same | GPU sandbox's Volume view misses files written later by CPU sandboxes (pilot direction bank "not found"). | `tools/extract_bank.py` → download → `modal_gpu.py put` into the GPU sandbox. |
| 2026-11-19 | same | `modal_gpu.py oneshot` overwrote the interactive sandbox id, redirecting exec/put/live to a finished CPU sandbox. | oneshot no longer writes the state file. |
| 2026-11-19 | v1 overnight (8×B300) | A crash sat unnoticed for over 10 minutes: only completion waiters were armed. | Run 2 used a Monitor on the live log: errors, milestones, STALL after 5 min without log growth while busy, unreachable after 3 failed polls; re-armed every 30 min. No unnoticed stalls in run 2. |
| 2026-11-19 | run 2 forks (8×B300) | The first branch test confirmed only 1 in 4 refusals as forks: the runner-up token is usually a synonym ("I can't" / "I cannot"). | Several alternatives per position, the first answer tokens as well as the candidate window, and a control branch (the original token re-fed: detects nondeterminism). |
| 2026-11-19 | same | Then the branch pass was far too slow (~62k substitutions planned). | Interrupted twice; tightened to 2 alternatives, 2 head tokens, ±1; stopped once the borderline set was done (user approved); `--no-branch` rebuilds rows from disk. |
| 2026-11-19 | same | Smoke rerun skipped: the DONE marker from the first smoke run. | `--force`. |
| 2026-11-19 | run 2 probes | `nt_fork` direction is position-confounded: refused rows fork inside the answer, answered rows at the prompt end, so the direction mostly encodes position. No effect on refusal, more over-refusal. | Extension conditions switched to `nt_prompt_end`. A position-matched fork direction is still to do. |
| 2026-11-19 | same | Edited greedy runs are not deterministic across batches (unedited repeats: 0 flips; edited repeats: up to ±4/50 harmful). | Treat differences under ~8 points on 50 prompts (~4 on 150) as noise; repeat the leaders. |
| 2026-11-19 | same | First reasoning-mode check capped at 1,024 tokens: 90% never closed their reasoning (mean answer 11 tokens), so the result was meaningless. Rerun at 4,096 tokens: still 36–59% cut off on edited runs, because probes ran at **effort "max"**, and Jev then judged the empty answer (mostly as a refusal). | Re-judged the cut-off traces on where the reasoning was heading (`tools/rejudge_truncated.py`, validated 92% on finished traces): corrected reasoning-mode refusal 8–11% instead of 42–64%. `quick_probe` now defaults to effort "low" and judges cut-off traces with `response_judge.HEADING_Q`, reporting them apart. |

## CPU sandboxes

| Date | Job | CPU / RAM | Wall (min) | Notes |
|---|---|---|---|---|
| 2026-11-20 | `tools/modal_hf_upload.py`: `/vol/Kimi-K3.1-Pro-H3` → private HF repo `moeadham/Kimi-K3.1-Pro-abliterated` (rev `c570709`), then hub-side verify | 16 / 32 GiB | ~40 | 116 files, 25 commits (small files, then 4 shards each) at ~1 GB/s; group 16 was refused at ~0.94 TB ("set up automatic credit recharge to use more storage": PRO includes 1 TB private) until billing was enabled, then went through on retry 4. Verify: all files present with local sizes (except `.gitattributes`, where the hub appended LFS lines for the two figure PNGs); 59/59 unedited shards sha256-identical to `moonshotai/Kimi-K3.1-Pro`; 111/111 edited tensors range-read from the hub match `EDIT.json`. The two earlier attempts (`hf upload` of the whole folder, `/vol/logs/hfup*.log`) put all 100 large files in one batch and died with 0 commits (Xet 429, then Xet "xorb not found") |
| 2026-11-20 | `tools/materialize_edit.py` → `/vol/Kimi-K3.1-Pro-H3` (full edited checkpoint, run-3 λ 1.5 edit) | 32 / 64 GiB | 27.7 | 96 shards / 1,561 GB copied in 1,432 s (~1.1 GB/s, 8 workers); 111 tensors byte-patched in 37 shards; verify 111/111 exact, 0 problems |
| 2026-11-18 | analyze_refusal + directions ×2 (serial) | 16 / 64 GiB | ~16 | 400 rows, 43 GB |
| 2026-11-18 | same, after AUC fix | 16 / 64 GiB | 14 | |
| 2026-11-18 | refusal `--phase judge` (compliance onsets, 194 rows) | 4 / 16 GiB | 0.9 | Jev only, 19 s of judging |
| 2026-11-19 | v1 pairs `directions` + analyze (fanout) | 16 / 128 GiB | 18 (load) + | **CPU sandboxes read capture files at ~2/s (~80 MB/s)**: 2,122 files took 18 min; analyze crashed (KeyError, fixed) |
| 2026-11-19 | v1 analyze_refusal rerun (pairs / borderline) | 32 / 256 · 336 GiB | 6.2 / 28.0 | loads 188 s / 875 s. **Volume read speed scales with sandbox CPUs, not RAM:** 16 CPU ≈ 80–90 MB/s (pairs 85 GB: 18 min at 128 GiB, 15.6 min at 64 GiB), 32 CPU ≈ 400 MB/s (3 min). Use 32 CPUs even when RAM can be small. CPU-only sandboxes are capped at 336 GiB |
| 2026-11-19 | v1 `directions --label-by both` (first try) | 32 / 336 GiB | stopped at 30 | full SVD PCA took 7–13 min per family; borderline would not finish in 3 h. Fixed: randomized top-k PCA, no duplicate /en split |
| 2026-11-19 | streaming `directions` (layer chunks of 24), pairs, verification | 16 / 64 GiB | 20.4 | **peak RSS 27.3 GB** (in-memory: ~125 GB); first chunk 15.6 min (network read at 16 CPU), later chunks ~40 s; bank identical to the in-memory one (`tools/compare_banks.py`) |
| 2026-11-19 | `tools/quick_bank.py` inside the GPU sandbox (pairs / borderline) | GPU host CPUs | 0.4 / ~1.5 | same reads, 2,122 files in 25 s: compute banks where the captures were written |
| 2026-11-19 | run 2 analyses on the GPU host's CPUs while probes ran (`fork_bank`, `analyze_forks` ×2, `score_recovery`) | GPU host (1.28 TB RAM) | 0.25 / ~25 each / ~2 | `fork_bank` 15 s for 3,122 rows; token analysis (3,122 rows × 81 offsets × 48 streams) ~25 min per anchor; recovery scoring 1,800 files in ~2 min. No interference with the probes |
| 2026-11-19 | fanout ×4: analyze + directions `--label-by both`, v1 pilots (40 + 160 rows) | 8 / 32 GiB each | 1.2–6.6 | parallel; pairs analyze 1.2, pairs directions 1.7, border analyze 2.5, border directions (both labelings) 6.6 |
