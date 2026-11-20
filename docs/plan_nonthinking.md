# Plan: token-level refusal onset with reasoning off (T = 0)

Goal (goal B, general refusal, as a lesson for goal A): find where in K3.1's activations the decision to refuse
forms, token by token, with reasoning off and greedy decoding, and test whether a direction found this way removes
refusal in non-thinking *and* reasoning mode. Background: `docs/brainstorms.md` (token-by-token onset entry),
`results/refusal_styles_probe/` (hosted catalogue: 193 refusals, decision mostly in sentence 0, fork at answer
token 1–2 for direct refusals and ~12–18 for "I can help with X, but not Y"; Jev's word judge agrees with the
logprob fork on 139/140).

## Two measurements on one token timeline
- **Behavioral fork**: the answer token after which the outcome is fixed. Found by
  1. Jev sentence onset on the full answer → decision sentence,
  2. Jev word judge on that sentence → candidate fork word (highest score ≥ 0.4),
  3. branch test on our engine: at the candidate token (± a few), substitute the runner-up token from the greedy
     logprobs, continue greedily ~128 tokens, Jev-judge; the fork is the commitment point: the last position
     whose substitution still flips the outcome (an earlier flip only shows the model could still go either way). The probability at the fork is the graded refusal score.
- **Internal onset**: the earliest token, prompt included, whose activations predict the eventual outcome
  (cross-validated readout per token position × stream index, aligned by distance to the fork and to the prompt
  end).

## Code (built; CPU-tested, not yet run on the GPU)
| Piece | Where |
|---|---|
| Non-thinking mode: prompt (ends in `<|open|>response<|sep|>`), parse, replay with answer char→token map | `harness/k3chat.py`, `harness/replay.py` |
| Greedy generation keeping each token's logprob + top-20 alternatives, resumable; greedy branch continuations | `Evaluator(thinking=False)`, `generate_progressive(logprobs=20)`, `continue_greedy` (`harness/evals.py`) |
| Answer judging during generation: verdict, decision sentence, per-word scores → candidate fork | `harness/forks.py` (`judge_answer`), `refusal.StreamJudge(judge_fn=...)` |
| Branch test; fork = commitment point (last tested position whose runner-up still flips the outcome); generation, judging and branch test overlap | `harness/forks.py` (`run_pipeline`, `BackgroundJev`), stage `forks` |
| Capture of the exact generated sequence: prompt window → fork + margin, every 4th stream index; plus all stream indices at the prompt end and fork states | stage `forks` phase `capture`, `tp.capture_one(streams=...)` |
| Edit directions at the fork / non-thinking prompt end | `tools/fork_bank.py` |
| Token × layer readout (internal onset), streaming | `tools/analyze_forks.py` |
| Probe per step in non-thinking or reasoning mode, greedy, with paired flips vs a baseline step | `tools/quick_probe.py` (`thinking`, `temperature`, `compare_to`) |

Run: `stage forks --harmful data/refusal_v1/borderline_harmful.jsonl --harmless data/refusal_v1/borderline_harmless.jsonl
--extra-harmful data/refusal_v1/pairs_harmful.jsonl --out /vol/session_nt` (add `--n-per-class` to cap).
Unverified until the GPU run: greedy determinism on our engine, vLLM's logprob output on this build, branch-test
throughput, and capture size (~5–8 MB per window row at stride 4 for typical prompt lengths, + ~8 MB anchors).

## GPU session (8×B300, ~1–1.5 h, ~$75–110)
| Step | Work | Est. |
|---|---|---|
| 0 | engine load (`NCCL_NVLS_ENABLE=0` on restarts), greedy determinism check | 15–28 min |
| 1–2 | `stage forks --phase generate`: greedy answers (T=0, ≤512 tokens, top-20 logprobs), Jev judging and the branch test **pipelined on one engine** (`forks.run_pipeline`): each answer is judged as it finishes, its branch substitutions join the engine while other answers still generate, each branch result is judged in the background; every answer, judgment and branch result is saved as it completes | ~15–25 min for ~2,000–3,000 prompts (judging is the pace: ~20–25 Jev calls per refusal) |
| 3 | capture prompt → fork + 16 | ~5–10 min |
| 4 | `tools/fork_bank.py` inside the GPU sandbox (fast reads) → bank copied out / into `/root/banks` | ~2 min |
| 5 | probe, greedy paired, in non-thinking and reasoning mode: new fork direction vs reasoning-mode `onset_vs_comply` and `prompt_end` | 20–30 min |
| 6 | stop the sandbox; token-level analysis on CPU (32 CPU sandbox) | — |

## Questions it answers
- At which token and layer is the refusal decision first readable: while reading the request, at the prompt end, or
  only at the fork?
- Is the non-thinking decision direction the reasoning-mode one?
- Does an edit found with reasoning off reduce refusal with reasoning on?
- Are partial refusals ("I can help with X, but not Y") a separate mechanism from direct ones?
