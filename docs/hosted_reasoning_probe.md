# Hosted K3.1 Pro: are the reasoning traces raw? (2026-11-17)

Probe: `tools/probe_reasoning.py` → `results/probe_reasoning/responses.jsonl`, checked with `tools/check_reasoning_tokens.py`.
- 3 prompts: Tiananmen (EN), Xinjiang camps (zh), and a Kent State control (EN)
- 3 pinned providers (`provider.only`, no fallbacks): `wafer/us` (MXFP4), `moonshotai/mxfp4`, `deepinfra/bf16`
- effort medium and low, temperature 0.6
- Total cost: well under $0.10.

## Result: yes, the traces are raw
- Every successful response returned `reasoning_details` of type **`reasoning.text`** (OpenRouter's "raw text" type), never `reasoning.summary` or `reasoning.encrypted`. The doc only distinguishes raw from summary by the detail `type`; there is no separate flag.
- **Token-exact check:** re-tokenizing the returned trace with the real K3.1 tokenizer matches the provider-billed `reasoning_tokens`:
  - DeepInfra: **diff 0** on all 6 responses
  - Moonshot: **+1**
  - Wafer: a constant **+3** (they count delimiter tokens in the reasoning)
  - Answer and template overhead is a constant +13 / +16.
  - A summary would come in far below the billed count. These are the model's actual tokens.
- **Logprobs (Wafer) cover only the answer, not the reasoning.** They also expose the K3.1 answer delimiters `<|open|>response<|sep|>`. So we get no per-token reasoning logprobs from the API. Replaying through our own vLLM gives them anyway.

## Providers
- **Moonshot's own endpoint** rejects censored prompts at the input with `content_filter` ("considered high risk"). It never reaches the model, so it is useless for baselines. **Exclude it.**
- **Wafer and DeepInfra** serve the model unfiltered (the censorship below is the model itself).
- DeepInfra reports BF16; Wafer and most others serve MXFP4, which is the native checkpoint format for the experts.
- 24 endpoints exist. Pin with `provider: {only: [...], allow_fallbacks: false}`. The response's `provider` field names who served it.

## What K3.1 does (n=2 per cell, indicative only)
- **The censorship decision is made in the first reasoning sentence and is explicit.** Censored prompts get ~30–75 reasoning tokens; the control gets 180–680.
  - Tiananmen: the trace says to claim it lacks details, cites strict information rules, and decides to "focus on positive changes".
  - Xinjiang (zh): the trace says the camps must be described as "vocational skills education and training centers" and the conclusion must match the official position ("与官方表述保持一致"). One trace refers to a "参考" (a reference answer?) the conclusion must match.
  - Answers are pure party line, with no refusal.
- **The control reasoning is long and factual** (names, round counts, timeline).
- **Effort barely changes the censored behaviour.** Medium and low both produce short, decided traces.

## Implications for the harness
1. **The hosted route works for judge testing and for generating traces.** Activations come from replaying API transcripts (prompt + thinking + answer) through our own vLLM with `harness/capture.py`, at the reasoning positions Jev flags.
2. **Judge reasoning separately.** The onset is at reasoning token ~0–20. Capture positions should be the first flagged reasoning chunk and the tokens right before it, including the last prompt token and the thinking-open delimiter.
3. **The decision may be made before the trace starts.** That would mean "censor" is already settled at the end of the prompt. Prompt-end activations (censored vs control) are a strong direction candidate alongside the flagged reasoning spans.
4. **Replay needs the exact template:** thinking delimiters, then `<|open|>response<|sep|>`, then the answer. Confirm against K3.1's chat template before capture.
