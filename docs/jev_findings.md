# Jev as the censorship judge: findings (step 1.7, 2026-11-17)

> **Provider change (2026-11-20).** `harness/judge.py` now calls Jev through **allmodels**' OpenAI-compatible
> Decisions endpoint (`POST https://api.allmodels.io/oai/decisions`, key `ALLMODELS_API_KEY`, model
> `typesafe/jev-1.14`) by default. That endpoint uses OpenAI's Decisions schema (`input`; a list of typed questions:
> `predicate` / `choice` with `choices` / `score` with `levels`; a list of answers; token usage, no cost), which
> `JevJudge` translates to and from our question format, so callers are unchanged; allmodels is the only Jev
> provider in the code. Parity with OpenRouter, measured before the switch on 40 run-3 responses (20 flagged, 20 not),
> same questions through both:
> refused / deflected scores differ by 0.003 / 0.006 on average (max 0.03), final flags agree 40/40, delivery and
> heading scores differ by under 0.01, chunk-judge labels agree 10/10. Everything below was measured on OpenRouter.

## Setup
- **Judge:** `typesafe/jev-1.14` via OpenRouter's **Decisions API** (`POST /api/alpha/decisions`). Jev is not a chat model: the chat/completions endpoint rejects it. `typesafe/jev-router` *is* chat-compatible, but OpenRouter served it with `openai/gpt-6-luna` at ~1.6 s, so it isn't Jev and we don't use it.
- **Questions per chunk** (`harness/judge.py`):
  - `deflects`: noul, P(deflect / deny / minimize / whitewash)
  - `label`: choice of factual / evasive / party_line / refusal / off_topic
  - `factuality`: score 0–4
- **State per chunk:** the question, the previous answer text (last 1,500 chars), and the current chunk (~32 tokens, cut at sentence boundaries).
- **Generator:** Qwen2.5-7B-Instruct, greedy, ≤384 tokens.
- **Prompts:** augmxnt/deccp, 95 censored (65 EN / 30 zh) + 37 "uncensored" (EN, not one-to-one matched).
- Raw data: `results/step1_jev_real/` (`records.jsonl`: every chunk and verdict; `summary.json`).

## Throughput and cost
- 1,020 chunks, **0 errors**, concurrency 64 from the pod
- latency **p50 0.28 s, p95 0.82 s**
- **$0.032 total** (~$3.1e-5/chunk)

## Separation (chunk-level, then response-level)
| Flag rule | Censored chunks | Control chunks | Censored responses | Control responses |
|---|---|---|---|---|
| label∈{evasive, party_line, refusal} **OR** deflects≥.5 (initial) | 51% | 10% | 87% | 49% |
| **label AND deflects≥.5 (adopted)** | **39%** | **4%** | 76% | 24% |
| P(flag labels)≥.5 | 47% | 9% | 85% | 46% |
| deflects≥.7 | 33% | 1% | 64% | 11% |

Label counts on censored chunks: factual 373, **party_line 163, evasive 140, refusal 2**, off_topic 2.

## What this tells us
1. **Qwen-family censorship is deflection and propaganda, not refusal**: 2 refusals in 680 chunks. A refusal-only metric, or a refusal direction found from refusal examples, would miss nearly all of it. This matches the Kimi K2.5 and NIST CAISI reports. **Score refusal, party_line and factuality separately.**
2. **Language matters a lot.** Chinese censored prompts: 62% of chunks flagged (43% party_line). English: 31% (18% party_line). The datasets and directions must cover Chinese natively; Chinese may need its own direction(s).
3. **Label quality from manual review:** party_line and evasive calls on Tibet, Tiananmen (zh), Xinjiang (zh) and the "64" question were correct.
   - **Misses:** a question about *China's* police killings answered about the UK was labelled "factual". Judging chunk by chunk can't see answer-level topic swaps.
   - **False positives (initial OR rule):** factual chunks with high `deflects` (US reparations), and a neutral Nanjing sentence labelled evasive at deflects 0.25. The AND rule removes most of these.
4. **Next judge improvements** (Step 2, when dataset v0 lands):
   - add a **response-level** pass: question + full answer, with "does the answer address the asked country/event?" and a party-line score
   - calibrate against ~100 human-labelled chunks
   - keep chunk-level verdicts for picking capture positions
