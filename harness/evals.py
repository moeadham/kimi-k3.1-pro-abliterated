"""Step-2 evaluation runner on an in-process vLLM K3.1 engine (pod only).

Steering/edit state is set by the caller (harness.steering / collective_rpc); this module only
generates, scores and compares. Every comparison is paired against the unedited model run in
the same session with identical items and sampling seeds.

Tier 1 (every sweep trial):
  kl()            whole-response KL(P_base || P_edit), estimated without bias from reference
                  responses SAMPLED from the base model at T=1, top_p=1:
                  KL = E_{y~P_base}[ log P_base(y) - log P_edit(y) ]  (per token, averaged).
                  Both log-probs come from teacher-forced prefill in this runtime
                  (prompt_logprobs=0 returns the logprob of each actual token).
  censorship()    regenerate val replay prompts; one Jev call each (refused / deflected,
                  harness.response_judge); censored_rate = share flagged
  controls()      matched non-sensitive controls, same judge; flagged rate must stay low
  harmful()       refusal retention: share of harmful requests refused or deflected
  + health metrics on every generation (harness.k3chat.health)
Tier 2/3: benchmark(name) for mmlu_pro, gpqa_diamond, cmmlu, math500, ifeval, livecodebench,
  compared with paired_bootstrap().
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from harness import k3chat, scorers

TASKS = {
    "mmlu_pro": (scorers.mc_prompt, scorers.mc_score),
    "gpqa_diamond": (scorers.mc_prompt, scorers.mc_score),
    "cmmlu": (scorers.mc_prompt, scorers.mc_score),
    "math500": (scorers.math_prompt, scorers.math_score),
    "ifeval": (lambda x: x["prompt"], scorers.ifeval_score),
    "livecodebench": (scorers.code_prompt, scorers.code_score),
}


def load_set(name: str, root: str | Path = "data/evals") -> list[dict]:
    return [json.loads(l) for l in open(Path(root) / f"{name}.jsonl")]


@dataclass
class Gen:
    item_id: str
    sample: int
    prompt_ids: list[int]
    out_ids: list[int]
    reasoning: str
    answer: str
    finish: str
    health: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    lp: list = field(default_factory=list)  # logprob of each out token (when requested)
    top: list = field(default_factory=list)  # per out token: [[token_id, logprob], ...] best first (when requested)


GEN_SAVE = ("item_id", "sample", "prompt_ids", "out_ids", "reasoning", "answer", "finish", "health", "lp", "top")


def _token_logprobs(c, k: int):
    """vLLM CompletionOutput -> (chosen logprobs, top-k alternatives per token)."""
    lp, top = [], []
    for tid, d in zip(c.token_ids, getattr(c, "logprobs", None) or []):
        lp.append(float(d[tid].logprob) if tid in d else None)
        top.append([[int(t), float(x.logprob)] for t, x in sorted(d.items(), key=lambda kv: -kv[1].logprob)[:k]])
    return lp, top


class Evaluator:
    def __init__(self, llm, tok, max_tokens: int = 16000, seed: int = 0, thinking: bool = True):
        """thinking=False: the template's non-thinking mode (no reasoning; generation starts in the answer)."""
        self.llm, self.tok, self.max_tokens, self.seed = llm, tok, max_tokens, seed
        self.thinking = thinking
        self.m = k3chat.Markers(tok)

    def continue_greedy(self, prefixes: list[list[int]], max_tokens: int) -> list[list[int]]:
        """Greedy continuation of raw token prefixes (e.g. prompt + answer up to a substituted token)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[self.m.eom],
                                skip_special_tokens=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prefixes], params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]

    # ---------- generation ----------
    def generate(self, items: list[dict], prompt_fn, *, effort="max", temperature=1.0, top_p=0.95, n=1,
                 max_tokens=None, id_key="id") -> list[Gen]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        pids = [k3chat.prompt_ids(self.tok, prompt_fn(it), effort, thinking=self.thinking) for it in items]
        # one request per (item, sample) with a fixed per-sample seed: identical across conditions
        reqs, meta = [], []
        for it, p in zip(items, pids):
            for s in range(n):
                reqs.append(TokensPrompt(prompt_token_ids=p))
                meta.append((it, p, s))
        params = [SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                                 max_tokens=max_tokens or self.max_tokens, seed=self.seed + s,
                                 stop_token_ids=[self.m.eom], skip_special_tokens=False)
                  for _, _, s in meta]
        outs = self.llm.generate(reqs, params, use_tqdm=False)
        gens = []
        for (it, p, s), o in zip(meta, outs):
            c = o.outputs[0]
            parsed = k3chat.parse(list(c.token_ids), self.m, thinking=self.thinking)
            reasoning = self.tok.decode(parsed.reasoning_ids)
            answer = self.tok.decode(parsed.answer_ids)
            g = Gen(it[id_key], s, p, list(c.token_ids), reasoning, answer, c.finish_reason or "")
            g.health = k3chat.health(parsed, g.finish, answer, it.get("lang", "en"))
            gens.append(g)
        return gens

    def _to_gen(self, it, p, s, token_ids, finish, id_key="id") -> Gen:
        parsed = k3chat.parse(list(token_ids), self.m, thinking=self.thinking)
        g = Gen(it[id_key], s, p, list(token_ids), self.tok.decode(parsed.reasoning_ids),
                self.tok.decode(parsed.answer_ids), finish or "")
        g.health = k3chat.health(parsed, g.finish, g.answer, it.get("lang", "en"))
        return g

    def generate_progressive(self, items: list[dict], prompt_fn, path, *, effort="max", temperature=1.0, top_p=0.95,
                             n=1, max_tokens=None, id_key="id", log=print, log_every=10, on_done=None,
                             logprobs: int | None = None) -> list[Gen]:
        """Like generate(), but every finished response is appended to `path` (JSONL, flushed + fsynced) the
        moment it completes, and responses already in `path` are not regenerated. A stopped run (interrupt,
        crash, sandbox loss) therefore keeps every finished response; only in-flight ones restart.
        Same per-sample seeds as generate(), so a resumed response is sampled exactly as it would have been.
        on_done(gen), if given, is called for every response already in `path` and then for each one as soon as
        it is saved (e.g. refusal.StreamJudge.submit, which judges while generation continues); it must not block.
        logprobs=k keeps each out token's logprob and its top-k alternatives (Gen.lp, Gen.top), saved with it.
        temperature=0 is greedy (the seed is then irrelevant)."""
        import os
        import time as _t

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        path = Path(path)
        done: dict[tuple[str, int], Gen] = {}
        if path.exists():
            for line in open(path):
                if line.strip():
                    x = json.loads(line)
                    g = Gen(**{k: x[k] for k in ("item_id", "sample", "prompt_ids", "out_ids", "reasoning", "answer",
                                                 "finish")})
                    g.health, g.lp, g.top = x.get("health", {}), x.get("lp", []), x.get("top", [])
                    done[(g.item_id, g.sample)] = g
        by_key, todo = {}, []
        for it in items:
            p = k3chat.prompt_ids(self.tok, prompt_fn(it), effort, thinking=self.thinking)
            for s in range(n):
                by_key[(it[id_key], s)] = (it, p, s)
                if (it[id_key], s) not in done:
                    todo.append((it, p, s))
        log(f"[gen] {len(done)} already done in {path.name}, {len(todo)} to generate")
        if on_done:
            for g in done.values():
                on_done(g)
        eng = self.llm.llm_engine
        rid_to_key = {}
        stamp = int(_t.time())
        for it, p, s in todo:
            params = SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                                    max_tokens=max_tokens or self.max_tokens, seed=self.seed + s,
                                    stop_token_ids=[self.m.eom], skip_special_tokens=False,
                                    **({"logprobs": logprobs} if logprobs else {}))
            params.output_kind = RequestOutputKind.FINAL_ONLY
            rid = f"{it[id_key]}#{s}#{stamp}"
            eng.add_request(rid, TokensPrompt(prompt_token_ids=p), params)
            rid_to_key[rid] = (it[id_key], s)
        t0, n_done, n_tok = _t.time(), 0, 0
        with open(path, "a") as f:
            try:
                while eng.has_unfinished_requests():
                    for o in eng.step():
                        if not getattr(o, "finished", False) or o.request_id not in rid_to_key:
                            continue
                        it, p, s = by_key[rid_to_key.pop(o.request_id)]
                        c = o.outputs[0]
                        g = self._to_gen(it, p, s, c.token_ids, c.finish_reason, id_key)
                        if logprobs:
                            g.lp, g.top = _token_logprobs(c, logprobs)
                        done[(g.item_id, g.sample)] = g
                        f.write(json.dumps({k: getattr(g, k) for k in GEN_SAVE}, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        if on_done:
                            on_done(g)
                        n_done += 1
                        n_tok += len(g.out_ids)
                        if n_done % log_every == 0 or not rid_to_key:
                            dt = _t.time() - t0
                            log(f"[gen] {n_done}/{len(todo)} done in {dt:.0f}s, {n_tok} tokens, "
                                f"{n_tok / max(dt, 1e-9):.1f} tok/s (finished responses only)")
            except BaseException:
                if rid_to_key:  # leave the engine clean; finished responses are already on disk
                    try:
                        eng.abort_request(list(rid_to_key), internal=False)
                    except Exception:  # noqa: BLE001 -- e.g. already aborted by the live step guard
                        pass
                raise
        return [done[k] for k in by_key]

    # ---------- KL ----------
    def seq_logprobs(self, seqs: list[tuple[list[int], int]]) -> list[list[float]]:
        """Per-token log-probs of ids[start:] under the current model (teacher-forced)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids, _ in seqs],
                                 SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        res = []
        for (ids, start), o in zip(seqs, outs):
            pl = o.prompt_logprobs
            res.append([pl[t][ids[t]].logprob for t in range(start, len(ids))])
        return res

    def kl_reference(self, prompts: list[dict], effort="low", max_tokens=2048, n=1) -> list[dict]:
        """Sample reference responses from the CURRENT (must be unedited) model at T=1, top_p=1."""
        gens = self.generate(prompts, lambda x: x["prompt"], effort=effort, temperature=1.0, top_p=1.0, n=n,
                             max_tokens=max_tokens)
        seqs = [(g.prompt_ids + g.out_ids, len(g.prompt_ids)) for g in gens]
        base = self.seq_logprobs(seqs)
        return [{"item_id": g.item_id, "ids": s[0], "start": s[1], "base_lp": lp} for g, s, lp in zip(gens, seqs, base)]

    def kl(self, ref: list[dict]) -> dict:
        cur = self.seq_logprobs([(r["ids"], r["start"]) for r in ref])
        per_seq = [statistics.fmean(b - c for b, c in zip(r["base_lp"], lp)) for r, lp in zip(ref, cur) if lp]
        toks = [b - c for r, lp in zip(ref, cur) for b, c in zip(r["base_lp"], lp)]
        return {"kl_per_token": statistics.fmean(toks), "kl_seq_mean": statistics.fmean(per_seq),
                "kl_seq_p90": sorted(per_seq)[int(0.9 * (len(per_seq) - 1))], "n_seq": len(per_seq), "n_tok": len(toks)}

    # ---------- Jev-scored tier-1 sets ----------
    def _jev(self, coro_factory):
        from harness.judge import JevJudge

        async def run():
            async with JevJudge(concurrency=64) as j:
                return await coro_factory(j)
        return asyncio.run(run())

    def censorship(self, items: list[dict], effort="max", n=1) -> dict:
        """items: {id, question, lang}. Regenerate and judge (one Jev call per response)."""
        from harness.response_judge import RESPONSE_Q, THRESHOLD, flag_score, is_flagged, judge_response

        gens = self.generate(items, lambda x: x["question"], effort=effort, n=n)
        q = {it["id"]: it["question"] for it in items}
        verdicts = self._jev(lambda j: asyncio.gather(*(judge_response(j, q[g.item_id], g.reasoning, g.answer) for g in gens)))
        for g, v in zip(gens, verdicts):
            g.score = {**v, "censored": is_flagged(v), "flag_score": flag_score(v)}
        return {"censored_rate": _mean(g.score["censored"] for g in gens),
                "mean_flag_score": _mean(g.score["flag_score"] for g in gens),
                **{f"{k}_rate": _mean(g.score.get(k, 0) >= THRESHOLD for g in gens) for k in RESPONSE_Q},
                "by_lang": _by_lang(items, gens, lambda g: g.score["censored"]),
                "health": health_summary(gens), "gens": gens}

    def controls(self, items: list[dict], effort="max", n=1) -> dict:
        return self.censorship(items, effort, n)

    def harmful(self, items: list[dict], effort="max", n=1) -> dict:
        """Refusal retention: a harmful request counts as declined if refused or deflected."""
        from harness.response_judge import is_flagged, judge_response

        gens = self.generate(items, lambda x: x["prompt"], effort=effort, n=n)
        q = {it["id"]: it["prompt"] for it in items}
        verdicts = self._jev(lambda j: asyncio.gather(*(judge_response(j, q[g.item_id], g.reasoning, g.answer) for g in gens)))
        for g, v in zip(gens, verdicts):
            g.score = v
        return {"refusal_rate": _mean(is_flagged(g.score) for g in gens),
                "health": health_summary(gens), "gens": gens}

    # ---------- tool calling (BFCL subset) ----------
    def bfcl(self, items: list[dict], *, effort="low", n=1, temperature=1.0, top_p=0.95, max_tokens=4096) -> dict:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from harness import bfcl

        reqs, meta, params = [], [], []
        for it in items:
            p = k3chat.prompt_ids(self.tok, it["messages"], effort, tools=bfcl.to_openai_tools(it["functions"]))
            for s in range(n):
                reqs.append(TokensPrompt(prompt_token_ids=p))
                meta.append((it, p, s))
                params.append(SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                                             max_tokens=max_tokens, seed=self.seed + s, stop_token_ids=[self.m.eom],
                                             skip_special_tokens=False))
        outs = self.llm.generate(reqs, params, use_tqdm=False)
        gens = []
        for (it, p, s), o in zip(meta, outs):
            c = o.outputs[0]
            parsed = k3chat.parse(list(c.token_ids), self.m)
            full = self.tok.decode(list(c.token_ids))
            g = Gen(it["id"], s, p, list(c.token_ids), self.tok.decode(parsed.reasoning_ids),
                    self.tok.decode(parsed.answer_ids), c.finish_reason or "")
            g.health = k3chat.health(parsed, g.finish, g.answer, it.get("lang", "en"))
            g.score = bfcl.score(it, bfcl.calls_from_xtml(full))
            gens.append(g)
        per_item = item_accuracy(gens)
        return {"accuracy": statistics.fmean(per_item.values()), "per_item": per_item,
                "health": health_summary(gens), "gens": gens}

    # ---------- benchmarks ----------
    def benchmark(self, name: str, items: list[dict], *, effort="low", n=1, temperature=1.0, top_p=0.95,
                  max_tokens=None, code_workers=32) -> dict:
        prompt_fn, score_fn = TASKS[name]
        gens = self.generate(items, prompt_fn, effort=effort, n=n, temperature=temperature, top_p=top_p,
                             max_tokens=max_tokens)
        by_id = {it["id"]: it for it in items}
        if name == "livecodebench":
            with ThreadPoolExecutor(code_workers) as ex:
                scores = list(ex.map(lambda g: score_fn(by_id[g.item_id], g.answer), gens))
        else:
            scores = [score_fn(by_id[g.item_id], g.answer) for g in gens]
        for g, s in zip(gens, scores):
            g.score = s
        per_item = item_accuracy(gens)
        return {"accuracy": statistics.fmean(per_item.values()), "per_item": per_item,
                "health": health_summary(gens), "gens": gens}


# ---------- statistics ----------

def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return statistics.fmean(xs) if xs else float("nan")


def _by_lang(items, gens, f) -> dict:
    lang = {it["id"]: it.get("lang", "en") for it in items}
    out = {}
    for lg in sorted(set(lang.values())):
        out[lg] = _mean(f(g) for g in gens if lang[g.item_id] == lg)
    return out


def item_accuracy(gens: list[Gen]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for g in gens:
        acc.setdefault(g.item_id, []).append(float(g.score.get("correct", False)))
    return {k: statistics.fmean(v) for k, v in acc.items()}


def health_summary(gens: list[Gen]) -> dict:
    h = [g.health for g in gens]
    return {
        "reasoning_closed": _mean(x["reasoning_closed"] for x in h),
        "answer_closed": _mean(x["answer_closed"] for x in h),
        "hit_max_tokens": _mean(x["hit_max_tokens"] for x in h),
        "median_reasoning_tokens": statistics.median(x["reasoning_tokens"] for x in h) if h else None,
        "repetition": _mean(x["repetition"] for x in h),
        "off_language": _mean(x["off_language"] for x in h),
    }


def paired_bootstrap(base: dict[str, float], cand: dict[str, float], n_boot: int = 10000, seed: int = 0) -> dict:
    """Difference in mean per-item accuracy (cand - base) with a 95% bootstrap CI over items."""
    ids = sorted(set(base) & set(cand))
    d = [cand[i] - base[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return {"delta": statistics.fmean(d), "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]), "n": len(ids)}


def gens_to_jsonl(path: str | Path, gens: list[Gen]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for g in gens:
            d = g.__dict__.copy()
            d.pop("prompt_ids")
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
