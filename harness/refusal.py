"""Refusal-direction dataset: harmful vs harmless prompts -> generate -> judge -> onset -> capture rows.

The pure parts (prompt loading, folds, turning judged generations into capture rows) live here so they
can be tested on CPU; harness.session.stage_refusal drives the GPU parts (generation, stream capture).

Each capture row has the same schema as a data/step2/replay_set.jsonl row, so the existing `directions`
stage reads the refusal captures unchanged (`directions --capture-from refusal`):
  censored      = the judge flagged the response (refused or deflected): the "bad" class
  dataset_split = "harmful" | "harmless" (the prompt's label; kept so prompt-label contrasts are possible)
  fold          = train/val by a stable hash of the prompt id, or of the `pair` id for matched twins (val ~20%)
  mixed_prompt  = the prompt's samples were both flagged and answered (--samples > 1)
  positions     = replay.Replay.positions(): prompt_end, reasoning_head, answer_head, + onset when flagged,
                  + comply_onset when answered (the sentence that settles on answering, response_judge.COMPLY_Q)

Why harmless rows matter: the refusal direction is a DIFFERENCE of means (harmful-refused minus
harmless-answered) at the same position type. The harmful mean alone is dominated by components every
prompt shares; ablating that would damage the model. The harmless mean is also what the `projected`
variant orthogonalizes against, and val AUC needs a negative class.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VAL_SHARE = 0.2


def _question(x: dict) -> str:
    for k in ("question", "prompt", "instruction", "goal", "behavior"):
        if x.get(k):
            return x[k]
    raise KeyError(f"no prompt text in {sorted(x)}")


def load_prompts(path: str | Path, split: str, n: int | None = None) -> list[dict]:
    """JSONL -> [{id, question, lang, split, source_split}]. Accepts question/prompt/instruction fields."""
    out = []
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        x = json.loads(line)
        q = _question(x)
        lang = x.get("lang") or ("zh" if any("一" <= c <= "鿿" for c in q) else "en")
        out.append({"id": f"{split}:{x.get('id', i)}", "question": q, "lang": lang, "split": split,
                    "source_split": x.get("kind") or x.get("split") or x.get("category"), "pair": x.get("pair")})
    if len({x["id"] for x in out}) != len(out):
        raise ValueError(f"duplicate ids in {path}")
    return out[:n] if n else out


def fold_key(item: dict) -> str:
    """What the fold is hashed on: the `pair` id when the prompt has a matched twin (both twins land in the same
    fold, so val never holds one side of a pair that train saw), else the prompt id."""
    return f"pair:{item['pair']}" if item.get("pair") else item["id"]


def mark_mixed(rows: list[dict]) -> int:
    """mixed_prompt = the prompt's samples include both flagged and answered responses (--samples > 1): the
    within-prompt contrast the directions stage uses (mixed/* families). In place; returns how many prompts."""
    outcomes: dict = {}
    for r in rows:
        outcomes.setdefault(r["prompt_id"], set()).add(bool(r["censored"]))
    mixed = {p for p, o in outcomes.items() if len(o) == 2}
    for r in rows:
        r["mixed_prompt"] = r["prompt_id"] in mixed
    return len(mixed)


def fold_of(prompt_id: str, val_share: float = VAL_SHARE) -> str:
    h = int(hashlib.sha1(prompt_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_share else "train"


def _chars(o: dict | None):
    if o and o.get("onset_sentence") is not None:
        return tuple(o["spans"][o["onset_sentence"]])
    return None


def capture_row(tok, item: dict, gen, verdict: dict, onset: dict | None, effort: str, sample: int,
                flagged: bool, comply: dict | None = None) -> dict:
    """One judged generation -> a replay-set-shaped row (ids, named positions, labels).
    `onset`: refusal-onset result (flagged rows); `comply`: compliance-onset result (answered rows)."""
    from harness.replay import build

    rep = build(tok, item["question"], gen.reasoning, gen.answer, effort=effort)
    pos = rep.positions(onset_chars=_chars(onset), comply_chars=_chars(comply))
    return {"key": f"{item['id']}|{effort}|{sample}", "prompt_id": item["id"], "dataset_split": item["split"],
            "source_split": item.get("source_split"), "lang": item["lang"], "effort": effort, "sample": sample,
            "censored": bool(flagged), "mixed_prompt": False, "fold": fold_of(fold_key(item)),
            "pair": item.get("pair"),
            "verdict": verdict, "onset_sentence": onset and onset.get("onset_sentence"),
            "comply_sentence": None if flagged else (comply or {}).get("onset_sentence"),
            "finish": gen.finish, "health": gen.health,
            "prompt_ids_match": rep.ids[: rep.prompt_len] == list(gen.prompt_ids),
            "prompt_len": rep.prompt_len, "n_tokens": len(rep.ids), "ids": rep.ids, "positions": pos}


GEN_FIELDS = ("item_id", "sample", "prompt_ids", "out_ids", "reasoning", "answer", "finish", "health")


def save_gens(path: str | Path, gens: list) -> None:
    """Checkpoint raw generations (before judging), so a judge/build failure never costs a regeneration."""
    with open(path, "w") as f:
        for g in gens:
            f.write(json.dumps({k: getattr(g, k) for k in GEN_FIELDS}, ensure_ascii=False) + "\n")


def load_gens(path: str | Path) -> list:
    import types

    return [types.SimpleNamespace(**json.loads(l)) for l in open(path)]


def gen_key(item_id: str, sample: int) -> str:
    return f"{item_id}#{sample}"


async def judge_one(judge, question: str, gen) -> dict:
    """One response: Jev verdict, then the refusal onset (flagged) or the compliance onset (answered)."""
    from harness.response_judge import (COMPLY_Q, COMPLY_THRESHOLD, find_onset, flag_score, is_flagged,
                                        judge_response)

    v = await judge_response(judge, question, gen.reasoning, gen.answer)
    fl = is_flagged(v)
    on = None
    if gen.reasoning.strip():
        on = (await find_onset(judge, question, gen.reasoning) if fl else
              await find_onset(judge, question, gen.reasoning, questions=COMPLY_Q, threshold=COMPLY_THRESHOLD))
    return {"key": gen_key(gen.item_id, gen.sample), "item_id": gen.item_id, "sample": gen.sample,
            "verdict": {**v, "flag_score": flag_score(v)}, "flagged": fl, "onset": on}


def load_judged(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return {x["key"]: x for x in map(json.loads, (l for l in open(p) if l.strip()))}


class StreamJudge:
    """Judge responses while generation continues. submit(gen) is non-blocking: the response goes to an asyncio
    loop in a background thread (its own Jev client, same concurrency limit as Evaluator._jev), and each
    finished judgment is appended to `path` (fsynced), so a stopped run never re-judges. Responses already in
    `path` are skipped. A judgment whose Jev verdict errored is not saved: judge_and_build retries it.
    close() waits for what is in flight (or cancels it with cancel=True) and returns {key: judgment}."""

    def __init__(self, items: list[dict], path: str | Path, judge_factory=None, log=print, judge_fn=None):
        """judge_fn(judge, question, gen) -> judgment dict with "key" and "verdict" (default judge_one)."""
        import asyncio
        import threading

        self.judge_fn = judge_fn or judge_one
        self.question = {x["id"]: x["question"] for x in items}
        self.path, self.log = Path(path), log
        self.done = load_judged(self.path)
        self.resumed = len(self.done)
        self.pending: dict = {}
        self.failed = 0
        if judge_factory is None:
            from harness.judge import JevJudge

            judge_factory = lambda: JevJudge(concurrency=64)  # noqa: E731
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="stream-judge", daemon=True)
        self.thread.start()
        self.judge = asyncio.run_coroutine_threadsafe(self._open(judge_factory), self.loop).result()
        self.f = open(self.path, "a")

    async def _open(self, factory):
        j = factory()
        if hasattr(j, "__aenter__"):
            await j.__aenter__()
        return j

    def submit(self, gen) -> None:
        import asyncio

        k = gen_key(gen.item_id, gen.sample)
        if k in self.done or k in self.pending:
            return
        self.pending[k] = asyncio.run_coroutine_threadsafe(self._one(gen), self.loop)

    async def _one(self, gen) -> None:
        import os

        try:
            rec = await self.judge_fn(self.judge, self.question[gen.item_id], gen)
        except Exception as e:  # noqa: BLE001 -- retried by judge_and_build
            self.failed += 1
            self.log(f"[judge] {gen.item_id}#{gen.sample} failed in the background: {e!r}")
            return
        if rec["verdict"].get("errors"):
            self.failed += 1
            return
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")  # only this loop's thread writes
        self.f.flush()
        os.fsync(self.f.fileno())
        self.done[rec["key"]] = rec

    async def _cancel_all(self) -> None:
        import asyncio

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def close(self, cancel: bool = False) -> dict:
        import asyncio
        from concurrent.futures import wait

        futs = list(self.pending.values())
        if cancel:  # in the loop, after every queued submit has become a task: cancel them and let them unwind
            asyncio.run_coroutine_threadsafe(self._cancel_all(), self.loop).result()
        else:
            if any(not fu.done() for fu in futs):
                self.log(f"[judge] waiting for {sum(not fu.done() for fu in futs)} judgments still in flight")
            wait(futs)
        if hasattr(self.judge, "__aexit__"):
            asyncio.run_coroutine_threadsafe(self.judge.__aexit__(None, None, None), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.f.close()
        return dict(self.done)


def judge_and_build(ev, tok, items: list[dict], gens: list, effort: str,
                    gens_path: str | Path | None, judged: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Judgments (judge_one) for every response, then capture rows. `judged` ({gen_key: judgment}, e.g. from
    StreamJudge) is reused; only responses missing from it are judged here (all concurrent under the judge's
    limit). Full texts + verdicts go to gens_path. A response whose canonical ids can't be rebuilt is returned in
    `failures`, not raised."""
    import asyncio

    by_id = {x["id"]: x for x in items}
    judged = dict(judged or {})
    missing = [g for g in gens if gen_key(g.item_id, g.sample) not in judged]
    if missing:
        async def run(j):
            return await asyncio.gather(*(judge_one(j, by_id[g.item_id]["question"], g) for g in missing))
        for rec in ev._jev(run):
            judged[rec["key"]] = rec
    rows, log, failures = [], [], []
    for g in gens:
        it = by_id[g.item_id]
        rec = judged[gen_key(g.item_id, g.sample)]
        v, fl, on = rec["verdict"], rec["flagged"], rec["onset"]
        try:
            rows.append(capture_row(tok, it, g, v, on if fl else None, effort, g.sample, fl,
                                    comply=None if fl else on))
        except Exception as e:  # noqa: BLE001
            failures.append({"stage": "build", "key": f"{it['id']}|{effort}|{g.sample}", "error": repr(e)})
        log.append({"prompt_id": it["id"], "split": it["split"], "question": it["question"], "sample": g.sample,
                    "reasoning": g.reasoning, "answer": g.answer, "finish": g.finish, "health": g.health,
                    "verdict": v, "flagged": fl, "onset": on if fl else None, "comply": None if fl else on})
    mark_mixed(rows)
    if gens_path:
        with open(gens_path, "w") as f:
            for x in log:
                f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    return rows, failures


def add_compliance_onsets(ev, tok, items: list[dict], gens: list, rows: list[dict]) -> int:
    """Upgrade rows judged before compliance onset existed: run COMPLY_Q on answered rows that have no
    `comply_sentence` key and add their comply_onset positions (in place). Returns how many were judged."""
    import asyncio

    from harness.replay import build
    from harness.response_judge import COMPLY_Q, COMPLY_THRESHOLD, find_onset

    by_id = {x["id"]: x for x in items}
    gen_of = {(g.item_id, g.sample): g for g in gens}
    todo = [r for r in rows if not r["censored"] and "comply_sentence" not in r]
    if not todo:
        return 0

    async def run(j):
        async def one(r):
            g = gen_of[(r["prompt_id"], r["sample"])]
            if not g.reasoning.strip():
                return None
            return await find_onset(j, by_id[r["prompt_id"]]["question"], g.reasoning, questions=COMPLY_Q,
                                    threshold=COMPLY_THRESHOLD)
        return await asyncio.gather(*(one(r) for r in todo))

    for r, c in zip(todo, ev._jev(run)):
        g = gen_of[(r["prompt_id"], r["sample"])]
        r["comply_sentence"] = (c or {}).get("onset_sentence")
        chars = _chars(c)
        if chars is not None:
            rep = build(tok, by_id[r["prompt_id"]]["question"], g.reasoning, g.answer, effort=r["effort"])
            r["positions"]["comply_onset"] = rep.positions(comply_chars=chars)["comply_onset"]
    return len(todo)


def trim_positions(position_sets: dict, per_type: int) -> dict:
    """At most `per_type` evenly spaced positions per type (what the directions stage reads anyway);
    0 = keep all. Storage is ~2.7 MB per position on K3.1 (187 stream indices x 7168 x bf16)."""
    if not per_type:
        return position_sets
    out = {}
    for k, ps in position_sets.items():
        if len(ps) <= per_type:
            out[k] = list(ps)
        elif per_type == 1:
            out[k] = [ps[-1]]
        else:
            out[k] = [ps[round(i * (len(ps) - 1) / (per_type - 1))] for i in range(per_type)]
    return out


def index_entry(i: int, row: dict, file: str, positions: list[int], agree,
                position_sets: dict | None = None) -> dict:
    """Capture index line, same schema as the capture stage's (read by session.load_capture_groups)."""
    return {"i": i, "key": row["key"], "file": file, "positions": positions, "rank_agreement": agree,
            "position_sets": position_sets or row["positions"], "censored": row["censored"], "lang": row["lang"],
            "fold": row["fold"], "mixed_prompt": bool(row.get("mixed_prompt")), "prompt_id": row["prompt_id"],
            "dataset_split": row["dataset_split"], "onset_sentence": row["onset_sentence"],
            "comply_sentence": row.get("comply_sentence")}


def truncate_for_capture(ids: list[int], positions: list[int]) -> list[int]:
    """The model is causal, so states at positions <= p do not depend on later tokens: prefill only up to
    the last captured position (saves prefill on long harmless answers)."""
    return ids[: max(positions) + 1]


def summarize(rows: list[dict]) -> dict:
    """Confusion of prompt label vs judge flag, plus position-set coverage."""
    s: dict = {}
    for r in rows:
        k = f"{r['dataset_split']}/{'flagged' if r['censored'] else 'answered'}"
        s[k] = s.get(k, 0) + 1
    cov = {}
    for r in rows:
        for pt, ps in r["positions"].items():
            if ps:
                cov[f"{r['dataset_split']}/{pt}"] = cov.get(f"{r['dataset_split']}/{pt}", 0) + 1
    return {"rows": len(rows), "label_vs_flag": s, "rows_with_position": cov,
            "truncated_generations": sum(r["finish"] == "length" for r in rows),
            "mixed_prompts": len({r["prompt_id"] for r in rows if r.get("mixed_prompt")}),
            "prompt_ids_mismatch": sum(not r["prompt_ids_match"] for r in rows)}
