"""Calibrate the compliance-onset question (harness/response_judge.COMPLY_Q) on hosted K3.1 transcripts.

Scores COMPLY_Q on every reasoning sentence (no early stop) for a stratified sample of the already-judged
hosted transcripts in results/{xstest_safe,harmbench_standard}/calib.jsonl:
  - answered XSTest-safe   (should have a compliance onset)
  - answered HarmBench     (should have one; the rare harmful-but-answered cases)
  - refused HarmBench / XSTest (negatives: the question should NOT fire, at least not before the refusal
    onset, whose per-sentence scores are already stored in onset_all)
Answered/refused use the stored verdict (flag at response_judge.THRESHOLD).

Usage: uv run --env-file .env python tools/compliance_onset_calib.py results/compliance_onset/calib.jsonl [--n-answered 40] [--n-refused 15]
Resumable by key. Then: python tools/compliance_onset_calib.py out.jsonl --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import (COMPLY_Q, ONSET_THRESHOLD, is_flagged, score_all_sentences,  # noqa: E402
                                    split_sentences)

SRC = {"xstest": "results/xstest_safe/calib.jsonl", "harmbench": "results/harmbench_standard/calib.jsonl"}


def sample(n_answered: int, n_refused: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for ds, path in SRC.items():
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("reasoning", "").strip() and r.get("verdict")]
        ans = [r for r in rows if not is_flagged(r["verdict"])]
        ref = [r for r in rows if is_flagged(r["verdict"])]
        # one sample per prompt first (diversity), then the rest
        def pick(xs, n):
            rng.shuffle(xs)
            seen, first, rest = set(), [], []
            for x in xs:
                (rest if x["prompt_id"] in seen else first).append(x)
                seen.add(x["prompt_id"])
            return (first + rest)[:n]
        for cls, xs, n in (("answered", ans, n_answered), ("refused", ref, n_refused)):
            for r in pick(xs, n):
                out.append({**r, "ds": ds, "cls": cls})
    return out


async def run(out: Path, n_answered: int, n_refused: int, max_sentences: int, concurrency: int):
    todo = sample(n_answered, n_refused)
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = [r for r in todo if r["key"] not in done]
    print(f"{len(todo)} transcripts to score", flush=True)
    cost = errs = 0
    async with JevJudge(concurrency=concurrency) as j:
        with open(out, "a") as f:
            async def one(r):
                c = await score_all_sentences(j, r["question"], r["reasoning"], max_sentences, questions=COMPLY_Q)
                return {k: r[k] for k in ("key", "prompt_id", "split", "question", "reasoning", "content", "verdict",
                                          "onset_all", "ds", "cls")} | {"comply_all": c}
            for fut in asyncio.as_completed([one(r) for r in todo]):
                r = await fut
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                cost += r["comply_all"]["cost"]
                errs += len(r["comply_all"]["errors"])
    print(f"done cost=${cost:.3f} errors={errs}")


def first_at(scores, t):
    return next((i for i, p in enumerate(scores) if p is not None and p >= t), None)


def report(out: Path, show: int):
    rows = [json.loads(l) for l in open(out)]
    print(f"{len(rows)} transcripts")
    for t in (0.5, 0.7, 0.85, 0.9):
        line = []
        for ds in SRC:
            for cls in ("answered", "refused"):
                g = [r for r in rows if r["ds"] == ds and r["cls"] == cls]
                if not g:
                    continue
                hit = [first_at(r["comply_all"]["scores"], t) for r in g]
                n_hit = sum(h is not None for h in hit)
                extra = ""
                if cls == "refused":  # does compliance fire before the refusal onset?
                    before = 0
                    for r, h in zip(g, hit):
                        ro = first_at((r.get("onset_all") or {}).get("scores") or [], ONSET_THRESHOLD)
                        before += h is not None and (ro is None or h < ro)
                    extra = f", before refusal onset {before}"
                line.append(f"{ds}/{cls} {n_hit}/{len(g)}{extra}")
        print(f"t={t}: " + " | ".join(line))
    for r in rows[:show]:
        sp, sc = r["comply_all"]["spans"], r["comply_all"]["scores"]
        print(f"\n=== {r['ds']}/{r['cls']} {r['key']}\nQ: {r['question'][:200]}")
        for i, ((s, e), p) in enumerate(zip(sp, sc)):
            if i > 25:
                break
            print(f"  {i:2d} {p if p is None else f'{p:.2f}'}  {r['reasoning'][s:e].strip()[:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-answered", type=int, default=40)
    ap.add_argument("--n-refused", type=int, default=15)
    ap.add_argument("--max-sentences", type=int, default=80)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    if a.report:
        report(a.out, a.show)
    else:
        asyncio.run(run(a.out, a.n_answered, a.n_refused, a.max_sentences, a.concurrency))
