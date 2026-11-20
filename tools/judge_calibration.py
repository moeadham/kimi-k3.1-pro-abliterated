"""Score hosted transcripts for judge threshold calibration (no early stopping).

Per transcript: one main Jev call (refused / deflected, harness/response_judge.py) and, when there is
reasoning, the onset question on EVERY reasoning sentence (same state as find_onset, all sentences
sent at once, no early stop). With all scores stored, both the response threshold and the onset
threshold can be swept offline against human labels, and onset false positives on non-refusals are
visible.

Usage: uv run --env-file .env python tools/judge_calibration.py in.jsonl out.jsonl [--max-sentences N]
Resumable: transcripts already in out.jsonl (by key) are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import judge_response, score_all_sentences  # noqa: E402


async def one(j, r, max_sentences):
    v = await judge_response(j, r["question"], r["reasoning"], r["content"])
    o = None
    if r["reasoning"].strip():
        o = await score_all_sentences(j, r["question"], r["reasoning"], max_sentences=max_sentences)
    return {**r, "verdict": v, "onset_all": o}


async def main(src: Path, out: Path, max_sentences: int, concurrency: int):
    rows = [json.loads(l) for l in open(src)]
    rows = list({r["key"]: r for r in rows if r.get("error") is None}.values())
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    print(f"{len(rows)} transcripts, {len(todo)} to score", flush=True)
    cost = errs = n = 0
    async with JevJudge(concurrency=concurrency) as j:
        with open(out, "a") as f:
            for fut in asyncio.as_completed([one(j, r, max_sentences) for r in todo]):
                r = await fut
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                o = r["onset_all"] or {}
                cost += r["verdict"]["cost"] + o.get("cost", 0)
                errs += len(r["verdict"]["errors"]) + len(o.get("errors", []))
                n += 1
                if n % 100 == 0:
                    print(f"{n}/{len(todo)} cost=${cost:.3f} errors={errs}", flush=True)
    print(f"done n={n} cost=${cost:.3f} errors={errs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-sentences", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    asyncio.run(main(a.src, a.out, a.max_sentences, a.concurrency))
