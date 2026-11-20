"""Re-judge reasoning traces cut off by the token limit: is the reasoning heading toward refusing?

A probe generation whose reasoning never closed has an empty answer, and the normal response judge (refused /
deflected on the RESPONSE, reasoning capped at 8,000 chars) then mostly flags it as a refusal. This asks one Jev
question per trace about the direction of the full reasoning instead (response_judge.HEADING_Q, which
quick_probe now applies to cut-off traces itself). Validated first on finished traces with the
answer hidden: agreement with the normal verdict on the actual answer says how far the reasoning-only call can be
trusted.

  uv run --env-file ../../../.env python tools/rejudge_truncated.py out/run2/rejudge/gens/*.jsonl \
      --out out/run2/rejudge [--validate 60]
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
from harness.response_judge import HEADING_Q, THRESHOLD, cutoff_state  # noqa: E402

def closed(x: dict) -> bool:
    return bool(x["health"].get("reasoning_closed", True))


async def score(j, x: dict) -> dict:
    r = await j.decide(cutoff_state(x["question"], x["reasoning"]), HEADING_Q)
    s = None if r["error"] else r["answers"].get("heading_to_refuse", {}).get("noul")
    return {"id": x["id"], "split": x["split"], "score": s, "error": r["error"], "cost": r["cost"]}


async def main(files: list[str], out: Path, n_validate: int, concurrency: int = 32) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs = {Path(f).stem: [json.loads(l) for l in open(f)] for f in files}
    rng = random.Random(0)
    # validation sample: finished traces with a non-empty reasoning, balanced by the normal verdict
    fin = [(n, x) for n, xs in runs.items() for x in xs if closed(x) and x["reasoning"].strip()]
    ref = [p for p in fin if p[1]["flagged"]]
    ans = [p for p in fin if not p[1]["flagged"]]
    val = rng.sample(ref, min(n_validate // 2, len(ref))) + rng.sample(ans, min(n_validate // 2, len(ans)))
    cut = [(n, x) for n, xs in runs.items() for x in xs if not closed(x)]
    print(f"validation {len(val)} finished traces, re-judge {len(cut)} cut-off traces", flush=True)
    async with JevJudge(concurrency=concurrency) as j:
        vres = await asyncio.gather(*(score(j, x) for _, x in val))
        cres = await asyncio.gather(*(score(j, x) for _, x in cut))
    cost = sum(r["cost"] for r in vres + cres)
    errs = [r for r in vres + cres if r["score"] is None]
    # validation: reasoning-only call vs the normal verdict on the actual answer
    agree = tp = fp = fn = tn = 0
    rows = []
    for (n, x), r in zip(val, vres):
        if r["score"] is None:
            continue
        pred, truth = r["score"] >= THRESHOLD, bool(x["flagged"])
        agree += pred == truth
        tp += pred and truth; tn += (not pred) and (not truth); fp += pred and not truth; fn += truth and not pred
        rows.append({**r, "run": n, "kind": "validate", "verdict_flagged": truth})
    nv = tp + tn + fp + fn
    lines = [f"Validation on {nv} finished traces (answer hidden): agreement {agree}/{nv} = {agree / max(nv, 1):.0%}",
             f"  refused & predicted refusing {tp}, answered & predicted complying {tn}, "
             f"answered but predicted refusing {fp}, refused but predicted complying {fn}", ""]
    # corrected rates per run: finished traces keep the normal verdict, cut-off ones use the heading call
    by = {}
    for (n, x), r in zip(cut, cres):
        by[(n, x["id"])] = r
        rows.append({**r, "run": n, "kind": "cutoff", "verdict_flagged": bool(x["flagged"])})
    lines.append("run  | split | n | finished: refused/n | cut off: heading to refuse / complying / unscored "
                 "| old refusal rate | corrected refusal rate")
    for n, xs in runs.items():
        for split in ("harmful", "harmless"):
            s = [x for x in xs if x["split"] == split]
            f_ = [x for x in s if closed(x)]
            c_ = [by[(n, x["id"])] for x in s if not closed(x)]
            hr = sum(r["score"] is not None and r["score"] >= THRESHOLD for r in c_)
            hc = sum(r["score"] is not None and r["score"] < THRESHOLD for r in c_)
            un = sum(r["score"] is None for r in c_)
            old = sum(bool(x["flagged"]) for x in s) / len(s)
            new = (sum(bool(x["flagged"]) for x in f_) + hr) / max(len(s) - un, 1)
            lines.append(f"{n} | {split} | {len(s)} | {sum(bool(x['flagged']) for x in f_)}/{len(f_)} | "
                         f"{hr} / {hc} / {un} | {old:.1%} | {new:.1%}")
    lines.append(f"\nJev cost ${cost:.3f}, unscored {len(errs)}")
    (out / "rejudge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "rejudge.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="out/run2/rejudge")
    ap.add_argument("--validate", type=int, default=60)
    a = ap.parse_args()
    asyncio.run(main(a.files, Path(a.out), a.validate))
