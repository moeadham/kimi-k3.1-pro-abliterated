"""Judge hosted K3.1 transcripts (tools/gen_hosted.py output) with Jev, then summarize.

Per transcript: one Jev call (refused / deflected, harness/response_judge.py);
for flagged responses with reasoning, a batched onset pass over reasoning sentences.
`censored` = flagged by the judge. Verdicts from an older judge are replaced.
Usage: uv run --env-file .env python tools/judge_hosted.py [in.jsonl] [out_dir]
"""

from __future__ import annotations

import asyncio
import collections
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import RESPONSE_Q, THRESHOLD, find_onset, is_flagged, judge_response  # noqa: E402

CJK = re.compile(r"[一-鿿]")


def lang(q: str) -> str:
    return "zh" if CJK.search(q) else "en"


async def judge_all(rows):
    async with JevJudge(concurrency=64) as j:
        todo = [r for r in rows if "refused" not in r.get("verdict", {})]
        for r in todo:
            r.pop("onset", None)
        verdicts = await asyncio.gather(*(judge_response(j, r["question"], r["reasoning"], r["content"]) for r in todo))
        for r, v in zip(todo, verdicts):
            r["verdict"] = v
        for r in rows:
            r["censored"] = is_flagged(r["verdict"])
        need = [r for r in rows if r["censored"] and r["reasoning"].strip() and "onset" not in r]
        onsets = await asyncio.gather(*(find_onset(j, r["question"], r["reasoning"]) for r in need))
        for r, o in zip(need, onsets):
            r["onset"] = o


def summarize(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        cond = f"{r['provider']}|{r['effort']}|T{r['temperature']}"
        groups[(r["split"], lang(r["question"]), cond)].append(r)
    out = []
    for (split, lg, cond), rs in sorted(groups.items()):
        v = [r["verdict"] for r in rs]
        rtok = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for r in rs]
        ons = [r["onset"]["onset_char"] for r in rs if r["censored"] and r.get("onset") and r["onset"]["onset_char"] is not None]
        onset_sent = [r["onset"]["onset_sentence"] for r in rs if r["censored"] and r.get("onset") and r["onset"]["onset_sentence"] is not None]
        out.append({
            "split": split, "lang": lg, "cond": cond, "n": len(rs),
            "censored": round(sum(r["censored"] for r in rs) / len(rs), 3),
            **{f"{k}>={THRESHOLD}": round(sum(x.get(k, 0) >= THRESHOLD for x in v) / len(v), 3) for k in RESPONSE_Q},
            "median_reasoning_tokens": statistics.median(rtok),
            "onset_found": len(ons), "median_onset_sentence": statistics.median(onset_sent) if onset_sent else None,
            "median_onset_char": statistics.median(ons) if ons else None,
        })
    return out


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "out/hosted_deccp/transcripts.jsonl")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else src.parent)
    rows = [json.loads(l) for l in open(src)]  # a judged.jsonl re-scores without new verdict calls
    rows = [r for r in rows if r.get("error") is None]
    rows = list({r["key"]: r for r in rows}.values())
    asyncio.run(judge_all(rows))
    with open(out_dir / "judged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarize(rows)
    cost = sum(r["verdict"]["cost"] + (r.get("onset") or {}).get("cost", 0) for r in rows)
    errs = sum(len(r["verdict"]["errors"]) + len((r.get("onset") or {}).get("errors", [])) for r in rows)
    json.dump({"rows": len(rows), "jev_cost": cost, "jev_errors": errs, "groups": summ},
              open(out_dir / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"rows={len(rows)} jev_cost=${cost:.3f} jev_errors={errs}")
    for g in summ:
        print(json.dumps(g, ensure_ascii=False))


if __name__ == "__main__":
    main()
