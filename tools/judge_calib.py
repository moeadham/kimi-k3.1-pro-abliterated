"""Calibrate the delivery second pass (response_judge.DELIVERY_Q) on run-2 probe generations.

Two checks: (1) unedited flagged harmful responses (genuine refusals) must stay flagged: the delivery score on them
gives the false-unflag rate per threshold; (2) deflection-only flags under the best edits: how many the second pass
would clear, with the texts written out for reading. Run with the probe gens downloaded locally:

  uv run --env-file ../../../.env python tools/judge_calib.py GENS_DIR OUT_DIR base_name edited_name...
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import THRESHOLD, judge_delivery  # noqa: E402


async def main(gdir: Path, out: Path, base: str, edited: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    load = lambda n: [json.loads(l) for l in open(gdir / f"{n}.jsonl")]  # noqa: E731
    rows = [("base", x) for x in load(base) if x["flagged"]]
    for n in edited:
        rows += [(n, x) for x in load(n) if x["flagged"]]
    async with JevJudge(concurrency=32) as j:
        res = await asyncio.gather(*(judge_delivery(j, x["question"], x["answer"]) for _, x in rows))
    recs = [{"run": n, "id": x["id"], "split": x["split"], "refused": x["verdict"].get("refused"),
             "deflected": x["verdict"].get("deflected"), "delivered": r.get("delivered"), "answer": x["answer"][:1500],
             "question": x["question"][:400]} for (n, x), r in zip(rows, res)]
    (out / "calib.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    lines = [f"cost ${sum(r['cost'] for r in res):.3f}, errors {sum(bool(r['errors']) for r in res)}"]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        b = [r for r in recs if r["run"] == "base" and r["split"] == "harmful"]
        bh = [r for r in b if (r["refused"] or 0) >= THRESHOLD]
        e = [r for r in recs if r["run"] != "base" and r["split"] == "harmful"]
        h = [r for r in recs if r["split"] == "harmless"]
        cl = lambda rs: sum((r["delivered"] or 0) >= t for r in rs)  # noqa: E731
        lines.append(f"delivered>={t}: base flagged harmful cleared {cl(b)}/{len(b)} (of them hard refusals "
                     f"{cl(bh)}/{len(bh)}); edited deflection flags cleared {cl(e)}/{len(e)}; harmless flags cleared "
                     f"{cl(h)}/{len(h)}")
    (out / "calib.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]))
