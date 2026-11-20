"""Does a 512-token reasoning generation give the same refusal label as the full one? (run 3 label pass check)

Run 3 labels its H-build prompts from reasoning-mode generations capped at 512 tokens. This replays that cap on
finished, already-judged traces (run 1 borderline: temperature 1, effort max; run 2 reasoning probes: greedy, effort
max) and compares the label it would give with the label from the full trace:
  full fits in the cap          -> identical output, same label (counted, not re-judged)
  reasoning longer than the cap -> Jev's heading question on the first CAP reasoning tokens (judge_cutoff)
  reasoning fits, answer cut    -> the normal judge on the cut answer, + the delivery pass if deflection-only
Truth = the full trace's flag, minus deflection-only flags whose content the delivery pass finds delivered (>= 0.7),
the same final label quick_probe / pe_capture use. Max-effort traces are longer than low-effort ones, so this is a
harder test than the run itself.

  uv run --env-file ../../../.env python tools/heading_check.py OUT_DIR --cap 512 FILE.jsonl [FILE.jsonl ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import JevJudge  # noqa: E402
from harness.response_judge import (DELIVERY_THRESHOLD, deflection_only, is_flagged, judge_cutoff,  # noqa: E402
                                    judge_delivery, judge_response)


def load(files: list[str], per_file: int | None, seed: int = 0) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    for f in files:
        xs = []
        for line in open(f):
            x = json.loads(line)
            if not x["health"].get("reasoning_closed", True) or not x.get("answer", "").strip():
                continue  # truth needs a finished trace
            if x.get("sample", 0) != 0:
                continue  # one sample per prompt
            xs.append({"id": x.get("id") or x.get("prompt_id"), "src": Path(f).stem, "split": x["split"],
                       "question": x["question"], "reasoning": x["reasoning"], "answer": x["answer"],
                       "verdict": x["verdict"], "flagged": bool(x["flagged"])})
        if per_file and len(xs) > per_file:
            xs = rng.sample(xs, per_file)
        rows += xs
    return rows


async def final_flag(j, q: str, reasoning: str, answer: str, v: dict | None = None) -> tuple[bool, dict]:
    if v is None:
        v = await judge_response(j, q, reasoning, answer)
    if deflection_only(v):
        d = await judge_delivery(j, q, answer)
        v = {**v, "delivered": d.get("delivered")}
    return is_flagged(v) and not (v.get("delivered") or 0) >= DELIVERY_THRESHOLD, v


async def one(j, tok, x: dict, cap: int) -> dict:
    r_ids = tok.encode(x["reasoning"], add_special_tokens=False)
    a_ids = tok.encode(x["answer"], add_special_tokens=False)
    budget = cap - 3  # think close / answer open markers
    truth, tv = await final_flag(j, x["question"], x["reasoning"], x["answer"], dict(x["verdict"]))
    if len(r_ids) + len(a_ids) <= budget:
        case, pred, pv = "fits", truth, tv
    elif len(r_ids) >= budget:
        case = "reasoning_cut"
        pv = await judge_cutoff(j, x["question"], tok.decode(r_ids[:budget]))
        pred = is_flagged(pv)
    else:
        case = "answer_cut"
        pred, pv = await final_flag(j, x["question"], x["reasoning"], tok.decode(a_ids[: budget - len(r_ids)]))
    return {"id": x["id"], "src": x["src"], "split": x["split"], "case": case, "reasoning_tokens": len(r_ids),
            "truth": truth, "pred": pred, "truth_verdict": tv, "pred_verdict": pv,
            "deflection": bool(x["flagged"]) and deflection_only(x["verdict"])}


async def main(out: Path, files: list[str], cap: int, per_file: int | None, concurrency: int) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("moonshotai/Kimi-K3.1-Pro", trust_remote_code=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(files, per_file)
    print(f"{len(rows)} finished traces from {len(files)} files", flush=True)
    path = out / "heading_check.jsonl"
    have = {(r["src"], r["id"]) for r in map(json.loads, open(path))} if path.exists() else set()
    todo = [x for x in rows if (x["src"], x["id"]) not in have]
    async with JevJudge(concurrency=concurrency) as j:
        with open(path, "a") as f:
            async def go(x):
                try:
                    r = await one(j, tok, x, cap)
                except Exception as e:  # noqa: BLE001
                    print(f"failed {x['id']}: {e!r}", flush=True)
                    return
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            done = 0
            for i in range(0, len(todo), 200):
                await asyncio.gather(*(go(x) for x in todo[i:i + 200]))
                done += len(todo[i:i + 200])
                print(f"{done}/{len(todo)}", flush=True)
    res = [json.loads(l) for l in open(path)]
    rep = {"cap": cap, "n": len(res), "cases": Counter(r["case"] for r in res)}
    for name, sel in [("all", res), ("cut_only", [r for r in res if r["case"] != "fits"]),
                      ("reasoning_cut", [r for r in res if r["case"] == "reasoning_cut"]),
                      ("answer_cut", [r for r in res if r["case"] == "answer_cut"]),
                      ("deflection_truth", [r for r in res if r["deflection"]]),
                      *[(f"src:{s}", [r for r in res if r["src"] == s]) for s in sorted({r["src"] for r in res})]]:
        if not sel:
            continue
        c = Counter((r["truth"], r["pred"]) for r in sel)
        rep[name] = {"n": len(sel), "agree": round(sum(r["truth"] == r["pred"] for r in sel) / len(sel), 4),
                     "refused_true": c[(True, True)], "refused_missed": c[(True, False)],
                     "answered_true": c[(False, False)], "answered_false_flag": c[(False, True)],
                     "truth_refusal_rate": round(sum(r["truth"] for r in sel) / len(sel), 4),
                     "pred_refusal_rate": round(sum(r["pred"] for r in sel) / len(sel), 4)}
    (out / "heading_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--per-file", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    asyncio.run(main(Path(a.out), a.files, a.cap, a.per_file, a.concurrency))
