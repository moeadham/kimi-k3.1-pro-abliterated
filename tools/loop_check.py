"""Loop check at max reasoning effort (live driver): does the edited model still finish its reasoning on hard prompts?

Hard prompts (MATH-500 level 5, MMLU-Pro, IFEval; samples other than the capability panel's) at effort "max",
temperature 1, a long token cap, per condition (unedited, or a quick_probe-style edit). Per condition and set:
accuracy, reasoning closed rate, hit-the-cap rate, reasoning-length quantiles, repetition; every response saved.
Each (condition, set) is written the moment it finishes and skipped on a rerun.

Config JSON at $K3_LOOP_CONFIG:
  {"out": "/vol/session_r3/loop", "max_tokens": 16384, "n": {"math500": 40, "mmlu_pro": 30, "ifeval": 20},
   "conditions": [{"name": "unedited"}, {"name": "H_l1.5", "edit": {"components": [...], "layers": [48, 84], "lam": 1.5}}]}
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

CONFIG = os.environ.get("K3_LOOP_CONFIG", "/vol/session_r3/loop_config.json")


def items_for(name: str, n: int) -> list[dict]:
    from harness.evals import load_set
    from harness.session import ROOT
    xs = load_set(name, ROOT / "data/evals")
    if name == "math500":
        xs = [x for x in xs if int(x["level"]) == 5]
    return random.Random(1).sample(xs, min(n, len(xs)))  # seed 1: not the panel's seed-0 sample


def stats(gens) -> dict:
    rt = sorted(int(g.health.get("reasoning_tokens", 0)) for g in gens)
    q = lambda p: rt[min(len(rt) - 1, int(p * len(rt)))]  # noqa: E731
    return {"n": len(gens), "reasoning_closed": round(sum(bool(g.health.get("reasoning_closed", True)) for g in gens) / len(gens), 3),
            "hit_max_tokens": round(sum(g.finish == "length" for g in gens) / len(gens), 3),
            "reasoning_tokens_p50": q(0.5), "reasoning_tokens_p90": q(0.9), "reasoning_tokens_max": rt[-1],
            "mean_repetition": round(statistics.fmean(float(g.health.get("repetition", 0)) for g in gens), 4)}


def run(llm, tok, log, cfg: dict) -> list[dict]:
    import torch

    from harness import tp
    from harness.evals import Evaluator
    from harness.session import ROOT as root

    sys.path.insert(0, str(root / "tools"))
    import quick_probe as qp

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    mt = int(cfg.get("max_tokens", 16384))
    sets = cfg.get("n", {"math500": 40, "mmlu_pro": 30, "ifeval": 20})
    res = []
    for cond in cfg["conditions"]:
        edited = False
        try:
            for name, n in sets.items():
                path = out / f"{cond['name']}__{name}.json"
                if path.exists():
                    log(f"[loop] {path.name} exists, skipped")
                    res.append(json.loads(path.read_text()))
                    continue
                if cond.get("edit") and not edited:
                    step = cond["edit"]
                    banks = {c["bank"]: torch.load(c["bank"], map_location="cpu") for c in step["components"]}
                    Q = qp.step_basis(banks, step)
                    tp.edit_writers(llm, Q, qp.lam_for_layers(Q.shape[0], tuple(step["layers"]), float(step["lam"]),
                                                              step.get("writers", "both")))
                    edited = True
                    log(f"[loop] edit applied for {cond['name']}")
                t0 = time.time()
                ev = Evaluator(llm, tok, max_tokens=mt, thinking=True)
                r = ev.benchmark(name, items_for(name, int(n)), effort="max", temperature=1.0, max_tokens=mt)
                row = {"condition": cond["name"], "set": name, "accuracy": round(r["accuracy"], 3), **stats(r["gens"]),
                       "seconds": round(time.time() - t0)}
                with open(out / f"{cond['name']}__{name}.gens.jsonl", "w") as f:
                    for g in r["gens"]:
                        f.write(json.dumps({"id": g.item_id, "score": g.score, "finish": g.finish, "health": g.health,
                                            "reasoning": g.reasoning, "answer": g.answer}, ensure_ascii=False) + "\n")
                path.write_text(json.dumps(row))
                log(f"[loop] {json.dumps(row)}")
                res.append(row)
        finally:
            if edited:
                tp.restore_writers(llm)
    return res


if __name__ == "__live__":
    run(llm, tok, log, json.loads(Path(CONFIG).read_text()))  # noqa: F821
