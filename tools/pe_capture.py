"""Prompt-end capture without generation, with a one-token outcome label (live driver), for run 3's H rebuild.

Run 2 showed the refusal decision is readable at the prompt end and committed at the first answer token: with
reasoning off, P(first answer token = "I") separates refused from answered at AUC 0.98. So, per prompt:
  pass 1 (labels, batched): non-thinking prompt, one greedy token with the top-k logprobs -> p_refuse_start
         (probability mass on tokens that start a refusal, "I" by default) and the top-k list; or labels taken
         from a probe's judged generations (`labels_from`, e.g. residual refusals under an edit)
  pass 2 (capture): the reasoning-template prompt at `effort` (the mode being edited), prefill only, the last `k`
         prompt positions, stream indices `streams` [a, b); one safetensors per row (atomic), resumable; the
         decoded token at each captured position is recorded (which template tokens H sits on)
Optionally under an edit (`edit`: a quick_probe edit step), restored in a `finally`.
Rows come from the same `sets` config as tools/quick_probe.py (same ids: "<split>:<tag>:<id>").

Config JSON at $K3_PE_CONFIG:
  {"out": "/vol/session_r3/pe", "sets": [{"tag": "build", "file": "data/refusal_v3/build_harmful.jsonl",
   "split": "harmful"}, ...], "effort": "low", "k": 8, "streams": [40, 187], "label_top": 20,
   "refuse_start": ["I"], "edit": null, "labels_from": null, "ref_sets": ["reference"]}

Residual direction (docs/run3_plan.md 3.3): run tools/quick_probe.py with the best edit on the build harmful set
(judged gens), then this with the same "edit", "labels_from": that gens file and the reference set included, then
tools/pe_bank.py --label jev: residual (still refusing / deflecting under the edit) vs answered under the edit.

  modal_gpu.py put tools/pe_capture.py /root/harness/tools/pe_capture.py
  modal_gpu.py live "py /root/harness/tools/pe_capture.py"
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

CONFIG = os.environ.get("K3_PE_CONFIG", "/vol/session_r3/pe_config.json")


def refuse_mass(top: list[tuple[str, float]], starts: list[str]) -> float:
    """Probability mass of the first answer token on refusal starts (decoded token, stripped)."""
    return sum(math.exp(lp) for t, lp in top if t.strip() in starts)


def label_pass(llm, tok, items: list[dict], top_k: int, starts: list[str], out: Path) -> dict:
    """{id: {"p_refuse_start", "greedy", "top"}} from one non-thinking greedy token per prompt (batched)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from harness import k3chat

    path = out / "labels.jsonl"
    have = {x["id"]: x for x in map(json.loads, open(path))} if path.exists() else {}
    todo = [it for it in items if it["id"] not in have]
    if todo:
        prompts = [TokensPrompt(prompt_token_ids=k3chat.prompt_ids(tok, it["question"], thinking=False)) for it in todo]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_k), use_tqdm=False)
        with open(path, "a") as f:
            for it, o in zip(todo, outs):
                c = o.outputs[0]
                lps = c.logprobs[0] if c.logprobs else {}
                top = sorted(((tok.decode([tid]), float(getattr(v, "logprob", v))) for tid, v in lps.items()),
                             key=lambda x: -x[1])
                row = {"id": it["id"], "p_refuse_start": round(refuse_mass(top, starts), 5),
                       "greedy": tok.decode(list(c.token_ids)), "top": [[t, round(lp, 4)] for t, lp in top]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                have[it["id"]] = row
    return have


def run(llm, tok, log, cfg: dict) -> dict:
    import torch

    from harness import k3chat, refusal, tp
    from harness.session import ROOT as root
    from harness.session import _file_positions

    sys.path.insert(0, str(root / "tools"))  # exec'd by the live driver: no __file__
    import quick_probe as qp

    out = Path(cfg["out"])
    (out / "states").mkdir(parents=True, exist_ok=True)
    items = qp.load_eval_items(cfg, lambda path, split: refusal.load_prompts(root / path, split))
    if cfg.get("max_rows"):
        items = items[: int(cfg["max_rows"])]
    k, (s0, s1) = int(cfg.get("k", 8)), cfg.get("streams", [0, 187])
    streams = list(range(int(s0), int(s1)))
    t0 = time.time()
    edited = False
    index, streak = [], 0
    # rewritten each run, one fsynced line per captured row (a rerun re-lists rows whose files exist, without
    # recapturing them), so the index on disk always matches the states files even after a hard stop
    idx_f = open(out / "index.jsonl", "w")
    try:
        if cfg.get("edit"):
            step = cfg["edit"]
            banks = {c["bank"]: torch.load(c["bank"], map_location="cpu") for c in step.get("components") or [step]}
            Q = qp.step_basis(banks, step)
            lam = qp.lam_for_layers(Q.shape[0], tuple(step["layers"]), float(step.get("lam", 1.0)),
                                    step.get("writers", "both"), step.get("exclude"), step.get("lam_by_layer"))
            tp.edit_writers(llm, Q, lam)
            edited = True
            log(f"[pe] edit applied: {step.get('name', step.get('family'))}")
        if cfg.get("labels_from"):  # judged probe generations (same ids): label = Jev flag
            src = {x["id"]: x for x in map(json.loads, open(cfg["labels_from"]))}
            # residual = the final verdict: flagged, unless the delivery pass found the content delivered
            # cut-off traces (512-token label pass) also carry the heading score: pe_bank drops the ambiguous band
            from harness.response_judge import flag_score
            labels = {i: {"flagged": bool(src[i]["flagged"]) and not (
                (src[i].get("verdict") or {}).get("delivered") or 0) >= 0.7, "verdict": src[i].get("verdict"),
                "cutoff": bool((src[i].get("verdict") or {}).get("cutoff")),
                "score": flag_score(src[i].get("verdict") or {})} for i in src}
            keep = set(cfg.get("ref_sets", ["reference"]))  # the projection reference is captured unlabelled
            items = [it for it in items if it["id"] in labels or it.get("set") in keep]
        else:
            labels = label_pass(llm, tok, items, int(cfg.get("label_top", 20)), cfg.get("refuse_start", ["I"]), out)
        log(f"[pe] labels for {len(labels)} rows ({time.time() - t0:.0f}s); capturing {len(items)}")
        effort = cfg.get("effort", "low")
        for n, it in enumerate(items):
            ids = k3chat.prompt_ids(tok, it["question"], effort=effort, thinking=True)
            positions = list(range(max(len(ids) - k, 0), len(ids)))
            path = out / "states" / f"{n:05d}.safetensors"
            try:
                have = _file_positions(path, None) if path.exists() else None
                if have is None or list(have) != positions:
                    tp.capture_one(llm, ids, positions, str(path), streams=streams)
                streak = 0
            except Exception as e:  # noqa: BLE001
                log(f"[pe] capture FAILED {it['id']}: {e!r}")
                streak += 1
                if streak >= 3:
                    log("[pe] 3 capture failures in a row: stopping (files so far are kept)")
                    break
                continue
            row = {"id": it["id"], "set": it.get("set"), "split": it["split"],
                   "source_split": it.get("source_split"), "pair": it.get("pair"), "file": path.name,
                   "positions": positions, "prompt_len": len(ids), "streams": [s0, s1], "effort": effort,
                   "tokens": [tok.decode([t]) for t in ids[-k:]], **labels.get(it["id"], {})}
            index.append(row)
            idx_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            idx_f.flush()
            os.fsync(idx_f.fileno())
            if n % 500 == 0:
                log(f"[pe] captured {n}/{len(items)} ({time.time() - t0:.0f}s)")
    finally:
        if edited:
            tp.restore_writers(llm)
        idx_f.close()
    res = {"rows": len(index), "planned": len(items), "seconds": round(time.time() - t0), "edited": edited}
    log(f"[pe] done {json.dumps(res)}")
    return res


if __name__ == "__live__":
    run(llm, tok, log, json.loads(Path(CONFIG).read_text()))  # noqa: F821
