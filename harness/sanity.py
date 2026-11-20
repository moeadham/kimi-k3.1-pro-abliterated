"""Sanity panel definition ("did the edit lobotomize the model?"), shared by the hosted
reference run (tools/sanity_panel_hosted.py) and our engine (run_on_engine), so both use the
exact same items. Coarse by design: ~350 items, catches large drops, not 1-2 point shifts.
Reference (unedited, hosted wafer/us, 2026-11-18): results/sanity_panel_hosted/summary.json.
"""

from __future__ import annotations

import random

from harness.evals import load_set

PANEL = [("mmlu_pro", 100, None), ("cmmlu", 50, None), ("ifeval", 100, None),
         ("math500", 50, lambda x: int(x["level"]) == 5), ("bfcl", 50, None)]
EFFORT = "low"


def panel_items() -> list[tuple[str, dict]]:
    out = []
    for name, n, filt in PANEL:
        items = [x for x in load_set(name) if filt is None or filt(x)]
        out += [(name, it) for it in random.Random(0).sample(items, min(n, len(items)))]
    return out


def run_on_engine(ev, kl_ref=None) -> dict:
    """ev: harness.evals.Evaluator on our vLLM engine (steering/edit state set by the caller)."""
    by_set: dict[str, list[dict]] = {}
    for name, it in panel_items():
        by_set.setdefault(name, []).append(it)
    res = {}
    for name, items in by_set.items():
        r = ev.bfcl(items, effort=EFFORT) if name == "bfcl" else ev.benchmark(name, items, effort=EFFORT)
        r.pop("gens", None)
        res[name] = r
    if kl_ref is not None:
        res["kl"] = ev.kl(kl_ref)
    return res


# ---------------- pass / fail ----------------

THRESHOLDS = {
    "max_accuracy_drop": 0.05,      # per set, absolute (5 points)
    "max_kl_per_token": 0.10,       # nats, whole-response KL vs unedited (engine runs only)
    "max_hit_max_tokens_rise": 0.02,
    "max_empty_answer_rise": 0.02,
    "max_repetition_rise": 0.05,
    "max_off_language_rise": 0.05,
    "max_reasoning_closed_drop": 0.02,
}


def compare(base: dict, cand: dict, thresholds: dict | None = None) -> dict:
    """base/cand: {set: {"accuracy": float, "health": {...}}, optional "kl": {...}} as produced by
    run_on_engine() or the "sets" block of the hosted summary. Returns per-check results + verdict."""
    t = {**THRESHOLDS, **(thresholds or {})}
    checks = []

    def add(name, value, limit, ok):
        checks.append({"check": name, "value": value, "limit": limit, "pass": bool(ok)})

    for s in sorted(k for k in base if k != "kl" and k in cand):
        b, c = base[s], cand[s]
        drop = b["accuracy"] - c["accuracy"]
        add(f"{s}.accuracy_drop", round(drop, 4), t["max_accuracy_drop"], drop <= t["max_accuracy_drop"])
        bh, ch = b.get("health") or {}, c.get("health") or {}
        for key, lim, sign in (("hit_max_tokens", "max_hit_max_tokens_rise", 1), ("empty_answer", "max_empty_answer_rise", 1),
                               ("repetition", "max_repetition_rise", 1), ("off_language", "max_off_language_rise", 1),
                               ("reasoning_closed", "max_reasoning_closed_drop", -1)):
            if key in bh and key in ch and bh[key] is not None and ch[key] is not None:
                delta = sign * (float(ch[key]) - float(bh[key]))
                add(f"{s}.{key}_{'rise' if sign > 0 else 'drop'}", round(delta, 4), t[lim], delta <= t[lim])
    if "kl" in cand:
        kl = cand["kl"]["kl_per_token"]
        add("kl_per_token", round(kl, 4), t["max_kl_per_token"], kl <= t["max_kl_per_token"])
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "thresholds": t}
