"""Run 3 configs (docs/run3_plan.md §5). Writes runs/run3/NAME.json + NAME.py (live-driver wrapper) per phase.

  python runs/run3/make_configs.py fixed                       # p1_base, labels, pe_build, p2_bar, p3_h3
  python runs/run3/make_configs.py think C1 C2 [...]           # p4_think: reasoning screens of candidates
  python runs/run3/make_configs.py push_lam C1 C2              # p5_push_lam: lambda 1 / 1.5 / 2 / 2.5 at 48-84
  python runs/run3/make_configs.py push_band C1@LAM C2@LAM     # p6_push_band: bands 48-71 / 60-84 / 36-84 at LAM
  python runs/run3/make_configs.py precision C@LAM@A-B [-n N]  # pN_prec: one change at a time from the point
  python runs/run3/make_configs.py final C@LAM@A-B[@opts] ...  # p9_final: full gate, both modes, panel, KL, T=1
A candidate C is BANK_KEY:FAMILY:VARIANT, BANK_KEY in BANKS (e.g. h3:h3/6:1:projected_ref, v1:prompt_end/all:projected).
Every config is greedy (temperature 0) except the final gate's T=1 reasoning check; reasoning runs at effort low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOL = "/vol/session_r3"
PROBE_OUT = f"{VOL}/probe"  # every probe phase shares it, so all edits pair with the unedited baselines by prompt
BANKS = {"v1": "/vol/session_v1_pairs/directions_refusal/directions.pt", "h3": "/vol/banks/r3_h3.pt",
         "r": "/vol/banks/r3_r.pt"}
NT_TOKENS, THINK_TOKENS, LABEL_TOKENS = 1024, 4096, 512

EVAL_SETS = [
    {"tag": "wjb_dev", "file": "data/refusal_v1/heldout_wjb_harmful.jsonl", "split": "harmful", "n": 150},
    {"tag": "wjb_test", "file": "data/refusal_v1/heldout_wjb_harmful.jsonl", "split": "harmful", "skip": 150, "n": 400},
    {"tag": "harmbench", "file": "results/harmbench_standard/prompts.jsonl", "split": "harmful"},
    {"tag": "xstest", "file": "results/xstest_safe/prompts.jsonl", "split": "harmless"},
    {"tag": "wjb_benign", "file": "data/refusal_v1/heldout_wjb_benign.jsonl", "split": "harmless"},
    {"tag": "xs_dev", "file": "results/xstest_safe/prompts.jsonl", "split": "harmless", "n": 100},
]
TIERS = {"screen": ["wjb_dev", "xs_dev"], "gate": ["wjb_dev", "wjb_test", "harmbench", "xstest", "wjb_benign"]}
BUILD_SETS = [
    {"tag": "build", "file": "data/refusal_v3/build_harmful.jsonl", "split": "harmful"},
    {"tag": "build", "file": "data/refusal_v3/build_harmless.jsonl", "split": "harmless"},
]
REFERENCE = {"tag": "reference", "file": "data/refusal_v3/reference_alpaca.jsonl", "split": "harmless"}


def probe_cfg(steps: list[dict], out: str = PROBE_OUT, sets=None, tiers=TIERS) -> dict:
    return {"out": out, "sets": sets or EVAL_SETS, "tiers": tiers, "temperature": 0.0, "effort": "low",
            "kl_n": 200, "kl_max_tokens": 1024, "steps": steps}


def cand(spec: str) -> dict:
    """BANK_KEY:FAMILY:VARIANT (FAMILY may contain ':', e.g. h3/6:1) -> edit component."""
    key, rest = spec.split(":", 1)
    family, variant = rest.rsplit(":", 1)
    return {"bank": BANKS[key], "family": family, "variant": variant}


def short(spec: str) -> str:
    key, rest = spec.split(":", 1)
    family, variant = rest.rsplit(":", 1)
    return f"{key}_{family.split('/', 1)[-1].replace(':', '-')}_{variant}"


def point(spec: str) -> tuple[str, float, list[int], dict]:
    """C@LAM@A-B[@writers=attn][@exclude=60-71] -> (C, lam, [A, B], extra step options)."""
    parts = spec.split("@")
    c, lam = parts[0], float(parts[1])
    layers = [int(x) for x in parts[2].split("-")] if len(parts) > 2 else [48, 84]
    extra = {}
    for p in parts[3:]:
        k, v = p.split("=", 1)
        extra[k] = [[int(a) for a in r.split("-")] for r in v.split(",")] if k == "exclude" else v
    return c, lam, layers, extra


def edit(name: str, c: str, lam: float, layers: list[int], mode: str, tier: str = "screen", **extra) -> dict:
    think = mode == "think"
    s = {"kind": "edit", "name": name, "components": [cand(c)], "layers": layers, "lam": lam, "tier": tier,
         "thinking": think, "max_tokens": THINK_TOKENS if think else NT_TOKENS,
         "compare_to": "base_think_gate" if think else "base_nt_gate", "kl": True, **extra}
    if think:
        s["effort"] = "low"
    return s


def write(name: str, cfg: dict) -> None:
    (HERE / f"{name}.json").write_text(json.dumps(cfg, indent=1) + "\n")
    (HERE / f"{name}.py").write_text(
        "import os\n"
        f'os.environ["{"K3_PE_CONFIG" if name == "pe_build" else "K3_PROBE_CONFIG"}"] = "/root/harness/runs/run3/{name}.json"\n'
        f'exec(compile(open("/root/harness/tools/{"pe_capture" if name == "pe_build" else "quick_probe"}.py").read(), '
        f'"{"pe_capture" if name == "pe_build" else "quick_probe"}.py", "exec"))\n')
    print(f"wrote runs/run3/{name}.json ({len(cfg.get('steps', []))} steps)")


H3_SCREEN = ["h3:h3/6:1:projected", "h3:h3/6:1:projected_ref", "h3:h3/6:1:pc4", "h3:h3/6:1:pc8",
             "h3:h3/1:projected_ref", "h3:h3/4:1:projected_ref"]
V1_H = "v1:prompt_end/all:projected"


def fixed() -> None:
    write("p1_base", probe_cfg([
        {"kind": "check"},
        {"kind": "kl_reference", "name": "kl_ref"},
        {"kind": "baseline", "name": "base_nt_gate", "tier": "gate", "thinking": False, "max_tokens": NT_TOKENS},
        {"kind": "baseline", "name": "base_think_gate", "tier": "gate", "thinking": True, "effort": "low",
         "max_tokens": THINK_TOKENS, "panel": True},
    ]))
    # label pass: the unedited model's own outcome on every build prompt, reasoning mode at low effort, <= 512 tokens,
    # judged as each finishes (cut-off traces on where they were heading; tools/heading_check.py)
    write("labels", probe_cfg([{"kind": "baseline", "name": "labels", "thinking": True, "effort": "low",
                                "max_tokens": LABEL_TOKENS}], out=f"{VOL}/labels", sets=BUILD_SETS, tiers={}))
    write("pe_build", {"out": f"{VOL}/pe_build", "effort": "low", "k": 6, "streams": [72, 187],
                       "labels_from": f"{VOL}/labels/gens/labels.jsonl", "ref_sets": ["reference"],
                       "sets": BUILD_SETS + [REFERENCE]})
    write("p2_bar", probe_cfg([edit("bar_nt_v1H_48_84_l1.5", V1_H, 1.5, [48, 84], "nt"),
                               edit("bar_th_v1H_48_84_l1.5", V1_H, 1.5, [48, 84], "think")]))
    write("p3_h3", probe_cfg([edit(f"h3_nt_{short(c)}_48_84_l1.5", c, 1.5, [48, 84], "nt") for c in H3_SCREEN]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fixed", "think", "push_lam", "push_band", "precision", "final"])
    ap.add_argument("cands", nargs="*")
    ap.add_argument("-n", type=int, default=1, help="precision round number (config name p7_prec_N)")
    ap.add_argument("--name", default="p9_final", help="config name for `final` (runs/recipe uses recipe_final)")
    a = ap.parse_args()
    if a.what == "fixed":
        return fixed()
    if a.what == "think":  # reasoning-mode screen of the best non-thinking candidates, same edit
        steps = []
        for spec in a.cands:
            c, lam, layers, extra = point(spec if "@" in spec else f"{spec}@1.5@48-84")
            steps.append(edit(f"th_{short(c)}_{layers[0]}_{layers[1]}_l{lam}", c, lam, layers, "think", **extra))
        return write("p4_think", probe_cfg(steps))
    if a.what == "push_lam":
        return write("p5_push_lam", probe_cfg([edit(f"push_nt_{short(c)}_48_84_l{lam}", c, lam, [48, 84], "nt")
                                               for c in a.cands for lam in (1.0, 2.0, 2.5)]))  # 1.5 is in p3/p2
    if a.what == "push_band":
        steps = []
        for spec in a.cands:
            c, lam, _, _ = point(spec)
            steps += [edit(f"push_nt_{short(c)}_{lo}_{hi}_l{lam}", c, lam, [lo, hi], "nt")
                      for lo, hi in ((48, 71), (60, 84), (36, 84))]
        return write("p6_push_band", probe_cfg(steps))
    if a.what == "precision":  # one change at a time from the chosen point (§3.5); non-thinking screen
        c, lam, (lo, hi), extra = point(a.cands[0])
        tag = f"prec{a.n}_{short(c)}"
        steps = [edit(f"{tag}_l{lam - 0.25}", c, lam - 0.25, [lo, hi], "nt", **extra)]
        starts = list(range(lo, hi + 1, 12))
        if len(starts) > 1 and hi - starts[-1] < 6:  # a short tail joins the block before it (48-84: 72-84)
            starts.pop()
        blocks = [(b, (starts[i + 1] - 1) if i + 1 < len(starts) else hi) for i, b in enumerate(starts)]
        steps += [edit(f"{tag}_drop{b0}-{b1}", c, lam, [lo, hi], "nt",
                       **{**extra, "exclude": extra.get("exclude", []) + [[b0, b1]]}) for b0, b1 in blocks]
        steps += [edit(f"{tag}_{w}only", c, lam, [lo, hi], "nt", **{**extra, "writers": w}) for w in ("attn", "mlp")
                  if extra.get("writers") is None]
        return write(f"p7_prec_{a.n}", probe_cfg(steps))
    if a.what == "final":  # the full gate on 1-2 accepted edits, plus a T=1 reasoning check vs unedited T=1
        steps = [{"kind": "baseline", "name": "base_think_screen_t1", "tier": "screen", "thinking": True,
                  "effort": "low", "max_tokens": THINK_TOKENS, "temperature": 1.0}]
        for spec in a.cands:
            c, lam, layers, extra = point(spec)
            nm = f"final_{short(c)}_{layers[0]}_{layers[1]}_l{lam}" + "".join(
                f"_{k}" + ("-".join(f"{a}to{b}" for a, b in v) if k == "exclude" else str(v))
                for k, v in sorted(extra.items()))
            steps += [edit(f"{nm}_nt", c, lam, layers, "nt", tier="gate", **extra),
                      {**edit(f"{nm}_th", c, lam, layers, "think", tier="gate", **extra), "panel": True},
                      {**edit(f"{nm}_th_t1", c, lam, layers, "think", **extra), "temperature": 1.0,
                       "compare_to": "base_think_screen_t1"}]
        return write(a.name, probe_cfg(steps))


if __name__ == "__main__":
    main()
