"""Figures for the README (docs/figures/*.png) from committed results.

  uv run --with matplotlib python tools/make_figures.py
Sources: results/run3/probe_results.jsonl (run-3 screens and gates), docs/refusal_run2_results.md §3 (decision
readout AUCs, copied below), docs/refusal_run3_results.md.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = {r["name"]: r for r in map(json.loads, open(ROOT / "results/run3/probe_results.jsonl"))}

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
RAMP = ["#b3cdf0", "#7aa9e6", "#2a78d6", "#18508f"]  # sequential blue, shallow -> deep layers
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def pct(x):
    return 100 * x


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def bar_labels(ax, bars, fmt="{:.1f}%", dy=1.2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()), ha="center",
                va="bottom", fontsize=8.5, color=INK)


# 1. Headline: refusal by eval set, unedited vs edited, both modes ---------------------------------------------------
def by_set(name):
    """Harmful sets only: on harmless prompts the edited model refused none when every judge flag was read
    (README 2.5), so a harmless flag rate would show judge noise, not refusals."""
    r = R[name]["by_set"]
    wjb = (r["wjb_dev"]["final"] * r["wjb_dev"]["n"] + r["wjb_test"]["final"] * r["wjb_test"]["n"]) / (
        r["wjb_dev"]["n"] + r["wjb_test"]["n"])
    return {"HarmBench\n(200)": r["harmbench"]["final"], "WildJailbreak\nheld-out (550)": wjb,
            "All harmful\n(750)": R[name]["harmful_refusal_final"]}


fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (mode, b, e) in zip(axes, [("Non-thinking", "base_nt_gate", "final_h3_6-1_pc4_48_84_l1.5_nt"),
                                   ("Reasoning (effort low)", "base_think_gate", "final_h3_6-1_pc4_48_84_l1.5_th")]):
    ub, ue = by_set(b), by_set(e)
    xs = range(len(ub))
    w = 0.38
    b1 = ax.bar([x - w / 2 - 0.01 for x in xs], [pct(v) for v in ub.values()], w, color=GRAY, label="Unedited")
    b2 = ax.bar([x + w / 2 + 0.01 for x in xs], [pct(v) for v in ue.values()], w, color=BLUE, label="Edited (R, λ 1.5)")
    bar_labels(ax, b1, "{:.0f}%")
    bar_labels(ax, b2)
    ax.set_xticks(list(xs), list(ub.keys()))
    ax.set_ylim(0, 112)
    ax.set_title(mode, loc="left")
    grid(ax)
axes[0].set_ylabel("Harmful prompts refused (%)")
axes[1].legend(loc="upper right")
fig.suptitle("Refusal of harmful prompts on the held-out gate, greedy", x=0.07, ha="left", fontsize=12,
             fontweight="bold", color=INK, y=1.03)
save(fig, "fig1_gate_refusal.png")

# 2. Direction variants, same edit -------------------------------------------------------------------------------------
var = [("Run-1 R (plain pairs, answered-mean projection)", "bar_nt_v1H_48_84_l1.5"),
       ("Run-3 R, answered-mean projection", "h3_nt_h3_6-1_projected_48_84_l1.5"),
       ("Run-3 R, last token only", "h3_nt_h3_1_projected_ref_48_84_l1.5"),
       ("Run-3 R, last 4 tokens", "h3_nt_h3_4-1_projected_ref_48_84_l1.5"),
       ("Run-3 R, minus top-8 reference PCs", "h3_nt_h3_6-1_pc8_48_84_l1.5"),
       ("Run-3 R, minus reference mean", "h3_nt_h3_6-1_projected_ref_48_84_l1.5"),
       ("Run-3 R, minus top-4 reference PCs (chosen)", "h3_nt_h3_6-1_pc4_48_84_l1.5")]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ys = range(len(var))[::-1]
vals = [pct(R[n]["harmful_refusal_final"]) for _, n in var]
kls = [R[n]["kl"]["kl_per_token"] for _, n in var]
cols = [GRAY] + [BLUE] * (len(var) - 2) + [ORANGE]
bars = ax.barh(list(ys), vals, 0.62, color=cols)
for y, v, k in zip(ys, vals, kls):
    ax.text(v + 0.12, y, f"{v:.1f}%   KL {k:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(list(ys), [v[0] for v in var])
ax.set_xlim(0, 8.5)
ax.set_xlabel("Harmful prompts refused (%), non-thinking screen: 150 WildJailbreak + 100 XSTest")
ax.set_title("Same edit (layers 48–84, λ 1.5), different directions", loc="left")
grid(ax, "x")
save(fig, "fig2_direction_variants.png")

# 3. Strength curve on the full gate -----------------------------------------------------------------------------------
lams = [1.25, 1.5, 1.75]
nt = [pct(R[f"final_h3_6-1_pc4_48_84_l{l}_nt"]["harmful_refusal_final"]) for l in lams]
th = [pct(R[f"final_h3_6-1_pc4_48_84_l{l}_th"]["harmful_refusal_final"]) for l in lams]
kl = [R[f"final_h3_6-1_pc4_48_84_l{l}_th"]["kl"]["kl_per_token"] for l in lams]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))
a1.axhline(5, color=INK2, lw=1, ls=(0, (4, 3)))
a1.text(1.22, 5.1, "target 5%", fontsize=8.5, color=INK2, ha="left", va="bottom")
a1.plot(lams, nt, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
a1.plot(lams, th, color=ORANGE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
a1.text(1.79, nt[-1], "non-thinking", color=INK, fontsize=9, va="center")
a1.text(1.79, th[-1], "reasoning", color=INK, fontsize=9, va="center")
for x, a, b in zip(lams, nt, th):
    a1.text(x, a + 0.25, f"{a:.1f}", fontsize=8, color=INK2, ha="center")
    a1.text(x, b - 0.42, f"{b:.1f}", fontsize=8, color=INK2, ha="center")
a1.set_xlim(1.2, 1.97)
a1.set_ylim(2, 7.5)
a1.set_xticks(lams)
a1.set_xlabel("λ (edit strength)")
a1.set_ylabel("Harmful refused (%)")
a1.set_title("Refusal on the full gate", loc="left")
grid(a1)
a2.axhline(0.1, color=INK2, lw=1, ls=(0, (4, 3)))
a2.text(1.76, 0.101, "limit 0.1", fontsize=8.5, color=INK2, ha="right", va="bottom")
a2.plot(lams, kl, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
for x, k in zip(lams, kl):
    a2.text(x, k + 0.003, f"{k:.3f}", fontsize=8, color=INK2, ha="center")
a2.set_ylim(0, 0.115)
a2.set_xticks(lams)
a2.set_xlabel("λ (edit strength)")
a2.set_ylabel("KL, nats per token")
a2.set_title("Change on ordinary prompts", loc="left")
grid(a2)
fig.tight_layout()
save(fig, "fig3_strength_curve.png")

# 4. What can be removed from the edit ---------------------------------------------------------------------------------
ab = [("Full edit: layers 48–84, both writers, λ 1.5", "h3_nt_h3_6-1_pc4_48_84_l1.5"),
      ("λ 1.25", "p5_nt_pc4_48_84_l1.25"),
      ("Attention writers only (λ 1.25)", "p6_nt_pc4_48_84_l1.25_attn"),
      ("Layers 48–71 only", "p5_nt_pc4_48_71_l1.5"),
      ("Layers 60–84 only", "p5_nt_pc4_60_84_l1.5"),
      ("MoE writers only (λ 1.25)", "p6_nt_pc4_48_84_l1.25_mlp")]
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ys = range(len(ab))[::-1]
vals = [pct(R[n]["harmful_refusal_final"]) for _, n in ab]
kls = [R[n]["kl"]["kl_per_token"] for _, n in ab]
bars = ax.barh(list(ys), vals, 0.62, color=[ORANGE] + [BLUE] * (len(ab) - 1))
ax.axvline(5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(5.05, len(ab) - 0.45, "target 5%", fontsize=8, color=INK2)
for y, v, k in zip(ys, vals, kls):
    ax.text(v + 0.12, y, f"{v:.1f}%   KL {k:.3f}", va="center", fontsize=8.5, color=INK, zorder=3,
            bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"))
ax.set_yticks(list(ys), [a[0] for a in ab])
ax.set_xlim(0, 10.5)
ax.set_xlabel("Harmful prompts refused (%), non-thinking screen (the easier mode)")
ax.set_title("Shrinking the edit costs refusal", loc="left")
grid(ax, "x")
save(fig, "fig4_edit_ablations.png")

# 5. Where the decision is readable (run 2, non-thinking, cross-validated AUC) -----------------------------------------
pos = ["~40 tokens\nbefore end\n(mid-request)", "~17 before\n(last words\nof request)", "chat-template\ntokens\n(−13 to −1)",
       "prompt end\n(0)", "first answer\ntokens\n(+1 to +16)"]
auc = {"layer 8": [0.59, 0.74, 0.74, 0.79, 0.965], "layer 40": [0.59, 0.88, 0.95, 0.93, 0.98],
       "layer 56": [0.61, 0.90, 0.955, 0.98, 0.98], "layers 72–88": [0.59, 0.865, 0.96, 0.99, 0.98]}
fig, ax = plt.subplots(figsize=(8.6, 3.8))
for (lab, ys_), c in zip(auc.items(), RAMP):
    ax.plot(range(len(pos)), ys_, color=c, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=lab)
ax.legend(loc="lower right", fontsize=8.5, title="residual stream depth", title_fontsize=8.5)
ax.axhline(0.5, color=GRID, lw=1)
ax.text(0, 0.51, "chance", fontsize=8, color=INK2)
ax.axvspan(1.5, 3.5, color="#eef3fb", zorder=0)
ax.text(2.5, 0.53, "no answer token yet", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(pos)), pos, fontsize=8.5)
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(0.45, 1.02)
ax.set_ylabel("AUC: refused vs answered")
ax.set_title("The refusal decision is readable before the model writes anything", loc="left")
grid(ax)
save(fig, "fig5_decision_readout.png")

# 6. Capability panel --------------------------------------------------------------------------------------------------
pb = R["base_think_gate"]["panel"]
pe = R["final_h3_6-1_pc4_48_84_l1.5_th"]["panel"]
names = {"mmlu_pro": "MMLU-Pro (100)", "cmmlu": "CMMLU (50)", "ifeval": "IFEval (100)", "math500": "MATH-500 L5 (50)",
         "bfcl": "BFCL (50)"}
fig, ax = plt.subplots(figsize=(7.6, 2.9))
ys = range(len(names))[::-1]
for y, k in zip(ys, names):
    a, b = 100 * pb[k]["accuracy"], 100 * pe[k]["accuracy"]
    ax.plot([a, b], [y, y], color=GRID, lw=3, zorder=1)
    ax.scatter([a], [y], s=70, color=GRAY, zorder=2, edgecolor=SURF, linewidth=1.5)
    ax.scatter([b], [y], s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.text(max(a, b) + 0.6, y, f"{a:.0f} → {b:.0f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(list(ys), list(names.values()))
ax.set_xlim(84, 101)
ax.set_xlabel("Accuracy (%), reasoning effort low, greedy")
ax.scatter([], [], s=50, color=GRAY, label="Unedited")
ax.scatter([], [], s=50, color=BLUE, label="Edited (λ 1.5)")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.5)
ax.set_title("Capability panel: every set within 2 points", loc="left", pad=22)
grid(ax, "x")
save(fig, "fig6_capability_panel.png")
