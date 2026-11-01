"""Figure 1: change in Game24 test accuracy vs RL iteration, mean +- std over 5 RL seeds (seed-42 bases).
Data: runs/rl/game24_eval_curves.json (test/val evaluations from outputs/rl/*/log.jsonl)."""
import json, statistics as st
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

D = json.load(open("runs/rl/game24_eval_curves.json"))
RUNS = [  # (tag, label, color, marker) -- fixed categorical order
    ("game24cd_hier", "H-ELF (native)", "#2a78d6", "o"),
    ("game24cdAwd_flat", "AdamW+wd base", "#eb6834", "s"),
    ("game24cd_swapMA", "AdamW-trunk critic", "#1baf7a", "^"),
    ("game24cd_swapMM7", "indep. Muon critic", "#eda100", "D"),
    ("game24cd_flat", "native (Muon)", "#e87ba4", "v"),
]
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.25, 2.45))
ax.axhline(0, color="#52514e", lw=0.6, ls=(0, (2, 2)))
for tag, lab, c, mk in RUNS:
    rs = [D[f"{tag}_svg2d_s{s}"] for s in range(5)]
    it = [r["iter"] for r in rs[0]]
    d = [[100 * (r[i]["test_acc"] - r[0]["test_acc"]) for r in rs] for i in range(len(it))]
    mu = [st.mean(x) for x in d]; sd = [st.stdev(x) for x in d]
    ax.fill_between(it, [m - s for m, s in zip(mu, sd)], [m + s for m, s in zip(mu, sd)], color=c, alpha=0.15, lw=0)
    ax.plot(it, mu, color=c, lw=1.5, marker=mk, ms=3.5, label=lab)
ax.set_xlabel("RL iteration"); ax.set_ylabel(r"$\Delta$ test accuracy (points)")
ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500]); ax.set_xlim(-30, 1530)
ax.grid(axis="y", color="#e5e4e0", lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.6, columnspacing=0.9)
fig.tight_layout(pad=0.3)
fig.savefig("paper/figs/seed_curves.pdf"); fig.savefig("paper/figs/seed_curves.png", dpi=200)
