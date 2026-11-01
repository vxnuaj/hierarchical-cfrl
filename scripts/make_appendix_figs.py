"""All appendix figures and generated tables of the paper.

Inputs (all in the repo):
  runs/rl/rl_logs_all.json.gz   per-iteration RL logs of every run (outputs/rl/*/log.jsonl, 'ks' field dropped)
  runs/analysis/*.json, *.pt    analysis outputs (outputs/analysis on the cluster)
  runs/theory/*.json            theory toy outputs (hcfrl/theory)
Outputs: paper/figs/app_*.pdf and paper/gen_*.tex.
usage: python scripts/make_appendix_figs.py
"""
import gzip, json, math, statistics as st

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
                     "axes.titlesize": 8, "legend.fontsize": 6.5, "legend.frameon": False})
FIG = "paper/figs/"
C = {"hier": "#2a78d6", "awd": "#eb6834", "ma": "#1baf7a", "mm7": "#eda100", "native": "#e87ba4",
     "violet": "#6250d6", "green": "#008300", "red": "#e34948", "grey": "#8a8984"}
MK = {"hier": "o", "awd": "s", "ma": "^", "mm7": "D", "native": "v"}
GRID = dict(color="#e5e4e0", lw=0.5)

D = json.load(gzip.open("runs/rl/rl_logs_all.json.gz", "rt"))


def dedup(rows):
    """Resumed runs re-log iterations; keep the last record per (iter, kind)."""
    tr, ev = {}, {}
    for r in rows:
        (ev if r.get("test_acc") is not None else tr)[r["iter"]] = r
    return [tr[k] for k in sorted(tr)], [ev[k] for k in sorted(ev)]


def evals(run):
    return dedup(D[run])[1]


def train(run):
    return dedup(D[run])[0]


def dtest(run):
    e = evals(run)
    return [r["iter"] for r in e], [100 * (r["test_acc"] - e[0]["test_acc"]) for r in e]


def smooth(x, w=25):
    x = np.array([np.nan if v is None else v for v in x], float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1): i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg): out[i] = seg.mean()
    return out


def finish(ax, ylab=r"$\Delta$ test acc. (points)", zero=True, xl="RL iteration"):
    if zero: ax.axhline(0, color="#52514e", lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel(xl); ax.set_ylabel(ylab); ax.grid(axis="y", **GRID); ax.set_axisbelow(True)


HEAD = [("native", "game24cd_flat", "native critic (Muon policy)"), ("mm7", "game24cd_swapMM7", "independent-Muon critic"),
        ("ma", "game24cd_swapMA", "AdamW-trunk critic"), ("awd", "game24cdAwd_flat", "AdamW+wd policy, own critic"),
        ("hier", "game24cd_hier", "H-ELF policy, own critic")]

# ---------------------------------------------------------------- A1: per-seed trajectories
fig, axs = plt.subplots(2, 3, figsize=(6.75, 3.9), sharey=True)
for ax, (k, tag, title) in zip(axs.flat, HEAD):
    for s in range(5):
        it, d = dtest(f"{tag}_svg2d_s{s}")
        ax.plot(it, d, color=C[k], lw=1.0, marker=MK[k], ms=2.5, alpha=0.35 + 0.13 * s)
    ax.set_title(title); finish(ax)
ax = axs.flat[5]
for tag, lab, col, ls in [("game24cd_flat_ft43_svg2d_s0", "native, s43 base", C["native"], "-"), ("game24cd_flat_ft44_svg2d_s0", "native, s44 base", C["native"], "--"),
                          ("game24cd_swapMA43_svg2d_s0", "AdamW trunk, s43", C["ma"], "-"), ("game24cd_swapMA44_svg2d_s0", "AdamW trunk, s44", C["ma"], "--"),
                          ("game24cd_hier_ft43_svg2d_s0", "H-ELF, s43", C["hier"], "-"), ("game24cd_hier_ft44_svg2d_s0", "H-ELF, s44", C["hier"], "--")]:
    it, d = dtest(tag); ax.plot(it, d, color=col, ls=ls, lw=1.2, label=lab)
ax.set_title("other Muon bases (one RL seed)"); finish(ax); ax.legend(ncol=2, loc="lower center", fontsize=5.4, columnspacing=0.6, handlelength=1.5)
ax.set_ylim(-36, 9)
fig.text(0.5, 0.995, "line opacity increases with RL seed (0 to 4)", ha="center", va="top", fontsize=6.5)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.97)); fig.savefig(FIG + "app_seeds.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A2: closed-loop variables
fig, axs = plt.subplots(2, 3, figsize=(6.75, 3.8))
VARS = [("buf_succ", "buffer success (verified)", False), ("critic_auc", "critic AUC on new states", False),
        ("obj", "actor objective (critic logit)", True), ("grad_norm", "actor gradient norm", True), ("tok_norm", "generated-token norm", False),
        ("_gcos", "cos(actor gradient, previous iteration)", False)]
for ax, (v, lab, logy) in zip(axs.flat, VARS):
    if v == "_gcos":
        for t, lab_, col in [("game24cd_flat_gcos_s0", "native", C["native"]), ("game24cd_swapMA_gcos_s0", "AdamW trunk", C["ma"])]:
            tr = train(t); ax.plot([r["iter"] for r in tr], smooth([r.get("gcos_prev") for r in tr]), color=col, lw=1.1, label=lab_)
        ax.set_title(lab, fontsize=6.8); finish(ax, ylab="", zero=False); ax.legend(fontsize=5.6); continue
    for k, tag, title in HEAD:
        curves = []
        for s in range(5):
            tr = train(f"{tag}_svg2d_s{s}")
            if not any(v in r and r[v] is not None for r in tr): continue
            x = smooth([r.get(v) for r in tr]); curves.append(x[:1500])
        if not curves: continue
        n = min(len(c) for c in curves); m = np.nanmean(np.stack([c[:n] for c in curves]), 0)
        y = np.abs(m) if logy else m
        ax.plot(np.arange(n), y, color=C[k], lw=1.1, label=f"{title} ({len(curves)})")
    if logy: ax.set_yscale("log")
    ax.set_title(lab); finish(ax, ylab="", zero=False)
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90)); fig.savefig(FIG + "app_loop.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A3: interventions on the native model
fig, axs = plt.subplots(1, 3, figsize=(6.75, 3.0), sharey=True)
ax = axs[0]
for tag, lab, col in [("game24cd_flat_svg2d_lr1e-7_s0", r"lr $10^{-7}$", C["green"]), ("game24cd_flat_svg2d_lr3e-7_s0", r"lr $3{\times}10^{-7}$", C["violet"]),
                      ("game24cd_flat_svg2d_s0", r"lr $10^{-6}$ (publ.)", C["native"])]:
    it, d = dtest(tag); ax.plot(it, d, color=col, lw=1.2, marker="o", ms=2.5, label=lab)
ax.set_title("actor step size"); finish(ax); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=5.6)
ax = axs[1]
for tag, lab, col in [("game24cd_flat_svg2d_np10_s0", r"norm penalty $\lambda{=}10$", C["violet"]), ("game24cd_flat_svg2d_np100_s0", r"norm penalty $\lambda{=}100$", C["red"]),
                      ("game24cd_flat_svg2d_relstep_s0", "layer-wise relative step", C["green"]), ("game24cd_flat_svg2d_rsa1e-3_s0", "sharpness-aware objective", C["grey"]),
                      ("game24cd_flat_svg2d_s0", "published recipe", C["native"])]:
    it, d = dtest(tag); ax.plot(it, d, color=col, lw=1.2, marker="o", ms=2.5, label=lab)
ax.set_title("step and critic regularisers"); finish(ax, ylab=""); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=5.6)
ax = axs[2]
for pre, lab, col in [("game24cd_flat_bigcrit_svg2d_s", "critic on 16k rollouts", C["violet"]), ("game24cd_flat_svg2d_refit8_s", r"$4\times$ critic refit", C["green"]),
                      ("game24cd_flat_svg2d_s", "published recipe", C["native"])]:
    for s in range(3):
        it, d = dtest(f"{pre}{s}"); ax.plot(it, d, color=col, lw=1.0, alpha=0.8, marker="o", ms=2, label=lab if s == 0 else None)
ax.set_title("more critic training (3 seeds each)"); finish(ax, ylab=""); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=5.6)
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_nofix.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A4: margin (final delta vs actor lr)
fig, ax = plt.subplots(figsize=(3.3, 2.3))
def fin(t): return dtest(t)[1][-1]
mu = lambda tags: (st.mean(fin(t) for t in tags), st.stdev(fin(t) for t in tags) if len(tags) > 1 else 0)
flat = {1e-7: ["game24cd_flat_svg2d_lr1e-7_s0"], 3e-7: ["game24cd_flat_svg2d_lr3e-7_s0"], 1e-6: [f"game24cd_flat_svg2d_s{s}" for s in range(5)]}
hier = {1e-7: ["game24cd_hier_svg2d_lr1e-7_s0"], 3e-7: ["game24cd_hier_svg2d_lr3e-7_s0"], 1e-6: [f"game24cd_hier_svg2d_s{s}" for s in range(5)], 3e-6: ["game24cd_hier_svg2d_lr3e-6_s0"]}
adam = {1e-6: [f"game24cdA_flat_svg2d_s{s}" for s in range(3)], 1e-5: ["game24cdA_flat_svg2d_lr1e-5_s0"]}
for dd, lab, col, mk in [(flat, "flat Muon", C["native"], "v"), (hier, "H-ELF Muon", C["hier"], "o"), (adam, "flat AdamW", C["awd"], "s")]:
    xs = sorted(dd); ms = [mu(dd[x]) for x in xs]
    ax.errorbar(xs, [m for m, _ in ms], yerr=[s for _, s in ms], color=col, marker=mk, ms=4, lw=1.3, capsize=2, label=lab)
ax.set_xscale("log"); finish(ax, ylab=r"final $\Delta$ (points)", xl="actor learning rate")
ax.annotate("AdamW at $10^{-5}$: $-37.9$ at it. 750", (1e-5, -37.9), fontsize=5.8, xytext=(1.3e-6, -30), arrowprops=dict(arrowstyle="-", lw=0.5))
ax.legend(loc="lower left")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_margin.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A5: failure modes
CATS = [("ok", "correct", C["ma"]), ("arith", "false equation", C["native"]), ("operand", "number not given", C["mm7"]),
        ("target", "wrong target", C["violet"]), ("n_steps", "wrong #steps", C["hier"]), ("parse", "unparsable", C["grey"])]
fig, axs = plt.subplots(1, 4, figsize=(6.75, 2.2), sharey=True)
for ax, (f, title) in zip(axs, [("fm_game24cd_flat", "Game24, flat"), ("fm_game24cd_hier", "Game24, H-ELF"), ("fm_kk_flat", "K&K, flat"), ("fm_kk_hier", "K&K, H-ELF")]):
    d = json.load(open(f"runs/analysis/{f}.json")); its = sorted(d, key=int); x = np.arange(len(its)); bot = np.zeros(len(its))
    for key, lab, col in CATS:
        y = np.array([100 * d[i]["cats"].get(key, 0.0) for i in its])
        if y.sum() == 0: continue
        lab2 = "wrong assignment (K&K)" if (key == "target" and "kk" in f) else lab
        ax.bar(x, y, bottom=bot, color=col, width=0.8, edgecolor="white", lw=0.5, label=lab2); bot += y
    ax.set_xticks(x); ax.set_xticklabels(its, rotation=60, fontsize=6); ax.set_title(title); ax.set_xlabel("RL iteration")
axs[0].set_ylabel("% of 500 validation outputs")
hl = {}
for a in axs:
    for h_, l_ in zip(*a.get_legend_handles_labels()): hl.setdefault(l_, h_)
fig.legend(list(hl.values()), list(hl.keys()), loc="upper center", ncol=7, fontsize=6, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9)); fig.savefig(FIG + "app_failmodes.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A6: frozen attacks (accuracy + cross-critic logits)
ATT = [("attack600_native", "own Muon (native)", C["native"]), ("attack600_mm43", "sibling Muon", C["violet"]), ("attack600_ma", "AdamW", C["ma"]),
       ("attack600_mm7", "independent Muon", C["mm7"]), ("attack600_hier_native", "H-ELF own", C["hier"])]
NAMES = [("native", "own"), ("mm43", "sibling"), ("mm7", "indep."), ("big", "own 16k"), ("ma", "AdamW"), ("Anat", "AdamW pol.")]
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
for f, lab, col in ATT:
    d = json.load(open(f"runs/analysis/{f}.json")); h = d["hist"]
    axs[0].plot([r["iter"] for r in h], [100 * r["acc"] for r in h], color=col, marker="o", ms=2.5, lw=1.2, label=lab)
    att = d["attacker"] if d["attacker"] in h[0] else list(h[0].keys())[2]
    axs[1].plot([r["iter"] for r in h], [r[att]["auc"] for r in h], color=col, marker="o", ms=2.5, lw=1.2)
axs[0].set_title("accuracy under frozen ascent"); finish(axs[0], ylab="val. accuracy (%)", zero=False, xl="ascent step"); axs[0].legend(fontsize=5.6)
axs[1].set_title("attacked critic's AUC"); finish(axs[1], ylab="AUC", zero=False, xl="ascent step")
M = np.full((4, 6), np.nan)
for i, (f, _, _) in enumerate(ATT[:4]):
    h = json.load(open(f"runs/analysis/{f}.json"))["hist"]
    for j, (n, _) in enumerate(NAMES):
        if n in h[0] and n in h[-1]: M[i, j] = (h[-1][n]["logit"] - h[0][n]["logit"]) / abs(h[0][n]["logit"])
im = axs[2].imshow(M, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
axs[2].set_xticks(range(6)); axs[2].set_xticklabels([b for _, b in NAMES], rotation=40, fontsize=6)
axs[2].set_yticks(range(4)); axs[2].set_yticklabels([b for _, b, _ in ATT[:4]], fontsize=6)
for i in range(4):
    for j in range(6):
        if not np.isnan(M[i, j]): axs[2].text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=5.2, color="white" if abs(M[i, j]) > 0.8 else "black")
axs[2].set_title("relative logit change at step 600\n(rows: attacked critic; columns: scoring critic)", fontsize=6.5)
for sp in axs[2].spines.values(): sp.set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_attack.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A7: T1 self-consistency along the native attack
d = json.load(open("runs/analysis/T1_selfconsistency_native.json"))["hist"]
it = [r["iter"] for r in d]
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.2))
axs[0].plot(it, [100 * r["acc"] for r in d], color=C["grey"], marker="o", ms=2.5, label="accuracy (%)")
ax2 = axs[0].twinx(); ax2.plot(it, [r["sc_residual"] for r in d], color=C["violet"], marker="s", ms=2.5); ax2.set_ylabel(r"residual $\rho$", color=C["violet"])
ax2.spines["right"].set_visible(True); axs[0].set_title(r"accuracy and residual $\rho$"); finish(axs[0], ylab="val. accuracy (%)", zero=False, xl="ascent step")
for n, lab, col in [("native", "own (attacked)", C["native"]), ("mm7", "indep. Muon", C["mm7"]), ("ma", "AdamW", C["ma"])]:
    axs[1].plot(it, [r["sc_corr_logit"][n] for r in d], color=col, marker="o", ms=2.5, label=lab)
    axs[2].plot(it, [r[n]["logit"] for r in d], color=col, marker="o", ms=2.5, label=lab)
axs[1].plot(it, [r["sc_corr_reward"] for r in d], color=C["grey"], ls="--", label="verified reward")
axs[1].set_title(r"corr$(-\rho,\cdot)$ over probe samples"); finish(axs[1], ylab="correlation", xl="ascent step"); axs[1].legend(fontsize=5.6)
axs[2].set_title("mean critic logit"); finish(axs[2], ylab="logit", xl="ascent step")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_t1.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A8: T2 refit feedback + refit churn + agreement
t2 = json.load(open("runs/analysis/T2_refit_feedback.json"))
churn = json.load(open("runs/analysis/refit_churn_g24.json"))
mo = json.load(open("runs/analysis/multi_onestep_g24.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
CR = [("native", "own Muon", C["native"]), ("mm7", "indep. Muon", C["mm7"]), ("ma", "AdamW", C["ma"]), ("Anat", "AdamW pol.", C["awd"])]
w = 0.38
for j, (n, lab, col) in enumerate(CR):
    for e, (key, hatch) in enumerate([("0.5", ""), ("1", "////")]):
        axs[0].bar(j + (e - 0.5) * w, t2[n][key]["gamma_v_mean"], w, color=col, hatch=hatch, edgecolor="white", lw=0.5)
axs[0].set_xticks(range(4)); axs[0].set_xticklabels([l for _, l, _ in CR], rotation=25, fontsize=6)
axs[0].set_title(r"refit feedback $\Gamma_v$ (plain: $\epsilon{=}0.5$, hatched: 1)", fontsize=6.8); finish(axs[0], ylab=r"$\Gamma_v$ (mean over 64 probes)", xl="")
for j, (n, lab, col) in enumerate(CR):
    v = churn[n]
    axs[1].bar(j - w / 2, st.mean(v["cos_to_original"]), w, color=col, edgecolor="white", lw=0.5)
    axs[1].bar(j + w / 2, st.mean(v["cos_between_refits"]), w, color=col, hatch="////", edgecolor="white", lw=0.5)
axs[1].set_xticks(range(4)); axs[1].set_xticklabels([l for _, l, _ in CR], rotation=25, fontsize=6); axs[1].set_ylim(0, 1)
axs[1].set_title("input-gradient cosine after 200-step refits\n(plain: to original; hatched: between refits)", fontsize=6.5); finish(axs[1], ylab="cosine", zero=False, xl="")
names = [n for n in ["native", "mm43", "mm7", "big", "ma", "Anat"] if n in mo["cos_raw"]]
lab = {"native": "own", "mm43": "sibling", "mm7": "indep.", "big": "own 16k", "ma": "AdamW", "Anat": "AdamW pol."}
M = np.array([[mo["cos_raw"][a][b] for b in names] for a in names])
axs[2].imshow(M, cmap="Blues", vmin=0, vmax=1)
for i in range(len(names)):
    for j in range(len(names)): axs[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 0.6 else "black")
axs[2].set_xticks(range(len(names))); axs[2].set_xticklabels([lab[n] for n in names], rotation=40, fontsize=6)
axs[2].set_yticks(range(len(names))); axs[2].set_yticklabels([lab[n] for n in names], fontsize=6)
axs[2].set_title("cosine between critics' mean\npathwise gradients (parameter space)", fontsize=6.5)
for sp in axs[2].spines.values(): sp.set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_refit.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A9: pull effects and one-step tests
pe = json.load(open("runs/analysis/pull_effects_g24.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.2))
PN = [("native", "own", C["native"]), ("mm43", "sibling", C["violet"]), ("mm7", "indep.", C["mm7"]), ("big", "own 16k", C["grey"]), ("ma", "AdamW", C["ma"]), ("Anat", "AdamW pol.", C["awd"])]
for n, lab_, col in PN:
    sizes = [0.01, 0.03, 0.1]
    st_ = [pe["24"]["steps"].get(f"{n}@{s}") for s in sizes]
    if None in st_: continue
    axs[0].plot(sizes, [x["dlogit"][n] for x in st_], color=col, marker="o", ms=3, label=lab_)
    axs[1].plot(sizes, [100 * x["dacc"] for x in st_], color=col, marker="o", ms=3)
axs[0].set_xscale("log"); axs[1].set_xscale("log")
axs[0].set_title("own-logit gain along the pull ($k{=}24$)", fontsize=6.6); finish(axs[0], ylab=r"$\Delta$ logit", zero=False, xl="displacement (fraction of token norm)")
axs[1].set_title("accuracy change along the pull", fontsize=6.8); finish(axs[1], ylab=r"$\Delta$ acc. (points)", xl="displacement (fraction of token norm)")
axs[0].legend(fontsize=5.5)
steps = mo["steps"]; xs = []
for j, (n, lab_, col) in enumerate(PN):
    for e, (sz, hatch) in enumerate([("0.25", ""), ("1", "////")]):
        r = steps.get(f"{n}@{sz}")
        if r is None: continue
        val = r.get("dacc", r.get("acc", 0) - mo["acc0"]) if isinstance(r, dict) else r
        axs[2].bar(j + (e - 0.5) * 0.38, 100 * val, 0.38, color=col, hatch=hatch, edgecolor="white", lw=0.5)
axs[2].set_xticks(range(len(PN))); axs[2].set_xticklabels([l for _, l, _ in PN], rotation=35, fontsize=6)
axs[2].set_title("one actor step along each critic\n" + r"($\Lambda/4$ plain, $\Lambda$ hatched)", fontsize=6.4); finish(axs[2], ylab=r"$\Delta$ acc. (points)", xl="")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_pull.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A10: spectra / fingerprint
import torch
S = torch.load("runs/analysis/spectra_g24.pt", map_location="cpu", weights_only=False)
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.3), gridspec_kw=dict(width_ratios=[1.4, 1]))
DL = [("dM_g24", "Muon s42", C["native"]), ("dM43_g24", "Muon s43", C["violet"]), ("dM7_g24", "Muon s7", C["mm7"]), ("dA_g24", "AdamW", C["ma"]), ("dAwd_g24", "AdamW+wd", C["awd"]),
      ("drl_nat1500", "RL, native (it 1500)", C["grey"]), ("drl_ma1500", "RL, AdamW trunk (it 1500)", C["hier"])]
rng = np.random.default_rng(0)
for j, (k, lab_, col) in enumerate(DL):
    if k not in S["deltas"]: continue
    v = [m["stable_rank"] / m["n"] for m in S["deltas"][k].values() if isinstance(m, dict) and "stable_rank" in m]
    axs[0].scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=col, alpha=0.6)
    axs[0].plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="black", lw=1)
axs[0].set_xticks(range(len(DL))); axs[0].set_xticklabels([l for _, l, _ in DL], rotation=25, fontsize=6)
axs[0].set_yscale("log"); axs[0].set_title(r"stable rank $/n$ of fine-tuning updates $\Delta W$, per matrix (line: median)", fontsize=6.6)
finish(axs[0], ylab=r"$\|\Delta W\|_F^2/(\sigma_1^2 n)$", zero=False, xl="")
ML = [("M_g24", "Muon s42", C["native"]), ("M43_g24", "Muon s43", C["violet"]), ("M7_g24", "Muon s7", C["mm7"]), ("A_g24", "AdamW", C["ma"]), ("owt", "OWT init.", C["grey"])]
for j, (k, lab_, col) in enumerate(ML):
    if k not in S["models"]: continue
    v = [m["stable_rank"] / m["n"] for m in S["models"][k].values() if isinstance(m, dict) and "stable_rank" in m]
    axs[1].scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=col, alpha=0.6)
    axs[1].plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="black", lw=1)
axs[1].set_xticks(range(len(ML))); axs[1].set_xticklabels([l for _, l, _ in ML], rotation=25, fontsize=6); axs[1].set_yscale("log")
axs[1].set_title(r"stable rank $/n$ of the trained weights $W$", fontsize=6.6); finish(axs[1], ylab=r"$\|W\|_F^2/(\sigma_1^2 n)$", zero=False, xl="")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_spectra.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A11: theory toys
wh = json.load(open("runs/theory/whitening_v1.json"))
dose = json.load(open("runs/theory/dose_lam0.001.json"))
hc = json.load(open("runs/theory/head_churn_lam0.001.json"))
lt = json.load(open("runs/theory/loop_toy.json"))
fig, axs = plt.subplots(1, 4, figsize=(6.75, 2.1))
OPT = [("sgd", "SGD", C["hier"]), ("adamw", "AdamW", C["ma"]), ("muon", "Muon", C["native"])]
for k, lab_, col in OPT:
    rows = [r for r in wh if r["kind"] == k and r.get("blend", 1.0) == 1.0]
    lams = sorted({r["lam_off"] for r in rows})
    ys = [st.mean(r["ratio"] for r in rows if r["lam_off"] == l) for l in lams]
    axs[0].plot(lams, ys, color=col, marker="o", ms=3, label=lab_)
axs[0].set_xscale("log"); axs[0].invert_xaxis(); axs[0].set_title(r"off/on input gain (MLP)", fontsize=6.8); finish(axs[0], ylab="off/on gain", zero=False, xl=r"off-manifold variance $\lambda_{\rm off}$"); axs[0].legend(fontsize=5.6)
axs[1].plot([r["blend"] for r in dose], [r["ratio"] for r in dose], color=C["violet"], marker="o", ms=3)
axs[1].set_title("dose: share of polar update", fontsize=6.8); finish(axs[1], ylab="off/on gain", zero=False, xl="blend (0 = AdamW, 1 = Muon)")
for k, lab_, col in OPT:
    ks = sorted([key for key in hc if key.startswith(k + "_")], key=lambda s: int(s.split("_")[1]))
    axs[2].plot([int(s.split("_")[1]) for s in ks], [hc[s]["cos"] for s in ks], color=col, marker="o", ms=3, label=lab_)
axs[2].set_xscale("log"); axs[2].set_title("head refits: gradient cosine", fontsize=6.8); finish(axs[2], ylab="cos between refits", zero=False, xl="refit samples")
for k, lab_, col in OPT:
    for mode, ls in [("refit", "-"), ("frozen", "--")]:
        h = lt.get(f"{k}_{mode}")
        if h: axs[3].plot([r["iter"] for r in h], [r["reward"] for r in h], color=col, ls=ls, lw=1.1, label=f"{lab_}, {mode}" if k == "muon" or mode == "refit" else None)
axs[3].set_ylim(-3, 1.2); axs[3].set_title("toy actor-critic loop", fontsize=6.8); finish(axs[3], ylab="true reward", zero=False, xl="iteration"); axs[3].legend(fontsize=5.2, loc="lower left")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_theory_toys.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A12: hierarchy controls
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.3), sharey=False)
def band(ax, tags, col, lab):
    ds = [dtest(t)[1] for t in tags]; it = dtest(tags[0])[0]; n = min(len(x) for x in ds); m = np.mean([x[:n] for x in ds], 0)
    ax.plot(it[:n], m, color=col, marker="o", ms=2.5, lw=1.2, label=f"{lab} ({len(tags)})")
    if len(tags) > 1:
        s_ = np.std([x[:n] for x in ds], 0, ddof=1); ax.fill_between(it[:n], m - s_, m + s_, color=col, alpha=0.15, lw=0)
band(axs[0], [f"game24cd_hier_svg2d_s{s}" for s in range(5)], C["hier"], r"$\Gamma_{\rm task}$")
band(axs[0], [f"game24cdK_hier_svg2d_s{s}" for s in range(3)], C["violet"], r"$\Gamma_{\rm kmeans}$")
band(axs[0], [f"game24cd_rand_svg2d_s{s}" for s in range(3)], C["mm7"], r"$\Gamma_{\rm rand}$ (base 70)")
band(axs[0], [f"game24cd_randm_svg2d_s{s}" for s in range(3)], C["grey"], r"$\Gamma_{\rm rand}$ (base 58.5)")
band(axs[0], [f"game24cd_flat_svg2d_s{s}" for s in range(5)], C["native"], "flat")
axs[0].set_title("coarse map (mean $\\pm$ s.d.)"); finish(axs[0]); axs[0].legend(fontsize=5.4, loc="lower right")
band(axs[1], [f"game24cd_hier_svg2d_s{s}" for s in range(5)], C["hier"], "native critic")
band(axs[1], [f"game24cd_hier_blind_svg2d_s{s}" for s in range(3)], C["green"], "critic blind to code")
band(axs[1], [f"game24cd_hier_bigcrit_svg2d_s{s}" for s in range(3)], C["violet"], "16k-rollout critic")
band(axs[1], ["game24cd_hier_svg2d_tokgrad_s0"], C["mm7"], "grad. via tokens only")
band(axs[1], ["game24cd_hier_svg2d_clgrad_s0"], C["grey"], "grad. via code only")
band(axs[1], [f"game24cd_swapHMA_svg2d_s{s}" for s in range(2)], C["red"], "AdamW H-ELF trunk")
axs[1].set_title("H-ELF critic and gradient controls"); finish(axs[1], ylab=""); axs[1].legend(fontsize=5.4, loc="lower left")
ct = json.load(open("runs/analysis/cons_trace_ft43_hier.json"))
runs_ = sorted({k.split("/")[0] for k in ct})
for r, col in zip(runs_, [C["hier"], C["mm7"], C["violet"], C["green"]]):
    ks = sorted([k for k in ct if k.startswith(r + "/")], key=lambda s: int(s.split("/")[1]))
    kk = "40" if "40" in ct[ks[0]] else list(ct[ks[0]])[0]
    axs[2].plot([int(k.split("/")[1]) for k in ks], [ct[k][kk]["cons"] for k in ks], color=col, marker="o", ms=2.5, label=r.replace("game24cd_", "").replace("_svg2d", ""))
axs[2].set_title("code–token agreement, s43 H-ELF runs", fontsize=6.8); finish(axs[2], ylab="agreement", zero=False); axs[2].legend(fontsize=5.4)
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_hier.pdf"); plt.close(fig)

# ---------------------------------------------------------------- A13: K&K and Countdown
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.2))
band(axs[0], [f"kk_flat_svg2d_s{s}" for s in range(3)], C["native"], "flat, native critic")
band(axs[0], [f"kk_swapMM7_svg2d_s{s}" for s in range(3)], C["mm7"], "flat, indep.-Muon critic")
band(axs[0], [f"kk_hier_svg2d_s{s}" for s in range(3)], C["hier"], "H-ELF, native critic")

axs[0].set_title("Knights & Knaves"); finish(axs[0]); axs[0].legend(fontsize=5.4, loc="lower left")
for t, lab_, col in [("countdown_flat_svg2d_s0", "flat, SVG", C["native"]), ("countdown_hier_svg2d_s0", "H-ELF, SVG", C["hier"]),
                     ("countdown_flat_c16_svg2d_s0", "flat, SVG cheap critic", C["mm7"]), ("countdown_hier_c16_svg2d_s0", "H-ELF, SVG cheap critic", C["violet"]),
                     ("countdown_flat_grr_lr3e-7_s0", "flat, GRR", C["grey"]), ("countdown_hier_grr_lr3e-7_s0", "H-ELF, GRR", C["green"])]:
    if t in D: band(axs[1], [t], col, lab_)
axs[1].set_title("Countdown (one seed each)"); finish(axs[1], ylab=""); axs[1].legend(fontsize=5.4, loc="lower left")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_tasks.pdf"); plt.close(fig)


# ---------------------------------------------------------------- A14: critic-free GRR baseline
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.2), sharey=False)
for t, lab_, col, ls in [("game24cd_flat_grr_lr3e-7_s0", r"flat Muon, lr $3{\times}10^{-7}$", C["native"], "-"), ("game24cd_hier_grr_lr3e-7_s0", r"H-ELF Muon, lr $3{\times}10^{-7}$", C["hier"], "-"),
                         ("game24cdA_flat_grr_lr3e-7_s0", r"flat AdamW, lr $3{\times}10^{-7}$", C["awd"], "-"), ("game24cdA_hier_grr_lr3e-7_s0", r"H-ELF AdamW, lr $3{\times}10^{-7}$", C["violet"], "-")]:
    it, d = dtest(t); axs[0].plot(it, d, color=col, ls=ls, marker="o", ms=2.5, lw=1.2, label=lab_)
axs[0].set_title("Game24, GRR at a 10x smaller lr"); finish(axs[0], xl="GRR iteration"); axs[0].legend(fontsize=5.4, loc="upper left")
for t, lab_, col in [("game24cd_flat_grr_s0", "flat, Game24", C["native"]), ("game24cd_hier_grr_s0", "H-ELF, Game24", C["hier"]),
                     ("kk_flat_grr_s0", "flat, K&K", C["mm7"]), ("kk_hier_grr_s0", "H-ELF, K&K", C["violet"]),
                     ("countdown_flat_grr_s0", "flat, Countdown", C["grey"]), ("countdown_hier_grr_s0", "H-ELF, Countdown", C["green"])]:
    it, d = dtest(t); axs[1].plot(it, d, color=col, marker="o", ms=2.5, lw=1.2, label=lab_)
axs[1].set_title(r"GRR at the published lr $3{\times}10^{-6}$"); finish(axs[1], xl="GRR iteration"); axs[1].legend(fontsize=5.4, loc="lower left")
for t, lab_, col in [("countdown_flat_grr_lr3e-7_s0", "flat", C["native"]), ("countdown_hier_grr_lr3e-7_s0", "H-ELF", C["hier"])]:
    it, d = dtest(t); axs[2].plot(it, d, color=col, marker="o", ms=2.5, lw=1.2, label=lab_)
axs[2].set_title(r"Countdown, GRR lr $3{\times}10^{-7}$"); finish(axs[2], xl="GRR iteration"); axs[2].legend(fontsize=5.4)
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_grr.pdf"); plt.close(fig)


# ---------------------------------------------------------------- A15: base critic sensitivity vs outcome (negative result)
import glob
MAP = {"game24cd_flat": [f"game24cd_flat_svg2d_s{q}" for q in range(5)], "game24cd_hier": [f"game24cd_hier_svg2d_s{q}" for q in range(5)],
       "game24cd_flat_ft43": ["game24cd_flat_ft43_svg2d_s0"], "game24cd_flat_ft44": ["game24cd_flat_ft44_svg2d_s0"],
       "game24cd_hier_ft43": ["game24cd_hier_ft43_svg2d_s0"], "game24cd_hier_ft44": ["game24cd_hier_ft44_svg2d_s0"],
       "game24cd_flat_early": ["game24cd_flat_early_svg2d_s0"], "game24cd_rand": [f"game24cd_rand_svg2d_s{q}" for q in range(3)],
       "game24cd_randm": [f"game24cd_randm_svg2d_s{q}" for q in range(3)], "game24cdA_flat": [f"game24cdA_flat_svg2d_s{q}" for q in range(3)],
       "game24cdA_flat60": ["game24cdA_flat60_svg2d_s0"], "game24cdA_hier": [f"game24cdA_hier_svg2d_s{q}" for q in range(3)],
       "kk_flat": [f"kk_flat_svg2d_s{q}" for q in range(3)], "kk_hier": [f"kk_hier_svg2d_s{q}" for q in range(3)],
       "countdown_flat": ["countdown_flat_svg2d_s0"], "countdown_hier": ["countdown_hier_svg2d_s0"],
       "countdown_flat_c16": ["countdown_flat_c16_svg2d_s0"], "countdown_hier_c16": ["countdown_hier_c16_svg2d_s0"]}
pts = []
for dpath in sorted(glob.glob("runs/analysis/sharp_*")):
    base = dpath.split("sharp_")[1].replace("_rho1e-3", "")
    if base not in MAP: continue
    g = [json.loads(l).get("rsa_gap") for l in open(dpath + "/log.jsonl") if l.strip()]
    g = [x for x in g if x is not None]
    fins = [dtest(t)[1][-1] for t in MAP[base] if t in D]
    task = "Game24" if base.startswith("game24") else ("K&K" if base.startswith("kk") else "Countdown")
    pts.append((base, task, st.mean(g), st.mean(fins)))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for task, col, mk in [("Game24", C["hier"], "o"), ("K&K", C["mm7"], "s"), ("Countdown", C["violet"], "^")]:
    q = [p_ for p_ in pts if p_[1] == task]
    ax.scatter([p_[2] for p_ in q], [p_[3] for p_ in q], color=col, marker=mk, s=18, label=task)
    for p_ in q: ax.annotate(p_[0].replace("game24cd", "").replace("countdown_", "").replace("kk_", "").strip("_") or "flat", (p_[2], p_[3]), fontsize=4.8, xytext=(2, 2), textcoords="offset points")
ax.set_xscale("log"); finish(ax, ylab=r"final $\Delta$ (mean over seeds)", xl=r"worst-case logit drop at $\rho=10^{-3}$ (base)"); ax.legend(fontsize=5.6)
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_sharp.pdf"); plt.close(fig)


# ---------------------------------------------------------------- A16: off-manifold geometry (offspan)
O = torch.load("runs/analysis/offspan_g24.pt", map_location="cpu", weights_only=False)
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.3))
DR = [("nat", "native", C["native"]), ("mm7", "indep. Muon", C["mm7"]), ("ma", "AdamW trunk", C["ma"]), ("A", "AdamW policy", C["awd"]), ("Awd", "AdamW+wd policy", C["violet"])]
for k, lab_, col in DR:
    if k not in O["drift"]: continue
    its = sorted(O["drift"][k], key=int)
    axs[0].plot([int(i) for i in its], [O["drift"][k][i]["off_frac"] for i in its], color=col, marker="o", ms=2.5, label=lab_)
    axs[1].plot([int(i) for i in its], [O["drift"][k][i]["norm"] for i in its], color=col, marker="o", ms=2.5)
axs[0].set_title("off-manifold energy fraction of $\\hat x$ ($k{=}40$)", fontsize=6.8); finish(axs[0], ylab="fraction", zero=False); axs[0].legend(fontsize=5.4)
axs[1].set_title("token-embedding norm of $\\hat x$", fontsize=6.8); finish(axs[1], ylab="norm", zero=False)
CN = [("native", "own"), ("mm43", "sibling"), ("mm7", "indep."), ("big", "own 16k"), ("ma", "AdamW"), ("ma43", "AdamW s43"), ("am", "Muon, A-pol."), ("Anat", "AdamW pol."), ("Awd", "AdamW+wd")]
vals = []
for n, lab_ in CN:
    if n not in O["critics"]: continue
    c = O["critics"][n]["40"] if "40" in O["critics"][n] else O["critics"][n][40]
    on, off = c["dl_on"].abs().mean().item(), c["dl_off"].abs().mean().item()
    vals.append((lab_, off / max(on, 1e-12)))
axs[2].bar(range(len(vals)), [v for _, v in vals], color=[C["native"], C["violet"], C["mm7"], C["grey"], C["ma"], C["green"], C["hier"], C["awd"], C["red"]][:len(vals)], edgecolor="white", lw=0.5)
axs[2].set_xticks(range(len(vals))); axs[2].set_xticklabels([l for l, _ in vals], rotation=60, fontsize=5.6)
axs[2].set_title("critic logit response: off- / on-manifold\nperturbation of equal size ($k{=}40$)", fontsize=6.5); finish(axs[2], ylab="ratio", zero=False, xl="")
fig.tight_layout(pad=0.3); fig.savefig(FIG + "app_offspan.pdf"); plt.close(fig)

# ---------------------------------------------------------------- generated table: every RL run
DESC = {}
def desc(run):
    return run.replace("_", "\\_")
rows = []
for run in sorted(D):
    e = evals(run)
    if not e: continue
    t = [100 * r["test_acc"] for r in e]
    b = max(e, key=lambda r: (r["val_acc"], -r["iter"]))
    traj = " ".join(f"{x:.1f}" for x in t)
    rows.append(f"\\texttt{{{desc(run)}}} & {e[-1]['iter']} & {t[0]:.1f} & {traj} & {t[-1] - t[0]:+.1f} & {100 * b['test_acc'] - t[0]:+.1f}@{b['iter']} & {min(t) - t[0]:+.1f} \\\\")
open("paper/gen_allruns.tex", "w").write("\n".join(rows) + "\n")
print("figures and tables written;", len(rows), "runs")
