"""Gate 1 analysis: critic quality vs denoising step for several runs (e.g. flat vs hierarchical).

Reads outputs/critic/<tag>/log.json (from hcfrl.rl.pretrain_critic), takes each run's best epoch,
prints a table of Pearson / AUC / pick@V per step, and saves a plot.

usage: python -m hcfrl.analysis.critic_curves outputs/critic/game24_flat_2d outputs/critic/game24_hier_2d --out outputs/figs/game24_critic_curves.png
"""
from __future__ import annotations

import argparse, json, os


def load(run_dir: str):
    log = json.load(open(os.path.join(run_dir, "log.json")))
    best = log["best_epoch"]
    rec = next(r for r in log["log"] if r["epoch"] == best)
    ks = sorted(int(k[1:]) for k in rec if k.startswith("k") and k[1:].isdigit())
    return best, {k: rec[f"k{k}"] for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="pearson", choices=["pearson", "auc", "pick_at_v", "pearson_minpos"])
    a = ap.parse_args()
    curves = {}
    for r in a.runs:
        best, per_k = load(r)
        name = os.path.basename(r.rstrip("/"))
        curves[name] = per_k
        print(f"== {name} (best epoch {best})")
        print("   k   pearson    auc  pick@V   n_mixed")
        for k, m in per_k.items():
            print(f"  {k:3d}  {m.get('pearson', float('nan')):7.3f} {m.get('auc', float('nan')):6.3f} "
                  f"{m.get('pick_at_v', float('nan')):7.3f} {m.get('n_mixed_groups', 0):8d}")
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for name, per_k in curves.items():
            ks = sorted(per_k); ax.plot(ks, [per_k[k].get(a.metric, float("nan")) for k in ks], marker="o", label=name)
        ax.set_xlabel("denoising step k"); ax.set_ylabel(a.metric); ax.grid(alpha=0.3); ax.legend()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.tight_layout(); fig.savefig(a.out, dpi=150)
        print("saved", a.out)


if __name__ == "__main__":
    main()
