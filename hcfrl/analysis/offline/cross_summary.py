"""Offline summary of cross_critic output: per critic x policy x k, accuracy, AUC, mean logit (all / correct / wrong),
the inflation of wrong answers relative to the base policy, and the pairwise alignment of critic input gradients.
usage: python -m hcfrl.analysis.offline.cross_summary <cross_critic.pt> [base policy name]"""
import sys
import torch


def auc(v, r):
    pos, neg = v[r > 0.5], v[r <= 0.5]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return ((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()).item()


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    base = sys.argv[2] if len(sys.argv) > 2 else d["policies"][0][0]
    pols = [p for p, _ in d["policies"]]; crits = [c for c, _ in d["critics"]]
    print("accuracy:", "  ".join(f"{p} {100 * d['rewards'][p].mean():.1f}" for p in pols))
    for k in d["ks"]:
        print(f"\n== k={k}: AUC (rows critics, cols policies)")
        print(" " * 8 + "".join(f"{p:>10s}" for p in pols))
        for c in crits:
            print(f"{c:8s}" + "".join(f"{auc(d['logits'][(c, p, k)], d['rewards'][p]):10.3f}" for p in pols))
        print(f"== k={k}: mean logit shift vs {base} (all samples) | accuracy shift")
        print(" " * 8 + "".join(f"{p:>10s}" for p in pols))
        print(f"{'dacc':8s}" + "".join(f"{100 * (d['rewards'][p].mean() - d['rewards'][base].mean()):10.1f}" for p in pols))
        for c in crits:
            b = d["logits"][(c, base, k)].mean()
            print(f"{c:8s}" + "".join(f"{(d['logits'][(c, p, k)].mean() - b):10.2f}" for p in pols))
        print(f"== k={k}: wrong-answer inflation = mean logit on WRONG samples of policy minus on WRONG samples of {base}, in units of the base correct-wrong gap")
        print(" " * 8 + "".join(f"{p:>10s}" for p in pols))
        for c in crits:
            lb, rb = d["logits"][(c, base, k)], d["rewards"][base]
            gap = (lb[rb > 0.5].mean() - lb[rb <= 0.5].mean()).item()
            row = []
            for p in pols:
                l, r = d["logits"][(c, p, k)], d["rewards"][p]
                row.append((l[r <= 0.5].mean() - lb[rb <= 0.5].mean()).item() / gap if (r <= 0.5).any() else float("nan"))
            print(f"{c:8s}" + "".join(f"{x:10.2f}" for x in row))
    if d.get("pulls"):
        gm = d["gen_mask"][: next(iter(d["pulls"].values()))[0].shape[0]].bool()
        for k in d["ks"]:
            print(f"\n== k={k}: cosine between critic input gradients on {d['pull_policy']} states (generated positions, z and x_prev)")
            vec = {}
            for c in crits:
                gz, gx = d["pulls"][(c, k)]
                v = torch.cat([gz.float() * gm[..., None], gx.float() * gm[..., None]], -1).flatten(1)
                vec[c] = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            print(" " * 8 + "".join(f"{c:>8s}" for c in crits))
            for c in crits:
                print(f"{c:8s}" + "".join(f"{(vec[c] * vec[c2]).sum(1).mean().item():8.3f}" for c2 in crits))


if __name__ == "__main__":
    main()
