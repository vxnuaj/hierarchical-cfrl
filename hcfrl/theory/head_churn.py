"""Gradient-direction stability of a critic under refitting of its head (toy).

Trunk = hidden layer of an MLP trained (whitening.py data) with SGD / AdamW / Muon. Critic heads are fit by ridge
regression on the trunk's hidden features (width 256) to the task signal using DIFFERENT small samples of states
(n_fit each), as a refit on a moving buffer would. Reported: mean pairwise cosine of the heads' input gradients
(stability of the direction the critic asks the policy to move in), the on-manifold share of those gradients, and
the correlation of each head's input gradient with the true signal's input gradient (usefulness).
Prediction: Muon trunk -> unstable, off-manifold, less useful directions; AdamW/SGD -> stable and useful."""
import itertools, json, sys
import torch
from hcfrl.theory.whitening import make_data, train

torch.set_num_threads(8)
lam_off = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
X, Y, T, Nn, sample, target = make_data(lam_off=lam_off)
out = {}
for kind, lr in [("sgd", 0.05), ("adamw", 1e-3), ("muon", 0.02)]:
    rows = []
    for s in range(3):
        m = train(X, Y, kind, 3000, lr, seed=s)
        def hid(x):
            h = torch.relu(m.inp(x)); return h + torch.relu(m.hid[0](h))
        signal = lambda x: target(x).sum(1)
        g = torch.Generator().manual_seed(50 + s)
        for n_fit in (128, 512, 2048):
            heads = []
            for j in range(5):
                Xf = sample(n_fit)
                with torch.no_grad(): H = hid(Xf); y = signal(Xf) + 0.3 * torch.randn(n_fit, generator=g)
                A = H.T @ H + 1e-2 * n_fit * torch.eye(H.shape[1]); w = torch.linalg.solve(A, H.T @ (y - y.mean()))
                heads.append(w)
            x = sample(256).requires_grad_(True)
            gs = []
            for w in heads:
                gx, = torch.autograd.grad((hid(x) @ w).sum(), x); gs.append(gx / gx.norm(dim=1, keepdim=True).clamp_min(1e-12))
            gt, = torch.autograd.grad(signal(x).sum(), x); gt = gt / gt.norm(dim=1, keepdim=True)
            cos = sum((gs[i] * gs[j]).sum(1).mean().item() for i, j in itertools.combinations(range(5), 2)) / 10
            on = sum((gq @ T).pow(2).sum(1).mean().item() for gq in gs) / 5
            use = sum((gq * gt).sum(1).mean().item() for gq in gs) / 5
            rows.append({"n_fit": n_fit, "cos": cos, "on": on, "use": use})
    for n_fit in (128, 512, 2048):
        r = [x for x in rows if x["n_fit"] == n_fit]
        res = {k: round(sum(x[k] for x in r) / len(r), 3) for k in ("cos", "on", "use")}
        out[f"{kind}_{n_fit}"] = res
        print(f"{kind:5s} n_fit {n_fit:5d}: head-to-head gradient cosine {res['cos']:.3f}  on-manifold share {res['on']:.3f}  cosine with true gradient {res['use']:.3f}", flush=True)
json.dump(out, open(f"runs/theory/head_churn_lam{lam_off:g}.json", "w"), indent=1)
