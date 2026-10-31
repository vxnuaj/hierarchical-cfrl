"""Off-manifold gain of a network as a function of its training optimizer (numerical check of the theory).

Data x in R^n live mostly on an m-dim subspace T ("manifold": variance lam_on per direction) with small variance
lam_off on the complement N. Targets depend on the T-coordinates only, plus label noise. A model (linear W, or a
2-layer MLP) is trained by GD, AdamW, or Muon (orthogonalised updates; `blend` interpolates to the raw direction).
We report the model's input sensitivity on N relative to T:
    G_off / G_on = E|J u_off| / E|J u_on|,  u random unit vectors in N resp. T, J the input Jacobian,
and, for a scalar critic-like readout r(x) = a^T f(x) with a random, the fraction of |grad_x r|^2 lying on N.

Prediction (whitening argument): GD/AdamW learn direction i at a rate proportional to its variance lam_i, so the
gain on N stays small; Muon's update gives every singular direction of the gradient the same step, so the low-variance
directions are learned (fit to noise) at the same rate and G_off / G_on grows like 1/sqrt(lam_off).
"""
from __future__ import annotations

import argparse, json, math

import torch
import torch.nn as nn

from hcfrl.theory.toy import Muon


def make_data(n=64, m=8, d=32, N=1024, lam_on=1.0, lam_off=1e-3, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    T, Nn = Q[:, :m], Q[:, m:]
    def sample(k):
        return (torch.randn(k, m, generator=g) * math.sqrt(lam_on)) @ T.T + (torch.randn(k, n - m, generator=g) * math.sqrt(lam_off)) @ Nn.T
    Wt = torch.randn(d, m, generator=g) / math.sqrt(m)
    def target(x):
        return torch.tanh(x @ T) @ Wt.T
    X = sample(N); Y = target(X) + noise * torch.randn(N, d, generator=g)
    return X, Y, T, Nn, sample, target


class MLP(nn.Module):
    def __init__(self, n, d, width, depth):
        super().__init__()
        self.inp = nn.Linear(n, width); self.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)]); self.out = nn.Linear(width, d)
    def forward(self, x):
        h = torch.relu(self.inp(x))
        for l in self.hid: h = h + torch.relu(l(h))
        return self.out(h)


def train(X, Y, kind, steps, lr, width=256, depth=1, blend=1.0, seed=0, bs=128, wd=0.0):
    torch.manual_seed(seed)
    model = MLP(X.shape[1], Y.shape[1], width, depth)
    if kind == "muon":   # all matrices with Muon (the input layer is where off-manifold gain is written)
        mats = [p for p in model.parameters() if p.ndim == 2]; vecs = [p for p in model.parameters() if p.ndim < 2]
        opts = [Muon(mats, lr=lr, blend=blend, wd=wd), torch.optim.AdamW(vecs, lr=1e-3)]
    elif kind == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)]
    else:
        opts = [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)]
    g = torch.Generator().manual_seed(seed + 1)
    for i in range(steps):
        idx = torch.randint(0, len(X), (bs,), generator=g)
        loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
        for o in opts: o.zero_grad()
        loss.backward()
        f = max(0.0, 1 - i / steps)
        for o in opts:
            for gq in o.param_groups: gq.setdefault("lr0", gq["lr"]); gq["lr"] = gq["lr0"] * f
            o.step()
    return model


def measure(model, sample, target, T, Nn, n_dir=64, seed=0):
    g = torch.Generator().manual_seed(seed + 2)
    x = sample(256).requires_grad_(True)
    with torch.no_grad():
        xs = sample(4096); test_mse = ((model(xs) - target(xs)) ** 2).mean().item()
    def gain(B):
        tot = 0.0
        for _ in range(n_dir):
            u = (torch.randn(B.shape[1], generator=g) @ B.T); u = u / u.norm()
            _, jvp = torch.autograd.functional.jvp(model, (x.detach(),), (u.expand_as(x),))
            tot += jvp.norm(dim=1).mean().item()
        return tot / n_dir
    a = torch.randn(model.out.out_features, generator=g)
    r = (model(x) @ a).sum(); gx, = torch.autograd.grad(r, x)
    off_frac = ((gx @ Nn).pow(2).sum(1) / gx.pow(2).sum(1)).mean().item()
    G_on, G_off = gain(T), gain(Nn)
    return {"test_mse": test_mse, "G_on": G_on, "G_off": G_off, "ratio": G_off / G_on, "readout_off_frac": off_frac,
            "chance_off_frac": Nn.shape[1] / (T.shape[1] + Nn.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(8)
    res = []
    for lam_off in [1e-2, 1e-3, 1e-4]:
        X, Y, T, Nn, sample, target = make_data(lam_off=lam_off)
        for kind, lr, blend in [("sgd", 0.05, 1.0), ("adamw", 1e-3, 1.0), ("muon", 0.02, 1.0), ("muon", 0.02, 0.5), ("muon", 0.02, 0.0)]:
            for s in range(a.seeds):
                m = train(X, Y, kind, 3000, lr, blend=blend, seed=s)
                r = measure(m, sample, target, T, Nn, seed=s)
                r.update(lam_off=lam_off, kind=kind, blend=blend, seed=s); res.append(r)
            sub = [r for r in res if r["lam_off"] == lam_off and r["kind"] == kind and r["blend"] == blend]
            print(f"lam_off {lam_off:g} {kind:5s} blend {blend:.2f}: G_off/G_on {sum(r['ratio'] for r in sub)/len(sub):.3f}  "
                  f"readout grad off-fraction {sum(r['readout_off_frac'] for r in sub)/len(sub):.3f} (chance {sub[0]['chance_off_frac']:.3f})  test mse {sum(r['test_mse'] for r in sub)/len(sub):.4f}", flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
