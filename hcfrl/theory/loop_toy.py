"""Minimal pathwise actor-critic loop on the whitening data (theory sandbox for the refit route).

Policy: a population of points x = mu_c + sigma*eps for prompts c (mu_c learned per prompt, a "policy" that can move
its output in any direction), initialised near the data manifold. True reward: the task signal r(x) = s(x) on the
manifold coordinates, minus a penalty for leaving the manifold (an answer that is not well-formed is wrong):
r(x) = s(P_T x) - beta |P_N x|^2.
Critic: frozen trunk (hidden layer of an MLP trained on the whitening data with SGD / AdamW / Muon) + ridge head fit to
the true reward of the current samples. Actor: mu <- mu + eta * normalised grad_x critic(mu). Variants: head refit
every iteration on fresh samples ("refit") or fit once on the initial samples ("frozen").
Reported: true reward and off-manifold norm of mu over iterations, and the cosine between successive actor steps.
"""
from __future__ import annotations

import json, sys

import torch

from hcfrl.theory.whitening import make_data, train

torch.set_num_threads(8)


def run(kind, lr, refit, lam_off=1e-3, iters=300, eta=0.02, beta=2.0, n_pts=64, n_fit=256, seed=0, aug=0.0, aug_dirs="off"):
    X, Y, T, Nn, sample, target = make_data(lam_off=lam_off)
    m = train(X, Y, kind, 3000, lr, seed=seed)
    def hid(x):
        h = torch.relu(m.inp(x)); return h + torch.relu(m.hid[0](h))
    s = lambda x: target(x).sum(1) / 4
    def reward(x):
        return s(x @ T @ T.T) - beta * (x @ Nn).pow(2).sum(1)
    g = torch.Generator().manual_seed(100 + seed)
    mu = sample(n_pts).detach()
    sigma = 0.1

    def fit_head(center):
        xs = (center.repeat(n_fit // n_pts, 1) + sigma * torch.randn(n_fit, center.shape[1], generator=g))
        with torch.no_grad():
            y = reward(xs)
            if aug > 0:   # invariance augmentation: copies perturbed along N (or random directions) keep the original label
                B = Nn if aug_dirs == "off" else torch.eye(xs.shape[1])
                xa = xs + aug * (torch.randn(len(xs), B.shape[1], generator=g) @ B.T) / B.shape[1] ** 0.5
                xs, y = torch.cat([xs, xa]), torch.cat([y, y])
            H = hid(xs)
        A = H.T @ H + 1e-2 * len(xs) * torch.eye(H.shape[1])
        return torch.linalg.solve(A, H.T @ (y - y.mean()))

    w = fit_head(mu); hist = []; prev = None
    for it in range(iters + 1):
        if it % 25 == 0:
            with torch.no_grad():
                hist.append({"iter": it, "reward": reward(mu).mean().item(), "off": (mu @ Nn).norm(dim=1).mean().item()})
        if it == iters: break
        if refit: w = fit_head(mu)
        x = mu.clone().requires_grad_(True)
        gx, = torch.autograd.grad((hid(x) @ w).sum(), x)
        step = gx / gx.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if prev is not None and it % 25 == 1: hist[-1]["step_cos"] = (step * prev).sum(1).mean().item()
        prev = step
        mu = (mu + eta * step).detach()
    return hist


if __name__ == "__main__":
    out = {}
    for kind, lr in [("sgd", 0.05), ("adamw", 1e-3), ("muon", 0.02)]:
        for refit in (True, False):
            hs = [run(kind, lr, refit, seed=s) for s in range(3)]
            avg = [{k: sum(h[i][k] for h in hs) / 3 for k in ("iter", "reward", "off")} for i in range(len(hs[0]))]
            out[f"{kind}_{'refit' if refit else 'frozen'}"] = avg
            print(f"{kind:5s} {'refit ' if refit else 'frozen'}: reward " + " ".join(f"{a['reward']:+.2f}" for a in avg[::2]) +
                  "   off-manifold |P_N mu| " + " ".join(f"{a['off']:.2f}" for a in avg[::4]), flush=True)
    json.dump(out, open("runs/theory/loop_toy.json", "w"), indent=1)
