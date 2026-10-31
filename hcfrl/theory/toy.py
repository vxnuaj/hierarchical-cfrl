"""Toy pathwise-RL sandbox: does an optimizer-matched critic get exploited?

A small conditional flow model with the same structure as the real setup:
  policy   D_theta(z, t, c): MLP denoiser, x-prediction; K-step Euler sampler from z_0 ~ N(0, I) at t = 0
  task     prompt c ~ N(0, I_dc); target y*(c) = tanh(A2 tanh(A1 c)); verifier R = 1[max_i |y_i - y*_i| < tau]
  SFT      flow matching E|D(t y* + (1-t) eps, t, c) - y*|^2, optimizer Muon (hidden matrices) or AdamW
  critic   trunk = hidden layers of an SFT'd denoiser (its features h(z, t, c)), head MLP; pre-trained on rollouts
           of the policy with BCE on R (trunk lr small, head lr larger, AdamW) -- as in CFRL
  RL       actor: roll out to z_{k-1} without grad, one denoiser call with grad, z_k = advance(z_{k-1}, x_hat),
           ascend the critic logit at z_k (AdamW); critic head refit on fresh rollouts every iteration.

Every component can be switched (policy optimizer, critic trunk source) so the swap experiments of the paper can be
replayed in seconds, and quantities that are too expensive on the real model (exact Jacobians, kernels) are exact here.
"""
from __future__ import annotations

import copy, math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- optimizers
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = G.shape[0] > G.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden matrices (Nesterov momentum, orthogonalised update, shape-scaled lr).
    `blend` in [0, 1] interpolates the update direction between the orthogonalised momentum (1) and the
    Frobenius-normalised raw momentum with the same norm (0): the dose-response knob."""

    def __init__(self, params, lr=0.02, momentum=0.95, wd=0.0, blend=1.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd, blend=blend))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            for p in gr["params"]:
                if p.grad is None: continue
                st = self.state[p]
                buf = st.setdefault("buf", torch.zeros_like(p))
                buf.mul_(gr["momentum"]).add_(p.grad)
                u = p.grad + gr["momentum"] * buf
                o = newton_schulz(u)
                if gr["blend"] < 1.0:
                    r = u * (o.norm() / (u.norm() + 1e-12))
                    o = gr["blend"] * o + (1 - gr["blend"]) * r
                if gr["wd"]: p.mul_(1 - gr["lr"] * gr["wd"])
                p.add_(o, alpha=-gr["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


def make_opt(model: "Denoiser", kind: str, lr_muon=0.02, lr_adam=3e-3, wd=0.0, blend=1.0):
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and n.startswith("hid")]
    other = [p for n, p in model.named_parameters() if not (p.ndim == 2 and n.startswith("hid"))]
    if kind == "muon":
        return [Muon(hidden, lr=lr_muon, wd=wd, blend=blend), torch.optim.AdamW(other, lr=lr_adam, weight_decay=0.0)]
    return [torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=wd)]


# ---------------------------------------------------------------- models
class Denoiser(nn.Module):
    def __init__(self, d=32, dc=8, width=256, depth=3):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(d + dc + 1, width)
        self.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, d)

    def features(self, z, t, c):
        h = F.gelu(self.inp(torch.cat([z, c, t.expand(z.shape[0], 1)], -1)))
        for l in self.hid:
            h = h + F.gelu(l(h))
        return h

    def forward(self, z, t, c):
        return self.out(self.features(z, t, c))


class Critic(nn.Module):
    def __init__(self, trunk: Denoiser):
        super().__init__()
        self.trunk = copy.deepcopy(trunk)
        width = trunk.inp.out_features
        self.head = nn.Sequential(nn.Linear(width + 1, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, z, t, c):
        h = self.trunk.features(z, t, c)
        return self.head(torch.cat([h, t.expand(z.shape[0], 1)], -1)).squeeze(-1)


# ---------------------------------------------------------------- task / sampler
@dataclass
class Task:
    """Multimodal task: each prompt has `modes` valid solutions y_j(c) = s_j * sin(freq * A2 tanh(A1 c)) with fixed
    sign patterns s_j; a sample is correct if it is within tau (max-norm) of ANY valid solution. Training data draw
    j uniformly, so the flow model must learn a mixture, and its samples can fall between modes (wrong)."""
    d: int = 16; dc: int = 8; tau: float = 0.1; K: int = 8; seed: int = 0; freq: float = 2.0; modes: int = 4
    A1: torch.Tensor = field(init=False); A2: torch.Tensor = field(init=False); S: torch.Tensor = field(init=False)

    def __post_init__(self):
        g = torch.Generator().manual_seed(1000 + self.seed)
        self.A1 = torch.randn(32, self.dc, generator=g) / math.sqrt(self.dc)
        self.A2 = torch.randn(self.d, 32, generator=g) * 1.5 / math.sqrt(32)
        self.S = torch.where(torch.rand(self.modes, self.d, generator=g) < 0.5, -1.0, 1.0); self.S[0] = 1.0

    def prompts(self, n, g=None):
        return torch.randn(n, self.dc, generator=g)

    def solutions(self, c):                       # (n, modes, d)
        base = torch.sin(self.freq * torch.tanh(c @ self.A1.T) @ self.A2.T)
        return base[:, None, :] * self.S[None]

    def target(self, c, g=None):                  # one valid solution per prompt, mode drawn uniformly
        j = torch.randint(0, self.modes, (c.shape[0],), generator=g)
        return self.solutions(c)[torch.arange(c.shape[0]), j]

    def reward(self, y, c):
        return ((y[:, None, :] - self.solutions(c)).abs().amax(-1).amin(-1) < self.tau).float()

    def t(self, k):
        return k / self.K

    def advance(self, z, x, k):
        t0, t1 = self.t(k), self.t(k + 1)
        return z + (t1 - t0) * (x - z) / max(1 - t0, 1e-3)

    def rollout(self, model, c, g=None, record=()):
        z = torch.randn(c.shape[0], self.d, generator=g)
        states = {}
        for k in range(self.K):
            if k in record: states[k] = z
            x = model(z, torch.tensor([[self.t(k)]]), c)
            z = self.advance(z, x, k)
        return x, states   # final clean estimate is the sample


def sft(task: Task, kind: str, steps=4000, seed=0, bs=256, wd=0.0, blend=1.0, width=256, init=None, **kw):
    torch.manual_seed(seed)
    m = Denoiser(task.d, task.dc, width) if init is None else copy.deepcopy(init)
    opts = make_opt(m, kind, wd=wd, blend=blend, **kw)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda i: min(1.0, (i + 1) / 200) * max(0.0, 1 - i / steps)) for o in opts]
    for i in range(steps):
        c = task.prompts(bs); y = task.target(c); t = torch.rand(bs, 1); e = torch.randn_like(y)
        loss = ((m(t * y + (1 - t) * e, t, c) - y) ** 2).mean()
        for o in opts: o.zero_grad()
        loss.backward()
        for o, s in zip(opts, scheds): o.step(); s.step()
    return m


@torch.no_grad()
def accuracy(task, model, n=4096, seed=123):
    g = torch.Generator().manual_seed(seed); c = task.prompts(n, g)
    y, _ = task.rollout(model, c, g)
    return task.reward(y, c).mean().item()


def pretrain_critic(task, policy, trunk, n=8192, epochs=6, lr_trunk=1e-4, lr_head=1e-3, seed=0, ks=(1, 2, 3)):
    torch.manual_seed(seed); critic = Critic(trunk)
    g = torch.Generator().manual_seed(seed + 7); c = task.prompts(n, g)
    with torch.no_grad():
        y, st = task.rollout(policy, c, g, record=set(ks)); r = task.reward(y, c)
    Z = torch.cat([st[k] for k in ks]); C = c.repeat(len(ks), 1); R = r.repeat(len(ks))
    T = torch.cat([torch.full((n, 1), task.t(k)) for k in ks])
    opt = torch.optim.AdamW([{"params": critic.trunk.parameters(), "lr": lr_trunk}, {"params": critic.head.parameters(), "lr": lr_head}])
    for ep in range(epochs):
        perm = torch.randperm(len(R))
        for b in range(0, len(R), 256):
            i = perm[b: b + 256]
            loss = F.binary_cross_entropy_with_logits(critic(Z[i], T[i], C[i]), R[i])
            opt.zero_grad(); loss.backward(); opt.step()
    return critic


def auc(v, r):
    pos, neg = v[r > 0.5], v[r <= 0.5]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return ((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()).item()


def rl(task, policy, critic, iters=1500, B=64, lr=3e-4, lr_min=3e-5, lr_head=1e-3, refit=2, buf=4096, seed=0,
       eval_every=100, ks=(1, 2, 3), log=None):
    """Pathwise actor-critic on the toy (CFRL Algorithm 1 in miniature). Returns accuracy trajectory and loop stats."""
    torch.manual_seed(seed); random_g = torch.Generator().manual_seed(seed)
    policy = copy.deepcopy(policy); critic = copy.deepcopy(critic)
    for p in critic.trunk.parameters(): p.requires_grad_(False)
    opt_a = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    opt_c = torch.optim.AdamW(critic.head.parameters(), lr=lr_head)
    BZ, BC, BR, BT = [], [], [], []
    hist = []
    k_hi = max(ks)
    for it in range(iters + 1):
        if it % eval_every == 0:
            hist.append({"iter": it, "acc": accuracy(task, policy)})
        if it == iters: break
        for gq in opt_a.param_groups: gq["lr"] = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * it / iters))
        c = task.prompts(B, random_g); k = int(torch.randint(min(ks), k_hi + 1, (1,), generator=random_g))
        with torch.no_grad():
            z = torch.randn(B, task.d, generator=random_g)
            for j in range(k - 1):
                z = task.advance(z, policy(z, torch.tensor([[task.t(j)]]), c), j)
        x = policy(z, torch.tensor([[task.t(k - 1)]]), c)
        zk = task.advance(z, x, k - 1)
        obj = critic(zk, torch.tensor([[task.t(k)]]), c).mean()
        opt_a.zero_grad(); (-obj).backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_a.step()
        # buffer: fresh rollouts, states at k_hi
        c2 = task.prompts(B, random_g)
        with torch.no_grad():
            y2, st = task.rollout(policy, c2, random_g, record={k_hi}); r2 = task.reward(y2, c2)
        BZ.append(st[k_hi]); BC.append(c2); BR.append(r2)
        if len(BZ) * B > buf: BZ.pop(0); BC.pop(0); BR.pop(0)
        Z, C, R = torch.cat(BZ), torch.cat(BC), torch.cat(BR); T = torch.full((len(R), 1), task.t(k_hi))
        for _ in range(refit):
            i = torch.randint(0, len(R), (min(256, len(R)),), generator=random_g)
            loss = F.binary_cross_entropy_with_logits(critic(Z[i], T[i], C[i]), R[i])
            opt_c.zero_grad(); loss.backward(); opt_c.step()
        if it % eval_every == 0:
            with torch.no_grad():
                hist[-1].update({"buf_succ": r2.mean().item(), "obj": obj.item(),
                                 "auc": auc(critic(Z[-512:], T[-512:], C[-512:]), R[-512:])})
            if log: log(hist[-1])
    return hist
