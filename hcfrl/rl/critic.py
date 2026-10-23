"""CFRL critic (arXiv 2610.19080, Sec. 4 + App. B.4).

Backbone: a copy of the task-SFT denoiser trunk, evaluated WITHOUT CFG, reading the same
(z~_k, t~_k, x_{k-1}, c) the sampler's denoiser reads, returning per-position features h_i.
Head f_phi: MLP (hidden+1) -> 256 -> 64 -> 1 with GELU; the extra input is the time t~_k.

2D critic (recommended):  r_{k,i} = sigmoid(f(h_i)),  V = sigmoid( sum_{i in G} f(h_i) )
1D critic:                V = sigmoid( f( mean_{i in G} h_i ) )
Loss 2D: BCE(V, R) + (1/|G|) sum_i BCE(r_i, R)     (all per-position readings get the SEQUENCE reward)
Loss 1D: BCE(V, R)

The actor maximizes the PRE-SIGMOID logit of V (mean over positions), see `Critic.logit`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import hcfrl.elf_path  # noqa: F401
from utils.sampling_utils import restore_cond  # type: ignore
from hcfrl.rl.sampler import DenoiserInput


class ValueHead(nn.Module):
    def __init__(self, hidden: int, width: tuple[int, int] = (256, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + 1, width[0]), nn.GELU(),
            nn.Linear(width[0], width[1]), nn.GELU(),
            nn.Linear(width[1], 1),
        )
        # zero-init the output layer so every position starts at logit 0 (V = 0.5);
        # with ~60 summed positions a random init saturates the sequence value
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden), t: broadcastable to (..., 1). Returns (...,) logits."""
        t = t.expand(*h.shape[:-1], 1)
        return self.net(torch.cat([h, t], dim=-1)).squeeze(-1)


@dataclass
class CriticOutput:
    logit: torch.Tensor        # (B,) pre-sigmoid sequence value (actor ascends this)
    value: torch.Tensor        # (B,) sigmoid(logit)
    pos_logit: torch.Tensor | None   # (B, L) per-position logits (2D only; masked positions = 0)


class Critic(nn.Module):
    """Frozen trunk copy + trainable head. `mode` in {"2d", "1d"}."""

    def __init__(self, denoiser: nn.Module, config, mode: str = "2d", trunk_trainable: bool = False,
                 blind_clusters: bool = False):
        super().__init__()
        assert mode in ("2d", "1d")
        self.mode = mode
        self.config = config
        # ablation: critic cannot see the coarse stream (cluster channels zeroed in z and x_prev)
        self.blind_clusters = blind_clusters
        self.n_clusters = int(getattr(config, "n_clusters", 0) or 0)
        self.trunk = copy.deepcopy(denoiser)
        self.trunk.requires_grad_(trunk_trainable)
        self.head = ValueHead(self.trunk.hidden_size)

    # -- features ---------------------------------------------------------------
    def features(self, inp: DenoiserInput, cond_seq, cond_mask) -> torch.Tensor:
        """Per-position trunk features for the denoiser input, no CFG, self-cond input = x_prev."""
        B = inp.z.shape[0]
        t_batch = torch.full((B,), inp.t, dtype=inp.z.dtype, device=inp.z.device)
        x_prev = inp.x_prev
        if x_prev is None:
            x_prev = restore_cond(torch.zeros_like(inp.z), cond_seq, cond_mask)
        z, x_prev = inp.z, x_prev
        if self.blind_clusters and self.n_clusters > 0:
            D = z.shape[-1] - self.n_clusters
            keep = torch.ones(z.shape[-1], dtype=z.dtype, device=z.device); keep[D:] = 0
            z, x_prev = z * keep, x_prev * keep
        z_in = torch.cat([z, x_prev], dim=-1) if self.config.self_cond_prob > 0 else z
        sc = (torch.full((B,), 1.0, dtype=inp.z.dtype, device=inp.z.device)
              if self.config.num_self_cond_cfg_tokens > 0 else None)
        return self.trunk(z_in, t_batch, deterministic=True, self_cond_cfg_scale=sc, return_hidden=True)

    # -- value ------------------------------------------------------------------
    def forward(self, inp: DenoiserInput, cond_seq, cond_mask, gen_mask: torch.Tensor,
                h: torch.Tensor | None = None) -> CriticOutput:
        """gen_mask: (B, L) 1 on generated (scored) positions, 0 on cond/pad."""
        if h is None:
            h = self.features(inp, cond_seq, cond_mask)
        t = torch.full((h.shape[0], 1, 1), inp.t, dtype=h.dtype, device=h.device)
        m = gen_mask.to(h.dtype)
        if self.mode == "2d":
            pos = self.head(h, t) * m                              # (B, L)
            logit = pos.sum(dim=1)
            return CriticOutput(logit=logit, value=torch.sigmoid(logit), pos_logit=pos)
        hbar = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        logit = self.head(hbar, t[:, 0])
        return CriticOutput(logit=logit, value=torch.sigmoid(logit), pos_logit=None)

    def loss(self, out: CriticOutput, reward: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
        """reward: (B,) in {0,1}."""
        r = reward.to(out.logit.dtype)
        l = F.binary_cross_entropy_with_logits(out.logit, r)
        if self.mode == "2d":
            m = gen_mask.to(out.pos_logit.dtype)
            per = F.binary_cross_entropy_with_logits(out.pos_logit, r[:, None].expand_as(out.pos_logit), reduction="none")
            l = l + (per * m).sum(1).div(m.sum(1).clamp_min(1.0)).mean()
        return l

    def head_parameters(self):
        return self.head.parameters()


@torch.no_grad()
def rank_metrics(values: torch.Tensor, rewards: torch.Tensor, groups: torch.Tensor | None = None) -> dict:
    """Pearson(value, reward); AUC; and within-prompt pick@V (fraction of prompt groups where the
    top-valued sample is a success, among groups with at least one success and one failure)."""
    v = values.float().cpu(); r = rewards.float().cpu()
    out = {}
    if v.numel() > 1 and r.std() > 0 and v.std() > 0:
        out["pearson"] = float(((v - v.mean()) * (r - r.mean())).mean() / (v.std() * r.std()))
    pos, neg = v[r > 0.5], v[r <= 0.5]
    if len(pos) and len(neg):
        out["auc"] = float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())
    if groups is not None:
        g = groups.cpu(); hits = n = 0
        for gid in g.unique():
            idx = (g == gid).nonzero().squeeze(1)
            if r[idx].min() < 0.5 < r[idx].max():
                n += 1; hits += int(r[idx][v[idx].argmax()] > 0.5)
        if n:
            out["pick_at_v"] = hits / n
            out["n_mixed_groups"] = n
    return out
