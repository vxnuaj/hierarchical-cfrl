"""ELF churn sampler re-expressed as an MDP for CFRL-style RL.

Conventions (match CFRL, arXiv 2610.19080, App. B.1, and ELF's `_sde_step`):
  uniform grid t_0=0 < t_1 < ... < t_K=1, step h = 1/K, churn gamma,
  alpha = clip(1 - gamma*h, 0, 1), t_eps = config.t_eps.
  Loop state before step k:  (z_k, x_{k-1})   [x_{-1} = 0 on generated positions]
  Step k (k = 0..K-2, SDE):  z~_k = alpha*z_k + (1-alpha)*eps_k            (eps ~ N(0, c_sigma^2))
                             t~_k = alpha*t_k
                             x_k  = d_theta(z~_k, t~_k, x_{k-1}, c)          (CFG inside)
                             z_{k+1} = z~_k + (t_{k+1} - t~_k) * (x_k - z~_k) / max(1 - t~_k, t_eps)
  Last step (ODE, no churn): same with alpha = 1.
  Decode: argmax of decoder head on (z_K, t_K).

The critic reads exactly what the denoiser reads at a call: (z~_k, t~_k, x_{k-1}, c).
`DenoiserInput` holds that triple. `rollout` records DenoiserInputs at requested
steps under no_grad; `grad_step` recomputes step k-1 -> k's denoiser call WITH grad
from a recorded DenoiserInput at k-1 and returns the DenoiserInput at k, through
which the critic's gradient reaches theta via two channels (z~_k via x_{k-1}, and
x_{k-1} itself as the self-conditioning input).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import hcfrl.elf_path  # noqa: F401  (sys.path side effect)
from utils.sampling_utils import _forward_sample, restore_cond  # type: ignore


@dataclass
class DenoiserInput:
    k: int                      # step index of the denoiser call this input feeds
    z: torch.Tensor             # (B, L, D) churned latent z~_k (cond positions restored)
    t: float                    # lowered time t~_k
    x_prev: torch.Tensor        # (B, L, D) previous clean estimate x_{k-1} (cond restored)


@dataclass
class Rollout:
    z_final: torch.Tensor                     # z_K
    inputs: dict[int, DenoiserInput]          # recorded denoiser inputs by k
    tokens: Optional[torch.Tensor] = None     # (B, gen_len) decoded ids (after shift/eos mask)


class ChurnSampler:
    def __init__(self, config, K: int, gamma: float, cfg_scale: float = 2.0,
                 self_cond_cfg_scale: float = 1.0):
        self.config = config
        self.K = K
        self.h = 1.0 / K
        self.gamma = gamma
        self.alpha = max(0.0, min(1.0, 1.0 - gamma * self.h))
        assert self.alpha > 0.0, "gamma*h must be < 1 (alpha > 0) or gradients vanish"
        self.cfg_scale = cfg_scale
        self.sc_cfg = self_cond_cfg_scale
        self.noise_scale = config.denoiser_noise_scale
        self.t_eps = config.t_eps

    # ---- single-step pieces -------------------------------------------------
    def t_grid(self, k: int) -> float:
        return k * self.h

    def churn(self, z: torch.Tensor, k: int, eps: torch.Tensor, cond_seq, cond_mask):
        """Return (z~_k, t~_k). Last step (k == K-1) is ODE: no churn."""
        if k == self.K - 1:
            return z, self.t_grid(k)
        a = self.alpha
        z_t = restore_cond(a * z + (1.0 - a) * eps, cond_seq, cond_mask)
        return z_t, a * self.t_grid(k)

    def denoise(self, model, inp: DenoiserInput, cond_seq, cond_mask):
        """One denoiser call (with CFG). Returns x_k = clean estimate. Grad flows if enabled."""
        B = inp.z.shape[0]
        t_batch = torch.full((B,), inp.t, dtype=inp.z.dtype, device=inp.z.device)
        _, x = _forward_sample(
            model=model, z=inp.z, t_batch=t_batch, x_pred_prev=inp.x_prev, config=self.config,
            cfg_scale=self.cfg_scale, self_cond_cfg_scale=self.sc_cfg,
            cond_seq=cond_seq, cond_seq_mask=cond_mask,
        )
        return x

    def advance(self, inp: DenoiserInput, x_k: torch.Tensor, cond_seq, cond_mask):
        """z_{k+1} from (z~_k, t~_k, x_k)."""
        t_next = self.t_grid(inp.k + 1)
        denom = max(1.0 - inp.t, self.t_eps)
        z_next = inp.z + (t_next - inp.t) * (x_k - inp.z) / denom
        return restore_cond(z_next, cond_seq, cond_mask)

    def sample_eps(self, like: torch.Tensor, generator=None):
        if like.is_cuda or generator is None:
            return torch.randn_like(like) * self.noise_scale
        return torch.randn(like.shape, generator=generator, dtype=like.dtype).to(like.device) * self.noise_scale

    # ---- rollout --------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, model, cond_seq, cond_mask, z0: torch.Tensor, record: set[int] = frozenset(),
                stop_before: Optional[int] = None, generator=None,
                x_prev: Optional[torch.Tensor] = None, start_k: int = 0) -> Rollout:
        """Run steps start_k..K-1 (or until stop_before, exclusive). Records DenoiserInput at k in `record`."""
        z = restore_cond(z0, cond_seq, cond_mask)
        if x_prev is None:
            x_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        inputs: dict[int, DenoiserInput] = {}
        end = self.K if stop_before is None else stop_before
        for k in range(start_k, end):
            eps = self.sample_eps(z, generator)
            z_t, t_t = self.churn(z, k, eps, cond_seq, cond_mask)
            inp = DenoiserInput(k, z_t, t_t, x_prev)
            if k in record:
                inputs[k] = inp
            x_k = self.denoise(model, inp, cond_seq, cond_mask)
            z = self.advance(inp, x_k, cond_seq, cond_mask)
            x_prev = x_k
        return Rollout(z_final=z, inputs=inputs)

    def grad_step(self, model, inp_prev: DenoiserInput, cond_seq, cond_mask,
                  eps_next: Optional[torch.Tensor] = None, generator=None) -> DenoiserInput:
        """Differentiable transition: from the denoiser input at k-1 (treated as constant),
        one denoiser call WITH grad, then churn to the denoiser input at k.
        Both outputs (z~_k, x_{k-1}) carry grad w.r.t. theta."""
        inp_c = DenoiserInput(inp_prev.k, inp_prev.z.detach(), inp_prev.t, inp_prev.x_prev.detach())
        x_k = self.denoise(model, inp_c, cond_seq, cond_mask)              # grad
        z_next = self.advance(inp_c, x_k, cond_seq, cond_mask)             # grad via x_k
        k = inp_prev.k + 1
        if eps_next is None:
            eps_next = self.sample_eps(z_next, generator)
        z_t, t_t = self.churn(z_next, k, eps_next, cond_seq, cond_mask)    # grad via z_next
        return DenoiserInput(k, z_t, t_t, x_k)

    # ---- decoding -------------------------------------------------------------
    @torch.no_grad()
    def decode(self, model, z_final: torch.Tensor) -> torch.Tensor:
        """Token ids (B, L) from the decoder head at t=1 (ELF `_dlm_decode_batch`)."""
        B = z_final.shape[0]
        t_final = torch.full((B,), 1.0, dtype=z_final.dtype, device=z_final.device)
        sc = (torch.full((B,), float(self.sc_cfg), dtype=z_final.dtype, device=z_final.device)
              if self.config.num_self_cond_cfg_tokens > 0 else None)
        z_in = torch.cat([z_final, torch.zeros_like(z_final)], dim=-1) if self.config.self_cond_prob > 0 else z_final
        _, logits = model(z_in, t_final, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=True)
        return logits.argmax(dim=-1)
