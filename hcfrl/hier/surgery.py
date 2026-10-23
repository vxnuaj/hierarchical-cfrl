"""Widen a flat ELF checkpoint (latent dim D) to an H-ELF model (latent dim D + n) so that, at
initialization, the widened model computes EXACTLY the same token-stream outputs as the original and
ignores the cluster channels (zero-initialized input/output weights), except that the self-conditioning
input projection passes the cluster latent through (identity), so the coarse stream reaches the trunk
only through weights that start at zero and are learned.

Affected parameters (ELF naming):
  self_cond_proj.weight (D', 2D'), .bias (D')
  text_proj.proj1.weight (bottleneck, D')
  final_layer.linear.weight (D', hidden), .bias (D')
  proj_kernel (hidden, D'), proj_bias (D')
  unembed_kernel (D', vocab)
"""
from __future__ import annotations

import torch


def widen_state_dict(sd: dict, D: int, n: int) -> dict:
    Dp = D + n
    out = dict(sd)

    W = sd["self_cond_proj.weight"]                      # (D, 2D)
    Wn = torch.zeros((Dp, 2 * Dp), dtype=W.dtype)
    Wn[:D, :D] = W[:, :D]                                 # z1 -> rows 0..D
    Wn[:D, Dp:Dp + D] = W[:, D:]                          # x1_prev -> rows 0..D
    Wn[D:, D:Dp] = torch.eye(n, dtype=W.dtype)            # z2 passthrough
    out["self_cond_proj.weight"] = Wn
    b = sd["self_cond_proj.bias"]
    out["self_cond_proj.bias"] = torch.cat([b, torch.zeros(n, dtype=b.dtype)])

    P = sd["text_proj.proj1.weight"]                     # (bottleneck, D)
    out["text_proj.proj1.weight"] = torch.cat([P, torch.zeros((P.shape[0], n), dtype=P.dtype)], dim=1)

    Fw = sd["final_layer.linear.weight"]                 # (D, hidden)
    out["final_layer.linear.weight"] = torch.cat([Fw, torch.zeros((n, Fw.shape[1]), dtype=Fw.dtype)], dim=0)
    Fb = sd["final_layer.linear.bias"]
    out["final_layer.linear.bias"] = torch.cat([Fb, torch.zeros(n, dtype=Fb.dtype)])

    K = sd["proj_kernel"]                                # (hidden, D)
    out["proj_kernel"] = torch.cat([K, torch.zeros((K.shape[0], n), dtype=K.dtype)], dim=1)
    kb = sd["proj_bias"]
    out["proj_bias"] = torch.cat([kb, torch.zeros(n, dtype=kb.dtype)])
    U = sd["unembed_kernel"]                             # (D, vocab)
    out["unembed_kernel"] = torch.cat([U, torch.zeros((n, U.shape[1]), dtype=U.dtype)], dim=0)
    return out


def maybe_widen(sd: dict, model) -> dict:
    """Widen `sd` if the model's latent dim exceeds the checkpoint's."""
    D_ckpt = sd["final_layer.linear.weight"].shape[0]
    Dp = model.text_encoder_dim
    if Dp == D_ckpt:
        return sd
    if Dp < D_ckpt:
        raise ValueError(f"model latent dim {Dp} < checkpoint {D_ckpt}")
    return widen_state_dict(sd, D_ckpt, Dp - D_ckpt)
