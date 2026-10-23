"""Cluster-augmented encoder for H-ELF.

ELF's latent per position is the normalized T5 encoding (dim D=512). H-ELF appends n extra channels
holding the coarse stream: cluster_scale * onehot(Gamma(token)). Both streams are then noised, denoised,
self-conditioned and restored by the UNCHANGED ELF machinery, because they live in one (B, L, D+n) tensor.

`encode_text` in ELF computes (encoder(...) - latent_mean) / latent_std, so this wrapper emits the cluster
channels pre-scaled such that after that normalization they equal cluster_scale * onehot.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_gamma(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu")
    return d["gamma"].long()


class ClusterAugmentedEncoder(nn.Module):
    def __init__(self, base: nn.Module, gamma: torch.Tensor, n: int, cluster_scale: float,
                 latent_mean: float, latent_std: float):
        super().__init__()
        self.base = base
        self.register_buffer("gamma", gamma.long(), persistent=False)
        self.n, self.cluster_scale = n, cluster_scale
        self.latent_mean, self.latent_std = latent_mean, latent_std

    def forward(self, input_ids=None, attention_mask=None, deterministic=True, **kw):
        lat = self.base(input_ids=input_ids, attention_mask=attention_mask, deterministic=deterministic, **kw)
        ids = input_ids.clamp(0, self.gamma.numel() - 1)
        c = F.one_hot(self.gamma[ids], self.n).to(lat.dtype) * self.cluster_scale
        c = c * self.latent_std + self.latent_mean   # undo encode_text's normalization in advance
        return torch.cat([lat, c], dim=-1)

    def cluster_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gamma[input_ids.clamp(0, self.gamma.numel() - 1)]


def wrap_encoder_if_hier(encoder: nn.Module, config) -> nn.Module:
    n = int(getattr(config, "n_clusters", 0) or 0)
    if n <= 0:
        return encoder
    gamma = load_gamma(config.cluster_map)
    return ClusterAugmentedEncoder(encoder, gamma, n, float(config.cluster_scale),
                                   float(config.latent_mean), float(config.latent_std))
