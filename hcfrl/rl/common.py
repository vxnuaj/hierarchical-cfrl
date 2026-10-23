"""Shared loading helpers for RL scripts."""
from __future__ import annotations

import torch

import hcfrl.elf_path  # noqa: F401
from transformers import AutoTokenizer
from configs.config import load_config_from_yaml, apply_config_overrides  # type: ignore
from modules.t5_encoder import get_encoder  # type: ignore
from modules.model import ELF_models  # type: ignore
from utils.checkpoint_utils import load_weights_only, find_latest_checkpoint  # type: ignore
from utils.train_utils import TrainState  # type: ignore


def build_model(cfg, tok, enc_dim: int, device):
    return ELF_models[cfg.model](
        text_encoder_dim=enc_dim, max_length=cfg.max_length,
        attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
        num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
        bottleneck_dim=cfg.bottleneck_dim,
    ).to(device)


def load_policy(config_path: str, ckpt: str, device, overrides: list[str] | None = None, use_ema: bool = True):
    """Returns (cfg, tokenizer, encoder, model) with the model in eval mode loaded from `ckpt`
    (a checkpoint file, an output dir with checkpoint_* files, or an HF repo id). EMA weights by default."""
    cfg = load_config_from_yaml(config_path)
    if overrides:
        cfg = apply_config_overrides(cfg, overrides)
    tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name)
    enc_cfg, encoder = get_encoder(cfg.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    latent_dim = enc_cfg.d_model
    if int(getattr(cfg, "n_clusters", 0) or 0) > 0:  # H-ELF
        from hcfrl.hier.encoder import wrap_encoder_if_hier
        encoder = wrap_encoder_if_hier(encoder, cfg).to(device).eval()
        latent_dim += cfg.n_clusters
    model = build_model(cfg, tok, latent_dim, device)
    state = TrainState(model=model, optimizer=None, ema_params1={}, step=0, epoch=0)
    path = ckpt
    if not ckpt.startswith("embedded-language-flows/") and "checkpoint_" not in ckpt:
        path = find_latest_checkpoint(ckpt) or ckpt
    load_weights_only(path, state, use_ema=use_ema)
    model.eval()
    return cfg, tok, encoder, model
