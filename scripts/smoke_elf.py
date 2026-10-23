"""Smoke test: load ELF-B (OWT) on CPU/MPS, encode a Countdown prompt, roll out the
churn sampler with recorded states, take one differentiable step, decode.
usage: .venv/bin/python scripts/smoke_elf.py [--ckpt embedded-language-flows/ELF-B-owt-torch]
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import hcfrl.elf_path  # noqa
from transformers import AutoTokenizer
from configs.config import load_config_from_yaml, apply_config_overrides
from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.checkpoint_utils import load_checkpoint
from utils.train_utils import TrainState, get_optimizer
from utils.encoder_utils import build_self_attn_cond_masks, encode_text
from hcfrl.rl.sampler import ChurnSampler

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="embedded-language-flows/ELF-B-owt-torch")
ap.add_argument("--device", default="cpu")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--gamma", type=float, default=4.0)
a = ap.parse_args()
dev = torch.device(a.device)

cfg = load_config_from_yaml("configs/elf_b_countdown.yml")
cfg = apply_config_overrides(cfg, ["max_length=72", "max_input_length=16", "use_wandb=false"])
tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name)
enc_cfg, encoder = get_encoder(cfg.encoder_model_name, torch.float32)
encoder = encoder.to(dev).eval()
for p in encoder.parameters(): p.requires_grad_(False)
model = ELF_models[cfg.model](
    text_encoder_dim=enc_cfg.d_model, max_length=cfg.max_length,
    attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
    num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
    vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
    bottleneck_dim=cfg.bottleneck_dim).to(dev)
print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
state = TrainState(model=model, optimizer=get_optimizer(model, cfg, lr=1e-4), lr_scheduler=None,
                   ema_params1=TrainState.init_ema(model), step=0, epoch=0,
                   dropout_generator=torch.Generator(device="cpu").manual_seed(0))
t0 = time.time(); state, step = load_checkpoint(a.ckpt, state); print("loaded ckpt step", step, "in %.1fs" % (time.time() - t0))
model = state.model.to(dev)
if state.ema_params1: model.load_state_dict(state.ema_params1)
model.eval()

prompts = ["71 : 46 75 28 65 18", "24 : 7 9 1 7"]
L, Lc = cfg.max_length, cfg.max_input_length
ids = np.full((len(prompts), L), tok.eos_token_id, dtype=np.int64); cond_lens = []
for i, p in enumerate(prompts):
    c = tok(p, add_special_tokens=False)["input_ids"][:Lc]; ids[i, :len(c)] = c; cond_lens.append(len(c))
pos = np.arange(L)[None]; is_cond = pos < np.array(cond_lens)[:, None]; is_valid = np.ones_like(is_cond)
enc_mask, _, cond_mask = build_self_attn_cond_masks(is_cond, is_valid)
ids_t = torch.from_numpy(ids).to(dev); enc_mask_t = torch.from_numpy(enc_mask).to(dev); cond_mask_t = torch.from_numpy(cond_mask).to(dev)
cond_seq = encode_text(ids_t, enc_mask_t, encoder, cfg.latent_mean, cfg.latent_std, use_bf16=False).float()
print("cond_seq", tuple(cond_seq.shape))

S = ChurnSampler(cfg, K=a.K, gamma=a.gamma, cfg_scale=2.0)
g = torch.Generator().manual_seed(0)
z0 = torch.randn((len(prompts), L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale
z0 = z0.to(dev)
t0 = time.time()
ro = S.rollout(model, cond_seq, cond_mask_t, z0, record={4, 8}, generator=g)
print("rollout %d steps: %.1fs" % (a.K, time.time() - t0))
toks = S.decode(model, ro.z_final)
for i in range(len(prompts)):
    gen = toks[i, cond_lens[i]:].tolist()
    print(repr(prompts[i]), "->", repr(tok.decode(gen, skip_special_tokens=True)[:80]))

# differentiable step from the recorded input at k=4 -> input at k=5
model.train(False)
for p in model.parameters(): p.requires_grad_(True)
t0 = time.time()
nxt = S.grad_step(model, ro.inputs[4], cond_seq, cond_mask_t, generator=g)
assert nxt.k == 5 and nxt.z.requires_grad and nxt.x_prev.requires_grad
loss = (nxt.z[:, Lc:].pow(2).mean() + nxt.x_prev[:, Lc:].pow(2).mean())
loss.backward()
gn = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
print("grad_step + backward: %.1fs, grad norm %.3e" % (time.time() - t0, gn.item()))
print("SMOKE OK")
