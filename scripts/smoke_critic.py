"""Smoke: critic features/value/loss on recorded states, and the actor objective
(ascend critic logit through one differentiable sampler step). CPU, OWT checkpoint."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import hcfrl.elf_path  # noqa
from transformers import AutoTokenizer
from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.checkpoint_utils import load_weights_only
from utils.train_utils import TrainState
from utils.encoder_utils import build_self_attn_cond_masks, encode_text
from hcfrl.rl.sampler import ChurnSampler
from hcfrl.rl.critic import Critic, rank_metrics

dev = torch.device("cpu")
cfg = load_config_from_yaml("configs/elf_b_game24.yml")
tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name)
enc_cfg, encoder = get_encoder(cfg.encoder_model_name, torch.float32); encoder = encoder.to(dev).eval()
model = ELF_models[cfg.model](text_encoder_dim=enc_cfg.d_model, max_length=cfg.max_length, attn_drop=0, proj_drop=0,
    num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
    vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens, bottleneck_dim=cfg.bottleneck_dim).to(dev)
state = TrainState(model=model, optimizer=None, ema_params1={}, step=0, epoch=0)
load_weights_only(cfg.init_from, state); model.eval()

prompts = ["24 : 7 9 1 7", "24 : 1 1 4 6", "24 : 2 3 4 5", "24 : 8 8 3 3"]
L, Lc = cfg.max_length, cfg.max_input_length
ids = np.full((len(prompts), L), tok.eos_token_id, dtype=np.int64); cl = []
for i, p in enumerate(prompts):
    c = tok(p, add_special_tokens=False)["input_ids"][:Lc]; ids[i, :len(c)] = c; cl.append(len(c))
pos = np.arange(L)[None]; is_cond = pos < np.array(cl)[:, None]
enc_mask, _, cond_mask = build_self_attn_cond_masks(is_cond, np.ones_like(is_cond))
cond_seq = encode_text(torch.from_numpy(ids), torch.from_numpy(enc_mask), encoder, cfg.latent_mean, cfg.latent_std, use_bf16=False).float()
cond_mask = torch.from_numpy(cond_mask); gen_mask = 1.0 - cond_mask

S = ChurnSampler(cfg, K=16, gamma=4.0)
g = torch.Generator().manual_seed(0)
z0 = torch.randn((len(prompts), L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale
ro = S.rollout(model, cond_seq, cond_mask, z0, record={4, 8, 12}, generator=g)
reward = torch.tensor([1., 0., 1., 0.])  # fake labels

critic = Critic(model, cfg, mode="2d")
print("critic head params:", sum(p.numel() for p in critic.head_parameters()))
t0 = time.time()
out = critic(ro.inputs[8], cond_seq, cond_mask, gen_mask)
loss = critic.loss(out, reward, gen_mask); loss.backward()
print("critic fwd/bwd %.2fs loss %.3f value %s" % (time.time() - t0, loss.item(), out.value.detach().numpy().round(3)))
assert all(p.grad is not None for p in critic.head_parameters())
assert all(p.grad is None for p in critic.trunk.parameters())
print("rank metrics:", rank_metrics(out.value.detach(), reward, groups=torch.tensor([0, 0, 1, 1])))

# actor objective: ascend critic logit through one differentiable step (k=8 -> 9)
for p in model.parameters(): p.requires_grad_(True)
model.zero_grad()
nxt = S.grad_step(model, ro.inputs[8], cond_seq, cond_mask, generator=g)
out2 = critic(nxt, cond_seq, cond_mask, gen_mask)
(-out2.logit.mean()).backward()
gn = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
print("actor grad norm through critic: %.3e; critic head grads untouched by actor step: %s" % (gn.item(), all(p.grad is not None for p in critic.head_parameters())))
# 1d variant
c1 = Critic(model, cfg, mode="1d"); o1 = c1(ro.inputs[8], cond_seq, cond_mask, gen_mask); c1.loss(o1, reward, gen_mask).backward()
print("1d ok, value", o1.value.detach().numpy().round(3))
print("CRITIC SMOKE OK")
