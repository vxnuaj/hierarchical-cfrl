"""Debug: gradient norm of the actor objective with actor_channels in {all, tokens, clusters}."""
import sys, torch
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts, split_channels
cfgp, ckpt, crit, prompts = sys.argv[1:5]
dev = torch.device("cuda")
cfg, tok, enc, model = load_policy(cfgp, ckpt, dev); env = TaskEnv(tok, enc, cfg, dev); S = ChurnSampler(cfg, K=64, gamma=56.0)
ck = torch.load(crit, map_location="cpu")
critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False, blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
critic.trunk.load_state_dict(ck["trunk"]); critic.head.load_state_dict(ck["head"]); critic.eval()
print("blind_clusters", critic.blind_clusters, "n_clusters", critic.n_clusters, "self_cond_prob", cfg.self_cond_prob)
model.train(False)
for p in model.parameters(): p.requires_grad_(True)
n_cl = int(cfg.n_clusters); pb = load_train_prompts(prompts)[:4]
cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
g = torch.Generator(device="cpu").manual_seed(0)
z0 = (torch.randn((4, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
for keep in ["all", "tokens", "clusters"]:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ro = S.rollout(model, cond_seq, cond_mask, z0, record={7}, stop_before=8, generator=g)
        prev = ro.inputs[7]
        nxt = S.grad_step(model, DenoiserInput(7, prev.z, prev.t, prev.x_prev), cond_seq, cond_mask, generator=g)
        print(keep, "pre-split requires_grad z/x:", nxt.z.requires_grad, nxt.x_prev.requires_grad, nxt.z.dtype, nxt.x_prev.dtype, nxt.z.shape)
        if keep != "all": nxt = split_channels(nxt, n_cl, keep)
        print(keep, "post-split requires_grad z/x:", nxt.z.requires_grad, nxt.x_prev.requires_grad)
        out = critic(nxt, cond_seq, cond_mask, gen_mask)
    print(keep, "logit requires_grad", out.logit.requires_grad, "grad_fn", out.logit.grad_fn)
    model.zero_grad(set_to_none=True)
    if out.logit.requires_grad:
        out.logit.mean().backward()
        print(keep, "grad norm", torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item())
