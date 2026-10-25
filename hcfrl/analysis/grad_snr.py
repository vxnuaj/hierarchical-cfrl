"""Quality of the pathwise actor gradient at a fixed policy (CFRL Sec. 6 diagnostic, flat vs H-ELF).
For N prompts at denoiser step k: per-prompt gradients g_i of the critic logit w.r.t. theta.
Reports  snr = ||mean g|| / sqrt(sum Var)  (per-coordinate variance summed),
         cos_half(N') = cosine between the mean gradients of two disjoint halves of N' prompts (3 random splits),
         and, for H-ELF, the share of the critic's input-gradient norm that falls on the cluster channels of (z_k, x_{k-1}).
usage: python -m hcfrl.analysis.grad_snr --config ... --ckpt ... --critic outputs/critic/<tag>_2d/critic_epN.pt \
          --prompts data/game24/train_prompts.jsonl --n 256 --ks 8,24,40 --out outputs/analysis/gsnr_<tag>.json
"""
from __future__ import annotations

import argparse, json, os, random

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--critic", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ks", default="8,24,40"); ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    ck = torch.load(a.critic, map_location="cpu")
    critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False, blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
    if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
    critic.head.load_state_dict(ck["head"]); critic.eval()
    for p in critic.parameters(): p.requires_grad_(False)
    model.train(False)
    for p in model.parameters(): p.requires_grad_(True)
    params = [p for p in model.parameters()]
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0)
    prompts = random.sample(load_train_prompts(a.prompts), a.n)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    res = {}
    for k in (int(x) for x in a.ks.split(",")):
        s1 = None; s2 = None; grads = []; cl_share_z, cl_share_x, n_in = 0.0, 0.0, 0
        for b0 in range(0, a.n, a.bs):
            pb = prompts[b0: b0 + a.bs]
            cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1}, stop_before=k, generator=g)
            prev = ro.inputs[k - 1]
            for i in range(len(pb)):
                sel = torch.tensor([i], device=dev)
                sub = DenoiserInput(k - 1, prev.z[sel], prev.t, prev.x_prev[sel])
                with ac:
                    nxt = S.grad_step(model, sub, cond_seq[sel], cond_mask[sel], generator=g)
                    zz = nxt.z.detach().float().requires_grad_(True); xx = nxt.x_prev.detach().float().requires_grad_(True)
                    out = critic(nxt, cond_seq[sel], cond_mask[sel], gen_mask[sel])
                model.zero_grad(set_to_none=True); out.logit.sum().backward()
                gi = flat_grad(params)
                grads.append(gi.half().cpu())
                s1 = gi if s1 is None else s1 + gi; s2 = gi * gi if s2 is None else s2 + gi * gi
                if n_cl > 0:   # input-gradient split: recompute critic on leaf copies of the state
                    with ac:
                        o2 = critic(DenoiserInput(nxt.k, zz, nxt.t, xx), cond_seq[sel], cond_mask[sel], gen_mask[sel])
                    gz, gx = torch.autograd.grad(o2.logit.sum(), [zz, xx])
                    cl_share_z += (gz[..., -n_cl:].norm() ** 2 / gz.norm() ** 2).item(); cl_share_x += (gx[..., -n_cl:].norm() ** 2 / gx.norm() ** 2).item(); n_in += 1
        N = len(grads); mean = s1 / N; var = s2 / N - mean * mean
        snr = (mean.norm() / var.clamp_min(0).sum().sqrt()).item()
        cos = {}
        for Np in [n for n in (64, 128, 256, 512, 1024) if n <= N]:
            cs = []
            for rep in range(3):
                idx = random.sample(range(N), Np); A, B = idx[: Np // 2], idx[Np // 2:]
                gA = sum(grads[i].float() for i in A) / len(A); gB = sum(grads[i].float() for i in B) / len(B)
                cs.append(torch.nn.functional.cosine_similarity(gA, gB, dim=0).item())
            cos[str(Np)] = sum(cs) / len(cs)
        mean_norm_i = sum(gi.float().norm().item() for gi in grads) / N
        res[str(k)] = {"snr": snr, "cos_half": cos, "mean_grad_norm": mean.norm().item(), "mean_per_prompt_norm": mean_norm_i,
                       "cluster_share_z": cl_share_z / n_in if n_in else None, "cluster_share_x": cl_share_x / n_in if n_in else None}
        print(f"k={k}: snr {snr:.4f} cos_half {cos} |mean g| {mean.norm().item():.3f} mean|g_i| {mean_norm_i:.3f} "
              f"cluster share z/x {res[str(k)]['cluster_share_z']}/{res[str(k)]['cluster_share_x']}", flush=True)
        del grads
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
