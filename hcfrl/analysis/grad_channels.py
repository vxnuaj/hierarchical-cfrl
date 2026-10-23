"""Diagnostics on the pathwise actor gradient (CFRL App. + our hierarchy question):
  (a) channel decomposition: how much of ||grad_theta V|| flows through the re-noised state z~_k,
      the self-conditioning estimate x_{k-1}, and (for H-ELF) the cluster channels of each;
  (b) gradient SNR / half-split cosine across independent batches at several steps k.

usage:
  python -m hcfrl.analysis.grad_channels --config configs/helf_b_game24.yml --ckpt outputs/helf_b-game24 \
      --critic outputs/critic/game24_hier_2d/critic_ep6.pt --prompts data/game24/train_prompts.jsonl \
      --ks 8,16,24,32,40 --B 32 --reps 4 --out outputs/diag/game24_hier_grad.json
"""
from __future__ import annotations

import argparse, json, os, random

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--critic", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--ks", default="8,16,24,32,40")
    ap.add_argument("--B", type=int, default=32); ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    ck = torch.load(a.critic, map_location="cpu")
    critic = Critic(model, cfg, mode=ck["mode"], blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
    if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
    critic.head.load_state_dict(ck["head"]); critic.eval()
    model.train(False)
    params = [p for p in model.parameters()]
    for p in params: p.requires_grad_(True)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0); D = model.text_encoder_dim - n_cl
    prompts = load_prompts(a.prompts)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    results = {}

    def masked(t, keep_tok: bool, keep_cl: bool):
        m = torch.zeros(t.shape[-1], device=t.device, dtype=t.dtype)
        if keep_tok: m[:D] = 1
        if keep_cl and n_cl: m[D:] = 1
        return t * m + (t * (1 - m)).detach()   # gradient flows only through kept channels

    for k in [int(x) for x in a.ks.split(",")]:
        grads = {"full": [], "z": [], "x": [], "z_tok": [], "z_cl": [], "x_tok": [], "x_cl": []}
        for rep in range(a.reps):
            pb = random.sample(prompts, a.B)
            cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
            z0 = (torch.randn((a.B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1}, stop_before=k, generator=g)
            prev = ro.inputs[k - 1]
            eps = S.sample_eps(prev.z, g)
            variants = {"full": (True, True, True, True), "z": (True, True, False, False), "x": (False, False, True, True),
                        "z_tok": (True, False, False, False), "z_cl": (False, True, False, False),
                        "x_tok": (False, False, True, False), "x_cl": (False, False, False, True)}
            for name, (zt, zc, xt, xc) in variants.items():
                if n_cl == 0 and name in ("z_cl", "x_cl", "z_tok", "x_tok"): continue
                model.zero_grad(set_to_none=True)
                with ac:
                    nxt = S.grad_step(model, prev, cond_seq, cond_mask, eps_next=eps)
                    nz = masked(nxt.z, zt, zc); nx = masked(nxt.x_prev, xt, xc)
                    out = critic(DenoiserInput(nxt.k, nz, nxt.t, nx), cond_seq, cond_mask, gen_mask)
                    (-out.logit.mean()).backward()
                grads[name].append(flat_grad(params).float().cpu())
        res = {}
        for name, gl in grads.items():
            if not gl: continue
            G = torch.stack(gl); mean = G.mean(0)
            res[name] = {"norm_mean": mean.norm().item(), "norm_per_batch": [x.norm().item() for x in gl]}
            if len(gl) >= 2:
                h = len(gl) // 2; g1, g2 = G[:h].mean(0), G[h:].mean(0)
                res[name]["half_split_cos"] = torch.nn.functional.cosine_similarity(g1, g2, dim=0).item()
                res[name]["snr"] = (mean.norm() / (G - mean).norm(dim=1).mean().clamp_min(1e-12)).item()
        full = res["full"]["norm_mean"]
        res["share"] = {n: (res[n]["norm_mean"] / max(full, 1e-12)) for n in res if n != "full"}
        results[k] = res
        print(json.dumps({"k": k, "share": {n: round(v, 3) for n, v in res["share"].items()},
                          "snr_full": round(res["full"].get("snr", float("nan")), 3),
                          "cos_full": round(res["full"].get("half_split_cos", float("nan")), 3)}), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
