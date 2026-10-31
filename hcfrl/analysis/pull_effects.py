"""What does ascending each critic do to the ANSWER? Output-space version of the actor step, policy-independent.

For fixed probe states (val prompts, fixed noise) the base policy is rolled out to step k-1; x = d_theta(s_{k-1}) is the
clean estimate the actor would move. For each critic C: G_C = d l_C / d x through the sampler transition (both paths,
as in the actor). Then x' = x + eps |x| G_C / |G_C| (per sequence, generated positions), the state at k is rebuilt with
the same churn noise, the sampler is continued to the end with fixed noise, and the answer is decoded and verified.
Reported per critic and eps: logit gain of C and of every other critic, fraction of decoded tokens changed, accuracy
change, and the radial share of G_C (cos^2 between G_C and x per position: moving along x only rescales the latent,
which does not change the argmax decode of a nearly one-hot-like latent).

usage: python -m hcfrl.analysis.pull_effects --config ... --ckpt ... --policy <policy_it0.pt> --critics a=...,b=... \
          --val data/game24/val.jsonl --out outputs/analysis/pull_effects.json
"""
from __future__ import annotations

import argparse, json, time

import torch

from hcfrl.analysis.cross_critic import best_critic, parse
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", default="outputs/elf_b-game24-cd")
    ap.add_argument("--policy", required=True); ap.add_argument("--critics", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--ks", default="24,40"); ap.add_argument("--eps", default="0.01,0.03,0.1")
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    sd = torch.load(a.policy, map_location="cpu"); model.load_state_dict(sd["params"] if "params" in sd else sd); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    critics = {}
    for name, path in parse(a.critics):
        ck = torch.load(best_critic(path), map_location="cpu")
        c = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False).to(dev)
        if ck.get("trunk") is not None: c.trunk.load_state_dict(ck["trunk"])
        c.head.load_state_dict(ck["head"]); c.eval()
        for p in c.parameters(): p.requires_grad_(False)
        critics[name] = c
    names = list(critics); ks = [int(x) for x in a.ks.split(",")]; epss = [float(x) for x in a.eps.split(",")]
    val = load_prompts(a.val)[: a.n]

    def state_from(x_k, prev, eps_next, cs, cm):
        z_next = S.advance(prev, x_k, cs, cm)
        z_t, t_t = S.churn(z_next, prev.k + 1, eps_next, cs, cm)
        return DenoiserInput(prev.k + 1, z_t, t_t, x_k), z_next

    def finish(z_next, x_k, k, cs, cm, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ro = S.rollout(model, cs, cm, z_next, generator=g, x_prev=x_k, start_k=k)
        return S.decode(model, ro.z_final)

    res = {k: {"base": {}, "steps": {}} for k in ks}
    t0 = time.time()
    for k in ks:
        acc_tot = {}; base_correct = 0; ntok = 0; nseq = 0
        agg = {(c, e): {"dacc": 0.0, "tok_changed": 0.0, "dlogit": {m: 0.0 for m in names}} for c in names for e in epss}
        radial = {c: [] for c in names}
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        for b0 in range(0, len(val), a.bs):
            pb = val[b0: b0 + a.bs]
            cs, cm, gm, cl = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad():
                ro = S.rollout(model, cs, cm, z0, record={k - 1}, stop_before=k, generator=g)
                prev = ro.inputs[k - 1]; x0 = S.denoise(model, prev, cs, cm).float()
            eps_next = S.sample_eps(prev.z, g); fseed = int(torch.randint(0, 2**31 - 1, (1,), generator=g))
            M3 = gm[..., None]
            with torch.no_grad():
                s0, zn0 = state_from(x0, prev, eps_next, cs, cm)
                l0 = {m: critics[m](s0, cs, cm, gm).logit.float() for m in names}
                dec0 = finish(zn0, x0, k, cs, cm, fseed)
                r0 = env.reward(pb, env.decode_text(dec0, cl))
            base_correct += r0.sum().item(); nseq += len(pb); ntok += (gm.sum()).item()
            xn = (x0 * M3).norm(dim=(1, 2), keepdim=True)
            for c in names:
                xv = x0.clone().requires_grad_(True)
                st, _ = state_from(xv, prev, eps_next, cs, cm)
                G, = torch.autograd.grad(critics[c](st, cs, cm, gm).logit.float().sum(), xv)
                G = G * M3
                cosp = ((G * x0).sum(-1) ** 2 / (G.pow(2).sum(-1) * x0.pow(2).sum(-1)).clamp_min(1e-30))
                radial[c].append(cosp[gm.bool()].cpu())
                u = G / G.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                for e in epss:
                    x1 = x0 + e * xn * u
                    with torch.no_grad():
                        s1, zn1 = state_from(x1, prev, eps_next, cs, cm)
                        for m in names:
                            agg[(c, e)]["dlogit"][m] += (critics[m](s1, cs, cm, gm).logit.float() - l0[m]).sum().item()
                        dec1 = finish(zn1, x1, k, cs, cm, fseed)
                        r1 = env.reward(pb, env.decode_text(dec1, cl))
                        agg[(c, e)]["dacc"] += (r1 - r0).sum().item()
                        agg[(c, e)]["tok_changed"] += ((dec1 != dec0) & gm.bool()).sum().item()
        res[k]["base"] = {"acc": base_correct / nseq}
        for (c, e), v in agg.items():
            res[k]["steps"][f"{c}@{e:g}"] = {"dacc": v["dacc"] / nseq, "tok_changed": v["tok_changed"] / ntok,
                                            "dlogit": {m: v["dlogit"][m] / nseq for m in names}}
        res[k]["radial_share"] = {c: torch.cat(radial[c]).mean().item() for c in names}
        print(f"k={k} base acc {res[k]['base']['acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        print("  radial share: " + "  ".join(f"{c} {res[k]['radial_share'][c]:.3f}" for c in names), flush=True)
        for e in epss:
            for c in names:
                r = res[k]["steps"][f"{c}@{e:g}"]
                print(f"  eps {e:g} along {c:7s}: dacc {100 * r['dacc']:+5.1f}  tokens changed {100 * r['tok_changed']:5.2f}%  dlogit "
                      + " ".join(f"{m}:{r['dlogit'][m]:+.0f}" for m in names), flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("saved", a.out)


if __name__ == "__main__":
    main()
