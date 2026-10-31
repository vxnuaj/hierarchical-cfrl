"""Theory item T2: sign of the refit feedback Gamma (notes/theory.md, Prop. 5.3 / 5.5) on ELF.

The policy drifts from theta_0 (base) to theta_1 (a later RL checkpoint of a collapsing run). Interpolate
theta(eps) = theta_0 + eps (theta_1 - theta_0). For each critic and each eps: roll out theta(eps), cache trunk features of
the states at k_hi with verified rewards, refit the critic head from its pretrained weights with the RL recipe
(AdamW 1e-4, minibatch 64, --steps steps). On fixed probe states (theta_0 rollouts) measure the critic's input gradient
g_eps w.r.t. the self-conditioning input x_{k-1}, and the policy's displacement direction there,
v = d_{theta_1}(s_{k-1}) - d_{theta_0}(s_{k-1}) (the change of the clean estimate the drift produced).
Feedback along the drift: Gamma_v(eps) = <g_eps - g_0, v> / (eps |g_0| |v|)  (per probe state, averaged).
Positive: refitting on drifted states makes the critic reward further drift (destabilising, Prop. 5.3).
Negative: restoring (the linear-Gaussian prediction, Prop. 5.5).

usage: python -m hcfrl.analysis.refit_feedback --config configs/elf_b_game24_cd.yml --theta0 <policy_it0.pt> --theta1 <policy_it250.pt>
          --critics native=...,ma=... --train data/game24/train.jsonl --val data/game24/val.jsonl --out outputs/analysis/T2_refit_feedback.json
"""
from __future__ import annotations

import argparse, copy, json, random

import torch

from hcfrl.analysis.cross_critic import best_critic, parse
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", default="outputs/elf_b-game24-cd")
    ap.add_argument("--theta0", required=True); ap.add_argument("--theta1", required=True)
    ap.add_argument("--critics", required=True); ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--eps", default="0,0.5,1"); ap.add_argument("--n_data", type=int, default=1536); ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--k_hi", type=int, default=40); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    p0 = torch.load(a.theta0, map_location="cpu")["params"]; p1 = torch.load(a.theta1, map_location="cpu")["params"]

    def set_eps(e):
        sd = {k: (p0[k].float() + e * (p1[k].float() - p0[k].float())).to(p0[k].dtype) if p0[k].dtype.is_floating_point else p0[k] for k in p0}
        model.load_state_dict(sd); model.eval()

    # probe states from theta_0 and the displacement of the clean estimate theta_0 -> theta_1 on them
    set_eps(0.0)
    val = load_prompts(a.val)[: a.n_probe]; g = torch.Generator(device="cpu").manual_seed(777); probes = []
    for b0 in range(0, len(val), a.bs):
        pb = val[b0: b0 + a.bs]; cs, cm, gm, _ = env.encode(pb)
        z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad(), ac:
            ro = S.rollout(model, cs, cm, z0, record={a.k_hi - 1, a.k_hi}, stop_before=a.k_hi + 1, generator=g)
        probes.append(dict(prev=ro.inputs[a.k_hi - 1], cur=ro.inputs[a.k_hi], cs=cs, cm=cm, gm=gm))
    with torch.no_grad(), ac:
        x0s = [S.denoise(model, pr["prev"], pr["cs"], pr["cm"]).float() for pr in probes]
    set_eps(1.0)
    with torch.no_grad(), ac:
        vs = [(S.denoise(model, pr["prev"], pr["cs"], pr["cm"]).float() - x0) * pr["gm"][..., None] for pr, x0 in zip(probes, x0s)]

    # rollouts (features computed per critic) at each eps
    train = load_train_prompts(a.train); data_prompts = random.sample(train, a.n_data)
    states = {}
    for e in (float(x) for x in a.eps.split(",")):
        set_eps(e); g = torch.Generator(device="cpu").manual_seed(a.seed + 1); rec = []
        for b0 in range(0, len(data_prompts), a.bs):
            pb = data_prompts[b0: b0 + a.bs]; cs, cm, gm, cl = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cs, cm, z0, record={a.k_hi}, generator=g)
                r = env.reward(pb, env.decode_text(S.decode(model, ro.z_final), cl))
            rec.append((ro.inputs[a.k_hi], gm, r))
        states[e] = rec
        print(f"eps {e:g}: success {torch.cat([x[2] for x in rec]).mean():.3f}", flush=True)

    res = {}
    for cname, cpath in parse(a.critics):
        ck = torch.load(best_critic(cpath), map_location="cpu")
        critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False).to(dev)
        if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
        head0 = copy.deepcopy(ck["head"]); critic.eval()
        for p in critic.trunk.parameters(): p.requires_grad_(False)

        def grads():
            out = []
            for pr in probes:
                xi = pr["cur"].x_prev.detach().float().requires_grad_(True)
                o = critic(DenoiserInput(pr["cur"].k, pr["cur"].z.float(), pr["cur"].t, xi), None, None, pr["gm"])
                gx, = torch.autograd.grad(o.logit.float().sum(), xi); out.append(gx * pr["gm"][..., None])
            return out

        gs = {}
        for e, rec in states.items():
            critic.head.load_state_dict(head0)
            with torch.no_grad(), ac:
                H = torch.cat([critic.features(inp, None, None).float() for inp, _, _ in rec])
            Mm = torch.cat([gm for _, gm, _ in rec]); R = torch.cat([r for _, _, r in rec]).to(dev); t_hi = rec[0][0].t
            for p in critic.head.parameters(): p.requires_grad_(True)
            opt = torch.optim.AdamW(critic.head.parameters(), lr=1e-4)
            gg = torch.Generator(device="cpu").manual_seed(a.seed + 5)
            for _ in range(a.steps):
                i = torch.randint(0, len(R), (64,), generator=gg).to(dev)
                o = critic(DenoiserInput(a.k_hi, H[i], t_hi, None), None, None, Mm[i], h=H[i])
                loss = critic.loss(o, R[i], Mm[i]); opt.zero_grad(); loss.backward(); opt.step()
            for p in critic.head.parameters(): p.requires_grad_(False)
            gs[e] = grads(); del H
        e_list = sorted(gs)
        out = {}
        for e in e_list[1:]:
            vals = []
            for g0, ge, v in zip(gs[e_list[0]], gs[e], vs):
                num = ((ge - g0) * v).sum((1, 2)); den = g0.flatten(1).norm(dim=1) * v.flatten(1).norm(dim=1)
                vals.append((num / den.clamp_min(1e-12) / e).cpu())
            vals = torch.cat(vals)
            out[f"{e:g}"] = {"gamma_v_mean": vals.mean().item(), "gamma_v_median": vals.median().item(), "frac_positive": (vals > 0).float().mean().item()}
        # baseline: gradient alignment with the drift before any refit
        base = torch.cat([((g0 * v).sum((1, 2)) / (g0.flatten(1).norm(dim=1) * v.flatten(1).norm(dim=1)).clamp_min(1e-12)).cpu() for g0, v in zip(gs[e_list[0]], vs)])
        out["cos_g0_drift"] = base.mean().item()
        res[cname] = out
        print(f"critic {cname}: " + json.dumps(out), flush=True)
        del critic; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("saved", a.out)


if __name__ == "__main__":
    main()
