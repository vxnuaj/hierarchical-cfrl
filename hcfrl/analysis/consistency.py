"""Tests of the "coarse rail" conjecture (paper Sec. 4.5): does the critic reward inconsistency between the
coarse channels and the decoded tokens, and does the hierarchical policy keep its coarse channels consistent
during RL while runs that collapse lose that consistency?

Two modes.

--mode trace: for each policy_it*.pt of one or more RL runs (H-ELF), roll out --n val prompts under the
  deployment sampler, record the clean estimate x_{k-1} at the steps in --ks, and report per step
    cons   : fraction of generated positions where argmax(coarse channels) == Gamma(decoded token)
    margin : mean (coarse channel at Gamma(decoded token) - max other coarse channel) / cluster_scale
    tok_nrm: mean per-position norm of the token channels of x_{k-1} (drift of the token stream)
  A run that stays consistent while collapsing refutes the conjecture; a run that loses consistency as it
  collapses supports it.

--mode probe: base policy + its critic (flat or hier). Roll out --n prompts to step k-1, take the model's
  clean estimate x_k with the denoiser call at k-1, build the critic's input at k from a displaced x_k
  (advance + churn with the same noise, so the displacement reaches both z~_k and the self-conditioning
  input), and report the change in the pooled critic logit l_phi (mean over sequences) for:
    recons    : coarse channels re-set to beta*onehot(Gamma(decode(x_k)))         (hier; should be ~0)
    wrongcls  : coarse channels set to beta*onehot(random other class)            (hier; inconsistent)
    zerocl    : coarse channels zeroed                                           (hier)
    tok_rand  : token channels += eps*|x_tok,i| * random unit direction, coarse fixed  (both)
    tok_pull  : token channels += eps*|x_tok,i| * (P_tok G)_i / |(P_tok G)_i|, coarse fixed  (both; G = dl/dx_k)
    full_pull : same along G on all channels                                     (hier)
  with eps in --eps, plus the fraction of generated positions whose decode changed under each displacement
  and the coarse share rho_cl = |P_cl G|^2 / |G|^2.

usage:
  python -m hcfrl.analysis.consistency --mode trace --config configs/helf_b_game24_cd.yml \
      --ckpt outputs/helf_b-game24-cd/checkpoint_2800 --prompts data/game24/val.jsonl --n 256 \
      --runs outputs/rl/game24cd_hier_svg2d_s0,outputs/rl/game24cd_hier_svg2d_clgrad_s0 --out outputs/analysis/cons_trace.json
  python -m hcfrl.analysis.consistency --mode probe --config configs/helf_b_game24_cd.yml --ckpt ... \
      --critic outputs/critic/game24cd_hier_2d/critic_ep2.pt --prompts data/game24/val.jsonl --n 256 --out outputs/analysis/cons_probe_hier.json
"""
from __future__ import annotations

import argparse, glob, json, os, random, re

import torch
import torch.nn.functional as F

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


def batches(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def coarse_stats(x, dec_ids, gen_mask, eos, gamma, n_cl, beta):
    """x: (B,L,D) clean estimate; dec_ids: (B,L) decoded ids of x. Returns dict of sums and the count."""
    D = x.shape[-1] - n_cl
    m = gen_mask.bool() & (dec_ids != eos)
    if n_cl == 0:                                               # flat run: token-norm only
        tok_nrm = x[..., :D].float().norm(dim=-1) * m
        return {"cons": float("nan"), "margin": float("nan"), "tok_nrm": tok_nrm.sum().item()}, m.sum().item()
    cl = x[..., D:].float()                                     # (B,L,n)
    g = gamma.to(x.device)[dec_ids]                             # (B,L) class of decoded token
    cons = (cl.argmax(-1) == g) & m
    at = cl.gather(-1, g[..., None]).squeeze(-1)
    other = cl.scatter(-1, g[..., None], float("-inf")).max(-1).values
    margin = ((at - other) / beta) * m
    tok_nrm = x[..., :D].float().norm(dim=-1) * m
    return {"cons": cons.sum().item(), "margin": margin.sum().item(), "tok_nrm": tok_nrm.sum().item()}, m.sum().item()


def mode_trace(a, cfg, tok, encoder, model, dev):
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0)
    gamma = torch.load(cfg.cluster_map, map_location="cpu")["gamma"].long() if n_cl else None
    beta = float(getattr(cfg, "cluster_scale", 1.0) or 1.0)
    eos = tok.eos_token_id
    prompts = load_prompts(a.prompts)[: a.n]
    ks = [int(x) for x in a.ks.split(",")]
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    results = {}
    runs = [r for r in a.runs.split(",") if r]
    for run in runs:
        fs = sorted(glob.glob(os.path.join(run, "policy_it*.pt")), key=lambda f: int(re.findall(r"policy_it(\d+)", f)[0]))
        for f in fs:
            it = int(re.findall(r"policy_it(\d+)", f)[0])
            model.load_state_dict(torch.load(f, map_location=dev)["params"]); model.eval()
            g = torch.Generator(device="cpu").manual_seed(a.seed)
            agg = {k: [{"cons": 0.0, "margin": 0.0, "tok_nrm": 0.0}, 0] for k in ks}
            with torch.no_grad(), ac:
                for pb in batches(prompts, a.B):
                    cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
                    z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
                    ro = S.rollout(model, cond_seq, cond_mask, z0, record=set(ks), generator=g)
                    for k in ks:
                        x = ro.inputs[k].x_prev                 # clean estimate x_{k-1}
                        dec = S.decode(model, x.float())
                        st, cnt = coarse_stats(x, dec, gen_mask, eos, gamma, n_cl, beta)
                        for key in st: agg[k][0][key] += st[key]
                        agg[k][1] += cnt
            row = {str(k): {key: agg[k][0][key] / max(agg[k][1], 1) for key in agg[k][0]} for k in ks}
            results[f"{os.path.basename(run)}/{it}"] = row
            print(json.dumps({"run": os.path.basename(run), "it": it,
                              **{f"k{k}": {key: round(v, 4) for key, v in row[str(k)].items()} for k in ks}}), flush=True)
    return results


def mode_probe(a, cfg, tok, encoder, model, dev):
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0); D = model.text_encoder_dim - n_cl
    gamma = torch.load(cfg.cluster_map, map_location="cpu")["gamma"].long().to(dev) if n_cl else None
    beta = float(getattr(cfg, "cluster_scale", 0.0) or 0.0)
    eos = tok.eos_token_id
    ck = torch.load(a.critic, map_location="cpu")
    critic = Critic(model, cfg, mode=ck["mode"], blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
    if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
    critic.head.load_state_dict(ck["head"]); critic.eval()
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    prompts = load_prompts(a.prompts)[: a.n]
    ks = [int(x) for x in a.ks.split(",")]
    epss = [float(x) for x in a.eps.split(",")]
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    results = {}

    def state_from(x_k, inp_prev, eps_next, cond_seq, cond_mask):
        z_next = S.advance(inp_prev, x_k, cond_seq, cond_mask)
        z_t, t_t = S.churn(z_next, inp_prev.k + 1, eps_next, cond_seq, cond_mask)
        return DenoiserInput(inp_prev.k + 1, z_t, t_t, x_k)

    for k in ks:
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        sums = {}; counts = {}; rho = []; base_logit = []
        def add(name, dl, changed, cnt):
            s = sums.setdefault(name, [0.0, 0.0]); s[0] += dl; s[1] += changed; counts[name] = counts.get(name, 0) + cnt
        for pb in batches(prompts, a.B):
            B = len(pb)
            cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
            z0 = (torch.randn((B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1}, stop_before=k, generator=g)
                prev = ro.inputs[k - 1]
                x_k = S.denoise(model, prev, cond_seq, cond_mask).float()
            eps_next = S.sample_eps(prev.z, g)
            # critic pull G = dl/dx_k through both paths
            x_var = x_k.clone().requires_grad_(True)
            with ac:
                out = critic(state_from(x_var, prev, eps_next, cond_seq, cond_mask), cond_seq, cond_mask, gen_mask)
                out.logit.float().mean().backward()
            G = x_var.grad.detach().float()
            if n_cl:
                rho.append((G[..., D:].pow(2).sum() / G.pow(2).sum().clamp_min(1e-12)).item())
            with torch.no_grad(), ac:
                base = critic(state_from(x_k, prev, eps_next, cond_seq, cond_mask), cond_seq, cond_mask, gen_mask).logit.float()
                dec0 = S.decode(model, x_k)
            base_logit.append(base.mean().item())
            m = gen_mask.bool() & (dec0 != eos); cnt = m.sum().item()
            tok_nrm = x_k[..., :D].norm(dim=-1, keepdim=True)                      # (B,L,1)

            def evaluate(name, x_mod):
                with torch.no_grad(), ac:
                    lg = critic(state_from(x_mod, prev, eps_next, cond_seq, cond_mask), cond_seq, cond_mask, gen_mask).logit.float()
                    dec = S.decode(model, x_mod)
                changed = ((dec != dec0) & m).sum().item()
                add(name, (lg - base).sum().item(), changed, cnt)

            variants = []
            if n_cl:
                g0 = gamma[dec0]
                recons = x_k.clone(); recons[..., D:] = beta * F.one_hot(g0, n_cl).float()
                variants.append(("recons", recons))
                wrong = (g0 + torch.randint(1, n_cl, g0.shape, device=dev)) % n_cl
                wc = x_k.clone(); wc[..., D:] = beta * F.one_hot(wrong, n_cl).float()
                variants.append(("wrongcls", wc))
                zc = x_k.clone(); zc[..., D:] = 0
                variants.append(("zerocl", zc))
            for eps in epss:
                nrand = torch.randn_like(x_k[..., :D]); nrand = nrand / nrand.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                xr = x_k.clone(); xr[..., :D] = x_k[..., :D] + eps * tok_nrm * nrand
                variants.append((f"tok_rand@{eps}", xr))
                gt = G[..., :D]; gt = gt / gt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                xp = x_k.clone(); xp[..., :D] = x_k[..., :D] + eps * tok_nrm * gt
                variants.append((f"tok_pull@{eps}", xp))
                if n_cl:
                    gf = G / G.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    xf = x_k + eps * x_k.norm(dim=-1, keepdim=True) * gf
                    variants.append((f"full_pull@{eps}", xf))
            for name, xm in variants:
                evaluate(name, xm)
        res = {name: {"dlogit": s[0] / max(len(prompts), 1), "decode_changed": s[1] / max(counts[name], 1)}
               for name, s in sums.items()}
        res["base_logit"] = sum(base_logit) / len(base_logit)
        if rho: res["rho_cl"] = sum(rho) / len(rho)
        results[str(k)] = res
        print(json.dumps({"k": k, "base_logit": round(res["base_logit"], 3), "rho_cl": round(res.get("rho_cl", float("nan")), 4),
                          **{n: (round(v["dlogit"], 3), round(v["decode_changed"], 3)) for n, v in res.items() if isinstance(v, dict)}}),
              flush=True)
    return results


def mode_survive(a, cfg, tok, encoder, model, dev):
    """Rollout-level survival of a critic-pleasing displacement. At step k displace the clean estimate x_k along the
    critic pull (token channels, eps of the per-position norm) or a random direction, then run the REMAINING steps
    with the same noise as the undisplaced control; report accuracy (verifier), decoded-token change vs control,
    and the critic logit gap vs control at k, k+8, k+16, k+32 (does the inflation survive or get healed?)."""
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0); D = model.text_encoder_dim - n_cl
    gamma = torch.load(cfg.cluster_map, map_location="cpu")["gamma"].long().to(dev) if n_cl else None
    eos = tok.eos_token_id
    ck = torch.load(a.critic, map_location="cpu")
    critic = Critic(model, cfg, mode=ck["mode"], blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
    if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
    critic.head.load_state_dict(ck["head"]); critic.eval(); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    prompts = load_prompts(a.prompts); random.Random(777).shuffle(prompts); prompts = prompts[: a.n]
    ks = [int(x) for x in a.ks.split(",")]; epss = [float(x) for x in a.eps.split(",")]
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    results = {}
    for k in ks:
        laters = [kk for kk in (k, k + 8, k + 16, k + 32) if kk < a.K]
        agg = {}
        def add(name, key, val, cnt=1):
            r = agg.setdefault(name, {}); v = r.setdefault(key, [0.0, 0]); v[0] += val; v[1] += cnt
        for bi, b0 in enumerate(range(0, len(prompts), a.B)):
            pb = prompts[b0: b0 + a.B]; B = len(pb)
            cond_seq, cond_mask, gen_mask, cl = env.encode(pb)
            g1 = torch.Generator(device="cpu").manual_seed(a.seed * 1000 + bi)
            z0 = (torch.randn((B, env.L, model.text_encoder_dim), generator=g1) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1}, stop_before=k, generator=g1)
                prev = ro.inputs[k - 1]
                x0 = S.denoise(model, prev, cond_seq, cond_mask).float()
            eps_next = S.sample_eps(prev.z, g1)
            x_var = x0.clone().requires_grad_(True)
            with ac:
                st = DenoiserInput(k, S.churn(S.advance(prev, x_var, cond_seq, cond_mask), k, eps_next, cond_seq, cond_mask)[0],
                                   S.churn(S.advance(prev, x_var, cond_seq, cond_mask), k, eps_next, cond_seq, cond_mask)[1], x_var)
                critic(st, cond_seq, cond_mask, gen_mask).logit.float().mean().backward()
            G = x_var.grad.detach().float(); u = G[..., :D] / G[..., :D].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            tok_nrm = x0[..., :D].norm(dim=-1, keepdim=True)
            nrand = torch.randn_like(x0[..., :D]); nrand = nrand / nrand.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            variants = [("control", x0)]
            for eps in epss:
                xp = x0.clone(); xp[..., :D] = x0[..., :D] + eps * tok_nrm * u; variants.append((f"pull@{eps}", xp))
            xr = x0.clone(); xr[..., :D] = x0[..., :D] + epss[0] * tok_nrm * nrand; variants.append((f"rand@{epss[0]}", xr))
            outs = {}
            for name, xm in variants:
                g2 = torch.Generator(device="cpu").manual_seed(a.seed * 7919 + bi)   # same continuation noise for every variant
                with torch.no_grad(), ac:
                    z_next = S.advance(prev, xm, cond_seq, cond_mask)
                    ro2 = S.rollout(model, cond_seq, cond_mask, z_next, record=set(laters) | {a.K - 1}, generator=g2, x_prev=xm, start_k=k)
                    logits = {kk: critic(ro2.inputs[kk], cond_seq, cond_mask, gen_mask).logit.float() for kk in laters}
                    dec = S.decode(model, ro2.z_final)
                    rew = env.reward(pb, env.decode_text(dec, cl)).float()
                    cons = None
                    if n_cl and (a.K - 1) in ro2.inputs:
                        xl = ro2.inputs[a.K - 1].x_prev
                        cons = ((xl[..., D:].argmax(-1) == gamma[dec]) & gen_mask.bool()).float().sum() / gen_mask.sum()
                outs[name] = dict(logits=logits, dec=dec, rew=rew, cons=cons)
            c = outs["control"]; m = gen_mask.bool() & (c["dec"] != eos)
            for name, o in outs.items():
                add(name, "acc", o["rew"].sum().item(), B)
                add(name, "dec_changed", ((o["dec"] != c["dec"]) & m).sum().item(), m.sum().item())
                for kk in laters: add(name, f"dlogit@{kk}", (o["logits"][kk] - c["logits"][kk]).sum().item(), B)
                if o["cons"] is not None: add(name, "cons_final", o["cons"].item() * B, B)
        res = {name: {key: v[0] / max(v[1], 1) for key, v in r.items()} for name, r in agg.items()}
        results[str(k)] = res
        print(json.dumps({"k": k, **{n: {kk: round(vv, 4) for kk, vv in r.items()} for n, r in res.items()}}), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trace", "probe", "survive"], required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=256); ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--runs", default=""); ap.add_argument("--critic", default=None)
    ap.add_argument("--ks", default="24,40"); ap.add_argument("--eps", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    res = {"trace": mode_trace, "probe": mode_probe, "survive": mode_survive}[a.mode](a, cfg, tok, encoder, model, dev)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
