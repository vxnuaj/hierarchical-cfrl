"""Test of the revised rail conjecture (paper Conj. 2): per normalised actor step, how far does each policy's
clean estimate move along the critic's decode-preserving pull direction?

Procedure (one model + its critic):
  1. Mean pathwise gradient g over --n_prompts x --repeat rollouts at window steps --ks (as in one_step.py);
     d_adam = AdamW-first-step direction (unit norm), d_rand = Gaussian direction (unit norm);
     Lambda = parameter distance policy_it0 -> policy_it<lam_iter> of --run.
  2. Fixed probe set: --n_probe val prompts rolled out (fixed noise) to k-1 for k in --probe_ks; record the denoiser
     input at k-1 and the churn noise for k. At the base parameters compute the clean estimate x0 = d_theta(s_{k-1}),
     the critic pull G = dl/dx (through the sampler transition, both paths), u = P_tok G normalised per position,
     the pooled logit l0 and the decode dec0.
  3. For eta = frac * Lambda (--fracs) and each direction: theta <- theta0 + eta*d; recompute x = d_theta(s_{k-1})
     on the SAME inputs; report, averaged over generated positions,
       along_pull : <dx_tok, u> / |x0_tok|          (signed fraction of the token norm moved along the pull)
       total_tok  : |dx_tok| / |x0_tok|             (fraction moved in any direction)
       cos_pull   : cos(dx_tok, u)
       dlogit     : change of the pooled critic logit on the state rebuilt from x
       dec_changed: fraction of decoded tokens that changed
       cl_move    : |dx_cl| / beta                 (H-ELF: movement of the coarse channels)
       cons       : coarse-token consistency after the step (H-ELF)
  A smaller along_pull per eta on the hierarchical policy (at equal or larger total_tok) supports the conjecture.

usage: python -m hcfrl.analysis.jacobian_step --config ... --ckpt ... --critic ... --run outputs/rl/<svg run> \
          --train_prompts data/game24/train_prompts.jsonl --val_prompts data/game24/val.jsonl --out outputs/analysis/jac_<tag>.json
"""
from __future__ import annotations

import argparse, json, os, random, time

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--critic", required=True)
    ap.add_argument("--run", required=True); ap.add_argument("--lam_iter", type=int, default=750)
    ap.add_argument("--train_prompts", required=True); ap.add_argument("--val_prompts", required=True)
    ap.add_argument("--n_prompts", type=int, default=128); ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--ks", default="8,16,24,32"); ap.add_argument("--probe_ks", default="24,40")
    ap.add_argument("--n_probe", type=int, default=128); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--fracs", default="0.0625,0.25,0.5"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0); D = model.text_encoder_dim - n_cl
    gamma_map = torch.load(cfg.cluster_map, map_location="cpu")["gamma"].long().to(dev) if n_cl else None
    beta = float(getattr(cfg, "cluster_scale", 1.0) or 1.0); eos = tok.eos_token_id
    ck = torch.load(a.critic, map_location="cpu")
    critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False, blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
    if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
    critic.head.load_state_dict(ck["head"]); critic.eval()
    for p in critic.parameters(): p.requires_grad_(False)
    params = [p for p in model.parameters()]
    base = [p.detach().clone() for p in params]
    p0 = torch.load(os.path.join(a.run, "policy_it0.pt"), map_location="cpu")["params"]
    p1 = torch.load(os.path.join(a.run, f"policy_it{a.lam_iter}.pt"), map_location="cpu")["params"]
    Lam = torch.sqrt(sum(((p1[k].float() - p0[k].float()) ** 2).sum() for k in p0 if p0[k].dtype.is_floating_point)).item()
    del p0, p1
    print(f"Lambda = {Lam:.4f}", flush=True)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")

    # ---- 1. mean pathwise gradient and directions ----
    model.train(False)
    for p in params: p.requires_grad_(True)
    ks = [int(x) for x in a.ks.split(",")]
    prompts = random.sample(load_train_prompts(a.train_prompts), a.n_prompts)
    pb_all = [p for p in prompts for _ in range(a.repeat)]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    gsum = [torch.zeros_like(p) for p in params]; n_terms = 0; t0 = time.time()
    for b0 in range(0, len(pb_all), a.bs):
        pb = pb_all[b0: b0 + a.bs]
        cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
        z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad(), ac:
            ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1 for k in ks}, stop_before=max(ks), generator=g)
        obj = 0.0
        with ac:
            for k in ks:
                prev = ro.inputs[k - 1]
                nxt = S.grad_step(model, DenoiserInput(k - 1, prev.z, prev.t, prev.x_prev), cond_seq, cond_mask, generator=g)
                obj = obj + critic(nxt, cond_seq, cond_mask, gen_mask).logit.sum()
        model.zero_grad(set_to_none=True); obj.backward()
        for s_, p in zip(gsum, params):
            if p.grad is not None: s_ += p.grad.detach()
        n_terms += len(pb) * len(ks)
    gmean = [s_ / n_terms for s_ in gsum]; del gsum
    gnorm = torch.sqrt(sum((x ** 2).sum() for x in gmean)).item()
    print(f"mean gradient norm {gnorm:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for p in params: p.requires_grad_(False)

    def unit(vs):
        n = torch.sqrt(sum((v ** 2).sum() for v in vs)).clamp_min(1e-12); return [v / n for v in vs]
    clip = min(1.0, 1.0 / max(gnorm, 1e-12))
    d_adam = unit([torch.sign(v) * (v.abs() * clip) / (v.abs() * clip + 1e-8) for v in gmean])
    gr = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    d_rand = unit([torch.randn(v.shape, generator=gr).to(dev) for v in gmean])
    del gmean

    def set_params(direction, eta):
        with torch.no_grad():
            for p, b, d in zip(params, base, direction): p.copy_(b + eta * d)

    def state_from(x_k, inp_prev, eps_next, cond_seq, cond_mask):
        z_next = S.advance(inp_prev, x_k, cond_seq, cond_mask)
        z_t, t_t = S.churn(z_next, inp_prev.k + 1, eps_next, cond_seq, cond_mask)
        return DenoiserInput(inp_prev.k + 1, z_t, t_t, x_k)

    # ---- 2. fixed probe set at the base ----
    val = load_prompts(a.val_prompts); random.Random(777).shuffle(val); val = val[: a.n_probe]
    probe_ks = [int(x) for x in a.probe_ks.split(",")]
    probes = []   # per (k, batch): dict
    gp = torch.Generator(device="cpu").manual_seed(777)
    set_params(d_rand, 0.0)
    for k in probe_ks:
        for b0 in range(0, len(val), a.bs):
            pb = val[b0: b0 + a.bs]
            cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=gp) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1}, stop_before=k, generator=gp)
                prev = ro.inputs[k - 1]
                x0 = S.denoise(model, prev, cond_seq, cond_mask).float()
            eps_next = S.sample_eps(prev.z, gp)
            x_var = x0.clone().requires_grad_(True)
            with ac:
                out = critic(state_from(x_var, prev, eps_next, cond_seq, cond_mask), cond_seq, cond_mask, gen_mask)
                out.logit.float().mean().backward()
            G = x_var.grad.detach().float()
            u = G[..., :D] / G[..., :D].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            with torch.no_grad(), ac:
                l0 = critic(state_from(x0, prev, eps_next, cond_seq, cond_mask), cond_seq, cond_mask, gen_mask).logit.float()
                dec0 = S.decode(model, x0)
            m = gen_mask.bool() & (dec0 != eos)
            probes.append(dict(k=k, prev=prev, eps=eps_next, cond_seq=cond_seq, cond_mask=cond_mask, gen_mask=gen_mask,
                               x0=x0, u=u, l0=l0, dec0=dec0, m=m))
    print(f"probe states: {len(probes)} batches", flush=True)

    # ---- 3. displacement per step ----
    def measure(direction, eta):
        set_params(direction, eta)
        acc = {}
        for P in probes:
            with torch.no_grad(), ac:
                x = S.denoise(model, P["prev"], P["cond_seq"], P["cond_mask"]).float()
                l = critic(state_from(x, P["prev"], P["eps"], P["cond_seq"], P["cond_mask"]), P["cond_seq"], P["cond_mask"], P["gen_mask"]).logit.float()
                dec = S.decode(model, x)
            dx = x - P["x0"]; m = P["m"]; cnt = m.sum().item()
            x0n = P["x0"][..., :D].norm(dim=-1).clamp_min(1e-12)
            along = ((dx[..., :D] * P["u"]).sum(-1) / x0n)[m].sum().item()
            total = (dx[..., :D].norm(dim=-1) / x0n)[m].sum().item()
            cosv = torch.nn.functional.cosine_similarity(dx[..., :D], P["u"], dim=-1)[m].sum().item()
            dchg = ((dec != P["dec0"]) & m).sum().item()
            row = acc.setdefault(P["k"], {"along": 0.0, "total": 0.0, "cos": 0.0, "dchg": 0.0, "dlogit": 0.0, "cl_move": 0.0, "cons": 0.0, "n": 0, "nseq": 0})
            row["along"] += along; row["total"] += total; row["cos"] += cosv; row["dchg"] += dchg
            row["dlogit"] += (l - P["l0"]).sum().item(); row["n"] += cnt; row["nseq"] += x.shape[0]
            if n_cl:
                row["cl_move"] += (dx[..., D:].norm(dim=-1) / beta)[m].sum().item()
                cons = (x[..., D:].argmax(-1) == gamma_map[dec]) & m
                row["cons"] += cons.sum().item()
        out = {}
        for k, r in acc.items():
            n = max(r["n"], 1)
            out[str(k)] = {"along_pull": r["along"] / n, "total_tok": r["total"] / n, "cos_pull": r["cos"] / n,
                           "dec_changed": r["dchg"] / n, "dlogit": r["dlogit"] / max(r["nseq"], 1)}
            if n_cl: out[str(k)].update({"cl_move": r["cl_move"] / n, "cons": r["cons"] / n})
        set_params(direction, 0.0)
        return out

    res = {"Lambda": Lam, "grad_norm": gnorm, "steps": {}}
    for frac in (float(x) for x in a.fracs.split(",")):
        eta = frac * Lam
        row = {"adamw": measure(d_adam, eta), "random": measure(d_rand, eta)}
        res["steps"][str(frac)] = row
        for name in ("adamw", "random"):
            print(json.dumps({"frac": frac, "dir": name, **{k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in row[name].items()}}), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
