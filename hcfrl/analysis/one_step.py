"""One step along the pathwise gradient at the SFT base (CFRL App. "One step along each gradient").
Mean pathwise gradient of the critic logit over N rollouts x 4 window steps; the policy is moved once,
theta' = theta + eta * d, with d the AdamW-first-step direction (clip 1, then g/(|g|+eps) per coordinate, unit norm),
the raw direction g/|g|, or a Gaussian random direction; eta = (eta/Lambda) * Lambda, where Lambda is the
parameter distance an RL run travelled (policy_it0 -> policy_it<lam_iter> of --run). Paired evaluation with fixed noise.
usage: python -m hcfrl.analysis.one_step --config ... --ckpt ... --critic ... --run outputs/rl/<svg run> \
          --train_prompts data/game24/train_prompts.jsonl --val_prompts data/game24/val.jsonl --out outputs/analysis/onestep_<tag>.json
"""
from __future__ import annotations

import argparse, json, os, random, time

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--critic", required=True)
    ap.add_argument("--run", required=True, help="RL run dir with policy_it0.pt and policy_it<lam_iter>.pt (defines Lambda)")
    ap.add_argument("--lam_iter", type=int, default=750)
    ap.add_argument("--train_prompts", required=True); ap.add_argument("--val_prompts", required=True)
    ap.add_argument("--n_prompts", type=int, default=128); ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--ks", default="8,16,24,32"); ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--n_val", type=int, default=1000); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--fracs", default="0.0625,0.25,0.5,1"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
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
    params = [p for p in model.parameters()]
    base = [p.detach().clone() for p in params]
    # Lambda from the RL run
    p0 = torch.load(os.path.join(a.run, "policy_it0.pt"), map_location="cpu")["params"]
    p1 = torch.load(os.path.join(a.run, f"policy_it{a.lam_iter}.pt"), map_location="cpu")["params"]
    Lam = torch.sqrt(sum(((p1[k].float() - p0[k].float()) ** 2).sum() for k in p0 if p0[k].dtype.is_floating_point)).item()
    del p0, p1
    print(f"Lambda (policy_it0 -> policy_it{a.lam_iter}) = {Lam:.4f}", flush=True)

    # ---- mean pathwise gradient over N rollouts x window steps ----
    model.train(False)
    for p in params: p.requires_grad_(True)
    ks = [int(x) for x in a.ks.split(",")]
    prompts = random.sample(load_train_prompts(a.train_prompts), a.n_prompts)
    pb_all = [p for p in prompts for _ in range(a.repeat)]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
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
    print(f"mean gradient norm {gnorm:.4f} over {n_terms} terms ({time.time() - t0:.0f}s)", flush=True)
    for p in params: p.requires_grad_(False)

    # ---- directions (unit norm) ----
    def unit(vs):
        n = torch.sqrt(sum((v ** 2).sum() for v in vs)).clamp_min(1e-12); return [v / n for v in vs]
    clip = min(1.0, 1.0 / max(gnorm, 1e-12))
    d_adam = unit([torch.sign(v) * (v.abs() * clip) / (v.abs() * clip + 1e-8) for v in gmean])   # g/(|g|+eps) after clipping to norm 1
    d_raw = unit([v.clone() for v in gmean])
    gr = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    d_rand = unit([torch.randn(v.shape, generator=gr).to(dev) for v in gmean])
    del gmean

    # ---- paired evaluation ----
    val = load_prompts(a.val_prompts)[: a.n_val]
    def ev():
        geval = torch.Generator(device="cpu").manual_seed(777)
        with ac: return evaluate(model, S, env, val, a.eval_bs, geval, dev)
    def set_params(direction, eta):
        with torch.no_grad():
            for p, b, d in zip(params, base, direction): p.copy_(b + eta * d)
    res = {"Lambda": Lam, "grad_norm": gnorm, "n_val": len(val), "steps": {}}
    set_params(d_raw, 0.0); acc0 = ev(); res["base_acc"] = acc0
    print(f"base acc {acc0:.4f}", flush=True)
    for frac in (float(x) for x in a.fracs.split(",")):
        eta = frac * Lam; row = {}
        for name, d in [("adamw", d_adam), ("random", d_rand), ("raw", d_raw)]:
            set_params(d, eta); acc = ev(); row[name] = acc - acc0
        res["steps"][str(frac)] = row
        print(f"eta/Lambda={frac:g}: " + "  ".join(f"{k} {100 * v:+.1f}" for k, v in row.items()), flush=True)
    set_params(d_raw, 0.0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
