"""One actor step along each critic's pathwise gradient, all critics on the same rollouts (first-order usefulness).

For every critic C: the mean pathwise gradient of C's pooled logit over the same --n_prompts x --repeat rollouts and
window steps (identical prompts and noise for every critic), its AdamW-first-step direction d_C (unit norm), and
  * cosines between the directions of all critics (parameter space),
  * for eta = f * Lambda (Lambda = parameter distance of a reference RL run from it0 to it<lam_iter>): validation
    accuracy after theta + eta d_C (paired fixed noise), and every critic's mean logit change on fixed probe states
    rebuilt with the stepped policy (first-order transfer matrix).
A critic whose step raises accuracy carries reward signal in its gradient; one whose step raises only its own
logit carries an idiosyncratic direction.

usage: python -m hcfrl.analysis.multi_onestep --config ... --ckpt ... --critics a=...,b=... --run outputs/rl/<run> \
          --train_prompts data/game24/train.jsonl --val_prompts data/game24/val.jsonl --out outputs/analysis/multi_onestep.json
"""
from __future__ import annotations

import argparse, json, os, random, time

import torch

from hcfrl.analysis.cross_critic import best_critic, parse
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--critics", required=True)
    ap.add_argument("--run", required=True); ap.add_argument("--lam_iter", type=int, default=750)
    ap.add_argument("--train_prompts", required=True); ap.add_argument("--val_prompts", required=True)
    ap.add_argument("--n_prompts", type=int, default=64); ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--ks", default="8,16,24,32"); ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--n_val", type=int, default=512); ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--fracs", default="0.25,1"); ap.add_argument("--k_probe", type=int, default=40); ap.add_argument("--n_probe", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    critics = {}
    for name, path in parse(a.critics):
        ck = torch.load(best_critic(path), map_location="cpu")
        c = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False).to(dev)
        if ck.get("trunk") is not None: c.trunk.load_state_dict(ck["trunk"])
        c.head.load_state_dict(ck["head"]); c.eval()
        for p in c.parameters(): p.requires_grad_(False)
        critics[name] = c
    params = list(model.parameters()); base = [p.detach().clone() for p in params]
    p0 = torch.load(os.path.join(a.run, "policy_it0.pt"), map_location="cpu")["params"]
    p1 = torch.load(os.path.join(a.run, f"policy_it{a.lam_iter}.pt"), map_location="cpu")["params"]
    Lam = torch.sqrt(sum(((p1[k].float() - p0[k].float()) ** 2).sum() for k in p0 if p0[k].dtype.is_floating_point)).item(); del p0, p1
    print(f"Lambda = {Lam:.4f}", flush=True)

    # ---- mean pathwise gradient per critic, same rollouts ----
    ks = [int(x) for x in a.ks.split(",")]
    prompts = random.sample(load_train_prompts(a.train_prompts), a.n_prompts)
    pb_all = [p for p in prompts for _ in range(a.repeat)]
    for p in params: p.requires_grad_(True)
    gsum = {n: [torch.zeros_like(p) for p in params] for n in critics}
    g = torch.Generator(device="cpu").manual_seed(a.seed); t0 = time.time()
    for b0 in range(0, len(pb_all), a.bs):
        pb = pb_all[b0: b0 + a.bs]
        cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
        z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad(), ac:
            ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1 for k in ks}, stop_before=max(ks), generator=g)
        eps = {k: S.sample_eps(ro.inputs[k - 1].z, g) for k in ks}
        for n, c in critics.items():
            obj = 0.0
            with ac:
                for k in ks:
                    prev = ro.inputs[k - 1]
                    nxt = S.grad_step(model, DenoiserInput(k - 1, prev.z, prev.t, prev.x_prev), cond_seq, cond_mask, eps_next=eps[k])
                    obj = obj + c(nxt, cond_seq, cond_mask, gen_mask).logit.sum()
            gr = torch.autograd.grad(obj, params, allow_unused=True)
            for s_, q in zip(gsum[n], gr):
                if q is not None: s_ += q.detach()
    for p in params: p.requires_grad_(False)
    print(f"gradients done ({time.time() - t0:.0f}s)", flush=True)

    def unit(vs):
        nrm = torch.sqrt(sum((v.float() ** 2).sum() for v in vs)).clamp_min(1e-12); return [v / nrm for v in vs]
    dirs, raw = {}, {}
    for n in critics:
        gm = gsum[n]; gn = torch.sqrt(sum((v.float() ** 2).sum() for v in gm)).item(); clip = min(1.0, 1.0 / max(gn, 1e-12))
        dirs[n] = unit([torch.sign(v) * (v.abs() * clip) / (v.abs() * clip + 1e-8) for v in gm]); raw[n] = unit(gm)
    del gsum
    names = list(critics)
    cos_raw = {a_: {b_: sum((x * y).sum() for x, y in zip(raw[a_], raw[b_])).item() for b_ in names} for a_ in names}
    cos_adam = {a_: {b_: sum((x * y).sum() for x, y in zip(dirs[a_], dirs[b_])).item() for b_ in names} for a_ in names}
    del raw
    print("raw-gradient cosines:\n" + "\n".join(f"  {a_:7s} " + " ".join(f"{cos_raw[a_][b_]:6.3f}" for b_ in names) for a_ in names), flush=True)

    # ---- probe states and evaluation ----
    val = load_prompts(a.val_prompts)[: a.n_val]; probe_val = val[: a.n_probe]

    def set_params(direction, eta):
        with torch.no_grad():
            for p, b, d in zip(params, base, direction): p.copy_(b + eta * d)

    def ev():
        gq = torch.Generator(device="cpu").manual_seed(4242)
        with ac:
            return evaluate(model, S, env, val, a.eval_bs, gq, dev)

    @torch.no_grad()
    def probe_logits():
        gq = torch.Generator(device="cpu").manual_seed(777); out = {n: [] for n in critics}
        for b0 in range(0, len(probe_val), 64):
            pb = probe_val[b0: b0 + 64]; cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=gq) * cfg.denoiser_noise_scale).to(dev)
            with ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={a.k_probe}, stop_before=a.k_probe + 1, generator=gq)
                for n, c in critics.items(): out[n].append(c(ro.inputs[a.k_probe], cond_seq, cond_mask, gen_mask).logit.float().cpu())
        return {n: torch.cat(v).mean().item() for n, v in out.items()}

    model.eval(); set_params(dirs[names[0]], 0.0)
    acc0, l0 = ev(), probe_logits()
    print(f"base acc {acc0:.4f}", flush=True)
    res = {"Lambda": Lam, "acc0": acc0, "logit0": l0, "cos_raw": cos_raw, "cos_adam": cos_adam, "steps": {}}
    for f in (float(x) for x in a.fracs.split(",")):
        for n in names:
            set_params(dirs[n], f * Lam); acc, l = ev(), probe_logits()
            res["steps"][f"{n}@{f:g}"] = {"dacc": acc - acc0, "dlogit": {m: l[m] - l0[m] for m in names}}
            print(f"step {n} eta/Lambda={f:g}: dacc {100 * (acc - acc0):+.1f}  dlogit " + " ".join(f"{m}:{l[m] - l0[m]:+.0f}" for m in names), flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("saved", a.out)


if __name__ == "__main__":
    main()
