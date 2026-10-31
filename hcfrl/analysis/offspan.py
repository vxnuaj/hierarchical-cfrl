"""On- vs off-manifold geometry of policies and critics (the candidate exploitation rule).

The data manifold of clean latents: encode --n_manifold training solutions exactly as SFT does ([prompt][solution]
latents from the frozen encoder, normalised), pool the vectors of all generated positions, PCA. P_on = top-m
principal directions holding --var of the variance; P_off = orthogonal complement (per-position subspace of R^D).

Measured on fixed probe states of a base policy (rollouts of val prompts to step k-1 / k):
  critics : input gradient of the pooled logit w.r.t. the self-conditioning input x_{k-1} and the latent z~_k;
            fraction of its squared norm on P_on ("critic on-fraction"); and the trunk's response to on/off input
            perturbations of equal norm (|dh| / |h| and the logit change).
  policies: Jacobian of the clean estimate x_hat = d_theta(s_{k-1}) w.r.t. the parameters, probed with random output
            directions u restricted to P_on or P_off (unit norm over generated positions): |J^T u|_2^2 (gradient-
            descent gain) and |J^T u|_1 (gain of a sign/Adam-like step) -> the policy's cost of moving on vs off.
  drift   : for RL checkpoints, the off-manifold energy fraction of the clean estimate at --k_drift.

usage: python -m hcfrl.analysis.offspan --config configs/elf_b_game24_cd.yml --base_ckpt outputs/elf_b-game24-cd \
         --probe_policy <policy_it0.pt> --critics native=...,ma=... --policies M=<pt>,A=<pt> --drift nat=<run>:0,250,...
         --train data/game24/train.jsonl --val data/game24/val.jsonl --out outputs/analysis/offspan.pt
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch

import hcfrl.elf_path  # noqa: F401
from utils.encoder_utils import build_self_attn_cond_masks, encode_text  # type: ignore
from hcfrl.analysis.cross_critic import best_critic, parse
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


@torch.no_grad()
def manifold(env, rows, bs=256):
    tok, L, Lc = env.tok, env.L, env.Lc
    vecs = []
    for b0 in range(0, len(rows), bs):
        rb = rows[b0: b0 + bs]
        ids = np.full((len(rb), L), env.eos, dtype=np.int64); cl = np.zeros(len(rb), dtype=np.int64)
        for i, r in enumerate(rb):
            c = tok(r["input"], add_special_tokens=False)["input_ids"][:Lc]
            o = tok(r["output"], add_special_tokens=False)["input_ids"]
            seq = (c + o + [env.eos])[:L]
            ids[i, : len(seq)] = seq; cl[i] = len(c)
        is_cond = np.arange(L)[None] < cl[:, None]
        enc_mask, _, _ = build_self_attn_cond_masks(is_cond, np.ones_like(is_cond))
        x = encode_text(torch.from_numpy(ids).to(env.dev), torch.from_numpy(enc_mask).to(env.dev), env.encoder,
                        env.cfg.latent_mean, env.cfg.latent_std, use_bf16=False).float()
        vecs.append(x[torch.from_numpy(~is_cond).to(env.dev)].cpu())
    X = torch.cat(vecs).double()
    mu = X.mean(0); C = ((X - mu).T @ (X - mu)) / len(X)
    ev, U = torch.linalg.eigh(C); ev, U = ev.flip(0), U.flip(1)
    return mu.float(), ev.float(), U.float(), len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--base_ckpt", default="outputs/elf_b-game24-cd")
    ap.add_argument("--probe_policy", required=True); ap.add_argument("--critics", default=""); ap.add_argument("--policies", default="")
    ap.add_argument("--drift", default="", help="name=<run dir>:it,it,...;name=...")
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--n_manifold", type=int, default=4000); ap.add_argument("--var", type=float, default=0.99)
    ap.add_argument("--n_probe", type=int, default=64); ap.add_argument("--ks", default="24,40"); ap.add_argument("--k_drift", type=int, default=40)
    ap.add_argument("--n_dirs", type=int, default=8); ap.add_argument("--pert", type=float, default=0.02)
    ap.add_argument("--n_drift", type=int, default=256)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.base_ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    out = {"args": vars(a)}

    # ---- manifold ----
    rows = [json.loads(l) for l in open(a.train) if l.strip()][: a.n_manifold]
    mu, ev, U, n = manifold(env, rows)
    m = int((ev.cumsum(0) / ev.sum() < a.var).sum().item()) + 1
    Pon = U[:, :m].to(dev)
    out["manifold"] = {"eigvals": ev, "m": m, "n_vectors": n, "var": a.var}
    print(f"manifold: {n} vectors, D={len(ev)}, m={m} PCs hold {a.var:.0%} of the variance; "
          f"m90={int((ev.cumsum(0)/ev.sum() < 0.9).sum()) + 1}, m999={int((ev.cumsum(0)/ev.sum() < 0.999).sum()) + 1}", flush=True)

    def on_part(v):            # project the last dim onto P_on
        return (v @ Pon) @ Pon.T

    def set_policy(path):
        sd = torch.load(path, map_location="cpu"); model.load_state_dict(sd["params"] if "params" in sd else sd); model.eval()

    # ---- probe states of the probe policy ----
    set_policy(a.probe_policy)
    ks = [int(x) for x in a.ks.split(",")]
    val = load_prompts(a.val)[: a.n_probe]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    probes = []
    for b0 in range(0, len(val), a.bs):
        pb = val[b0: b0 + a.bs]
        cond_seq, cond_mask, gen_mask, cl = env.encode(pb)
        z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad():
            ro = S.rollout(model, cond_seq, cond_mask, z0, record={k for k in ks} | {k - 1 for k in ks}, generator=g)
        for k in ks:
            probes.append(dict(k=k, prev=ro.inputs[k - 1], cur=ro.inputs[k], cond_seq=cond_seq, cond_mask=cond_mask, gen_mask=gen_mask,
                               eps=S.sample_eps(ro.inputs[k].z, g)))
    print(f"probe states: {len(probes)} batches", flush=True)

    # ---- critics ----
    out["critics"] = {}
    for cname, cpath in parse(a.critics):
        ck = torch.load(best_critic(cpath), map_location="cpu")
        critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False).to(dev)
        if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
        critic.head.load_state_dict(ck["head"]); critic.eval()
        for p in critic.parameters(): p.requires_grad_(False)
        rec = {k: {"on_frac_x": [], "on_frac_z": [], "dh_on": [], "dh_off": [], "dl_on": [], "dl_off": []} for k in ks}
        gp = torch.Generator(device="cpu").manual_seed(a.seed + 1)
        for pr in probes:
            k, cur, gm = pr["k"], pr["cur"], pr["gen_mask"]
            zi = cur.z.detach().clone().requires_grad_(True); xi = cur.x_prev.detach().clone().requires_grad_(True)
            o = critic(DenoiserInput(k, zi, cur.t, xi), None, None, gm)
            gz, gx = torch.autograd.grad(o.logit.sum(), [zi, xi])
            M3 = gm[..., None]
            for v, key in ((gx, "on_frac_x"), (gz, "on_frac_z")):
                v = v * M3
                rec[k][key].append((on_part(v).pow(2).sum((1, 2)) / v.pow(2).sum((1, 2)).clamp_min(1e-30)).cpu())
            # trunk response to equal-norm on / off perturbations of the self-conditioning input
            with torch.no_grad():
                h0 = critic.features(cur, None, None); l0 = critic(cur, None, None, gm).logit
                for _ in range(2):
                    r = torch.randn(cur.x_prev.shape, generator=gp).to(dev) * M3
                    for key, d in (("on", on_part(r)), ("off", r - on_part(r))):
                        d = d / d.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12) * a.pert * (cur.x_prev * M3).norm(dim=(1, 2), keepdim=True)
                        inp = DenoiserInput(k, cur.z, cur.t, cur.x_prev + d)
                        h1 = critic.features(inp, None, None); l1 = critic(inp, None, None, gm).logit
                        rec[k]["dh_" + key].append((((h1 - h0) * M3).norm(dim=(1, 2)) / (h0 * M3).norm(dim=(1, 2))).cpu())
                        rec[k]["dl_" + key].append((l1 - l0).abs().cpu())
        out["critics"][cname] = {k: {kk: torch.cat(v) for kk, v in rec[k].items()} for k in ks}
        print(f"critic {cname}: " + "  ".join(
            f"k{k} on_x {out['critics'][cname][k]['on_frac_x'].mean():.3f} on_z {out['critics'][cname][k]['on_frac_z'].mean():.3f} "
            f"dh off/on {out['critics'][cname][k]['dh_off'].mean() / out['critics'][cname][k]['dh_on'].mean():.2f} "
            f"dl off/on {out['critics'][cname][k]['dl_off'].mean() / out['critics'][cname][k]['dl_on'].mean():.2f}" for k in ks), flush=True)
        del critic; torch.cuda.empty_cache()

    # ---- policies: Jacobian gains on vs off ----
    out["policies"] = {}
    params = [p for p in model.parameters()]
    for pname, ppath in parse(a.policies):
        set_policy(ppath)
        for p in params: p.requires_grad_(True)
        gq = torch.Generator(device="cpu").manual_seed(a.seed + 2)
        rec = {k: {"l2_on": [], "l2_off": [], "l1_on": [], "l1_off": []} for k in ks}
        t0 = time.time()
        for pr in probes[: max(2, len(probes) // 2)]:
            k, prev, gm = pr["k"], pr["prev"], pr["gen_mask"]
            M3 = gm[..., None]
            for _ in range(a.n_dirs):
                r = torch.randn(prev.z.shape, generator=gq).to(dev) * M3
                for key, u in (("on", on_part(r)), ("off", r - on_part(r))):
                    u = u / u.norm().clamp_min(1e-12)
                    x = S.denoise(model, DenoiserInput(prev.k, prev.z.detach(), prev.t, prev.x_prev.detach()), pr["cond_seq"], pr["cond_mask"])
                    gr = torch.autograd.grad((x * u).sum(), params, allow_unused=True)
                    rec[k]["l2_" + key].append(sum((q.float() ** 2).sum() for q in gr if q is not None).item())
                    rec[k]["l1_" + key].append(sum(q.float().abs().sum() for q in gr if q is not None).item())
        for p in params: p.requires_grad_(False)
        out["policies"][pname] = {k: {kk: torch.tensor(v) for kk, v in rec[k].items()} for k in ks}
        print(f"policy {pname} ({time.time() - t0:.0f}s): " + "  ".join(
            f"k{k} L2 off/on {out['policies'][pname][k]['l2_off'].mean() / out['policies'][pname][k]['l2_on'].mean():.3f} "
            f"L1 off/on {out['policies'][pname][k]['l1_off'].mean() / out['policies'][pname][k]['l1_on'].mean():.3f} "
            f"(L2 on {out['policies'][pname][k]['l2_on'].mean():.3g})" for k in ks), flush=True)

    # ---- drift of the clean estimate off the manifold along RL runs ----
    out["drift"] = {}
    dval = load_prompts(a.val)[: a.n_drift]
    for entry in [x for x in a.drift.split(";") if x]:
        name, spec = entry.split("=", 1); run, its = spec.split(":")
        res = {}
        for it in [int(x) for x in its.split(",")]:
            f = os.path.join(run, f"policy_it{it}.pt")
            if not os.path.exists(f): continue
            set_policy(f); gd = torch.Generator(device="cpu").manual_seed(a.seed + 3); fr, nrm = [], []
            for b0 in range(0, len(dval), 64):
                pb = dval[b0: b0 + 64]
                cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
                z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=gd) * cfg.denoiser_noise_scale).to(dev)
                with torch.no_grad():
                    ro = S.rollout(model, cond_seq, cond_mask, z0, record={a.k_drift + 1}, stop_before=a.k_drift + 2, generator=gd)
                    x = ro.inputs[a.k_drift + 1].x_prev - mu.to(dev)       # clean estimate after the denoiser call at k_drift
                    msk = gen_mask.bool()
                    v = x[msk]
                    fr.append((1 - on_part(v).pow(2).sum(-1) / v.pow(2).sum(-1).clamp_min(1e-30)).cpu()); nrm.append(v.norm(dim=-1).cpu())
            res[it] = {"off_frac": torch.cat(fr).mean().item(), "norm": torch.cat(nrm).mean().item()}
            print(f"drift {name} it{it}: off-manifold energy {res[it]['off_frac']:.4f}  |x - mu| {res[it]['norm']:.3f}", flush=True)
        out["drift"][name] = res
    torch.save(out, a.out); print("saved", a.out)


if __name__ == "__main__":
    main()
