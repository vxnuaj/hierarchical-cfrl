"""CFRL-GRR: group-relative REINFORCE baseline (arXiv 2610.19080, Sec. 3.2), score-function, no critic.

Per iteration: P prompts x G rollouts under the (cheap) rollout sampler (K_roll, gamma_roll);
A^g = (r^g - mean) / (std + eps) within each prompt group;
J = (1/(P G)) sum_g A^g [ (K_s / S) sum_{k in S} log p_theta(z~_{k+1} | input_k) + (1/L) log pi_theta(w | z_K) ]
where S is a random subset of the K_s = K_roll - 1 stochastic transitions (CFRL: 8 of 15), and
log p of a transition is the exact Gaussian kernel  N(z~_{k+1}; alpha*(z~_k + (t_{k+1}-t~_k)(x_k - z~_k)/D), (1-alpha)^2 c_sigma^2 I)
with theta entering through x_k = d_theta(input_k). Summed over generated positions only.
Eval under the deployment sampler (K, gamma). No KL, no ratio clipping, one step per batch.

usage:
  python -m hcfrl.rl.train_grr --config configs/elf_b_game24.yml --ckpt outputs/elf_b-game24 \
      --train_prompts data/game24/train.jsonl --val_prompts data/game24/val.jsonl --test_prompts data/game24/test.jsonl \
      --out outputs/rl/game24_flat_grr_s0 --iters 600 --seed 0
"""
from __future__ import annotations

import argparse, json, os, random, time

import torch
import torch.nn.functional as F

from hcfrl.rl.common import load_policy
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_prompts", required=True); ap.add_argument("--val_prompts", required=True)
    ap.add_argument("--test_prompts", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--K_roll", type=int, default=16); ap.add_argument("--gamma_roll", type=float, default=4.0)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=600); ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--G", type=int, default=8); ap.add_argument("--n_trans", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-6); ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_bs", type=int, default=128); ap.add_argument("--n_val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bf16", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev)
    S_eval = ChurnSampler(cfg, K=a.K, gamma=a.gamma, cfg_scale=a.cfg_scale)
    S = ChurnSampler(cfg, K=a.K_roll, gamma=a.gamma_roll, cfg_scale=a.cfg_scale)
    model.train(False)
    for p in model.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    train_prompts = load_train_prompts(a.train_prompts)
    val_prompts = load_prompts(a.val_prompts); val_prompts = val_prompts[: a.n_val] if a.n_val else val_prompts
    test_prompts = load_prompts(a.test_prompts) if a.test_prompts else None
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, "log.jsonl"), "a")
    g = torch.Generator(device="cpu").manual_seed(a.seed); geval = torch.Generator(device="cpu").manual_seed(12345)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16) and dev.type == "cuda")
    sigma2 = ((1.0 - S.alpha) * cfg.denoiser_noise_scale) ** 2
    Ks = a.K_roll - 1  # stochastic transitions 0..K-2
    best_val, t0 = -1.0, time.time()

    for it in range(a.iters + 1):
        if it % a.eval_every == 0:
            model.eval()
            with ac:
                val = evaluate(model, S_eval, env, val_prompts, a.eval_bs, geval, dev)
                test = evaluate(model, S_eval, env, test_prompts, a.eval_bs, geval, dev) if test_prompts else None
            model.train(False)
            rec = {"iter": it, "val_acc": val, "test_acc": test, "elapsed": time.time() - t0}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            torch.save({"params": model.state_dict(), "iter": it, "val_acc": val, "test_acc": test}, os.path.join(a.out, f"policy_it{it}.pt"))
            if val > best_val:
                best_val = val; torch.save({"params": model.state_dict(), "iter": it, "val_acc": val, "test_acc": test}, os.path.join(a.out, "policy_best.pt"))
            if it == a.iters: break

        # ---- rollouts: P prompts x G samples, record every denoiser input ----
        pb = [p for p in random.sample(train_prompts, a.P) for _ in range(a.G)]
        B = len(pb)
        cond_seq, cond_mask, gen_mask, cl = env.encode(pb)
        z0 = (torch.randn((B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad(), ac:
            ro = S.rollout(model, cond_seq, cond_mask, z0, record=set(range(a.K_roll)), generator=g)
            toks = S.decode(model, ro.z_final)
            r = env.reward(pb, env.decode_text(toks, cl)).to(dev)
        rg = r.view(a.P, a.G)
        A = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(B)
        gm = gen_mask  # (B, L)

        # ---- score-function surrogate on n_trans random stochastic transitions + decoder term ----
        ks = random.sample(range(Ks), min(a.n_trans, Ks))
        with ac:
            logp_sum = torch.zeros(B, device=dev)
            for k in ks:
                inp, nxt = ro.inputs[k], ro.inputs[k + 1]
                x_k = S.denoise(model, DenoiserInput(k, inp.z.detach(), inp.t, inp.x_prev.detach()), cond_seq, cond_mask)
                t_next = S.t_grid(k + 1); D = max(1.0 - inp.t, S.t_eps)
                mu = S.alpha * (inp.z + (t_next - inp.t) * (x_k - inp.z) / D)
                sq = ((nxt.z.detach().float() - mu.float()) ** 2).sum(-1)          # (B, L)
                logp_sum = logp_sum + (-(sq * gm).sum(1) / (2 * sigma2))
            logp_sum = logp_sum * (Ks / len(ks))
            # decoder term: (1/L) log pi(w | z_K)
            zK = ro.z_final.detach()
            t1 = torch.ones((B,), dtype=zK.dtype, device=dev)
            sc = torch.full((B,), 1.0, dtype=zK.dtype, device=dev) if cfg.num_self_cond_cfg_tokens > 0 else None
            z_in = torch.cat([zK, torch.zeros_like(zK)], -1) if cfg.self_cond_prob > 0 else zK
            _, logits = model(z_in, t1, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=True)
            lp = F.log_softmax(logits.float(), -1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)   # (B, L)
            dec_term = (lp * gm).sum(1) / gm.sum(1).clamp_min(1.0)
            J = (A * (logp_sum + dec_term)).mean()
        opt.zero_grad(set_to_none=True); (-J).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item(); opt.step()

        rec = {"iter": it, "J": J.item(), "grad_norm": gn, "roll_succ": r.mean().item(),
               "n_mixed_groups": int(((rg.max(1).values > 0.5) & (rg.min(1).values < 0.5)).sum()), "elapsed": time.time() - t0}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if it % 10 == 0: print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    print(f"done. best val {best_val:.4f}")


if __name__ == "__main__":
    main()
