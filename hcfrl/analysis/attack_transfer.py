"""Attack transfer between critics: does a policy that ascends one critic also raise the others?

Start from a base policy. Run pathwise actor ascent on ONE frozen critic (the attacker: no head refit, so the policy
optimises exactly this critic, like the frozen-critic probe) with the RL recipe's actor (B prompts, k ~ U[k_lo, k_hi],
AdamW, constant lr). Every --every iterations, roll out a fixed probe set (fixed prompts and noise), verify, and score
the recorded states with EVERY critic in --critics (all loaded at once). Saved per checkpoint: accuracy, and for each
critic its mean logit, its mean logit on correct and on wrong samples, and its AUC.

Reading: if the attack on Muon critic A raises Muon critic B's logit while accuracy falls, the two critics share the
exploited directions (transfer); a critic whose logit falls with accuracy is robust to that attack.

usage: python -m hcfrl.analysis.attack_transfer --config configs/elf_b_game24_cd.yml --ckpt outputs/elf_b-game24-cd/checkpoint_2450 \
          --attacker native --critics native=...,mm7=...,ma=... --train_prompts data/game24/train.jsonl --val_prompts data/game24/val.jsonl \
          --out outputs/analysis/attack_native.json
"""
from __future__ import annotations

import argparse, json, random, time

import torch

from hcfrl.analysis.cross_critic import best_critic, parse
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic, rank_metrics
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--policy", default=None, help="optional policy_itN.pt to start from instead of the checkpoint")
    ap.add_argument("--attacker", required=True); ap.add_argument("--critics", required=True)
    ap.add_argument("--train_prompts", required=True); ap.add_argument("--val_prompts", required=True)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--B", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--n_val", type=int, default=384); ap.add_argument("--k_rec", type=int, default=40)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    if a.policy:
        sd = torch.load(a.policy, map_location="cpu"); model.load_state_dict(sd["params"] if "params" in sd else sd)
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
    attacker = critics[a.attacker]
    k_lo, k_hi = a.K // 8, 5 * a.K // 8
    train = load_train_prompts(a.train_prompts); val = load_prompts(a.val_prompts)[: a.n_val]

    def probe(it):
        model.eval(); g = torch.Generator(device="cpu").manual_seed(777)
        R, L = [], {n: [] for n in critics}
        for b0 in range(0, len(val), a.bs):
            pb = val[b0: b0 + a.bs]
            cond_seq, cond_mask, gen_mask, cl = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record={a.k_rec}, generator=g)
                R.append(env.reward(pb, env.decode_text(S.decode(model, ro.z_final), cl)))
                for n, c in critics.items():
                    L[n].append(c(ro.inputs[a.k_rec], cond_seq, cond_mask, gen_mask).logit.float().cpu())
        R = torch.cat(R); rec = {"iter": it, "acc": R.mean().item()}
        for n in critics:
            v = torch.cat(L[n]); m = rank_metrics(torch.sigmoid(v), R)
            rec[n] = {"logit": v.mean().item(), "logit_correct": v[R > 0.5].mean().item() if (R > 0.5).any() else None,
                      "logit_wrong": v[R <= 0.5].mean().item() if (R <= 0.5).any() else None, "auc": m.get("auc")}
        print(json.dumps(rec), flush=True)
        return rec

    for p in model.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    hist = [probe(0)]; t0 = time.time()
    for it in range(1, a.iters + 1):
        pb = random.sample(train, a.B); ks = [random.randint(k_lo, k_hi) for _ in pb]
        cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
        z0 = (torch.randn((a.B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with ac:
            ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1 for k in ks}, stop_before=max(ks), generator=g)
            objs = []
            for k in sorted(set(ks)):
                sel = torch.tensor([i for i, kk in enumerate(ks) if kk == k], device=dev)
                prev = ro.inputs[k - 1]
                nxt = S.grad_step(model, DenoiserInput(k - 1, prev.z[sel], prev.t, prev.x_prev[sel]), cond_seq[sel], cond_mask[sel], generator=g)
                objs.append(attacker(nxt, cond_seq[sel], cond_mask[sel], gen_mask[sel]).logit)
            obj = torch.cat(objs).mean()
        opt.zero_grad(set_to_none=True); (-obj).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % a.every == 0:
            hist.append(probe(it)); print(f"  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"attacker": a.attacker, "critics": list(critics), "args": vars(a), "hist": hist}, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
