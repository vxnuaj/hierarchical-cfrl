"""CFRL-SVG joint actor/critic training (arXiv 2610.19080, Algorithm 1), see notes/cfrl.md.

Per iteration:
  Actor : B prompts; per-prompt k ~ U(W), W = [K/8, 5K/8]; roll out (no grad) to the denoiser input at
          k-1; ONE differentiable denoiser call -> denoiser input at k; critic logit; ascend mean logit.
  Buffer: B fresh prompts rolled to the end under the deployment sampler; record the critic's trunk
          features at k_hi = 5K/8 (trunk is frozen, so features are cached), decode, verify; push to a
          replay buffer of the newest N_B entries.
  Critic: m_c head-only AdamW steps on minibatches of B' buffer entries; every n_q iterations check the
          AUC of V on the newest N_q entries; if below a_0, do m'_c extra head steps.
Eval every `eval_every` iterations on val/test prompts under the deployment sampler; checkpoints saved.

usage:
  python -m hcfrl.rl.train_svg --config configs/elf_b_game24.yml --ckpt outputs/elf_b-game24 \
      --critic outputs/critic/game24_2d/critic_ep6.pt --train_prompts data/game24/train.jsonl \
      --val_prompts data/game24/val.jsonl --test_prompts data/game24/test.jsonl \
      --out outputs/rl/game24_svg2d_s0 --K 64 --gamma 56 --iters 1500 --seed 0
"""
from __future__ import annotations

import argparse, json, math, os, random, time
from collections import deque

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic, rank_metrics
from hcfrl.rl.env import TaskEnv, Prompt, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


def load_train_prompts(path: str) -> list[Prompt]:
    """Train JSONL has {input, output}; derive nums/target from the prompt text 'T : a b c d'."""
    out = []
    for line in open(path):
        if line.strip():
            r = json.loads(line); t, rest = r["input"].split(":")
            out.append(Prompt(r["input"], [int(x) for x in rest.split()], int(t)))
    return out


@torch.no_grad()
def evaluate(model, S, env, prompts, bs, g, device):
    correct = 0
    for b0 in range(0, len(prompts), bs):
        pb = prompts[b0 : b0 + bs]
        cond_seq, cond_mask, _, cl = env.encode(pb)
        z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * env.cfg.denoiser_noise_scale).to(device)
        ro = S.rollout(model, cond_seq, cond_mask, z0, generator=g)
        correct += env.reward(pb, env.decode_text(S.decode(model, ro.z_final), cl)).sum().item()
    return correct / len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--critic", required=True, help="critic_epN.pt from pretrain_critic")
    ap.add_argument("--train_prompts", required=True); ap.add_argument("--val_prompts", required=True)
    ap.add_argument("--test_prompts", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--Bp", type=int, default=64); ap.add_argument("--NB", type=int, default=16384)
    ap.add_argument("--mc", type=int, default=2); ap.add_argument("--mc_extra", type=int, default=8)
    ap.add_argument("--nq", type=int, default=10); ap.add_argument("--Nq", type=int, default=256)
    ap.add_argument("--a0", type=float, default=0.9)
    ap.add_argument("--lr_actor", type=float, default=1e-6); ap.add_argument("--lr_actor_min", type=float, default=1e-7)
    ap.add_argument("--lr_critic", type=float, default=1e-4)
    ap.add_argument("--window", default=None, help="k_lo,k_hi (default K/8, 5K/8)")
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--n_val", type=int, default=0, help="cap val prompts (0 = all)")
    ap.add_argument("--objective", default="logit", choices=["logit", "prob"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bf16", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev)
    S = ChurnSampler(cfg, K=a.K, gamma=a.gamma, cfg_scale=a.cfg_scale)
    k_lo, k_hi = (int(x) for x in a.window.split(",")) if a.window else (a.K // 8, 5 * a.K // 8)
    assert 1 <= k_lo <= k_hi <= a.K - 1

    # critic: frozen trunk (pretrained copy), trainable head
    ck = torch.load(a.critic, map_location="cpu")
    critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False, blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
    if ck.get("trunk") is not None:
        critic.trunk.load_state_dict(ck["trunk"])
    critic.head.load_state_dict(ck["head"]); critic.eval()
    opt_c = torch.optim.AdamW(critic.head.parameters(), lr=a.lr_critic, weight_decay=0.0)

    # actor
    model.train(False)
    for p in model.parameters(): p.requires_grad_(True)
    opt_a = torch.optim.AdamW(model.parameters(), lr=a.lr_actor, weight_decay=0.0)
    def lr_at(i):  # cosine from lr_actor to lr_actor_min
        return a.lr_actor_min + 0.5 * (a.lr_actor - a.lr_actor_min) * (1 + math.cos(math.pi * i / a.iters))

    train_prompts = load_train_prompts(a.train_prompts)
    val_prompts = load_prompts(a.val_prompts); val_prompts = val_prompts[: a.n_val] if a.n_val else val_prompts
    test_prompts = load_prompts(a.test_prompts) if a.test_prompts else None
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    geval = torch.Generator(device="cpu").manual_seed(12345)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16) and dev.type == "cuda")

    # replay buffer of cached critic features at k_hi: (h fp16 (L,H), gen_mask (L,), reward)
    buf_h, buf_m, buf_r = deque(maxlen=a.NB), deque(maxlen=a.NB), deque(maxlen=a.NB)
    t_hi = S.alpha * S.t_grid(k_hi) if k_hi < a.K - 1 else S.t_grid(k_hi)

    def critic_steps(n):
        if len(buf_r) < a.Bp: return None
        tot = 0.0
        for _ in range(n):
            idx = random.sample(range(len(buf_r)), a.Bp)
            h = torch.stack([buf_h[i] for i in idx]).to(dev).float()
            m = torch.stack([buf_m[i] for i in idx]).to(dev)
            r = torch.tensor([buf_r[i] for i in idx], device=dev)
            inp = DenoiserInput(k_hi, h, t_hi, None)  # z unused when h is given
            out = critic(inp, None, None, m, h=h)
            loss = critic.loss(out, r, m)
            opt_c.zero_grad(set_to_none=True); loss.backward(); opt_c.step(); tot += loss.item()
        return tot / n

    @torch.no_grad()
    def critic_auc(n):
        n = min(n, len(buf_r))
        if n < 8: return None
        h = torch.stack([buf_h[-i] for i in range(1, n + 1)]).to(dev).float()
        m = torch.stack([buf_m[-i] for i in range(1, n + 1)]).to(dev)
        r = torch.tensor([buf_r[-i] for i in range(1, n + 1)])
        out = critic(DenoiserInput(k_hi, h, t_hi, None), None, None, m, h=h)
        return rank_metrics(out.value.cpu(), r).get("auc")

    best_val, t_start = -1.0, time.time()
    for it in range(a.iters + 1):
        # ---- eval / checkpoint ----
        if it % a.eval_every == 0:
            model.eval()
            with ac:
                val = evaluate(model, S, env, val_prompts, a.eval_bs, geval, dev)
                test = evaluate(model, S, env, test_prompts, a.eval_bs, geval, dev) if test_prompts else None
            model.train(False)
            rec = {"iter": it, "val_acc": val, "test_acc": test, "elapsed": time.time() - t_start}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            torch.save({"params": model.state_dict(), "iter": it, "val_acc": val, "test_acc": test},
                       os.path.join(a.out, f"policy_it{it}.pt"))
            if val > best_val:
                best_val = val
                torch.save({"params": model.state_dict(), "iter": it, "val_acc": val, "test_acc": test},
                           os.path.join(a.out, "policy_best.pt"))
            if it == a.iters: break

        for gq in opt_a.param_groups: gq["lr"] = lr_at(it)

        # ---- actor step ----
        pb = random.sample(train_prompts, a.B)
        ks = [random.randint(k_lo, k_hi) for _ in pb]
        cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
        z0 = (torch.randn((a.B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with ac:
            ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1 for k in ks}, stop_before=max(ks), generator=g)
            logits = []
            for k in sorted(set(ks)):
                sel = torch.tensor([i for i, kk in enumerate(ks) if kk == k], device=dev)
                prev = ro.inputs[k - 1]
                sub = DenoiserInput(k - 1, prev.z[sel], prev.t, prev.x_prev[sel])
                nxt = S.grad_step(model, sub, cond_seq[sel], cond_mask[sel], generator=g)
                out = critic(nxt, cond_seq[sel], cond_mask[sel], gen_mask[sel])
                logits.append(out.logit if a.objective == "logit" else out.value)
            obj = torch.cat(logits).mean()
        opt_a.zero_grad(set_to_none=True); (-obj).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt_a.step()

        # ---- buffer rollouts (fresh prompts, full trajectory, features cached at k_hi) ----
        pb2 = random.sample(train_prompts, a.B)
        cond_seq2, cond_mask2, gen_mask2, cl2 = env.encode(pb2)
        z0 = (torch.randn((a.B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad(), ac:
            ro2 = S.rollout(model, cond_seq2, cond_mask2, z0, record={k_hi}, generator=g)
            r2 = env.reward(pb2, env.decode_text(S.decode(model, ro2.z_final), cl2))
            h2 = critic.features(ro2.inputs[k_hi], cond_seq2, cond_mask2).half().cpu()
        for i in range(a.B):
            buf_h.append(h2[i]); buf_m.append(gen_mask2[i].cpu()); buf_r.append(float(r2[i]))

        # ---- critic refit ----
        closs = critic_steps(a.mc)
        auc = None
        if it % a.nq == 0:
            auc = critic_auc(a.Nq)
            if auc is not None and auc < a.a0:
                critic_steps(a.mc_extra)

        rec = {"iter": it, "obj": obj.item(), "grad_norm": gn, "lr": lr_at(it), "buf_succ": float(r2.mean()),
               "buf_succ_all": sum(buf_r) / max(len(buf_r), 1), "critic_loss": closs, "critic_auc": auc,
               "ks": sorted(set(ks)), "elapsed": time.time() - t_start}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if it % 10 == 0:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    print(f"done. best val {best_val:.4f}")


if __name__ == "__main__":
    main()
