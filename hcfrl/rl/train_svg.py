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
from hcfrl.rl.critic import Critic, rank_metrics, group_rank_loss
from hcfrl.rl.env import TaskEnv, Prompt, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


def load_train_prompts(path: str) -> list[Prompt]:
    """Train JSONL has {input, output[, task]}."""
    from hcfrl.rl.env import prompt_from_train_row
    return [prompt_from_train_row(json.loads(l)) for l in open(path) if l.strip()]


def split_channels(inp: DenoiserInput, n_cl: int, keep: str) -> DenoiserInput:
    """Detach the gradient on the token channels (keep='clusters') or the cluster channels (keep='tokens')
    of both state components, so the actor gradient flows only through the kept channels."""
    def f(x):
        tok, cl = x[..., :-n_cl], x[..., -n_cl:]
        return torch.cat([tok, cl.detach()], -1) if keep == "tokens" else torch.cat([tok.detach(), cl], -1)
    return DenoiserInput(inp.k, f(inp.z), inp.t, f(inp.x_prev))


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
    ap.add_argument("--a0", type=float, default=0.9, help="AUC threshold for extra critic steps (0 disables)")
    ap.add_argument("--lr_actor", type=float, default=1e-6); ap.add_argument("--lr_actor_min", type=float, default=1e-7)
    ap.add_argument("--lr_critic", type=float, default=1e-4)
    ap.add_argument("--window", default=None, help="k_lo,k_hi (default K/8, 5K/8)")
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--n_val", type=int, default=0, help="cap val prompts (0 = all)")
    ap.add_argument("--objective", default="logit", choices=["logit", "prob"])
    ap.add_argument("--norm_pen", type=float, default=0.0,
                    help="latent-norm penalty: actor loss += norm_pen * |obj_0| * mean_gen_pos((|x_k,tok| / norm_0 - 1)^2), where norm_0 is the "
                         "mean per-position token-latent norm of the clean estimate at the first actor step and obj_0 the first objective. "
                         "Control for the flat model: is the collapse the latent inflation seen in the trace?")
    ap.add_argument("--rsa_rho", type=float, default=0.0,
                    help="reward-sharpness-aware actor step (Kim & Sim, CVPR 2026, input-space variant): the critic is evaluated at the "
                         "one-step worst-case displacement of its input, (z~_k, x_{k-1}) + delta with delta = -rho * |state| * grad / |grad| "
                         "(norms over generated positions, per sequence, delta detached), so the actor maximises the flattened logit "
                         "min_{|delta| <= rho |state|} l_phi. Every run logs rsa_gap = l(unperturbed) - l(perturbed) when rho > 0")
    ap.add_argument("--rel_step", type=int, default=0,
                    help="1: layer-wise relative actor step (LARS-style): each weight matrix gets lr_actor * rms(W)/mean_rms(all matrices), "
                         "so every matrix moves by the same fraction of its own size at the same average lr; vectors keep lr_actor. "
                         "Multipliers fixed from the base weights.")
    ap.add_argument("--rsa_apply", type=int, default=1, help="0: only measure rsa_gap at rsa_rho, optimise the unperturbed objective")
    ap.add_argument("--actor_mode", default="critic", choices=["critic", "random", "none"],
                    help="diagnostic: 'random' replaces the actor gradient by Gaussian noise of the same clipped norm; 'none' skips the actor step")
    ap.add_argument("--actor_channels", default="all", choices=["all", "tokens", "clusters"],
                    help="H-ELF ablation: let the actor gradient flow only through the token channels or only the cluster channels of (z_k, x_{k-1})")
    ap.add_argument("--G", type=int, default=1, help="rollouts per prompt in the replay buffer (B/G prompts x G)")
    ap.add_argument("--rank_w", type=float, default=0.0, help="within-prompt ranking weight in the critic refit (needs G>1)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--buf_dtype", choices=["bf16", "fp16"], default="bf16",
                    help="dtype of the cached critic features in the replay buffer. Runs before 2026-10-27 used fp16; the AdamW-trained ELF has trunk activations up to ~5e4 that overflow fp16 (inf -> NaN critic refit), so the default is now bf16")
    ap.add_argument("--log_gcos", type=int, default=0,
                    help="1: log the cosine of each actor gradient with the previous one (gcos_prev) and with an EMA (0.9) of past unit "
                         "gradients (gcos_ema): coherence of the direction the critic asks the policy to move in")
    ap.add_argument("--resume", action="store_true", help="resume from <out>/state.pt if present")
    ap.add_argument("--save_every", type=int, default=50, help="save full resumable state every N iters")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev)
    S = ChurnSampler(cfg, K=a.K, gamma=a.gamma, cfg_scale=a.cfg_scale)
    k_lo, k_hi = (int(x) for x in a.window.split(",")) if a.window else (a.K // 8, 5 * a.K // 8)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0)
    assert a.actor_channels == "all" or n_cl > 0, "--actor_channels needs an H-ELF config"
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
    if a.rel_step:
        mats = [(n, q) for n, q in model.named_parameters() if q.requires_grad and q.ndim >= 2]
        rms = {n: q.detach().float().pow(2).mean().sqrt().item() for n, q in mats}
        mean_rms = sum(rms.values()) / len(rms)
        groups = [{"params": [q], "mult": rms[n] / mean_rms} for n, q in mats]
        groups.append({"params": [q for q in model.parameters() if q.requires_grad and q.ndim < 2], "mult": 1.0})
        opt_a = torch.optim.AdamW(groups, lr=a.lr_actor, weight_decay=0.0)
        mults = sorted(g["mult"] for g in groups[:-1])
        print(f"rel_step: {len(mats)} matrices, mean rms {mean_rms:.4g}, mult min {mults[0]:.3g} median {mults[len(mults)//2]:.3g} max {mults[-1]:.3g}", flush=True)
    else:
        opt_a = torch.optim.AdamW(model.parameters(), lr=a.lr_actor, weight_decay=0.0)
        for gq in opt_a.param_groups: gq["mult"] = 1.0
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

    # replay buffer of cached critic features at k_hi: (h bf16/fp16 (L,H), gen_mask (L,), reward)
    buf_dt = torch.bfloat16 if a.buf_dtype == "bf16" else torch.float16
    buf_h, buf_m, buf_r, buf_p = deque(maxlen=a.NB), deque(maxlen=a.NB), deque(maxlen=a.NB), deque(maxlen=a.NB)
    prompt_counter = [0]
    ref = [None]   # (norm_0, |obj_0|) measured at the first actor step; used by --norm_pen and logged as tok_norm
    t_hi = S.alpha * S.t_grid(k_hi) if k_hi < a.K - 1 else S.t_grid(k_hi)

    def critic_steps(n):
        if n <= 0 or len(buf_r) < a.Bp: return None
        tot = 0.0
        for _ in range(n):
            if a.G > 1:   # sample whole prompt groups (entries of a group are contiguous in the buffer)
                starts = random.sample(range(0, len(buf_r) - a.G + 1, a.G), a.Bp // a.G)
                idx = [s_ + j for s_ in starts for j in range(a.G)]
            else:
                idx = random.sample(range(len(buf_r)), a.Bp)
            h = torch.stack([buf_h[i] for i in idx]).to(dev).float()
            m = torch.stack([buf_m[i] for i in idx]).to(dev)
            r = torch.tensor([buf_r[i] for i in idx], device=dev)
            inp = DenoiserInput(k_hi, h, t_hi, None)  # z unused when h is given
            out = critic(inp, None, None, m, h=h)
            loss = critic.loss(out, r, m)
            if a.rank_w > 0 and a.G > 1:
                loss = loss + a.rank_w * group_rank_loss(out.logit, r, torch.tensor([buf_p[i] for i in idx], device=dev))
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
        gp = torch.tensor([buf_p[-i] for i in range(1, n + 1)]) if a.G > 1 else None
        mets = rank_metrics(out.value.cpu(), r, groups=gp)
        last_wauc[0] = mets.get("wauc")
        return mets.get("auc")

    best_val, t_start, start_it = -1.0, time.time(), 0
    gstate = [None, None]   # previous unit actor gradient, EMA of unit gradients (--log_gcos)
    last_wauc = [None]
    state_path = os.path.join(a.out, "state.pt")
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        model.load_state_dict(st["model"]); opt_a.load_state_dict(st["opt_a"])
        critic.head.load_state_dict(st["critic_head"]); opt_c.load_state_dict(st["opt_c"])
        for h_, m_, r_, p_ in zip(st["buf_h"], st["buf_m"], st["buf_r"], st.get("buf_p", [0] * len(st["buf_r"]))):
            buf_h.append(h_); buf_m.append(m_); buf_r.append(r_); buf_p.append(p_)
        prompt_counter[0] = (max(buf_p) + 1) if len(buf_p) else 0
        start_it, best_val = st["iter"] + 1, st["best_val"]; ref[0] = st.get("ref")
        g.set_state(st["g"]); geval.set_state(st["geval"]); random.setstate(st["py_random"]); torch.set_rng_state(st["torch_rng"])
        print(f"resumed from iter {st['iter']} (buffer {len(buf_r)})", flush=True)

    def save_state(it):
        torch.save({"model": model.state_dict(), "opt_a": opt_a.state_dict(), "critic_head": critic.head.state_dict(),
                    "opt_c": opt_c.state_dict(), "buf_h": list(buf_h), "buf_m": list(buf_m), "buf_r": list(buf_r), "buf_p": list(buf_p),
                    "iter": it, "best_val": best_val, "g": g.get_state(), "geval": geval.get_state(),
                    "py_random": random.getstate(), "torch_rng": torch.get_rng_state(), "ref": ref[0]}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    for it in range(start_it, a.iters + 1):
        # ---- eval / checkpoint ----
        if it % a.eval_every == 0:
            model.eval()
            geval.manual_seed(12345)   # same noise stream at every eval -> paired comparisons across iters
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

        for gq in opt_a.param_groups: gq["lr"] = lr_at(it) * gq.get("mult", 1.0)

        # ---- actor step ----
        pb = random.sample(train_prompts, a.B)
        ks = [random.randint(k_lo, k_hi) for _ in pb]
        cond_seq, cond_mask, gen_mask, _ = env.encode(pb)
        z0 = (torch.randn((a.B, env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with ac:
            ro = S.rollout(model, cond_seq, cond_mask, z0, record={k - 1 for k in ks}, stop_before=max(ks), generator=g)
            logits, norms, gaps = [], [], []
            for k in sorted(set(ks)):
                sel = torch.tensor([i for i, kk in enumerate(ks) if kk == k], device=dev)
                prev = ro.inputs[k - 1]
                sub = DenoiserInput(k - 1, prev.z[sel], prev.t, prev.x_prev[sel])
                nxt = S.grad_step(model, sub, cond_seq[sel], cond_mask[sel], generator=g)
                if a.actor_channels != "all":
                    nxt = split_channels(nxt, n_cl, a.actor_channels)
                if a.rsa_rho > 0:   # one-step worst-case displacement of the critic input (sharpness-aware step)
                    zd = nxt.z.detach().requires_grad_(True); xd = nxt.x_prev.detach().requires_grad_(True)
                    with torch.enable_grad():
                        o0 = critic(DenoiserInput(nxt.k, zd, nxt.t, xd), cond_seq[sel], cond_mask[sel], gen_mask[sel])
                        l0 = o0.logit if a.objective == "logit" else o0.value
                        gz, gx = torch.autograd.grad(l0.sum(), [zd, xd])
                    gm3 = gen_mask[sel].to(gz.dtype).unsqueeze(-1)
                    gz, gx = gz * gm3, gx * gm3
                    gnorm = torch.sqrt((gz.float() ** 2).sum((1, 2)) + (gx.float() ** 2).sum((1, 2))).clamp_min(1e-12)
                    snorm = torch.sqrt(((nxt.z.detach().float() * gm3.float()) ** 2).sum((1, 2))
                                       + ((nxt.x_prev.detach().float() * gm3.float()) ** 2).sum((1, 2)))
                    scale = (a.rsa_rho * snorm / gnorm)[:, None, None]
                    dz, dx = -(gz.float() * scale).to(nxt.z.dtype).detach(), -(gx.float() * scale).to(nxt.x_prev.dtype).detach()
                    with torch.no_grad():
                        op = critic(DenoiserInput(nxt.k, zd + dz, nxt.t, xd + dx), cond_seq[sel], cond_mask[sel], gen_mask[sel])
                        lp = op.logit if a.objective == "logit" else op.value
                        gaps.append((l0.detach() - lp).float())
                    if a.rsa_apply:
                        nxt = DenoiserInput(nxt.k, nxt.z + dz, nxt.t, nxt.x_prev + dx)
                out = critic(nxt, cond_seq[sel], cond_mask[sel], gen_mask[sel])
                logits.append(out.logit if a.objective == "logit" else out.value)
                xk_tok = nxt.x_prev[..., : nxt.x_prev.shape[-1] - n_cl]          # clean estimate, token channels
                gm = gen_mask[sel].to(xk_tok.dtype)
                pos_norm = xk_tok.float().norm(dim=-1)                              # (b, L)
                norms.append(((pos_norm * gm).sum(1) / gm.sum(1).clamp_min(1.0)))  # per-sequence mean over generated positions
            obj = torch.cat(logits).mean()
            tok_norm_t = torch.cat(norms)                                            # (B,)
            tok_norm = tok_norm_t.mean().item()
            if ref[0] is None: ref[0] = (tok_norm, abs(obj.item()))                 # norm_0, |obj_0| from the first actor step
            pen = ((tok_norm_t / ref[0][0] - 1.0) ** 2).mean()
            loss_a = -obj + a.norm_pen * ref[0][1] * pen if a.norm_pen > 0 else -obj
        opt_a.zero_grad(set_to_none=True); loss_a.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        gcos_prev = gcos_ema = None
        if a.log_gcos:
            with torch.no_grad():
                gv = torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None]); gv = gv / gv.norm().clamp_min(1e-12)
                if gstate[0] is not None:
                    gcos_prev = float(gv @ gstate[0]); gcos_ema = float(gv @ gstate[1] / gstate[1].norm().clamp_min(1e-12))
                    gstate[1].mul_(0.9).add_(gv, alpha=0.1)
                else:
                    gstate[1] = gv.clone()
                gstate[0] = gv
        if not math.isfinite(gn):   # never step the actor on a non-finite gradient (it would poison the policy for good)
            raise RuntimeError(f"non-finite actor gradient at iter {it} (obj {obj.item()})")
        if a.actor_mode == "random":   # same clipped norm (1.0), random direction
            with torch.no_grad():
                ps = [p for p in model.parameters() if p.grad is not None]
                for p in ps: p.grad.normal_()
                tot = torch.sqrt(sum((p.grad ** 2).sum() for p in ps))
                for p in ps: p.grad.mul_(1.0 / tot)
        if a.actor_mode != "none":
            opt_a.step()

        # ---- buffer rollouts (fresh prompts, full trajectory, features cached at k_hi) ----
        pb2 = random.sample(train_prompts, a.B // a.G)
        pid2 = [prompt_counter[0] + j for j in range(len(pb2)) for _ in range(a.G)]; prompt_counter[0] += len(pb2)
        pb2 = [p_ for p_ in pb2 for _ in range(a.G)]
        cond_seq2, cond_mask2, gen_mask2, cl2 = env.encode(pb2)
        z0 = (torch.randn((len(pb2), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        with torch.no_grad(), ac:
            ro2 = S.rollout(model, cond_seq2, cond_mask2, z0, record={k_hi}, generator=g)
            r2 = env.reward(pb2, env.decode_text(S.decode(model, ro2.z_final), cl2))
            h2 = critic.features(ro2.inputs[k_hi], cond_seq2, cond_mask2).to(buf_dt).cpu()
            if not torch.isfinite(h2.float()).all():
                raise RuntimeError(f"non-finite critic features at iter {it} (buffer dtype {a.buf_dtype})")
        for i in range(len(pb2)):
            buf_h.append(h2[i]); buf_m.append(gen_mask2[i].cpu()); buf_r.append(float(r2[i])); buf_p.append(pid2[i])

        # ---- critic refit ----
        closs = critic_steps(a.mc)
        auc = None
        if it % a.nq == 0:
            auc = critic_auc(a.Nq)
            if auc is not None and auc < a.a0:
                critic_steps(a.mc_extra)

        rec = {"iter": it, "obj": obj.item(), "grad_norm": gn, "lr": lr_at(it), "buf_succ": float(r2.mean()), "tok_norm": tok_norm, "rsa_gap": (torch.cat(gaps).mean().item() if gaps else None),
               "norm_pen": (a.norm_pen * ref[0][1] * pen.item()) if a.norm_pen > 0 else None,
               "buf_succ_all": sum(buf_r) / max(len(buf_r), 1), "critic_loss": closs, "critic_auc": auc, "critic_wauc": last_wauc[0] if auc is not None else None,
               "ks": sorted(set(ks)), "gcos_prev": gcos_prev, "gcos_ema": gcos_ema, "elapsed": time.time() - t_start}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if it % 10 == 0:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if a.save_every and it % a.save_every == 0:
            save_state(it)
    print(f"done. best val {best_val:.4f}")


if __name__ == "__main__":
    main()
