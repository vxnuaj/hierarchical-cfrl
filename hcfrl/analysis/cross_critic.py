"""Cross-critic evaluation: do the states a policy reaches by exploiting one critic also fool the others?

For each policy checkpoint (base, collapsed, protected, improved), roll out --n validation prompts under the
deployment sampler with fixed noise, record the denoiser inputs at the steps in --ks, decode and verify. Then score
every recorded state with every critic (pretrained critic as RL starts from it: trunk + head, no refit) and save the
raw logits with the verified rewards, so all statistics (AUC per policy x critic, score shift vs. accuracy shift,
how wrong answers of an exploited policy are scored relative to wrong answers of the base) are computed offline.

Critic input gradients ("pulls") are also saved for a subset of base states, for the pairwise alignment analysis.

usage: python -m hcfrl.analysis.cross_critic --config configs/elf_b_game24_cd.yml \
          --policies base=outputs/rl/game24cd_flat_svg2d_s0/policy_it0.pt,... \
          --critics native=outputs/critic/game24cd_flat_2d,... \
          --prompts data/game24/val.jsonl --out outputs/analysis/cross_critic.pt
Policy entries are name=path[@config]; a critic entry is name=<critic dir> (best epoch by field score) or a .pt file.
"""
from __future__ import annotations

import argparse, glob, json, os, time

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput


def best_critic(path: str) -> str:
    if path.endswith(".pt"):
        return path
    if os.path.exists(f"{path}/log.json"):
        return f"{path}/critic_ep{json.load(open(f'{path}/log.json'))['best_epoch']}.pt"
    return max(glob.glob(f"{path}/critic_ep*.pt"), key=lambda f: torch.load(f, map_location="cpu")["eval"]["field_score"])


def parse(spec: str) -> list[tuple[str, str]]:
    return [tuple(x.split("=", 1)) for x in spec.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config used to build the model skeleton and the sampler")
    ap.add_argument("--base_ckpt", default="outputs/elf_b-game24-cd", help="any checkpoint of the architecture (weights replaced)")
    ap.add_argument("--policies", required=True); ap.add_argument("--critics", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ks", default="24,40"); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--n_pull", type=int, default=64, help="base states (per k) for which critic input gradients are saved")
    ap.add_argument("--pull_policy", default=None, help="policy whose states get pulls (default: first)")
    ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.base_ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    ks = [int(x) for x in a.ks.split(",")]
    prompts = load_prompts(a.prompts)[: a.n]
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    policies, critics = parse(a.policies), parse(a.critics)
    pull_policy = a.pull_policy or policies[0][0]

    # ---- 1. rollouts per policy (same prompts, same noise) ----
    states, rewards, texts = {}, {}, {}
    for name, path in policies:
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd["params"] if "params" in sd else sd); model.eval()
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        rec = {k: {"z": [], "x": []} for k in ks}; rs, ts, gms = [], [], []
        t0 = time.time()
        for b0 in range(0, len(prompts), a.bs):
            pb = prompts[b0: b0 + a.bs]
            cond_seq, cond_mask, gen_mask, cl = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cond_seq, cond_mask, z0, record=set(ks), generator=g)
                txt = env.decode_text(S.decode(model, ro.z_final), cl)
            rs.append(env.reward(pb, txt)); ts += txt; gms.append(gen_mask.cpu())
            for k in ks:
                rec[k]["z"].append(ro.inputs[k].z.to(torch.bfloat16).cpu()); rec[k]["x"].append(ro.inputs[k].x_prev.to(torch.bfloat16).cpu())
        states[name] = {k: (torch.cat(rec[k]["z"]), torch.cat(rec[k]["x"]), ro.inputs[k].t) for k in ks}
        rewards[name] = torch.cat(rs); texts[name] = ts; gen_mask_all = torch.cat(gms)
        print(f"policy {name}: acc {rewards[name].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. score every state with every critic; pulls on the pull policy's states ----
    logits, pos_logits, pulls = {}, {}, {}
    for cname, cpath in critics:
        f = best_critic(cpath); ck = torch.load(f, map_location="cpu")
        critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False, blind_clusters=bool(ck.get("blind_clusters", False))).to(dev)
        if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
        critic.head.load_state_dict(ck["head"]); critic.eval()
        for p in critic.parameters(): p.requires_grad_(False)
        t0 = time.time()
        for pname, _ in policies:
            for k in ks:
                Z, X, t = states[pname][k]; out_l, out_p = [], []
                for b0 in range(0, Z.shape[0], a.bs):
                    inp = DenoiserInput(k, Z[b0: b0 + a.bs].to(dev).float(), t, X[b0: b0 + a.bs].to(dev).float())
                    with torch.no_grad(), ac:
                        o = critic(inp, None, None, gen_mask_all[b0: b0 + a.bs].to(dev))
                    out_l.append(o.logit.float().cpu()); out_p.append(o.pos_logit.float().cpu().to(torch.bfloat16))
                logits[(cname, pname, k)] = torch.cat(out_l); pos_logits[(cname, pname, k)] = torch.cat(out_p)
        for k in ks:   # input gradient of the pooled logit on the first n_pull states of the pull policy
            Z, X, t = states[pull_policy][k]; gz, gx = [], []
            for b0 in range(0, a.n_pull, a.bs):
                zi = Z[b0: b0 + a.bs].to(dev).float().requires_grad_(True); xi = X[b0: b0 + a.bs].to(dev).float().requires_grad_(True)
                with ac:
                    o = critic(DenoiserInput(k, zi, t, xi), None, None, gen_mask_all[b0: b0 + a.bs].to(dev))
                dz, dx = torch.autograd.grad(o.logit.float().sum(), [zi, xi])
                gz.append(dz.to(torch.bfloat16).cpu()); gx.append(dx.to(torch.bfloat16).cpu())
            pulls[(cname, k)] = (torch.cat(gz), torch.cat(gx))
        print(f"critic {cname} ({os.path.basename(f)}, field {ck.get('eval', {}).get('field_score', float('nan')):.3f}) scored ({time.time() - t0:.0f}s)", flush=True)
        del critic; torch.cuda.empty_cache()

    torch.save({"policies": policies, "critics": critics, "ks": ks, "rewards": rewards, "texts": texts,
                "gen_mask": gen_mask_all, "logits": logits, "pos_logits": pos_logits, "pulls": pulls,
                "pull_policy": pull_policy, "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
