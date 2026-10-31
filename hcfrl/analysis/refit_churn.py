"""Does refitting a critic's head on the policy's newer states rotate its input gradient? (real-model churn test)

For each critic: cache trunk features (frozen trunk, as in RL) of states at k_hi of rollouts of --data_policy
(e.g. a later RL checkpoint), labelled by the verifier. Continue training the head from its pretrained weights with the
RL refit recipe (AdamW lr 1e-4, minibatch 64) for --steps steps on R independent subsets of these states. On fixed probe
states of the base policy, report the cosine between the input gradient (w.r.t. x_prev and z) of the refit heads and of
the original head, and between refit heads of different subsets. A critic whose gradient direction is set by its trunk
keeps it under refits; one whose direction is set by its head's fit to data rotates.

usage: python -m hcfrl.analysis.refit_churn --config ... --critics a=...,b=... --data_policy <pt> --probe_policy <pt> \
          --val data/game24/val.jsonl --train data/game24/train.jsonl --out outputs/analysis/refit_churn.json
"""
from __future__ import annotations

import argparse, copy, itertools, json, random, time

import torch

from hcfrl.analysis.cross_critic import best_critic, parse
from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler, DenoiserInput
from hcfrl.rl.train_svg import load_train_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", default="outputs/elf_b-game24-cd")
    ap.add_argument("--critics", required=True); ap.add_argument("--data_policy", required=True); ap.add_argument("--probe_policy", required=True)
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--n_data", type=int, default=2048); ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--k_hi", type=int, default=40); ap.add_argument("--steps", type=int, default=200); ap.add_argument("--subsets", type=int, default=3)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def set_policy(path):
        sd = torch.load(path, map_location="cpu"); model.load_state_dict(sd["params"] if "params" in sd else sd); model.eval()

    def collect(prompts, record_k, seed):
        g = torch.Generator(device="cpu").manual_seed(seed); out = []
        for b0 in range(0, len(prompts), a.bs):
            pb = prompts[b0: b0 + a.bs]; cs, cm, gm, cl = env.encode(pb)
            z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
            with torch.no_grad(), ac:
                ro = S.rollout(model, cs, cm, z0, record={record_k}, generator=g)
                r = env.reward(pb, env.decode_text(S.decode(model, ro.z_final), cl))
            out.append((ro.inputs[record_k], gm, r))
        return out

    set_policy(a.data_policy)
    data = collect(random.sample(load_train_prompts(a.train), a.n_data), a.k_hi, a.seed + 1)
    print(f"data policy success {torch.cat([d[2] for d in data]).mean():.3f}", flush=True)
    set_policy(a.probe_policy)
    probes = collect(load_prompts(a.val)[: a.n_probe], a.k_hi, a.seed + 2)

    res = {}
    for cname, cpath in parse(a.critics):
        ck = torch.load(best_critic(cpath), map_location="cpu")
        critic = Critic(model, cfg, mode=ck["mode"], trunk_trainable=False).to(dev)
        if ck.get("trunk") is not None: critic.trunk.load_state_dict(ck["trunk"])
        critic.head.load_state_dict(ck["head"]); critic.eval()
        for p in critic.parameters(): p.requires_grad_(False)
        with torch.no_grad(), ac:
            H = torch.cat([critic.features(inp, None, None).float() for inp, _, _ in data]); M = torch.cat([gm for _, gm, _ in data]); R = torch.cat([r for _, _, r in data]).to(dev)
        head0 = copy.deepcopy(critic.head.state_dict()); t_hi = data[0][0].t

        def grad_vec():
            vs = []
            for inp, gm, _ in probes:
                zi = inp.z.detach().float().requires_grad_(True); xi = inp.x_prev.detach().float().requires_grad_(True)
                with torch.enable_grad():
                    for p in critic.head.parameters(): p.requires_grad_(False)
                    o = critic(DenoiserInput(inp.k, zi, inp.t, xi), None, None, gm)
                    gz, gx = torch.autograd.grad(o.logit.float().sum(), [zi, xi])
                v = torch.cat([gz, gx], -1) * gm[..., None]; vs.append(v.flatten(1))
            v = torch.cat(vs); return v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

        g0 = grad_vec(); refits = []
        idx_all = torch.randperm(len(R))
        for s_ in range(a.subsets):
            critic.head.load_state_dict(head0)
            for p in critic.head.parameters(): p.requires_grad_(True)
            opt = torch.optim.AdamW(critic.head.parameters(), lr=1e-4)
            sub = idx_all[s_ * (len(R) // a.subsets): (s_ + 1) * (len(R) // a.subsets)]
            for _ in range(a.steps):
                i = sub[torch.randint(0, len(sub), (64,))]
                out = critic(DenoiserInput(a.k_hi, H[i], t_hi, None), None, None, M[i], h=H[i])
                loss = critic.loss(out, R[i], M[i]); opt.zero_grad(); loss.backward(); opt.step()
            critic.head.eval(); refits.append(grad_vec())
        cos0 = [float((g0 * gr).sum(1).mean()) for gr in refits]
        cosrr = [float((refits[i] * refits[j]).sum(1).mean()) for i, j in itertools.combinations(range(len(refits)), 2)]
        res[cname] = {"cos_to_original": cos0, "cos_between_refits": cosrr}
        print(f"critic {cname}: cos(refit, original) {sum(cos0)/len(cos0):.3f}  cos(refit, refit) {sum(cosrr)/len(cosrr):.3f}", flush=True)
        del critic, H; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("saved", a.out)


if __name__ == "__main__":
    main()
