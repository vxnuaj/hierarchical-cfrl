"""What does the policy turn into under RL? For each saved policy_it<N>.pt of an RL run, generate on
held-out prompts with the deployment sampler and report accuracy, failure categories (format / step
count / unavailable operand / wrong arithmetic / valid chain but wrong target), output diversity,
repetition, length, operator mix, how many prompts flipped vs iteration 0, and a few samples.

usage: python -m hcfrl.analysis.failure_modes --config configs/elf_b_game24_cd.yml --ckpt outputs/elf_b-game24-cd/checkpoint_2450 \
          --run outputs/rl/game24cd_flat_svg2d_s0 --prompts data/game24/val.jsonl --n 500 --out outputs/analysis/fm_game24cd_flat.json
"""
from __future__ import annotations

import argparse, collections, glob, json, os, re

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler
from hcfrl.tasks.arith import _STEP, _apply

CATS = ["ok", "n_steps", "parse", "operand", "arith", "target"]


def classify(text: str, nums, target) -> str:
    avail = sorted(int(n) for n in nums)
    steps = text.split(",")
    if len(steps) != len(avail) - 1:
        return "n_steps"
    last = None
    for s in steps:
        m = _STEP.match(s)
        if m is None:
            return "parse"
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if a not in avail:
            return "operand"
        avail.remove(a)
        if b not in avail:
            return "operand"
        avail.remove(b)
        r = _apply(a, op, b)
        if r is None or r != c:
            return "arith"
        avail.append(c); last = c
    return "ok" if last == target else "target"


@torch.no_grad()
def generate(model, S, env, prompts, bs, seed, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    texts, ids = [], []
    for b0 in range(0, len(prompts), bs):
        pb = prompts[b0: b0 + bs]
        cond_seq, cond_mask, _, cl = env.encode(pb)
        z0 = (torch.randn((len(pb), env.L, model.text_encoder_dim), generator=g) * env.cfg.denoiser_noise_scale).to(dev)
        ro = S.rollout(model, cond_seq, cond_mask, z0, generator=g)
        toks = S.decode(model, ro.z_final)
        texts += env.decode_text(toks, cl)
        gen = toks[:, : env.L]
        ids += [t.tolist() for t in gen.cpu()]
    return texts, ids


def stats(prompts, texts, ids, eos):
    n = len(prompts)
    cats = collections.Counter(classify(t, p.nums, p.target) for p, t in zip(prompts, texts))
    ok = [classify(t, p.nums, p.target) == "ok" for p, t in zip(prompts, texts)]
    lens = [len(t) for t in texts]
    # token repetition: fraction of generated (non-eos) positions equal to the previous position
    rep, tot = 0, 0
    for seq in ids:
        s = [x for x in seq if x != eos]
        tot += max(len(s) - 1, 0); rep += sum(1 for a, b in zip(s, s[1:]) if a == b)
    ops = collections.Counter(c for t in texts for c in t if c in "+-*/")
    nops = sum(ops.values()) or 1
    distinct = len(set(texts)) / n
    common = collections.Counter(texts).most_common(3)
    # how often the chain's steps are a permutation of the input numbers at all
    return {"acc": sum(ok) / n, "cats": {c: cats.get(c, 0) / n for c in CATS}, "mean_chars": sum(lens) / n,
            "distinct_frac": distinct, "tok_repeat_frac": rep / max(tot, 1),
            "op_mix": {k: ops.get(k, 0) / nops for k in "+-*/"}, "top_outputs": common}, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", default=None, help="RL run dir with policy_it*.pt (omit: base only)")
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
    prompts = load_prompts(a.prompts)[: a.n]
    pols = [("base", None)]
    if a.run:
        fs = sorted(glob.glob(os.path.join(a.run, "policy_it*.pt")), key=lambda f: int(re.findall(r"policy_it(\d+)", f)[0]))
        pols = [(int(re.findall(r"policy_it(\d+)", f)[0]), f) for f in fs]
    res, ok0, texts0 = {}, None, None
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    for name, f in pols:
        if f is not None:
            model.load_state_dict(torch.load(f, map_location=dev)["params"]); model.eval()
        with ac:
            texts, ids = generate(model, S, env, prompts, a.bs, a.seed, dev)
        st, ok = stats(prompts, texts, ids, env.eos)
        if ok0 is None:
            ok0, texts0 = ok, texts
        st["regressed_vs_first"] = sum(1 for x, y in zip(ok0, ok) if x and not y) / len(ok)
        st["fixed_vs_first"] = sum(1 for x, y in zip(ok0, ok) if (not x) and y) / len(ok)
        st["same_text_as_first"] = sum(1 for x, y in zip(texts0, texts) if x == y) / len(ok)
        st["samples"] = [{"prompt": p.text, "out": t, "cat": classify(t, p.nums, p.target)} for p, t in list(zip(prompts, texts))[:12]]
        res[str(name)] = st
        c = st["cats"]
        print(f"{str(name):>5} acc {st['acc']:.3f} | n_steps {c['n_steps']:.3f} parse {c['parse']:.3f} operand {c['operand']:.3f} "
              f"arith {c['arith']:.3f} target {c['target']:.3f} | distinct {st['distinct_frac']:.3f} rep {st['tok_repeat_frac']:.3f} "
              f"chars {st['mean_chars']:.1f} | regressed {st['regressed_vs_first']:.3f} fixed {st['fixed_vs_first']:.3f} same {st['same_text_as_first']:.3f} "
              f"| ops {' '.join(f'{k}{v:.2f}' for k, v in st['op_mix'].items())}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
