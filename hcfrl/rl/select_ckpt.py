"""Pick the best SFT checkpoint by validation accuracy under the deployment sampler.
Evaluates every checkpoint_* in a dir (optionally every k-th) on the first n val prompts and writes
<out>/select.json with per-checkpoint accuracy and the winner's path.

usage: python -m hcfrl.rl.select_ckpt --config configs/elf_b_game24_cd.yml --dir outputs/elf_b-game24-cd \
          --prompts data/game24/val.jsonl --n 600 --K 64 --gamma 56 --out outputs/eval/game24cd_flat_select
"""
from __future__ import annotations

import argparse, glob, json, os, time

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler
from hcfrl.rl.train_svg import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--dir", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--every", type=int, default=1, help="evaluate every k-th checkpoint (sorted by step)")
    ap.add_argument("--bs", type=int, default=300); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(a.dir, "checkpoint_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    ckpts = ckpts[a.every - 1 :: a.every] if a.every > 1 else ckpts
    prompts = load_prompts(a.prompts)[: a.n]
    res = {}
    for c in ckpts:
        cfg, tok, encoder, model = load_policy(a.config, c, dev)
        env = TaskEnv(tok, encoder, cfg, dev); S = ChurnSampler(cfg, K=a.K, gamma=a.gamma)
        g = torch.Generator(device="cpu").manual_seed(777)
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            acc = evaluate(model, S, env, prompts, a.bs, g, dev)
        res[c] = acc
        print(f"{c}: val acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del model
    best = max(res, key=res.get)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"acc": res, "best": best, "best_acc": res[best]}, open(os.path.join(a.out, "select.json"), "w"), indent=1)
    with open(os.path.join(a.out, "best_ckpt"), "w") as f:
        f.write(best + "\n")
    print(f"BEST {best} {res[best]:.4f}")


if __name__ == "__main__":
    main()
