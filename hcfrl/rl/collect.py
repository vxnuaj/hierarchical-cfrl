"""Collect rollouts of a policy for critic pretraining / diagnostics.

For each prompt: roll out under (K, gamma, cfg), record the denoiser input at the requested steps,
decode, verify. Saves shards: {k: int, z: fp16 (N,L,D), x_prev: fp16, t: float, reward: (N,), prompt_idx}.

usage (on a GPU node):
  python -m hcfrl.rl.collect --config configs/elf_b_game24.yml --ckpt outputs/elf_b-game24 \
      --prompts data/game24/train_prompts.jsonl --n 16000 --K 64 --gamma 56 --steps 1,8,16,32,48 \
      --out outputs/critic_data/game24_sft --batch 128
"""
from __future__ import annotations

import argparse, os, time
from pathlib import Path

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.env import TaskEnv, load_prompts
from hcfrl.rl.sampler import ChurnSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--repeat", type=int, default=1, help="rollouts per prompt (grouped for pick@V)")
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--gamma", type=float, default=56.0)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--steps", default="1,8,16,32,48"); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_states", action="store_true", help="only decode + verify (accuracy eval)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    env = TaskEnv(tok, encoder, cfg, dev)
    S = ChurnSampler(cfg, K=a.K, gamma=a.gamma, cfg_scale=a.cfg_scale)
    record = set() if a.no_states else {int(s) for s in a.steps.split(",")}

    prompts = load_prompts(a.prompts)[: a.n]
    prompts = [p for p in prompts for _ in range(a.repeat)]
    pidx = torch.tensor([i // a.repeat for i in range(len(prompts))])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    torch.manual_seed(a.seed)

    rewards, t0 = [], time.time()
    for b0 in range(0, len(prompts), a.batch):
        pb = prompts[b0 : b0 + a.batch]
        cond_seq, cond_mask, gen_mask, cl = env.encode(pb)
        z0 = (torch.randn((len(pb), cfg.max_length, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale).to(dev)
        ro = S.rollout(model, cond_seq, cond_mask, z0, record=record, generator=g)
        texts = env.decode_text(S.decode(model, ro.z_final), cl)
        r = env.reward(pb, texts)
        rewards.append(r)
        if record:
            shard = {"reward": r, "prompt_idx": pidx[b0 : b0 + len(pb)], "cond_lens": cl.cpu(),
                     "states": {k: {"z": inp.z.half().cpu(), "x_prev": inp.x_prev.half().cpu(), "t": inp.t}
                                for k, inp in ro.inputs.items()},
                     "texts": texts, "prompts": [p.text for p in pb]}
            torch.save(shard, os.path.join(a.out, f"shard_{b0 // a.batch:05d}.pt"))
        done = b0 + len(pb)
        print(f"{done}/{len(prompts)}  acc so far {torch.cat(rewards).mean():.4f}  {time.time() - t0:.0f}s", flush=True)
    acc = torch.cat(rewards).mean().item()
    print(f"FINAL accuracy {acc:.4f} on {len(prompts)} rollouts (K={a.K}, gamma={a.gamma}, cfg={a.cfg_scale})")
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(f"{acc:.4f}\t{len(prompts)}\tK={a.K}\tgamma={a.gamma}\tcfg={a.cfg_scale}\tckpt={a.ckpt}\n")


if __name__ == "__main__":
    main()
