"""Weight-spectrum fingerprint of the training optimizer (CPU job).

For every model and every weight matrix (ndim 2, both dims >= --min_dim) compute the singular values of W and of
the update dW = W - W_ref (W_ref = the model the fine-tune started from), and summary statistics:
  stable rank   |W|_F^2 / sigma_1^2
  effective rank exp(entropy of sigma / sum sigma)        (Roy & Vetterli)
  top-1 energy  sigma_1^2 / |W|_F^2
Muon makes every update's singular values equal (orthogonalised steps), so its fine-tuning deltas should have high
stable/effective rank; AdamW deltas should be dominated by a few directions.

Model entries: name=<ckpt>[@config]          (SFT checkpoint file/dir or HF id; EMA weights)
               name=critic:<critic .pt>      (critic trunk, skeleton from --config)
               name=policy:<policy_itN.pt>   (RL policy)
Delta entries: name=<model>-<ref model>

usage: python -m hcfrl.analysis.spectra --config configs/elf_b_game24_cd.yml --models ... --deltas ... --out outputs/analysis/spectra.pt
"""
from __future__ import annotations

import argparse, math, time

import torch

from hcfrl.rl.common import load_policy


def stats(s: torch.Tensor) -> dict:
    s = s.double(); e = s ** 2; p = s / s.sum()
    return {"stable_rank": float(e.sum() / e[0]), "eff_rank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum())),
            "top1_energy": float(e[0] / e.sum()), "fro": float(e.sum().sqrt()), "sigma1": float(s[0]), "n": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--deltas", default=""); ap.add_argument("--min_dim", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dev = torch.device("cpu")
    weights = {}
    skeleton = None
    for entry in a.models.split(","):
        name, spec = entry.split("=", 1)
        t0 = time.time()
        if spec.startswith("critic:") or spec.startswith("policy:"):
            kind, path = spec.split(":", 1)
            if skeleton is None:
                skeleton = load_policy(a.config, "outputs/elf_b-game24-cd", dev)[3]
            sd = torch.load(path, map_location="cpu")
            sd = sd["trunk"] if kind == "critic" else sd["params"]
        else:
            path, cfgp = (spec.split("@") + [a.config])[:2]
            sd = load_policy(cfgp, path, dev)[3].state_dict()
        weights[name] = {k: v.float() for k, v in sd.items() if v.ndim == 2 and min(v.shape) >= a.min_dim}
        print(f"loaded {name}: {len(weights[name])} matrices ({time.time() - t0:.0f}s)", flush=True)

    out = {"models": {}, "deltas": {}}
    for name, W in weights.items():
        rec = {}
        for k, w in W.items():
            s = torch.linalg.svdvals(w); rec[k] = {"sigma": s, **stats(s)}
        out["models"][name] = rec
        print(f"spectra {name} done", flush=True)
    for entry in [x for x in a.deltas.split(",") if x]:
        name, pair = entry.split("=", 1); m, r = pair.split("-")
        rec = {}
        for k, w in weights[m].items():
            if k in weights[r] and weights[r][k].shape == w.shape:
                d = w - weights[r][k]
                if d.abs().max() == 0: continue
                s = torch.linalg.svdvals(d); rec[k] = {"sigma": s, **stats(s), "rel": float(d.norm() / weights[r][k].norm())}
        out["deltas"][name] = rec
        print(f"delta {name}: {len(rec)} matrices", flush=True)
    torch.save(out, a.out); print("saved", a.out)


if __name__ == "__main__":
    main()
