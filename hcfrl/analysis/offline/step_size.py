"""Parameter displacement of RL runs: relative norm ||theta_it - theta_0|| / ||theta_0|| overall and per matrix (median),
from saved policy_it*.pt files. Compares how far the actor moves per iteration across model sizes.
usage: python -m hcfrl.analysis.offline.step_size <run_dir>[,<run_dir>...] [iters, default 250]"""
import sys, statistics as st
import torch

runs = sys.argv[1].split(","); its = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "250").split(",")]
for r in runs:
    p0 = torch.load(f"{r}/policy_it0.pt", map_location="cpu", weights_only=False)["params"]
    n_params = sum(v.numel() for v in p0.values() if v.dtype.is_floating_point)
    for it in its:
        try: p1 = torch.load(f"{r}/policy_it{it}.pt", map_location="cpu", weights_only=False)["params"]
        except FileNotFoundError: continue
        num = den = 0.0; rel = []; moved = 0
        for k, v in p0.items():
            if not v.dtype.is_floating_point: continue
            d = (p1[k].float() - v.float()); dn = d.norm().item(); vn = v.float().norm().item()
            num += dn ** 2; den += vn ** 2
            if dn > 0: moved += v.numel()
            if v.dim() == 2 and vn > 0: rel.append(dn / vn)
        print(f"{r.split('/')[-1]:32s} it{it:<5d} params {n_params/1e6:6.1f}M moved {moved/1e6:6.1f}M  rel|dtheta| {num**0.5/den**0.5:.3e}  "
              f"abs|dtheta| {num**0.5:.3f}  per-iter abs {num**0.5/it:.4f}  median matrix rel {st.median(rel):.3e}", flush=True)
