"""Dose-response: off-manifold gain vs the strength of Muon's orthogonalisation (blend 0 = Frobenius-normalised
momentum, 1 = full Muon), and vs AdamW/SGD, at fixed lam_off. Same data and measure as whitening.py."""
import json, sys
import torch
from hcfrl.theory.whitening import make_data, train, measure

torch.set_num_threads(8)
out = []
lam_off = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
X, Y, T, Nn, sample, target = make_data(lam_off=lam_off)
for blend in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    rs = [measure(train(X, Y, "muon", 3000, 0.02, blend=blend, seed=s), sample, target, T, Nn, seed=s) for s in range(3)]
    r = {k: sum(x[k] for x in rs) / 3 for k in ("ratio", "readout_off_frac", "test_mse")}; r["blend"] = blend; out.append(r)
    print(json.dumps(r), flush=True)
json.dump(out, open(f"runs/theory/dose_lam{lam_off:g}.json", "w"), indent=1)
