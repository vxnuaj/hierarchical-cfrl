"""Rotation-equivariance check: with the data manifold aligned to the input axes, AdamW's per-coordinate
normalisation also equalises learning rates across input directions, so AdamW should then show Muon-like
off-manifold gain; with a randomly rotated manifold it should not. Muon should be the same in both."""
import json
import torch
from hcfrl.theory.whitening import make_data, train, measure
torch.set_num_threads(8)
for rotate in [True, False]:
    X, Y, T, Nn, sample, target = make_data(lam_off=1e-3, rotate=rotate)
    for kind, lr in [("sgd", 0.05), ("adamw", 1e-3), ("muon", 0.02)]:
        rs = [measure(train(X, Y, kind, 3000, lr, seed=s), sample, target, T, Nn, seed=s) for s in range(2)]
        print(f"rotate={rotate} {kind:5s}: G_off/G_on {sum(r['ratio'] for r in rs)/2:.3f}  readout off-frac {sum(r['readout_off_frac'] for r in rs)/2:.3f}", flush=True)
