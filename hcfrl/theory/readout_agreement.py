"""Do gradients of different readouts agree because the trunk's Jacobian is low-rank? For trunks trained with
SGD / AdamW / Muon (whitening.py data), fit several independent linear readouts on the trunk features to DIFFERENT
targets (the true signal plus independent label noise, i.e. different critics of the same task) and report the
mean pairwise cosine of their input gradients, the on-manifold fraction of those gradients, and the effective
rank of the trunk's input Jacobian."""
import itertools, json
import torch
from hcfrl.theory.whitening import make_data, train

torch.set_num_threads(8)
X, Y, T, Nn, sample, target = make_data(lam_off=1e-3)
res = {}
for kind, lr in [("sgd", 0.05), ("adamw", 1e-3), ("muon", 0.02)]:
    rows = []
    for s in range(2):
        m = train(X, Y, kind, 3000, lr, seed=s)
        feat = lambda x: m.out(torch.relu(m.inp(x)) + torch.relu(m.hid[0](torch.relu(m.inp(x)))))  # trunk output = model output (32-d)
        g = torch.Generator().manual_seed(10 + s)
        Xc = sample(2048)
        with torch.no_grad(): F = feat(Xc)
        sig = target(Xc).sum(1)                              # task signal
        heads = []
        for j in range(4):                                   # 4 critics: same signal, independent noise
            yj = sig + 0.5 * torch.randn(len(sig), generator=g)
            w = torch.linalg.lstsq(F, yj[:, None]).solution.squeeze(1); heads.append(w)
        x = sample(256).requires_grad_(True)
        grads = []
        for w in heads:
            gx, = torch.autograd.grad((feat(x) @ w).sum(), x); grads.append(gx / gx.norm(dim=1, keepdim=True))
        cos = sum((grads[i] * grads[j]).sum(1).mean().item() for i, j in itertools.combinations(range(4), 2)) / 6
        on = sum(((gq @ T).pow(2).sum(1)).mean().item() for gq in grads) / 4
        J = torch.autograd.functional.jacobian(lambda z: feat(z[None])[0], sample(1)[0])   # (32, n)
        sv = torch.linalg.svdvals(J); p = sv / sv.sum(); er = torch.exp(-(p * p.log()).sum()).item()
        rows.append({"cos": cos, "on_frac": on, "jac_eff_rank": er})
    res[kind] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    print(kind, json.dumps({k: round(v, 3) for k, v in res[kind].items()}), flush=True)
