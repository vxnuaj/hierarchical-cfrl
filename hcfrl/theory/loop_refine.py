"""Trunk-side fix in the toy loop: Muon-trained trunk refined with AdamW for `refine` steps; off-manifold gain of the
refined trunk and the loop's true reward (refit heads)."""
import json
from hcfrl.theory.loop_toy import run
from hcfrl.theory.whitening import make_data, train, measure
out = {}
X, Y, T, Nn, sample, target = make_data(lam_off=1e-3)
for refine, rlr, wd in [(3000, 1e-3, 0.1), (3000, 1e-3, 1.0), (3000, 3e-3, 1.0)]:
    gains = []
    for s in range(3):
        m = train(X, Y, "muon", 3000, 0.02, seed=s)
        if refine: m = train(X, Y, "adamw", refine, rlr, seed=s, init=m, wd=wd)
        gains.append(measure(m, sample, target, T, Nn, seed=s)["ratio"])
    hs = [run("muon", 0.02, True, seed=s, refine=refine, refine_lr=rlr, refine_wd=wd) for s in range(3)]
    avg = [sum(h[i]["reward"] for h in hs) / 3 for i in range(len(hs[0]))]
    out[f"refine{refine}_lr{rlr:g}_wd{wd:g}"] = {"gain_ratio": sum(gains) / 3, "reward": avg}
    print(f"Muon trunk + AdamW refine {refine:4d} steps lr {rlr:g} wd {wd:g}: off/on gain {sum(gains)/3:.2f}  reward " + " ".join(f"{v:+.2f}" for v in avg[::2]), flush=True)
json.dump(out, open("runs/theory/loop_refine_wd.json", "w"), indent=1)
