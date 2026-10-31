"""Does an off-manifold invariance augmentation of the critic's head fit rescue a Muon-trunk critic in the toy loop?"""
import json
from hcfrl.theory.loop_toy import run
out = {}
for kind, lr in [("muon", 0.02), ("adamw", 1e-3)]:
    for aug, dirs in [(0.0, "off"), (0.3, "off"), (1.0, "off"), (1.0, "rand")]:
        hs = [run(kind, lr, True, seed=s, aug=aug, aug_dirs=dirs) for s in range(3)]
        avg = [sum(h[i]["reward"] for h in hs) / 3 for i in range(len(hs[0]))]
        out[f"{kind}_aug{aug}_{dirs}"] = avg
        print(f"{kind:5s} aug {aug:.1f} {dirs:4s}: reward " + " ".join(f"{v:+.2f}" for v in avg[::2]), flush=True)
json.dump(out, open("runs/theory/loop_fix.json", "w"), indent=1)
