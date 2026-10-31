"""Per-state agreement of critic input gradients: for each critic, cosine with the leave-one-out mean gradient of
the AdamW-trunk group and of the Muon-trunk group (per state, then averaged), and the per-position split of
the agreement (which generated positions carry the shared direction)."""
import sys
import torch

d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
groups = {"muon": ["native", "mm43", "mm7", "big", "am"], "adamw": ["ma", "ma43", "Anat"]}
gm = d["gen_mask"][: next(iter(d["pulls"].values()))[0].shape[0]].bool()
for k in d["ks"]:
    V = {}
    for c, _ in d["critics"]:
        gz, gx = d["pulls"][(c, k)]
        v = torch.cat([gz.float(), gx.float()], -1) * gm[..., None]
        V[c] = v.flatten(1) / v.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
    print(f"== k={k}: cosine with leave-one-out group mean (per state, averaged)")
    for c in V:
        row = []
        for gname, mem in groups.items():
            others = [V[o] for o in mem if o != c]
            m = torch.stack(others).mean(0); m = m / m.norm(dim=1, keepdim=True)
            row.append(f"{gname} {(V[c] * m).sum(1).mean():.3f}")
        print(f"  {c:7s} " + "  ".join(row))
