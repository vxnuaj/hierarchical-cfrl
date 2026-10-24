"""Token-commitment profile from stored rollout shards (hcfrl.rl.collect).

For every stored step k, decode the denoiser's clean estimate x_{k-1} with the decoder head and
measure, over generated positions whose final token is not EOS, the fraction already equal to the
decode at the last stored step (k_ref):
  tok      : decoded token at k == decoded token at k_ref
  tok_cl   : Γ(decoded token at k) == Γ(decoded token at k_ref)        (coarse commitment of the token stream)
  ch_cl    : argmax of the cluster channels at k == at k_ref            (H-ELF only: coarse stream itself)
  ch_vs_tok: argmax of the cluster channels at k == Γ(decoded token at k_ref)  (H-ELF: does the coarse stream
             predict the final coarse class earlier than the token stream does?)
Reported overall and split by trajectory reward.

usage: python -m hcfrl.analysis.commitment --config configs/helf_b_kk.yml --ckpt <ckpt> --data outputs/critic_data/kk_hier
"""
from __future__ import annotations

import argparse, glob, json, os

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.sampler import ChurnSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--max_shards", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)
    S = ChurnSampler(cfg, K=64, gamma=56.0)
    n_cl = int(getattr(cfg, "n_clusters", 0) or 0)
    gamma = torch.load(cfg.cluster_map, map_location="cpu")["gamma"] if getattr(cfg, "cluster_map", None) else None
    eos = tok.eos_token_id
    files = sorted(glob.glob(os.path.join(a.data, "shard_*.pt")))
    if a.max_shards: files = files[: a.max_shards]
    acc = {}  # (k, split, metric) -> [hits, count]

    def add(k, split, name, hits, cnt):
        h, c = acc.setdefault((k, split, name), [0, 0]); acc[(k, split, name)] = [h + hits, c + cnt]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for f in files:
            sh = torch.load(f)
            ks = sorted(sh["states"]); k_ref = ks[-1]
            r = sh["reward"]; cl = sh["cond_lens"]
            dec = {k: S.decode(model, sh["states"][k]["x_prev"].to(dev).float()).cpu() for k in ks}   # (N, L) ids
            L = dec[k_ref].shape[1]
            pos = torch.arange(L)[None]
            m = (pos >= cl[:, None]) & (dec[k_ref] != eos)
            ch = {k: sh["states"][k]["x_prev"][..., -n_cl:].float().argmax(-1) for k in ks} if n_cl else None
            for k in ks:
                for split, sel in (("all", torch.ones_like(r, dtype=torch.bool)), ("succ", r > 0.5), ("fail", r <= 0.5)):
                    mm = m & sel[:, None]; cnt = int(mm.sum())
                    if cnt == 0: continue
                    add(k, split, "tok", int(((dec[k] == dec[k_ref]) & mm).sum()), cnt)
                    if gamma is not None:
                        g_k, g_ref = gamma[dec[k]], gamma[dec[k_ref]]
                        add(k, split, "tok_cl", int(((g_k == g_ref) & mm).sum()), cnt)
                        if ch is not None:
                            add(k, split, "ch_cl", int(((ch[k] == ch[k_ref]) & mm).sum()), cnt)
                            add(k, split, "ch_vs_tok", int(((ch[k] == g_ref) & mm).sum()), cnt)
            print(f"{f}: n={len(r)} succ={r.mean():.3f}", flush=True)
    names = ["tok", "tok_cl", "ch_cl", "ch_vs_tok"]
    res = {}
    for split in ("all", "succ", "fail"):
        print(f"== {a.data} [{split}]  (fraction of generated positions committed, ref step {k_ref})")
        print("   k  " + "".join(f"{n:>11}" for n in names))
        for k in sorted({k for k, _, _ in acc}):
            row = []
            for n in names:
                v = acc.get((k, split, n)); row.append(v[0] / v[1] if v else float("nan"))
                res[f"{split}/{n}/{k}"] = row[-1]
            print(f"{k:4d}  " + "".join(f"{x:11.3f}" for x in row))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
