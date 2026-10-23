"""Critic pre-training on collected rollouts (CFRL App. C) + the critic-quality-vs-step curve
(the Gate 1 measurement for the research plan).

Reads shards from `hcfrl.rl.collect`, trains trunk (lr_trunk) + head (lr_head) jointly with the
2D/1D BCE loss, batch 64, for `--epochs`, evaluating on held-out shards after each epoch:
per-step Pearson / AUC / pick@V of the sequence value against the terminal reward, plus the
"field score" used for epoch selection (0.5 * Pearson(min-position reading, R) +
0.5 * within-prompt rank-corr(V, R), averaged over stored steps).

usage:
  python -m hcfrl.rl.pretrain_critic --config configs/elf_b_game24.yml --ckpt outputs/elf_b-game24 \
      --data outputs/critic_data/game24_sft --out outputs/critic/game24_2d --mode 2d --epochs 6
"""
from __future__ import annotations

import argparse, glob, json, os, time
from collections import defaultdict

import torch

from hcfrl.rl.common import load_policy
from hcfrl.rl.critic import Critic, rank_metrics
from hcfrl.rl.sampler import DenoiserInput
from hcfrl.rl.env import TaskEnv, Prompt


class StateDataset:
    """Flat list of (shard_id, k, row). Shards stay on CPU in fp16; batches move to device."""

    def __init__(self, shards: list[dict]):
        self.shards = shards
        self.items = [(si, k, i) for si, s in enumerate(shards) for k in s["states"] for i in range(len(s["reward"]))]

    def __len__(self):
        return len(self.items)

    def batch(self, idx: list[int], device):
        """Groups by (shard, k) are not required: we gather tensors row-wise."""
        zs, xs, ts, rs, ps, ks, cls = [], [], [], [], [], [], []
        for j in idx:
            si, k, i = self.items[j]
            s = self.shards[si]; st = s["states"][k]
            zs.append(st["z"][i]); xs.append(st["x_prev"][i]); ts.append(st["t"])
            rs.append(s["reward"][i]); ps.append(s["prompt_idx"][i]); ks.append(k); cls.append(s["cond_lens"][i])
        return (torch.stack(zs).to(device).float(), torch.stack(xs).to(device).float(), torch.tensor(ts),
                torch.stack(rs).to(device), torch.stack(ps), torch.tensor(ks), torch.stack(cls).to(device))


def make_masks(cond_lens: torch.Tensor, L: int):
    pos = torch.arange(L, device=cond_lens.device)[None]
    cond_mask = (pos < cond_lens[:, None]).float()
    return cond_mask, 1.0 - cond_mask


def forward_batch(critic, cfg, env_prompts, zb, xb, tb, clb, device):
    """The critic needs cond_seq to restore cond positions; states already have them restored, so
    pass the state's own cond positions as cond_seq (identity on masked positions)."""
    cond_mask, gen_mask = make_masks(clb, cfg.max_length)
    outs = []
    # t differs per row in general; group rows by t to keep the (B,) time tensor exact
    for t in tb.unique().tolist():
        sel = (tb == t).nonzero().squeeze(1).to(device)
        inp = DenoiserInput(-1, zb[sel], float(t), xb[sel])
        o = critic(inp, zb[sel], cond_mask[sel], gen_mask[sel])
        outs.append((sel, o))
    return outs, gen_mask


def evaluate(critic, cfg, ds: StateDataset, device, bs: int = 64) -> dict:
    critic.eval()
    per_k = defaultdict(lambda: {"v": [], "r": [], "p": [], "minpos": []})
    with torch.no_grad():
        for b0 in range(0, len(ds), bs):
            idx = list(range(b0, min(b0 + bs, len(ds))))
            zb, xb, tb, rb, pb, kb, clb = ds.batch(idx, device)
            outs, gen_mask = forward_batch(critic, cfg, None, zb, xb, tb, clb, device)
            for sel, o in outs:
                selc = sel.cpu()
                for j, row in enumerate(selc.tolist()):
                    k = int(kb[row]); d = per_k[k]
                    d["v"].append(o.value[j].item()); d["r"].append(rb[row].item()); d["p"].append(int(pb[row]))
                    if o.pos_logit is not None:
                        m = gen_mask[sel][j] > 0
                        d["minpos"].append(torch.sigmoid(o.pos_logit[j][m]).min().item())
    res, field = {}, []
    for k in sorted(per_k):
        d = per_k[k]
        v, r, p = torch.tensor(d["v"]), torch.tensor(d["r"]), torch.tensor(d["p"])
        m = rank_metrics(v, r, groups=p)
        if d["minpos"]:
            mp = rank_metrics(torch.tensor(d["minpos"]), r)
            m["pearson_minpos"] = mp.get("pearson", 0.0)
        res[k] = m
        field.append(0.5 * m.get("pearson_minpos", m.get("pearson", 0.0)) + 0.5 * m.get("pick_at_v", 0.0))
    res["field_score"] = sum(field) / max(len(field), 1)
    critic.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="2d"); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr_trunk", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4); ap.add_argument("--holdout_frac", type=float, default=0.1)
    ap.add_argument("--freeze_trunk", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_shards", type=int, default=0); ap.add_argument("--blind_clusters", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, encoder, model = load_policy(a.config, a.ckpt, dev)

    files = sorted(glob.glob(os.path.join(a.data, "shard_*.pt")))
    if a.max_shards: files = files[: a.max_shards]
    shards = [torch.load(f) for f in files]
    n_hold = max(1, int(len(shards) * a.holdout_frac))
    train_ds, hold_ds = StateDataset(shards[n_hold:]), StateDataset(shards[:n_hold])
    print(f"shards train/hold {len(shards) - n_hold}/{n_hold}; states {len(train_ds)}/{len(hold_ds)}; "
          f"success rate {torch.cat([s['reward'] for s in shards]).mean():.3f}", flush=True)

    critic = Critic(model, cfg, mode=a.mode, trunk_trainable=not a.freeze_trunk, blind_clusters=a.blind_clusters).to(dev)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")  # speed; head/loss stay fp32
    groups = [{"params": critic.head.parameters(), "lr": a.lr_head}]
    if not a.freeze_trunk:
        groups.append({"params": critic.trunk.parameters(), "lr": a.lr_trunk})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    os.makedirs(a.out, exist_ok=True)
    log = []
    best = (-1.0, None)
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(len(train_ds)).tolist(); t0 = time.time(); tot = 0.0; n = 0
        for b0 in range(0, len(perm), a.bs):
            idx = perm[b0 : b0 + a.bs]
            zb, xb, tb, rb, pb, kb, clb = train_ds.batch(idx, dev)
            with ac:
                outs, gen_mask = forward_batch(critic, cfg, None, zb, xb, tb, clb, dev)
            loss = sum(critic.loss(o, rb[sel], gen_mask[sel]) * len(sel) for sel, o in outs) / len(idx)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        ev = evaluate(critic, cfg, hold_ds, dev)
        rec = {"epoch": ep, "train_loss": tot / n, "time": time.time() - t0, **{f"k{k}": v for k, v in ev.items() if k != "field_score"}, "field_score": ev["field_score"]}
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save({"head": critic.head.state_dict(), "trunk": None if a.freeze_trunk else critic.trunk.state_dict(),
                    "mode": a.mode, "epoch": ep, "eval": ev, "blind_clusters": a.blind_clusters}, os.path.join(a.out, f"critic_ep{ep}.pt"))
        if ev["field_score"] > best[0]:
            best = (ev["field_score"], ep)
        json.dump({"log": log, "best_epoch": best[1], "best_field_score": best[0], "epochs_done": ep},
                  open(os.path.join(a.out, "log.json"), "w"), indent=1)  # written every epoch (survives job time limits)
    print(f"best epoch {best[1]} field score {best[0]:.3f}")


if __name__ == "__main__":
    main()
