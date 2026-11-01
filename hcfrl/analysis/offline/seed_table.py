"""Per-seed RL summary from outputs/rl/<tag>_svg2d_s<seed>/log.jsonl: base (it0 test), test at the val-selected
evaluation (ties -> earliest), final test, minimum test, last iteration. Prints per run and mean +- std per tag.
usage: python -m hcfrl.analysis.offline.seed_table game24cd_flat game24cd_swapMA ..."""
import json, statistics as st, sys, os

for tag in sys.argv[1:]:
    rows = []
    for s in range(10):
        f = f"outputs/rl/{tag}_svg2d_s{s}/log.jsonl"
        if not os.path.exists(f): continue
        ev = [r for r in map(json.loads, open(f)) if "test_acc" in r]
        if not ev: continue
        b = max(ev, key=lambda r: (r["val_acc"], -r["iter"]))
        rows.append((s, 100 * ev[0]["test_acc"], 100 * b["test_acc"], b["iter"], 100 * ev[-1]["test_acc"], 100 * min(r["test_acc"] for r in ev), ev[-1]["iter"]))
    for s, b0, bv, bi, fin, mn, li in rows:
        print(f"{tag:22s} s{s} base {b0:5.1f} sel {bv:5.1f}@{bi:<4d} ({bv-b0:+5.1f}) final {fin:5.1f}@{li:<4d} ({fin-b0:+5.1f}) min {mn:5.1f} ({mn-b0:+5.1f})")
    if len(rows) > 1:
        d = lambda i: [r[i] - r[1] for r in rows]
        f = lambda x: f"{st.mean(x):+5.1f} +- {st.stdev(x):4.1f}"
        print(f"{tag:22s} n={len(rows)}  sel {f(d(2))}  final {f(d(4))}  min {f(d(5))}\n")
