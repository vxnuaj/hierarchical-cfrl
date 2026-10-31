"""Windowed means of gradient coherence (gcos_prev, gcos_ema), buffer success, objective, from train_svg logs."""
import json, sys
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tr = [r for r in rows if "gcos_prev" in r and r["gcos_prev"] is not None]
    ev = [r for r in rows if "val_acc" in r]
    print(path, "evals:", [(r["iter"], round(100 * r["test_acc"], 1)) for r in ev])
    W = 50
    for w0 in range(0, max(r["iter"] for r in tr) + 1, W):
        sub = [r for r in tr if w0 <= r["iter"] < w0 + W]
        if not sub: continue
        m = lambda k: sum(r[k] for r in sub) / len(sub)
        print(f"  it {w0:4d}-{w0+W-1:4d}: gcos_prev {m('gcos_prev'):.3f}  gcos_ema {m('gcos_ema'):.3f}  buf_succ {m('buf_succ'):.3f}  obj {m('obj'):8.1f}  grad_norm {m('grad_norm'):7.1f}")
