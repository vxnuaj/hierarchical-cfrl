"""Print an attack_transfer trajectory (from a saved json, or from the job's stdout lines)."""
import json, sys

def rows(path):
    if path.endswith(".json"):
        return json.load(open(path))["hist"]
    return [json.loads(l) for l in open(path) if l.startswith("{")]

for path in sys.argv[1:]:
    hist = rows(path); crit = [k for k in hist[0] if isinstance(hist[0][k], dict)]
    print(path)
    print(f"{'iter':>5s} {'acc':>6s} " + " ".join(f"{c:>18s}" for c in crit))
    b = hist[0]
    for h in hist:
        print(f"{h['iter']:5d} {100*h['acc']:6.1f} " + " ".join(
            f"{h[c]['logit'] - b[c]['logit']:+9.0f}/{(h[c]['auc'] or 0):.2f}  " for c in crit))
