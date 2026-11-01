"""Build a Game24 training-prompt set whose rollout success under a given base is close to a target (theory: refit noise
scales with the reward variance pi(1-pi); matching pi across model sizes matches that factor).

Reads a per-prompt screen (collect --no_states --repeat R on chunked train_prompts.jsonl, generations in file order),
keeps every prompt with at least one failure and adds a random fraction of all-correct prompts so that the screened
success of the set equals --target. Writes <out>/{train_prompts,train,val,test}.jsonl (val/test copied unchanged).
usage: python -m hcfrl.analysis.offline.build_hard_set --chunks data/game24_screen/chunk_{0..5}.jsonl
          --screens outputs/screen/M_flat_{0..5} --repeat 2 --target 0.75 --train data/game24/train.jsonl --src data/game24 --out data/game24_hardM_flat
"""
import argparse, json, os, random, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", nargs="+", required=True); ap.add_argument("--screens", nargs="+", required=True)
ap.add_argument("--repeat", type=int, default=2); ap.add_argument("--target", type=float, default=0.75)
ap.add_argument("--train", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rows, succ = [], []
for cf, sd in zip(a.chunks, a.screens):
    pr = [json.loads(l) for l in open(cf) if l.strip()]
    gens = [json.loads(l)["reward"] for l in open(os.path.join(sd, "generations.jsonl")) if l.strip()]
    assert len(gens) == a.repeat * len(pr), (cf, len(gens), len(pr))
    for i, p in enumerate(pr):
        rows.append(p); succ.append(sum(gens[i * a.repeat:(i + 1) * a.repeat]) / a.repeat)
n = len(rows); pi_all = sum(succ) / n
hard = [i for i in range(n) if succ[i] < 1]; easy = [i for i in range(n) if succ[i] == 1]
s_hard = sum(succ[i] for i in hard)
# choose e easy prompts so that (s_hard + e) / (len(hard) + e) = target
e = int(round((a.target * len(hard) - s_hard) / (1 - a.target))) if a.target < 1 else len(easy)
e = max(0, min(e, len(easy)))
rnd = random.Random(a.seed); sel = hard + rnd.sample(easy, e); rnd.shuffle(sel)
pi_sel = sum(succ[i] for i in sel) / len(sel)
os.makedirs(a.out, exist_ok=True)
keep = {rows[i]["input"] for i in sel}
with open(os.path.join(a.out, "train_prompts.jsonl"), "w") as f:
    for i in sel: f.write(json.dumps(rows[i]) + "\n")
with open(os.path.join(a.out, "train.jsonl"), "w") as f:
    for l in open(a.train):
        if l.strip() and json.loads(l)["input"] in keep: f.write(l)
for fn in ("val.jsonl", "test.jsonl"): shutil.copy(os.path.join(a.src, fn), os.path.join(a.out, fn))
stats = {"n_screened": n, "pi_all": pi_all, "n_hard": len(hard), "n_easy_added": e, "n_selected": len(sel), "pi_selected_screen": pi_sel,
         "target": a.target, "repeat": a.repeat}
json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1); print(stats)
