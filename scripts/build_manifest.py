"""Build runs/manifest.json: one entry per RL run (tag → config, base checkpoint, critic, seed, iters, extra args,
slurm job id(s), eval curve, final/best test acc, base acc) from runs/rl/*/log.jsonl and the gzipped slurm logs."""
import glob, gzip, json, os, re

root = os.path.join(os.path.dirname(__file__), "..", "runs")
cmds = {}   # out_tag -> (jobid, jobname, argv)
for f in glob.glob(os.path.join(root, "slurm", "*.out.gz")):
    name = os.path.basename(f)[:-7]
    m = re.match(r"(.*)-(\d+)$", name); jobname, jobid = (m.group(1), m.group(2)) if m else (name, "")
    try:
        with gzip.open(f, "rt", errors="replace") as fh:
            head = fh.read(400000)
    except Exception:
        continue
    for line in head.splitlines():
        if line.startswith("+ python -m hcfrl.rl.train_svg") or line.startswith("+ python -m hcfrl.rl.train_grr"):
            mo = re.search(r"--out (\S+)", line)
            if mo:
                tag = os.path.basename(mo.group(1).rstrip("/"))
                cmds.setdefault(tag, []).append({"job_id": jobid, "job_name": jobname, "cmd": line[2:]})
manifest = {}
for d in sorted(glob.glob(os.path.join(root, "rl", "*"))):
    tag = os.path.basename(d); lf = os.path.join(d, "log.jsonl")
    if not os.path.exists(lf): continue
    evals, last_train = [], None
    with open(lf) as fh:
        for line in fh:
            try: r = json.loads(line)
            except Exception: continue
            if "test_acc" in r or "val_acc" in r:
                evals.append({"iter": r.get("iter"), "val_acc": r.get("val_acc"), "test_acc": r.get("test_acc")})
            else:
                last_train = r
    accs = [e["test_acc"] for e in evals if e.get("test_acc") is not None]
    entry = {"tag": tag, "n_evals": len(evals), "evals": evals,
             "base_test_acc": accs[0] if accs else None, "final_test_acc": accs[-1] if accs else None,
             "best_test_acc": max(accs) if accs else None, "last_iter": (last_train or {}).get("iter"),
             "jobs": cmds.get(tag, [])}
    for j in entry["jobs"]:
        c = j["cmd"]
        for key in ("config", "ckpt", "critic", "seed", "iters", "K", "gamma", "lr_actor", "actor_channels", "lr"):
            mo = re.search(rf"--{key} (\S+)", c)
            if mo: entry.setdefault("args", {})[key] = mo.group(1)
    manifest[tag] = entry
out = os.path.join(root, "manifest.json")
json.dump({"generated_from": "outputs/ on carc:/scratch1/verajuan/hcfrl", "runs": manifest}, open(out, "w"), indent=1)
print(f"manifest: {len(manifest)} runs -> {out}")
