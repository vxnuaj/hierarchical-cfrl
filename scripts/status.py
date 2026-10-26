#!/usr/bin/env python3
"""Terminal dashboard for every experiment on the cluster.

    python scripts/status.py              # one snapshot
    python scripts/status.py --watch 600  # refresh every 10 min
    python scripts/status.py --filter kk  # only runs/jobs whose name contains "kk"
    python scripts/status.py --json       # raw JSON (what the remote side returns)

The same file runs on the cluster login node with --remote (stdlib only, python 3.6 safe); it
gathers squeue + outputs/* into one JSON blob, and the local side renders it.
"""
import argparse, glob, json, os, re, subprocess, sys, time

REMOTE_DIR = "/scratch1/verajuan/hcfrl"
SSH_HOST = "carc"

# ----------------------------------------------------------------------------- remote side
def _age(path):
    try: return time.time() - os.path.getmtime(path)
    except OSError: return None

def _tail(path, n=60):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 20000))
            return f.read().decode("utf8", "replace").splitlines()[-n:]
    except OSError: return []

def gather():
    out = {"ts": time.time(), "queue": [], "rl": [], "sft": [], "critic": [], "analysis": [], "failed": []}
    try:
        q = subprocess.check_output(["squeue", "-u", "verajuan", "-h", "-o", "%i|%j|%t|%M|%R|%l"]).decode()
        for line in q.splitlines():
            jid, name, st, t, reason, lim = line.split("|")
            out["queue"].append({"id": jid, "name": name, "state": st, "time": t, "reason": reason, "limit": lim})
    except Exception as e:
        out["queue_error"] = str(e)
    for log in sorted(glob.glob("outputs/rl/*/log.jsonl")):
        name = os.path.basename(os.path.dirname(log)); evals = []; last = {}
        for line in open(log):
            try: r = json.loads(line)
            except ValueError: continue
            if "val_acc" in r:
                acc = r.get("test_acc") if r.get("test_acc") is not None else r.get("val_acc")
                evals.append([r["iter"], acc, r.get("val_acc")])
            elif "obj" in r: last = {k: r.get(k) for k in ("iter", "obj", "grad_norm", "lr", "buf_succ", "critic_auc")}
        out["rl"].append({"name": name, "evals": evals, "last": last, "age": _age(log)})
    for d in sorted(glob.glob("outputs/*_b-*")):
        cks = [int(p.rsplit("_", 1)[1]) for p in glob.glob(d + "/checkpoint_*") if p.rsplit("_", 1)[1].isdigit()]
        total = None
        try:
            cfg = open(d + "/config.yml").read()
            m = re.search(r"^\s*(?:total_steps|max_steps|steps)\s*:\s*(\d+)", cfg, re.M)
            if m: total = int(m.group(1))
        except OSError: pass
        out["sft"].append({"name": os.path.basename(d), "step": max(cks) if cks else 0, "total": total,
                           "age": _age(max(glob.glob(d + "/checkpoint_*"), key=os.path.getmtime)) if cks else _age(d)})
    for d in sorted(glob.glob("outputs/critic/*_2d")):
        eps = sorted(glob.glob(d + "/critic_ep*.pt")); best = None
        if os.path.exists(d + "/log.json"):
            try: best = json.load(open(d + "/log.json")).get("best_epoch")
            except ValueError: pass
        out["critic"].append({"name": os.path.basename(d)[:-3], "epochs": len(eps), "best": best,
                              "done": os.path.exists(d + "/log.json"), "age": _age(d)})
    for f in sorted(glob.glob("outputs/analysis/*.json"), key=os.path.getmtime):
        out["analysis"].append({"name": os.path.basename(f), "age": _age(f)})
    for f in sorted(glob.glob("outputs/slurm/*.out"), key=os.path.getmtime)[-60:]:
        tail = _tail(f)
        bad = [l for l in tail if re.search(r"Traceback|Error|CANCELLED|DUE TO TIME LIMIT|out of memory|Killed", l)]
        if bad and not any(("_DONE" in l) for l in tail):
            out["failed"].append({"name": os.path.basename(f)[:-4], "age": _age(f), "msg": bad[-1].strip()[:110]})
    return out

# ----------------------------------------------------------------------------- local side
BARS = "▁▂▃▄▅▆▇█"
def C(s, c): return "\033[%sm%s\033[0m" % (c, s) if sys.stdout.isatty() else s
def fmt_age(a):
    if a is None: return "   ?"
    a = int(a)
    return "%ds" % a if a < 60 else "%dm" % (a // 60) if a < 3600 else "%.1fh" % (a / 3600) if a < 86400 else "%.1fd" % (a / 86400)

def spark(vals, lo=None, hi=None):
    vals = [v for v in vals if v is not None]
    if not vals: return ""
    lo = min(vals) if lo is None else lo; hi = max(vals) if hi is None else hi
    rng = (hi - lo) or 1e-9
    return "".join(BARS[min(7, int((v - lo) / rng * 7.999))] for v in vals)

def target_iters(name):
    if "grr" in name: return 600
    if name.startswith("kk_") or name.startswith("kkA_"): return 1000
    return 1500

def task_of(name): return name.split("_")[0]

def render(d, filt=None, show_all=False):
    now = d["ts"]; lines = []
    P = lines.append
    P(C("hcfrl status  %s   (data %ds old)" % (time.strftime("%Y-%m-%d %H:%M"), int(time.time() - now)), "1"))
    # queue
    q = [j for j in d["queue"] if not filt or filt in j["name"]]
    run = [j for j in q if j["state"] == "R"]; pend = [j for j in q if j["state"] == "PD"]
    P(C("\nQUEUE", "1;36") + "  running %d   pending %d (%d waiting on a dependency, %d on priority)" % (
        len(run), len(pend), sum(j["reason"] == "(Dependency)" for j in pend), sum(j["reason"] == "(Priority)" for j in pend)))
    for j in sorted(run, key=lambda j: j["name"]):
        P("  %s %-28s %8s / %-9s %s" % (C("R", "32"), j["name"], j["time"], j["limit"], j["reason"]))
    if pend:
        P("  " + C("waiting:", "90") + " " + ", ".join(sorted(j["name"] for j in pend)))
    if d.get("queue_error"): P(C("  squeue failed: " + d["queue_error"], "31"))
    # RL runs
    P(C("\nRL RUNS", "1;36") + "   test acc (%); Δ = last − base; state: " + C("LIVE", "32") + " updated <20 min, " +
      C("DONE", "34") + " reached target iters, " + C("STALL", "33") + " stopped early, " + C("ENDED", "90") + " old short run")
    P("  %-36s %5s %12s %6s  %-10s %s" % ("run", "base", "last (iter)", "Δ", "trajectory", "state / buffer succ / grad"))
    rl = [r for r in d["rl"] if (not filt or filt in r["name"]) and (show_all or not r["name"].startswith(("probe", "smoke")))]
    cur_task = None
    for r in sorted(rl, key=lambda r: (task_of(r["name"]), r["name"])):
        if task_of(r["name"]) != cur_task:
            cur_task = task_of(r["name"]); P(C("  " + cur_task, "1"))
        ev = [e for e in r["evals"] if e[1] is not None]
        if not ev:
            P("  %-36s %s" % (r["name"], C("no eval yet", "90"))); continue
        base = ev[0][1]; it, last = ev[-1][0], ev[-1][1]
        tgt = target_iters(r["name"]); live = r["age"] is not None and r["age"] < 1200
        if live: state = C("LIVE", "32")
        elif it >= tgt: state = C("DONE", "34")
        elif r["age"] is not None and r["age"] > 6 * 3600: state = C("ENDED", "90")   # short/old run, no job attached
        else: state = C("STALL", "33")
        delta = (last - base) * 100
        dc = "32" if delta > 1 else "31" if delta < -1 else "0"
        traj = spark([e[1] for e in ev])
        extra = ""
        if r["last"]:
            l = r["last"]
            extra = "bs %.2f  g %.0f  it %s" % (l.get("buf_succ") or 0, l.get("grad_norm") or 0, l.get("iter"))
        P("  %-36s %5.1f %6.1f (%4d) %s  %-10s %s  %s" % (r["name"], base * 100, last * 100, it, C("%+6.1f" % delta, dc),
                                                         traj, state, C(extra, "90")))
    # SFT + critics
    P(C("\nSFT MODELS", "1;36") + "   (latest checkpoint step; age of last write)")
    for s in d["sft"]:
        if filt and filt not in s["name"]: continue
        live = s["age"] is not None and s["age"] < 1200
        tot = "/%d" % s["total"] if s["total"] else ""
        P("  %-28s step %6d%-7s  %s %s" % (s["name"], s["step"], tot, fmt_age(s["age"]), C("training", "32") if live else ""))
    P(C("\nCRITICS", "1;36") + "   (pretrained value heads)")
    for c in d["critic"]:
        if filt and filt not in c["name"]: continue
        st = C("done, best ep %s" % c["best"], "34") if c["done"] else C("%d/6 epochs" % c["epochs"], "32")
        P("  %-28s %s  %s" % (c["name"], st, fmt_age(c["age"])))
    # analysis + failures
    an = [a for a in d["analysis"] if not filt or filt in a["name"]]
    P(C("\nANALYSIS FILES", "1;36") + "  (newest last)")
    P("  " + "  ".join("%s (%s)" % (a["name"].replace(".json", ""), fmt_age(a["age"])) for a in an))
    fails = [f for f in d["failed"] if not filt or filt in f["name"]]
    if fails:
        P(C("\nJOBS WITH ERRORS IN THEIR LOG (last 60 slurm logs)", "1;31"))
        for f in fails: P("  %-34s %6s  %s" % (f["name"], fmt_age(f["age"]), f["msg"]))
    return "\n".join(lines)

def fetch(timeout=40):
    cmd = ["ssh", "-o", "ConnectTimeout=20", SSH_HOST, "cd %s && python3 scripts/status.py --remote" % REMOTE_DIR]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode().strip() or "ssh failed")
    return json.loads(p.stdout.decode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true"); ap.add_argument("--watch", type=int, default=0)
    ap.add_argument("--filter", default=None); ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true", help="also show smoke/probe runs")
    a = ap.parse_args()
    if a.remote:
        json.dump(gather(), sys.stdout); return
    while True:
        try:
            d = fetch()
        except Exception as e:
            d = None; print(C("cluster unreachable (%s). On USC VPN? try: ssh carc hostname" % str(e)[:120], "31"))
        if d is not None:
            if a.json: print(json.dumps(d, indent=1)); return
            if a.watch: os.system("clear")
            print(render(d, a.filter, a.all))
        if not a.watch: return
        print(C("\nnext refresh in %ds (ctrl-c to stop)" % a.watch, "90"))
        try: time.sleep(a.watch)
        except KeyboardInterrupt: return

if __name__ == "__main__":
    main()
