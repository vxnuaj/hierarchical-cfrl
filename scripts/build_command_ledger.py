"""Build runs/commands.jsonl: every `python -m hcfrl...` command executed by a Slurm job, in job order, with job name,
job id, node, and (for jobs started after 2026-11-02) the PROVENANCE line of scripts/common.sh (code fingerprint,
python/torch versions, precision flag). Source: the archived Slurm logs in runs/slurm/ (`set -x` echoes every command).
usage: python3 scripts/build_command_ledger.py"""
import glob, gzip, json, os, re

root = os.path.join(os.path.dirname(__file__), "..", "runs")
out = []
for f in sorted(glob.glob(os.path.join(root, "slurm", "*.out.gz")), key=lambda p: int(re.search(r"-(\d+)\.out\.gz$", p).group(1)) if re.search(r"-(\d+)\.out\.gz$", p) else 0):
    name = os.path.basename(f)[:-7]
    m = re.match(r"(.*)-(\d+)$", name); jobname, jobid = (m.group(1), m.group(2)) if m else (name, "")
    try:
        with gzip.open(f, "rt", errors="replace") as fh: text = fh.read(2_000_000)
    except Exception:
        continue
    prov = next((l.split("PROVENANCE ", 1)[1] for l in text.splitlines() if l.startswith("PROVENANCE ")), None)
    gpu = next((l.strip() for l in text.splitlines()[:5] if "MiB" in l), None)
    fp32 = "HCFRL_FP32=1" in text
    for l in text.splitlines():
        if l.startswith("+ python -m hcfrl"):
            out.append({"job_id": jobid, "job_name": jobname, "gpu": gpu, "fp32": fp32, "provenance": prov, "cmd": l[2:]})
with open(os.path.join(root, "commands.jsonl"), "w") as fh:
    for r in out: fh.write(json.dumps(r) + "\n")
print(f"{len(out)} commands from {len(set(r['job_id'] for r in out))} jobs -> runs/commands.jsonl")
