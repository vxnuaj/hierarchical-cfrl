#!/bin/bash
# Gate 0B on the 1-hour debug partition: same arguments as gate0b.slurm, but resumes from <out>/state.pt
# (saved every 50 iterations by train_svg) so a 1500-iteration run is completed by a chain of ~4 jobs:
#   j1=$(sbatch --parsable scripts/gate0b_debug.slurm <args>); j2=$(sbatch --parsable -d afterany:$j1 scripts/gate0b_debug.slurm <args>); ...
# A chained job exits immediately once the final evaluation is in the log.
# Self-relay: 4 min before the time limit Slurm sends USR1 to this script, which submits its own successor
# (same name and arguments, -d afterany on this job), retrying while the per-user debug cap is full. A chain
# therefore continues without anyone watching; it stops when the run finishes or the python process fails.
# If the cap stays full, the relay leaves an orphan file in outputs/relay_orphans/ (job name + arguments);
# every later debug job resubmits orphans when it starts and after its own relay, so a chain that lost the
# race for a slot is picked up by the next one.
#SBATCH --job-name=hcfrl-gate0b-dbg
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --signal=B:USR1@240
source scripts/common.sh
ARGS=("$@")
CFG=$1; CKPT=$2; TASK=$3; TAG=$4; SEED=${5:-0}; K=${6:-64}; GAMMA=${7:-56}; ITERS=${8:-1500}
OUT_TAG=${9:-${TAG}_svg2d_s${SEED}}; EXTRA="${@:10}"
set -x
if grep -q "\"iter\": $ITERS, \"val_acc\"" outputs/rl/${OUT_TAG}/log.jsonl 2>/dev/null; then echo "already finished"; echo GATE0B_DONE; exit 0; fi
[ -f outputs/eval/${TAG}_ckpt.txt ] && CKPT=$(sed 's/^CKPT=//' outputs/eval/${TAG}_ckpt.txt)
BEST=$(python - <<PY
import json, glob, os, torch
d = "outputs/critic/${TAG}_2d"
if os.path.exists(f"{d}/log.json"):
    print(json.load(open(f"{d}/log.json"))["best_epoch"])
else:
    best = max(glob.glob(f"{d}/critic_ep*.pt"), key=lambda f: torch.load(f, map_location="cpu")["eval"]["field_score"])
    print(int(best.split("critic_ep")[1].split(".pt")[0]))
PY
)
echo "using critic epoch $BEST"
adopt_orphans() {
    for f in outputs/relay_orphans/*; do
        [ -f "$f" ] || continue
        mapfile -t O < "$f"
        oid=${f##*_}; dep=""
        squeue -h -j "$oid" 2>/dev/null | grep -q . && dep="-d afterany:$oid"   # the orphan's job may still be ending
        nxt=$(sbatch --parsable -J "${O[0]}" $dep scripts/gate0b_debug.slurm "${O[@]:1}") && { echo "relay: adopted orphan ${O[0]} -> $nxt"; rm -f "$f"; }
    done
}
relay() {
    for i in $(seq 1 12); do
        nxt=$(sbatch --parsable -J "$SLURM_JOB_NAME" -d afterany:$SLURM_JOB_ID scripts/gate0b_debug.slurm "${ARGS[@]}") && { echo "relay: queued successor $nxt"; adopt_orphans; return; }
        sleep 15
    done
    mkdir -p outputs/relay_orphans
    printf '%s\n' "$SLURM_JOB_NAME" "${ARGS[@]}" > outputs/relay_orphans/${SLURM_JOB_NAME}_$SLURM_JOB_ID
    echo "relay: FAILED to queue a successor; left orphan outputs/relay_orphans/${SLURM_JOB_NAME}_$SLURM_JOB_ID"
}
adopt_orphans
trap relay USR1
python -m hcfrl.rl.train_svg --config $CFG --ckpt $CKPT --critic outputs/critic/${TAG}_2d/critic_ep${BEST}.pt \
    --train_prompts $TASK/train.jsonl --val_prompts $TASK/val.jsonl --test_prompts $TASK/test.jsonl \
    --out outputs/rl/${OUT_TAG} --K $K --gamma $GAMMA --iters $ITERS --seed $SEED --resume $EXTRA &
PID=$!
while kill -0 $PID 2>/dev/null; do wait $PID; done   # wait returns early when USR1 runs the trap
echo GATE0B_DONE
