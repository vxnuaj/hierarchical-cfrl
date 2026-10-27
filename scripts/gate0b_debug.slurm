#!/bin/bash
# Gate 0B on the 1-hour debug partition: same arguments as gate0b.slurm, but resumes from <out>/state.pt
# (saved every 50 iterations by train_svg) so a 1500-iteration run is completed by a chain of ~4 jobs:
#   j1=$(sbatch --parsable scripts/gate0b_debug.slurm <args>); j2=$(sbatch --parsable -d afterany:$j1 scripts/gate0b_debug.slurm <args>); ...
# A chained job exits immediately once the final evaluation is in the log.
#SBATCH --job-name=hcfrl-gate0b-dbg
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
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
python -m hcfrl.rl.train_svg --config $CFG --ckpt $CKPT --critic outputs/critic/${TAG}_2d/critic_ep${BEST}.pt \
    --train_prompts $TASK/train.jsonl --val_prompts $TASK/val.jsonl --test_prompts $TASK/test.jsonl \
    --out outputs/rl/${OUT_TAG} --K $K --gamma $GAMMA --iters $ITERS --seed $SEED --resume $EXTRA
echo GATE0B_DONE
