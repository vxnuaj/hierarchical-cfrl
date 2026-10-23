#!/bin/bash
# Gate 0, part B: CFRL-SVG RL from the pretrained critic (long; any A100/A40/L40S).
#   sbatch scripts/gate0b.slurm <config> <sft_ckpt_dir> <task_dir> <tag> [seed] [K] [gamma] [iters]
#SBATCH --job-name=hcfrl-gate0b
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; CKPT=$2; TASK=$3; TAG=$4; SEED=${5:-0}; K=${6:-64}; GAMMA=${7:-56}; ITERS=${8:-1500}
set -x
# best epoch from log.json if present, else from the per-epoch checkpoints' stored eval (field score)
# use the checkpoint gate0a selected, if any
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
    --out outputs/rl/${TAG}_svg2d_s${SEED} --K $K --gamma $GAMMA --iters $ITERS --seed $SEED
echo GATE0B_DONE
