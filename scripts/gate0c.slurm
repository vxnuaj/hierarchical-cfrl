#!/bin/bash
# Gate 0, part C: GRR (score-function) baseline from the selected SFT checkpoint, no critic.
#   sbatch scripts/gate0c.slurm <config> <sft_ckpt_dir> <task_dir> <tag> [seed] [K] [gamma] [iters]
#SBATCH --job-name=hcfrl-gate0c
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; CKPT=$2; TASK=$3; TAG=$4; SEED=${5:-0}; K=${6:-64}; GAMMA=${7:-56}; ITERS=${8:-600}
[ -f outputs/eval/${TAG}_ckpt.txt ] && CKPT=$(sed 's/^CKPT=//' outputs/eval/${TAG}_ckpt.txt)
set -x
python -m hcfrl.rl.train_grr --config $CFG --ckpt $CKPT \
    --train_prompts $TASK/train.jsonl --val_prompts $TASK/val.jsonl --test_prompts $TASK/test.jsonl \
    --out outputs/rl/${TAG}_grr_s${SEED} --K $K --gamma $GAMMA --iters $ITERS --seed $SEED
echo GATE0C_DONE
