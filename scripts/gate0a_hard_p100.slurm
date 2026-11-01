#!/bin/bash
# Critic data + critic pre-training on a custom training-prompt set (e.g. a difficulty-matched subset), P100 fp32.
#   sbatch scripts/gate0a_hard_p100.slurm <config> <ckpt> <task_dir> <tag> [seed] [accum]
#SBATCH --job-name=gate0a-hard
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --exclude=e22-[01-16],d23-14,e21-01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
export HCFRL_FP32=1
CFG=$1; CKPT=$2; TASK=$3; TAG=$4; SEED=${5:-0}; ACC=${6:-4}
set -x
echo "CKPT=$CKPT" > outputs/eval/${TAG}_ckpt.txt
python -m hcfrl.rl.collect --config $CFG --ckpt $CKPT --prompts $TASK/train_prompts.jsonl --n 2000 --repeat 4 \
    --K 64 --gamma 56 --steps 1,8,16,24,32,40,48 --batch 64 --out outputs/critic_data/${TAG} --seed $SEED
python -m hcfrl.rl.pretrain_critic --config $CFG --ckpt $CKPT --data outputs/critic_data/${TAG} \
    --out outputs/critic/${TAG}_2d --mode 2d --epochs 6 --seed $SEED --accum $ACC
[ -f outputs/critic/${TAG}_2d/log.json ] || { echo "critic pretraining failed"; exit 1; }
echo GATE0A_DONE
