#!/bin/bash
# Fallback for a gate0a that timed out during critic pre-training: re-run only pretrain_critic
# on the collected rollouts (outputs/critic_data/<tag>), on the gpu partition with a 3h limit.
#   sbatch scripts/critic_only.slurm <config> <tag> [seed]
#SBATCH --job-name=hcfrl-critic
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; TAG=$2; SEED=${3:-0}
CKPT=$(sed 's/^CKPT=//' outputs/eval/${TAG}_ckpt.txt)
set -x
python -m hcfrl.rl.pretrain_critic --config $CFG --ckpt $CKPT --data outputs/critic_data/${TAG} \
    --out outputs/critic/${TAG}_2d --mode 2d --epochs 6 --seed $SEED
echo CRITIC_DONE
