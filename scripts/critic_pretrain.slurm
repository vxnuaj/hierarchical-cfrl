#!/bin/bash
# Critic pretraining only (the last stage of gate0a.slurm), for when collection is done but pretraining must be rerun,
# e.g. with gradient accumulation for ELF-M on 16 GB P100s.
#   sbatch -p gpu --gres=gpu:p100:1 --time=2-00:00:00 --export=ALL,HCFRL_FP32=1 scripts/critic_pretrain.slurm <config> <tag> [seed] [extra pretrain args]
#SBATCH --job-name=critic-pretrain
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; TAG=$2; SEED=${3:-0}; EXTRA="${@:4}"
CKPT=$(sed 's/^CKPT=//' outputs/eval/${TAG}_ckpt.txt)
set -x
python -m hcfrl.rl.pretrain_critic --config $CFG --ckpt $CKPT --data outputs/critic_data/${TAG} \
    --out outputs/critic/${TAG}_2d --mode 2d --epochs 6 --seed $SEED $EXTRA
[ -f outputs/critic/${TAG}_2d/log.json ] || { echo "critic pretraining failed"; exit 1; }
echo CRITIC_DONE
