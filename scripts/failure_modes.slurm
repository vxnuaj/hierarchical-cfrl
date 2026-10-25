#!/bin/bash
# Failure-mode profile of the saved RL policies (base + every policy_it*.pt) on held-out prompts.
#   sbatch scripts/failure_modes.slurm <config> <ckpt> <run_dir> <prompts> <out_json>  (repeatable in groups of 5)
#SBATCH --job-name=hcfrl-fm
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
while [ $# -ge 5 ]; do
  python -m hcfrl.analysis.failure_modes --config $1 --ckpt $2 --run $3 --prompts $4 --n 500 --out $5
  shift 5
done
echo FM_DONE
