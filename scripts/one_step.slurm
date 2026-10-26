#!/bin/bash
# One step along the pathwise gradient from the SFT base (CFRL diagnostic), flat vs hier.
#   sbatch scripts/one_step.slurm <config> <ckpt> <critic.pt> <svg_run_dir> <task_dir> <out_json> [more groups of 6]
#SBATCH --job-name=hcfrl-onestep
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
while [ $# -ge 6 ]; do
  python -m hcfrl.analysis.one_step $ONESTEP_EXTRA --config $1 --ckpt $2 --critic $3 --run $4 --train_prompts $5/train_prompts.jsonl --val_prompts $5/val.jsonl --out $6
  shift 6
done
echo ONESTEP_DONE
