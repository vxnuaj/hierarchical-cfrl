#!/bin/bash
# Test accuracy of a supervised base under the deployment sampler (no RL).
#   sbatch scripts/eval_base.slurm <config> <ckpt> <task_dir> <out_name>
#SBATCH --job-name=eval-base
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
python -m hcfrl.rl.collect --config $1 --ckpt $2 --prompts $3/test.jsonl --n 100000 --K 64 --gamma 56 --batch 256 --out outputs/eval/$4 --no_states
echo EVAL_DONE
