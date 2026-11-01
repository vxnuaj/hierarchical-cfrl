#!/bin/bash
# Per-prompt difficulty screen: decode + verify <repeat> rollouts for every prompt of a prompt file (no states).
#   sbatch scripts/screen_p100.slurm <config> <ckpt> <prompts.jsonl> <out_dir> [repeat]
#SBATCH --job-name=screen
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --exclude=e22-[01-16],d23-14,e21-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
export HCFRL_FP32=1
set -x
python -m hcfrl.rl.collect --config $1 --ckpt $2 --prompts $3 --n 1000000 --repeat ${5:-2} --K 64 --gamma 56 --batch 128 --out $4 --no_states
[ -f $4/summary.txt ] || { echo "screen failed"; exit 1; }
echo SCREEN_DONE
