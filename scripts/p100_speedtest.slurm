#!/bin/bash
# Speed test of RL (ELF-B) in fp32 on a P100 (HCFRL_FP32=1 disables bf16 autocast).
#SBATCH --job-name=p100-speed
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
export HCFRL_FP32=1
set -x
python -m hcfrl.rl.train_svg --config configs/elf_b_game24_cd.yml --ckpt outputs/elf_b-game24-cd/checkpoint_2450 \
  --critic outputs/critic/game24cd_flat_2d/critic_ep6.pt --train_prompts data/game24/train.jsonl --val_prompts data/game24/val.jsonl \
  --out outputs/rl/p100_speedtest --K 64 --gamma 56 --iters 40 --eval_every 1000 --seed 0 --bf16 0 --buf_dtype bf16
