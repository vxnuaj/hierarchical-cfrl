#!/bin/bash
#SBATCH --job-name=hcfrl-smoke
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
python scripts/smoke_elf.py --device cuda --K 64 --gamma 56
