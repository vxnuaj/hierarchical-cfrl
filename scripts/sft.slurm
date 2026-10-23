#!/bin/bash
# usage: sbatch scripts/sft.slurm configs/elf_b_game24.yml [extra --config_override k=v ...]
#SBATCH --job-name=hcfrl-sft
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; shift
python hcfrl/elf/train.py --config $CFG --config_override num_workers=4 "$@"
