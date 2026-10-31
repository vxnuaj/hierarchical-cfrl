#!/bin/bash
# Generic 1-hour analysis job on the debug partition: sbatch -J <name> scripts/analysis.slurm <python module> [args...]
#SBATCH --job-name=analysis
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
mkdir -p outputs/analysis
MOD=$1; shift
python -m $MOD "$@"
