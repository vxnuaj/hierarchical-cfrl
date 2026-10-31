#!/bin/bash
# Generic CPU analysis job (main partition): sbatch -J <name> scripts/analysis_cpu.slurm <python module> [args...]
#SBATCH --job-name=analysis-cpu
#SBATCH --account=irahbari_832
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
mkdir -p outputs/analysis
export OMP_NUM_THREADS=16
MOD=$1; shift
python -m $MOD "$@"
