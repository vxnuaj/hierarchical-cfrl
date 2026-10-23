#!/bin/bash
# Self-resubmitting fine-tune chunk for the 1-hour debug partition (works around the per-user
# submitted-jobs cap). Runs train.py (auto-resumes from output_dir) under a 52-minute timeout;
# if it was cut off, resubmits itself. Finishes when train.py exits 0 on its own.
#   sbatch scripts/sft_chunk.slurm <config> [extra --config_override k=v ...]
#SBATCH --job-name=sft-chunk
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; shift
timeout 52m python hcfrl/elf/train.py --config $CFG --config_override num_workers=4 "$@"
rc=$?
if [ $rc -eq 0 ]; then
  echo "SFT_COMPLETE $CFG"
elif [ $rc -eq 124 ]; then
  echo "chunk timed out; resubmitting"
  sbatch --job-name=$SLURM_JOB_NAME scripts/sft_chunk.slurm $CFG "$@"
else
  echo "SFT_FAILED rc=$rc"; exit $rc
fi
