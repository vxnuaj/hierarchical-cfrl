#!/bin/bash
# Pathwise-gradient quality (SNR, half-split cosine, cluster-channel share) at the SFT policy.
#   sbatch scripts/grad_snr.slurm <config> <ckpt> <critic.pt> <prompts> <out_json> [more groups of 5]
#SBATCH --job-name=hcfrl-gsnr
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
  python -m hcfrl.analysis.grad_snr --config $1 --ckpt $2 --critic $3 --prompts $4 --n 256 --ks 8,24,40 --out $5
  shift 5
done
echo GSNR_DONE
