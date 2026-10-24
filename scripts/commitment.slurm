#!/bin/bash
# Token-commitment profiles for all collected critic datasets (debug partition).
#SBATCH --job-name=commitment
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
run() { # cfg tag
  CKPT=$(sed 's/^CKPT=//' outputs/eval/$2_ckpt.txt 2>/dev/null || true); [ -z "$CKPT" ] && CKPT=$3
  python -m hcfrl.analysis.commitment --config $1 --ckpt $CKPT --data outputs/critic_data/$2 --out outputs/analysis/commit_$2.json
}
mkdir -p outputs/analysis
run configs/elf_b_kk.yml kk_flat
run configs/helf_b_kk.yml kk_hier
run configs/elf_b_game24.yml game24_flat outputs/elf_b-game24
run configs/helf_b_game24.yml game24_hier outputs/helf_b-game24
echo COMMIT_DONE
