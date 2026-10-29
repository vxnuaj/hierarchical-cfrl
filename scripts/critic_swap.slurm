#!/bin/bash
# Critic-swap 2x2 (is the fragile part the policy or the critic?). Trains a critic whose frozen-at-RL trunk comes from
# one base model on the rollouts of ANOTHER policy (both ELF-B flat, same architecture and T5 latent space).
#   sbatch scripts/critic_swap.slurm <trunk_config> <trunk_tag> <policy_tag> <out_tag>
#   e.g. AdamW trunk on the Muon policy's rollouts:
#   sbatch scripts/critic_swap.slurm configs/elfA_b_game24_cd.yml game24cdA_flat game24cd_flat game24cd_swapMA
# Output: outputs/critic/<out_tag>_2d/critic_ep*.pt. Then RL with the policy's own config and checkpoint and
# --critic <that file> (train_svg loads the stored trunk into the critic).
#SBATCH --job-name=swap
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
TCFG=$1; TTAG=$2; PTAG=$3; OUT=$4; EXTRA="${@:5}"   # further args go to pretrain_critic, e.g. --epochs 12 --lr_trunk 3e-5
TCKPT=$(sed 's/^CKPT=//' outputs/eval/${TTAG}_ckpt.txt)
python -m hcfrl.rl.pretrain_critic --config $TCFG --ckpt $TCKPT --data outputs/critic_data/${PTAG} \
    --out outputs/critic/${OUT}_2d --mode 2d --epochs 6 --seed 0 $EXTRA
echo SWAP_DONE
