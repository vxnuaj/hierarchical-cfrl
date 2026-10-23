#!/bin/bash
# Gate 0 pipeline for one task + one model: SFT accuracy eval, critic data collection, critic
# pretraining (2D), then CFRL-SVG RL. Usage:
#   sbatch scripts/gate0.slurm <config> <sft_ckpt_dir> <task_dir> <tag> [seed] [K] [gamma]
# e.g. sbatch scripts/gate0.slurm configs/elf_b_game24.yml outputs/elf_b-game24 data/game24 game24_flat 0 64 56
#SBATCH --job-name=hcfrl-gate0
#SBATCH --account=irahbari_832
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; CKPT=$2; TASK=$3; TAG=$4; SEED=${5:-0}; K=${6:-64}; GAMMA=${7:-56}
set -x
# 1) base accuracy of the SFT policy under the deployment sampler (val + test)
python -m hcfrl.rl.collect --config $CFG --ckpt $CKPT --prompts $TASK/test.jsonl --n 100000 --K $K --gamma $GAMMA \
    --batch 256 --out outputs/eval/${TAG}_sft_test --no_states
# 2) rollouts of the SFT policy on train prompts for critic pretraining (states at several steps)
python -m hcfrl.rl.collect --config $CFG --ckpt $CKPT --prompts $TASK/train_prompts.jsonl --n 4000 --repeat 4 \
    --K $K --gamma $GAMMA --steps 1,8,16,24,32,40,48 --batch 128 --out outputs/critic_data/${TAG} --seed $SEED
# 3) critic pretraining + the critic-quality-vs-step curve
python -m hcfrl.rl.pretrain_critic --config $CFG --ckpt $CKPT --data outputs/critic_data/${TAG} \
    --out outputs/critic/${TAG}_2d --mode 2d --epochs 6 --seed $SEED
BEST=$(python -c "import json;print(json.load(open('outputs/critic/${TAG}_2d/log.json'))['best_epoch'])")
# 4) RL
python -m hcfrl.rl.train_svg --config $CFG --ckpt $CKPT --critic outputs/critic/${TAG}_2d/critic_ep${BEST}.pt \
    --train_prompts $TASK/train.jsonl --val_prompts $TASK/val.jsonl --test_prompts $TASK/test.jsonl \
    --out outputs/rl/${TAG}_svg2d_s${SEED} --K $K --gamma $GAMMA --iters 1500 --seed $SEED
