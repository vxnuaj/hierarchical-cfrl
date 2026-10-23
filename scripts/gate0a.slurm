#!/bin/bash
# Gate 0, part A (fits a 1h debug slot): SFT accuracy eval, rollout collection, critic pretraining.
#   sbatch scripts/gate0a.slurm <config> <sft_ckpt_dir> <task_dir> <tag> [seed] [K] [gamma]
#SBATCH --job-name=hcfrl-gate0a
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; CKPT=$2; TASK=$3; TAG=$4; SEED=${5:-0}; K=${6:-64}; GAMMA=${7:-56}; SELECT_EVERY=${8:-0}
set -x
if [ "$SELECT_EVERY" != "0" ]; then  # pick the best SFT checkpoint by val accuracy (CFRL: Game24 peaks mid-training)
  python -m hcfrl.rl.select_ckpt --config $CFG --dir $CKPT --prompts $TASK/val.jsonl --n 600 --K $K --gamma $GAMMA \
      --every $SELECT_EVERY --out outputs/eval/${TAG}_select
  CKPT=$(cat outputs/eval/${TAG}_select/best_ckpt)
fi
echo "CKPT=$CKPT" > outputs/eval/${TAG}_ckpt.txt
python -m hcfrl.rl.collect --config $CFG --ckpt $CKPT --prompts $TASK/test.jsonl --n 100000 --K $K --gamma $GAMMA \
    --batch 256 --out outputs/eval/${TAG}_sft_test --no_states
python -m hcfrl.rl.collect --config $CFG --ckpt $CKPT --prompts $TASK/train_prompts.jsonl --n 2000 --repeat 4 \
    --K $K --gamma $GAMMA --steps 1,8,16,24,32,40,48 --batch 128 --out outputs/critic_data/${TAG} --seed $SEED
python -m hcfrl.rl.pretrain_critic --config $CFG --ckpt $CKPT --data outputs/critic_data/${TAG} \
    --out outputs/critic/${TAG}_2d --mode 2d --epochs 6 --seed $SEED
echo GATE0A_DONE
