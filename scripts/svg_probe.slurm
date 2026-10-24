#!/bin/bash
# Short SVG diagnostic on the debug partition (1h). Extra args are passed to train_svg.
#   sbatch scripts/svg_probe.slurm <config> <sft_ckpt_dir> <task_dir> <critic_tag> <out_tag> [train_svg args...]
#SBATCH --job-name=svg-probe
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
CFG=$1; CKPT=$2; TASK=$3; CTAG=$4; OTAG=$5; shift 5
[ -f outputs/eval/${CTAG}_ckpt.txt ] && CKPT=$(sed 's/^CKPT=//' outputs/eval/${CTAG}_ckpt.txt)
BEST=$(python -c "import json; print(json.load(open('outputs/critic/${CTAG}_2d/log.json'))['best_epoch'])")
set -x
python -m hcfrl.rl.train_svg --config $CFG --ckpt $CKPT --critic outputs/critic/${CTAG}_2d/critic_ep${BEST}.pt \
    --train_prompts $TASK/train.jsonl --val_prompts $TASK/val.jsonl \
    --out outputs/rl/${OTAG} --K 64 --gamma 56 --save_every 0 "$@"
echo PROBE_DONE
