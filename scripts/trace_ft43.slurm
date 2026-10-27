#!/bin/bash
# Token-norm / consistency trace along the fine-tune-seed-43 RL runs (and hier seed-42 s1 for comparison):
# does the drifting hier seed-43 model lose code-token consistency, and how does its token norm compare?
#SBATCH --job-name=trace-ft43
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
mkdir -p outputs/analysis
python -m hcfrl.analysis.consistency --mode trace --config configs/helf_b_game24_cd.yml --ckpt outputs/helf_b-game24-cd-s43/checkpoint_2800 \
    --prompts data/game24/val.jsonl --n 256 --ks 24,40,63 \
    --runs outputs/rl/game24cd_hier_ft43_svg2d_s0,outputs/rl/game24cd_hier_svg2d_s1 \
    --out outputs/analysis/cons_trace_ft43_hier.json
python -m hcfrl.analysis.consistency --mode trace --config configs/elf_b_game24_cd.yml --ckpt outputs/elf_b-game24-cd-s43/checkpoint_2800 \
    --prompts data/game24/val.jsonl --n 256 --ks 24,40,63 \
    --runs outputs/rl/game24cd_flat_ft43_svg2d_s0 \
    --out outputs/analysis/cons_trace_ft43_flat.json
echo TRACE_DONE
