#!/bin/bash
# Revised rail conjecture test: displacement of the clean estimate along the critic pull per normalised actor step,
# flat vs hier, Game24-cd (debug partition, ~30 min).
#SBATCH --job-name=jac
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
mkdir -p outputs/analysis
FLAT=$(sed 's/^CKPT=//' outputs/eval/game24cd_flat_ckpt.txt); HIER=$(sed 's/^CKPT=//' outputs/eval/game24cd_hier_ckpt.txt)
best() { python - "$1" <<'EOF'
import glob, sys, torch
d = sys.argv[1]
print(max(glob.glob(f"{d}/critic_ep*.pt"), key=lambda f: torch.load(f, map_location="cpu")["eval"]["field_score"]))
EOF
}
python -m hcfrl.analysis.jacobian_step --config configs/elf_b_game24_cd.yml --ckpt $FLAT --critic $(best outputs/critic/game24cd_flat_2d) \
    --run outputs/rl/game24cd_flat_svg2d_s0 --train_prompts data/game24/train_prompts.jsonl --val_prompts data/game24/val.jsonl \
    --out outputs/analysis/jac_game24cd_flat.json
python -m hcfrl.analysis.jacobian_step --config configs/helf_b_game24_cd.yml --ckpt $HIER --critic $(best outputs/critic/game24cd_hier_2d) \
    --run outputs/rl/game24cd_hier_svg2d_s0 --train_prompts data/game24/train_prompts.jsonl --val_prompts data/game24/val.jsonl \
    --out outputs/analysis/jac_game24cd_hier.json
echo JAC_DONE
