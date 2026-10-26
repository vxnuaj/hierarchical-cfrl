#!/bin/bash
# Coarse-rail conjecture tests (paper Sec. 4.5): critic probe (flat + hier) and consistency trace along RL runs.
#   sbatch scripts/consistency.slurm probe     (~15 min)
#   sbatch scripts/consistency.slurm trace     (~45 min)
#SBATCH --job-name=cons
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
MODE=${1:-probe}
mkdir -p outputs/analysis
FLAT=$(sed 's/^CKPT=//' outputs/eval/game24cd_flat_ckpt.txt); HIER=$(sed 's/^CKPT=//' outputs/eval/game24cd_hier_ckpt.txt)
best() { python - "$1" <<'EOF'
import glob, sys, torch
d = sys.argv[1]
f = max(glob.glob(f"{d}/critic_ep*.pt"), key=lambda f: torch.load(f, map_location="cpu")["eval"]["field_score"])
print(f)
EOF
}
if [ "$MODE" = "probe" ]; then
  python -m hcfrl.analysis.consistency --mode probe --config configs/elf_b_game24_cd.yml --ckpt $FLAT \
      --critic $(best outputs/critic/game24cd_flat_2d) --prompts data/game24/val.jsonl --n 256 --out outputs/analysis/cons_probe_flat.json
  python -m hcfrl.analysis.consistency --mode probe --config configs/helf_b_game24_cd.yml --ckpt $HIER \
      --critic $(best outputs/critic/game24cd_hier_2d) --prompts data/game24/val.jsonl --n 256 --out outputs/analysis/cons_probe_hier.json
  python -m hcfrl.analysis.consistency --mode probe --config configs/helf_b_game24_cd.yml --ckpt $HIER \
      --critic $(best outputs/critic/game24cd_hier_blind_2d) --prompts data/game24/val.jsonl --n 256 --out outputs/analysis/cons_probe_hier_blind.json
elif [ "$MODE" = "trace" ]; then
  python -m hcfrl.analysis.consistency --mode trace --config configs/helf_b_game24_cd.yml --ckpt $HIER \
      --prompts data/game24/val.jsonl --n 256 --ks 24,40,63 \
      --runs outputs/rl/game24cd_hier_svg2d_s0,outputs/rl/game24cd_hier_svg2d_clgrad_s0,outputs/rl/game24cd_hier_svg2d_tokgrad_s0,outputs/rl/game24cd_hier_blind_svg2d_s0,outputs/rl/game24cd_hier_bigcrit_svg2d_s0 \
      --out outputs/analysis/cons_trace.json
elif [ "$MODE" = "survive" ]; then
  python -m hcfrl.analysis.consistency --mode survive --config configs/elf_b_game24_cd.yml --ckpt $FLAT \
      --critic $(best outputs/critic/game24cd_flat_2d) --prompts data/game24/val.jsonl --n 256 --ks 24,40 --eps 0.05,0.1 \
      --out outputs/analysis/cons_survive_flat.json
  python -m hcfrl.analysis.consistency --mode survive --config configs/helf_b_game24_cd.yml --ckpt $HIER \
      --critic $(best outputs/critic/game24cd_hier_2d) --prompts data/game24/val.jsonl --n 256 --ks 24,40 --eps 0.05,0.1 \
      --out outputs/analysis/cons_survive_hier.json
else  # trace2: flat runs (token norm only) + the hier runs cut by the limit
  python -m hcfrl.analysis.consistency --mode trace --config configs/elf_b_game24_cd.yml --ckpt $FLAT \
      --prompts data/game24/val.jsonl --n 256 --ks 24,40,63 \
      --runs outputs/rl/game24cd_flat_svg2d_s0,outputs/rl/game24cd_flat_bigcrit_svg2d_s0 \
      --out outputs/analysis/cons_trace_flat.json
  python -m hcfrl.analysis.consistency --mode trace --config configs/helf_b_game24_cd.yml --ckpt $HIER \
      --prompts data/game24/val.jsonl --n 256 --ks 24,40,63 \
      --runs outputs/rl/game24cd_hier_bigcrit_svg2d_s0,outputs/rl/game24cd_hier_blind_svg2d_s0 \
      --out outputs/analysis/cons_trace2.json
fi
echo CONS_DONE
