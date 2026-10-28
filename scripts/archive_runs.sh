#!/bin/bash
# Mirror the lightweight record of every run from the cluster into the repo (runs/), then build runs/manifest.json.
# Checkpoints are NOT included (see scripts/backup_checkpoints.sh); everything needed to re-plot and audit is.
#   bash scripts/archive_runs.sh
set -e
cd "$(dirname "$0")/.."
R="carc:/scratch1/verajuan/hcfrl/outputs"
SSH="ssh -o ControlPath=none -o ConnectTimeout=20"
mkdir -p runs/rl runs/analysis runs/eval runs/critic runs/slurm
rsync -az -e "$SSH" --include='*/' --include='log.jsonl' --exclude='*' "$R/rl/" runs/rl/
rsync -az -e "$SSH" --exclude "*.pt" --exclude "state.pt*" "$R/analysis/" runs/analysis/
rsync -az -e "$SSH" --include='*/' --include='*.txt' --include='*.json' --include='*.jsonl' --exclude='*' "$R/eval/" runs/eval/
rsync -az -e "$SSH" --include='*/' --include='log.json' --exclude='*' "$R/critic/" runs/critic/
rsync -az -e "$SSH" "$R/slurm/" runs/slurm/
# compress slurm logs (tqdm output) in place
find runs/slurm -name '*.out' -exec gzip -f {} \;
python3 scripts/build_manifest.py
echo "archived: $(ls runs/rl | wc -l) rl runs, $(ls runs/slurm | wc -l) slurm logs"
